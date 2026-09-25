"""Split v2: same FIT/VALIDATION/LOCKBOX assignment as v1, stress flag drawn from FIT only.

The assignment reuses v1's SEED, RATIOS, bucket() and hashkey() so it cannot drift; the build
proves it is identical to v1 row by row. Every check here returns errors instead of asserting,
so the build can run deliberate "must fail" negative tests against corrupted copies.
"""
from pathlib import Path

import polars as pl

from amlc.foundation.f7_split import RATIOS, SEED, bucket, hashkey

STRATUM_TOL_PP = 0.1
STRESS_RATIO_TOL = 0.01
SPLIT_V2_COLUMNS = ["s1_gid", "s1_id", "country", "split", "stress_hidden"]
LABEL_DERIVED_COLUMNS = {"bucket", "n_total", "n_s2", "n_s3", "is_singleton"}


def build_assignment(mc: pl.DataFrame) -> pl.DataFrame:
    """Identical algorithm to v1 f7_split.main(): stratify by (country, bucket), sort by
    sha256(SEED|split|s1_id), first floor(0.7n) FIT, next floor(0.2n) VALIDATION, rest LOCKBOX."""
    df = mc.select(["s1_gid", "s1_id", "country", "n_total"]).with_columns(
        pl.col("n_total").map_elements(bucket, return_dtype=pl.Utf8).alias("bucket"),
        pl.col("s1_id").map_elements(lambda x: hashkey("split", x), return_dtype=pl.Utf8).alias("split_key"),
    )
    parts = []
    for _, grp in df.group_by(["country", "bucket"]):
        grp = grp.sort("split_key")
        n = grp.height
        n_fit = int(n * RATIOS["FIT"])
        n_val = int(n * RATIOS["VALIDATION"])
        labels = ["FIT"] * n_fit + ["VALIDATION"] * n_val + ["LOCKBOX"] * (n - n_fit - n_val)
        parts.append(grp.with_columns(pl.Series("split", labels)))
    return pl.concat(parts).drop("split_key")


def assignment_diffs(new: pl.DataFrame, reference: pl.DataFrame) -> int:
    """Number of S1 whose split differs from the reference, or which exist on only one side."""
    j = new.select(["s1_id", "split"]).join(
        reference.select(["s1_id", pl.col("split").alias("split_ref")]),
        on="s1_id", how="full", coalesce=True,
    )
    return j.filter(
        pl.col("split").is_null() | pl.col("split_ref").is_null() | (pl.col("split") != pl.col("split_ref"))
    ).height


def country_counts(bronze_dir: Path, dataset: str) -> tuple[dict, dict]:
    """(S1 count per country, S2+S3 pool count per country), computed from bronze."""
    n1 = pl.read_parquet(bronze_dir / f"{dataset}_source1.parquet", columns=["country"]).group_by("country").len()
    pool = pl.concat([
        pl.read_parquet(bronze_dir / f"{dataset}_source{s}.parquet", columns=["country"]) for s in (2, 3)
    ]).group_by("country").len()
    return dict(zip(n1["country"], n1["len"])), dict(zip(pool["country"], pool["len"]))


def stress_params(train_n1: dict, train_pool: dict, test_n1: dict, test_pool: dict) -> tuple[dict, list]:
    """Per train country (open set, never a literal list): how many S1 to hide so the visible
    train pool ratio equals the test pool ratio. Countries absent from test are skipped and reported."""
    params, notes = {}, []
    for c in sorted(train_n1):
        if c not in test_n1 or c not in test_pool:
            notes.append(f"country {c!r} has train S1 but no test S1/pool: no stress target, skipped")
            continue
        r_train = train_pool.get(c, 0) / train_n1[c]
        r_test = test_pool[c] / test_n1[c]
        f = max(0.0, 1 - r_train / r_test)
        params[c] = {
            "n_s1_train": train_n1[c], "pool_train": train_pool.get(c, 0),
            "ratio_train": r_train, "ratio_test": r_test,
            "hide_fraction": f, "n_hide": round(f * train_n1[c]),
        }
    return params, notes


def build_stress(assign: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Hide the n_hide FIT S1 of each country with the lowest sha256(SEED|stress|s1_id)."""
    hidden = []
    for c, p in params.items():
        fit_c = assign.filter((pl.col("country") == c) & (pl.col("split") == "FIT")).select("s1_id").with_columns(
            pl.col("s1_id").map_elements(lambda x: hashkey("stress", x), return_dtype=pl.Utf8).alias("k")
        ).sort("k")
        if p["n_hide"] > fit_c.height:
            raise ValueError(f"{c}: need to hide {p['n_hide']} but FIT has only {fit_c.height}")
        hidden.extend(fit_c.head(p["n_hide"])["s1_id"].to_list())
    return assign.with_columns(pl.col("s1_id").is_in(pl.Series(hidden, dtype=pl.Utf8)).alias("stress_hidden"))


def check_stress(split_df: pl.DataFrame, params: dict) -> tuple[list, dict]:
    """Independent recount from the finished table (not a restatement of the formula)."""
    errs, meas = [], {}
    outside = split_df.filter(pl.col("stress_hidden") & (pl.col("split") != "FIT"))
    meas["hidden_outside_FIT"] = outside.height
    if outside.height:
        errs.append(f"{outside.height} stress-hidden S1 are not in FIT: {outside.group_by('split').len().to_dicts()}")
    for c in split_df.filter(pl.col("stress_hidden"))["country"].unique().to_list():
        if c not in params:
            errs.append(f"S1 hidden in country {c!r} which has no stress target")
    for c, p in params.items():
        sub = split_df.filter(pl.col("country") == c)
        hid = int(sub["stress_hidden"].sum())
        vis = sub.height - hid
        achieved = p["pool_train"] / vis if vis else float("inf")
        meas[c] = {"hidden": hid, "visible": vis, "achieved_ratio": achieved, "test_ratio": p["ratio_test"]}
        if hid != p["n_hide"]:
            errs.append(f"{c}: hidden={hid} but target n_hide={p['n_hide']}")
        if abs(achieved - p["ratio_test"]) > STRESS_RATIO_TOL:
            errs.append(f"{c}: achieved ratio {achieved:.4f} vs test {p['ratio_test']:.4f} (tol {STRESS_RATIO_TOL})")
    return errs, meas


def check_strata(split_df: pl.DataFrame) -> tuple[list, float]:
    """Every (country, bucket) stratum within STRATUM_TOL_PP of 70/20/10."""
    errs, max_dev = [], 0.0
    for (c, b), g in split_df.group_by(["country", "bucket"]):
        n = g.height
        for s, target in RATIOS.items():
            dev = abs(100 * (g["split"] == s).sum() / n - 100 * target)
            max_dev = max(max_dev, dev)
            if dev > STRATUM_TOL_PP:
                errs.append(f"stratum ({c},{b}) n={n}: {s} off by {dev:.4f}pp")
    return errs, max_dev
