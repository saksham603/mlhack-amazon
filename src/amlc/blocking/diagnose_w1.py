"""Why are true pairs missed by blocking? (user-approved diagnosis, 2026-09-26, before char-3gram.)

On the same 1% FIT S1 sample as dryrun_w1 (seed 20260925), at the most generous tried setting
(cap=5000, k=100), every FIT true link of a sampled S1 is put in exactly one class:
  hit          in that S1's top-k candidates
  ranked_out   generated as a candidate, but ranked below top-k (a ranking problem)
  cap_dropped  shares >=1 key with its S1, but every shared key is too common (df > cap)
  no_key       shares no key at all (name token, house|address token, ZIP) -- only fuzzy
               matching such as char-3grams can reach these
For no_key pairs, the char-3gram Jaccard of the two names shows whether char-3grams could help.
FIT labels only (G-L1). Saved in the repo so its numbers can be re-derived (CG-12).
"""
import json
import sys
import time
from collections import Counter

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access

CAP = 5000
K = 100
JACCARD_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01)


def char3(s: str) -> set[str]:
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def jaccard(a: str, b: str) -> float:
    x, y = char3(a), char3(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def all_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return pl.concat([
        C._name_tokens(df, id_col).with_columns(pl.lit("name").alias("key_type")),
        C.house_street_keys(df, id_col).with_columns(pl.lit("house").alias("key_type")),
        C.zip_keys(df, id_col).with_columns(pl.lit("zip").alias("key_type")),
    ], how="vertical")


def diagnose(country: str) -> dict:
    t0 = time.time()
    s1 = C.sample_fit_s1(country)
    s23 = C.load_s23_pool(country).rename({"gid": "s23_gid"})
    truth = (access.load_labels("FIT").select("s1_gid", "s23_gid")
             .join(s1.select("s1_gid"), on="s1_gid", how="semi"))

    s1_keys = all_keys(s1, "s1_gid")
    s23_keys = all_keys(s23, "s23_gid")
    df = C.document_frequency(s23_keys, "s23_gid")

    # keys shared by each TRUE pair (restrict the S2/S3 side to linked records first: cheap)
    s23_true_keys = s23_keys.join(truth.select("s23_gid").unique(), on="s23_gid", how="semi")
    shared = (truth.join(s1_keys, on="s1_gid", how="inner")
              .join(s23_true_keys.select("s23_gid", "key"), on=["s23_gid", "key"], how="inner")
              .unique(["s1_gid", "s23_gid", "key"])
              .join(df, on="key", how="left", validate="m:1"))
    per_pair = shared.group_by("s1_gid", "s23_gid").agg(
        pl.len().alias("n_shared"), (pl.col("df") <= CAP).sum().alias("n_rare"), pl.col("df").min().alias("min_df"))

    scored = C.candidates_for_cap(s1, s23, CAP, 6 * 2**30)
    topk = C.candidates_topk(scored, K)
    t = (truth.join(per_pair, on=["s1_gid", "s23_gid"], how="left")
         .join(scored.select("s1_gid", "s23_gid").with_columns(pl.lit(True).alias("scored")),
               on=["s1_gid", "s23_gid"], how="left")
         .join(topk.select("s1_gid", "s23_gid").with_columns(pl.lit(True).alias("in_topk")),
               on=["s1_gid", "s23_gid"], how="left")
         .with_columns(pl.col("n_shared", "n_rare").fill_null(0),
                       pl.col("scored", "in_topk").fill_null(False)))
    t = t.with_columns(
        pl.when(pl.col("in_topk")).then(pl.lit("hit"))
        .when(pl.col("scored")).then(pl.lit("ranked_out"))
        .when(pl.col("n_shared") > 0).then(pl.lit("cap_dropped"))
        .otherwise(pl.lit("no_key")).alias("cls"))
    if t.height != truth.height:
        raise AssertionError("diagnosis gained or lost true pairs")
    counts = dict(t.group_by("cls").len().iter_rows())

    # names for cap_dropped / no_key pairs
    names1 = s1.select("s1_gid", pl.col("name_clean").alias("n1"), pl.col("name_indic_untranslated").alias("u1"))
    names2 = s23.select("s23_gid", pl.col("name_clean").alias("n2"), pl.col("name_indic_untranslated").alias("u2"),
                        pl.col("addr_clean").alias("a2"))
    detail = (t.filter(pl.col("cls").is_in(["no_key", "cap_dropped", "ranked_out"]))
              .join(names1, on="s1_gid", how="left", validate="m:1")
              .join(names2, on="s23_gid", how="left", validate="m:1"))
    nk = detail.filter(pl.col("cls") == "no_key")
    jac = [jaccard(a, b) for a, b in nk.select("n1", "n2").iter_rows()]
    bins = Counter()
    for j in jac:
        for lo, hi in zip(JACCARD_BINS, JACCARD_BINS[1:]):
            if lo <= j < hi:
                bins[f"[{lo},{hi})"] += 1
                break

    # which too-common keys caused cap_dropped (top 25)
    cd_pairs = t.filter(pl.col("cls") == "cap_dropped").select("s1_gid", "s23_gid")
    common = (shared.join(cd_pairs, on=["s1_gid", "s23_gid"], how="semi")
              .group_by("key", "key_type").agg(pl.len().alias("pairs"), pl.col("df").first())
              .sort("pairs", descending=True).head(25))

    return {
        "country": country, "n_true_links": truth.height, "classes": counts,
        "shares": {k: v / truth.height for k, v in counts.items()},
        "no_key_name_char3_jaccard_bins": dict(sorted(bins.items())),
        "no_key_untranslated_indic_share": {
            "s1": float(nk["u1"].mean()) if nk.height else None,
            "s23": float(nk["u2"].mean()) if nk.height else None},
        "cap_dropped_top_keys": common.to_dicts(),
        "examples": {c: detail.filter(pl.col("cls") == c).head(15).select("n1", "n2", "a2").to_dicts()
                     for c in ("no_key", "cap_dropped", "ranked_out")},
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for country in sys.argv[1:] or ["India", "US"]:
        check_ram_budget()
        print(f"=== {country} (free RAM {free_ram_mb()} MB)", flush=True)
        out[country] = r = diagnose(country)
        print(json.dumps({k: r[k] for k in ("n_true_links", "classes", "shares",
                                             "no_key_name_char3_jaccard_bins", "no_key_untranslated_indic_share",
                                             "seconds")}, ensure_ascii=False, indent=1), flush=True)
    out["peak_ram_mb"] = peak_ram_mb()
    (OUT_DIR / "w1_diagnosis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    print(f"peak RAM {out['peak_ram_mb']} MB; written {OUT_DIR / 'w1_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
