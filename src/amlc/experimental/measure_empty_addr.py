"""FIT-holdout go/no-go measurement for the empty-address exact-name candidate key (Task 2 of
docs/superpowers/plans/2026-09-27-empty-address-exact-name-key.md). Reproduces and refreshes the
2026-09-27 08:42 OVERNIGHT_LOG estimate (~175/7272 links recovered, ~2.4%) with a fresh run, and
makes the go/no-go call against the pre-declared kill bar (net F0.5 gain >= 0.0005).

Scores new candidates with the v3.1 model (lgbm_v31.txt, its own threshold 0.68) rather than
recomputing the full v3.2 feature set (numnoise + pseudo) for them -- v3.1's block-rank + context
features are fully recomputable standalone (see amlc.features.w3.pair_features, which derives
ctx_* live from whatever candidate set it's given), the same "not found" sentinel convention as
amlc.pipeline.run_v3.union() (score 0.0, rank 999) is used for the new key's missing v1/v3/va
features. This is a conservative approximation, not a full v3.2 rescoring, done to keep this
measurement's cost proportional to its (expected to be small) payoff -- flagged explicitly rather
than silently claiming a v3.2-equivalent number.

Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.measure_empty_addr
"""
import json
import time

import lightgbm as lgb
import polars as pl

from amlc.eval import metric
from amlc.experimental.empty_addr_key import empty_addr_candidates, empty_addr_keys
from amlc.foundation import access
from amlc.model.predict import enforce_g_m3
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet

HOLDOUT = R.OUT.with_name("_v31") / "exp" / "holdout_p1.parquet"
OUT_DIR = access.DATA / "_helper" / "exp_empty_addr"
KILL_BAR = 0.0005
V31_THRESHOLD = 0.68


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def new_pairs_only(candidate: pl.DataFrame, existing: pl.DataFrame) -> pl.DataFrame:
    """candidate, existing: (s1_gid, s23_gid, ...). -> candidate rows not present in existing."""
    return candidate.join(existing.select("s1_gid", "s23_gid"), on=["s1_gid", "s23_gid"], how="anti")


def _f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.Series) -> float:
    return metric.macro(metric.per_s1_f05(pred.select("s1_gid", "s23_gid"), truth, s1))


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = pl.read_parquet(HOLDOUT, columns=["s1_gid", "s23_gid", "country", "label", *R.FEATURES])
    s1_df = d.select("s1_gid", "country").unique("s1_gid")
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").join(s1_df.select("s1_gid"), on="s1_gid", how="semi")
    log(f"holdout: {d.height:,} rows, {s1_df.height:,} S1")

    # baseline: v3.1 model's own prediction at its threshold, using the label column already in d
    # (d's rows ARE v3.1's candidates; "prob" isn't cached here, so re-derive the prediction from
    # the FIT truth join the same way run_v3's own reporting does -- label==1 rows ARE the true
    # positives among v3.1's candidates, which is exactly what f05_base needs at the CANDIDATE level.
    # For a real threshold-based prediction we still need v3.1's prob; recompute it once.)
    booster = lgb.Booster(model_file=str(access.ROOT / "models" / "lgbm_v31.txt"))
    d = d.with_columns(pl.Series("p1", booster.predict(d.select([pl.col(f).cast(pl.Float32) for f in R.FEATURES]).to_numpy())))
    base_pred = enforce_g_m3(d.filter(pl.col("p1") >= V31_THRESHOLD).select("s1_gid", "s23_gid", pl.col("p1").alias("prob"))) \
        .select("s1_gid", "s23_gid")
    f05_base = _f05(base_pred, truth, s1_df["s1_gid"])
    log(f"f05_base (v3.1 model, existing candidates only): {f05_base:.5f}")

    # new candidates from the empty-address exact-name key, over the FIT record pool for these S1s
    new_parts = []
    for country, grp in s1_df.group_by("country"):
        country = country[0]
        r1 = pl.read_parquet(R.REC / "train_s1.parquet", columns=["gid", "country", "ns"]).filter(pl.col("country") == country) \
            .join(grp.select("s1_gid").rename({"s1_gid": "gid"}), on="gid", how="semi")
        r23 = pl.concat([pl.read_parquet(R.REC / f"train_s{s}.parquet", columns=["gid", "country", "addr", "ns"])
                         .filter(pl.col("country") == country) for s in (2, 3)])
        keys = empty_addr_keys(r23)
        cand = empty_addr_candidates(r1, keys)
        new_parts.append(cand)
    new_cand = pl.concat(new_parts) if new_parts else pl.DataFrame({"s1_gid": [], "s23_gid": []})
    new_cand = new_pairs_only(new_cand, d.select("s1_gid", "s23_gid"))
    log(f"new candidate pairs from empty-address key: {new_cand.height:,}")

    truth_in_new = truth.join(new_cand, on=["s1_gid", "s23_gid"], how="semi")
    n_true_recovered_as_candidate = truth_in_new.height
    log(f"true links newly reachable as candidates: {n_true_recovered_as_candidate:,}")

    # score the new pairs: recompute w3 features standalone (they need no cross-candidate context
    # beyond the pair itself for these purposes -- ctx_* is over the S1's OWN new-only set here,
    # a slight simplification vs. recomputing over existing+new together, flagged, not hidden),
    # v1/v3/va block features get the same "not found" sentinel amlc.pipeline.run_v3.union() uses.
    from amlc.features import w3
    if new_cand.height:
        by_country_new = []
        for country, grp in new_cand.join(s1_df.select("s1_gid", "country"), on="s1_gid").group_by("country"):
            country = country[0]
            r1f = pl.read_parquet(R.REC / "train_s1.parquet", columns=["gid", "country", *w3.FIELDS]).filter(pl.col("country") == country)
            r23f = pl.concat([pl.read_parquet(R.REC / f"train_s{s}.parquet", columns=["gid", "country", *w3.FIELDS])
                              .filter(pl.col("country") == country) for s in (2, 3)])
            pairs = grp.select("s1_gid", "s23_gid")
            feats = w3.pair_features(pairs, r1f, r23f)
            by_country_new.append(feats)
        new_feats = pl.concat(by_country_new).with_columns(
            pl.lit(0.0).alias("v1_score"), pl.lit(999, pl.UInt32).alias("v1_rank"),
            pl.lit(0.0).alias("v3_score"), pl.lit(999, pl.UInt32).alias("v3_rank"),
            pl.lit(0.0).alias("va_score"), pl.lit(999, pl.UInt32).alias("va_rank"))
        new_feats = new_feats.with_columns(
            pl.Series("p1", booster.predict(new_feats.select([pl.col(f).cast(pl.Float32) for f in R.FEATURES]).to_numpy())))
        n_new_above_threshold = new_feats.filter(pl.col("p1") >= V31_THRESHOLD).height
        union_pred_raw = pl.concat([
            d.select("s1_gid", "s23_gid", "p1"),
            new_feats.select("s1_gid", "s23_gid", "p1")])
        union_pred = enforce_g_m3(union_pred_raw.filter(pl.col("p1") >= V31_THRESHOLD).rename({"p1": "prob"})) \
            .select("s1_gid", "s23_gid")
    else:
        n_new_above_threshold = 0
        union_pred = base_pred
    f05_with_new_key = _f05(union_pred, truth, s1_df["s1_gid"])
    log(f"new candidates crossing v3.1 threshold: {n_new_above_threshold:,} / {new_cand.height:,}")
    log(f"f05_with_new_key: {f05_with_new_key:.5f}")

    # per-S1 complete coverage, existing-only vs existing+new
    n_true = truth.group_by("s1_gid").len(name="n_true")
    found_existing = d.filter(pl.col("label") == 1).group_by("s1_gid").len(name="n_found")
    all_cand_keys = pl.concat([d.select("s1_gid", "s23_gid"), new_cand]).unique()
    found_union = all_cand_keys.join(truth, on=["s1_gid", "s23_gid"], how="semi").group_by("s1_gid").len(name="n_found")
    cov_existing = s1_df.join(n_true, on="s1_gid", how="left").join(found_existing, on="s1_gid", how="left").fill_null(0)
    cov_union = s1_df.join(n_true, on="s1_gid", how="left").join(found_union, on="s1_gid", how="left").fill_null(0)
    cov_existing = cov_existing.with_columns((pl.col("n_found") >= pl.col("n_true")).alias("complete"))
    cov_union = cov_union.with_columns((pl.col("n_found") >= pl.col("n_true")).alias("complete"))
    complete_coverage_existing = float(cov_existing["complete"].mean())
    complete_coverage_union = float(cov_union["complete"].mean())

    net_gain = f05_with_new_key - f05_base
    rep = {
        "n_new_candidate_pairs": new_cand.height,
        "n_true_links_recovered_as_candidates": n_true_recovered_as_candidate,
        "n_true_links_recovered_and_above_threshold": None,  # filled below
        "n_new_candidates_above_threshold": n_new_above_threshold,
        "f05_base": f05_base,
        "f05_with_new_key": f05_with_new_key,
        "net_f05_gain": net_gain,
        "complete_coverage_existing": complete_coverage_existing,
        "complete_coverage_union": complete_coverage_union,
        "kill_bar": KILL_BAR,
        "go_no_go": "go" if net_gain >= KILL_BAR else "no_go",
        "note": "scored with v3.1 model (lgbm_v31.txt, threshold 0.68), not a full v3.2 rescoring -- "
                "see module docstring.",
        "minutes": round((time.time() - t0) / 60, 1),
    }
    if new_cand.height:
        recovered_above = truth_in_new.join(
            new_feats.filter(pl.col("p1") >= V31_THRESHOLD).select("s1_gid", "s23_gid"),
            on=["s1_gid", "s23_gid"], how="semi").height
        rep["n_true_links_recovered_and_above_threshold"] = recovered_above
    (OUT_DIR / "measure_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    main()
