"""FIT-holdout go/no-go measurement for Kaggle-GPU embedding-based auxiliary retrieval (mirrors
measure_empty_addr.py's methodology and kill bar). Consumes the candidates Kaggle produced
(data/_helper/kaggle_embed/embed_candidates.parquet: s1_gid, s23_gid, sim), scores the NEW ones
(not already in the existing v3.1 candidate set) with the v3.1 model using the same "not found"
sentinel convention as measure_empty_addr.py, and reports complementary recall + net F0.5 gain.

Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.measure_embed
"""
import json
import time

import lightgbm as lgb
import polars as pl

from amlc.eval import metric
from amlc.experimental.measure_empty_addr import _f05, new_pairs_only
from amlc.foundation import access
from amlc.model.predict import enforce_g_m3
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet

HOLDOUT = R.OUT.with_name("_v31") / "exp" / "holdout_p1.parquet"
CANDIDATES = access.DATA / "_helper" / "kaggle_embed" / "embed_candidates.parquet"
OUT_DIR = access.DATA / "_helper" / "exp_embed"
KILL_BAR = 0.0005
V31_THRESHOLD = 0.68


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[{CANDIDATES.name}]", flush=True)


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = pl.read_parquet(HOLDOUT, columns=["s1_gid", "s23_gid", "country", "label", *R.FEATURES])
    s1_df = d.select("s1_gid", "country").unique("s1_gid")
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").join(s1_df.select("s1_gid"), on="s1_gid", how="semi")
    log(f"holdout: {d.height:,} rows, {s1_df.height:,} S1")

    booster = lgb.Booster(model_file=str(access.ROOT / "models" / "lgbm_v31.txt"))
    d = d.with_columns(pl.Series("p1", booster.predict(d.select([pl.col(f).cast(pl.Float32) for f in R.FEATURES]).to_numpy())))
    base_pred = enforce_g_m3(d.filter(pl.col("p1") >= V31_THRESHOLD).select("s1_gid", "s23_gid", pl.col("p1").alias("prob"))) \
        .select("s1_gid", "s23_gid")
    f05_base = _f05(base_pred, truth, s1_df["s1_gid"])
    log(f"f05_base (v3.1 model, existing candidates only): {f05_base:.5f}")

    embed_cand = pl.read_parquet(CANDIDATES).select(
        pl.col("s1_gid").cast(pl.UInt32), pl.col("s23_gid").cast(pl.UInt32)).unique()
    embed_cand = embed_cand.join(s1_df.select("s1_gid"), on="s1_gid", how="semi")  # holdout S1s only
    new_cand = new_pairs_only(embed_cand, d.select("s1_gid", "s23_gid"))
    log(f"embedding candidates (holdout S1s): {embed_cand.height:,}, new (not already candidates): {new_cand.height:,}")

    truth_in_new = truth.join(new_cand, on=["s1_gid", "s23_gid"], how="semi")
    n_true_recovered_as_candidate = truth_in_new.height
    log(f"true links newly reachable as candidates: {n_true_recovered_as_candidate:,}")

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
        union_pred_raw = pl.concat([d.select("s1_gid", "s23_gid", "p1"), new_feats.select("s1_gid", "s23_gid", "p1")])
        union_pred = enforce_g_m3(union_pred_raw.filter(pl.col("p1") >= V31_THRESHOLD).rename({"p1": "prob"})).select("s1_gid", "s23_gid")
        recovered_above = truth_in_new.join(new_feats.filter(pl.col("p1") >= V31_THRESHOLD).select("s1_gid", "s23_gid"),
                                            on=["s1_gid", "s23_gid"], how="semi").height
    else:
        n_new_above_threshold, recovered_above = 0, 0
        union_pred = base_pred
    f05_with_embed = _f05(union_pred, truth, s1_df["s1_gid"])
    log(f"new candidates crossing v3.1 threshold: {n_new_above_threshold:,} / {new_cand.height:,}")
    log(f"f05_with_embed: {f05_with_embed:.5f}")

    n_true = truth.group_by("s1_gid").len(name="n_true")
    found_existing = d.filter(pl.col("label") == 1).group_by("s1_gid").len(name="n_found")
    all_cand_keys = pl.concat([d.select("s1_gid", "s23_gid"), new_cand]).unique()
    found_union = all_cand_keys.join(truth, on=["s1_gid", "s23_gid"], how="semi").group_by("s1_gid").len(name="n_found")
    cov_existing = s1_df.join(n_true, on="s1_gid", how="left").join(found_existing, on="s1_gid", how="left").fill_null(0)
    cov_union = s1_df.join(n_true, on="s1_gid", how="left").join(found_union, on="s1_gid", how="left").fill_null(0)
    complete_coverage_existing = float((cov_existing["n_found"] >= cov_existing["n_true"]).mean())
    complete_coverage_union = float((cov_union["n_found"] >= cov_union["n_true"]).mean())

    net_gain = f05_with_embed - f05_base
    rep = {
        "n_embed_candidates_total": embed_cand.height, "n_new_candidate_pairs": new_cand.height,
        "n_true_links_recovered_as_candidates": n_true_recovered_as_candidate,
        "n_true_links_recovered_and_above_threshold": recovered_above,
        "n_new_candidates_above_threshold": n_new_above_threshold,
        "f05_base": f05_base, "f05_with_embed": f05_with_embed, "net_f05_gain": net_gain,
        "complete_coverage_existing": complete_coverage_existing, "complete_coverage_union": complete_coverage_union,
        "kill_bar": KILL_BAR, "go_no_go": "go" if net_gain >= KILL_BAR else "no_go",
        "note": "scored with v3.1 model (lgbm_v31.txt, threshold 0.68), same conservative sentinel approach as measure_empty_addr.py.",
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (OUT_DIR / "measure_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    main()
