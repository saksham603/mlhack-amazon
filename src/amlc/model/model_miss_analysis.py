"""Where does the model lose points, and do per-S1 context features help? FIT labels only.

Holdout = S1s with s1_gid % 10 == 0 (deterministic, S1-disjoint). Trains two analysis-only models on
the rest (base 9 features vs. base + context features), then on the holdout reports:
  - macro F0.5 against ALL true links of the holdout S1s (score proxy, includes blocking misses)
  - macro F0.5 against in-candidate true links only (isolates the model's own loss)
  - false-negative / false-positive profiles at the production threshold.
Analysis only: nothing here is used for the submission.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.model_miss_analysis
"""
import json

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.eval import metric
from amlc.foundation import access
from amlc.model import dataset
from amlc.model import train as T
from amlc.model.predict import enforce_g_m3

TRAIN_OUT = access.DATA / "_v1" / "training"
COUNTRIES = ("India", "US")
THRESHOLDS = [round(0.02 * i, 2) for i in range(10, 50)]

CONTEXT_COLS = ["n_cand_s1", "score_rel_max", "score_gap_top", "name_jac_rank", "name_jac_rel_max",
                "n_exact_name_s1", "addr_jac_rank", "n_s1_sharing_s23"]


def add_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 group context: how this candidate compares with the S1's other candidates."""
    g = "s1_gid"
    return df.with_columns(
        pl.len().over(g).alias("n_cand_s1"),
        (pl.col("blocking_score") / pl.col("blocking_score").max().over(g)).alias("score_rel_max"),
        (pl.col("blocking_score").max().over(g) - pl.col("blocking_score")).alias("score_gap_top"),
        pl.col("name_jaccard").rank("dense", descending=True).over(g).alias("name_jac_rank"),
        (pl.col("name_jaccard") / pl.col("name_jaccard").max().over(g)).fill_nan(0.0).alias("name_jac_rel_max"),
        (pl.col("name_jaccard") == 1.0).sum().over(g).alias("n_exact_name_s1"),
        pl.col("addr_token_jaccard").rank("dense", descending=True).over(g).alias("addr_jac_rank"),
        pl.len().over("s23_gid").alias("n_s1_sharing_s23"),
    )


def f05_at(scored: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, t: float) -> float:
    pred = enforce_g_m3(scored.filter(pl.col("prob") >= t)).select("s1_gid", "s23_gid")
    return metric.macro(metric.per_s1_f05(pred, truth, s1_ids))


def evaluate(name: str, fit_df, hold_df, feats, truth_all, truth_incand, s1_ids) -> dict:
    booster = T.train(fit_df, feature_cols=feats)
    scored = hold_df.select("s1_gid", "s23_gid", "label", "country").with_columns(
        pl.Series("prob", T.predict_proba(booster, hold_df, feature_cols=feats)))
    curve = [(t, f05_at(scored, truth_all, s1_ids, t)) for t in THRESHOLDS]
    best_t, best_f = max(curve, key=lambda x: x[1])
    by_country = {}
    for c in COUNTRIES:
        ids = scored.filter(pl.col("country") == c)["s1_gid"].unique()
        by_country[c] = f05_at(scored.filter(pl.col("country") == c), truth_all, ids, best_t)
    at = scored.filter(pl.col("prob") >= best_t)
    tp = at.filter(pl.col("label") == 1).height
    return {
        "model": name, "n_features": len(feats), "best_threshold": best_t,
        "holdout_f05_vs_all_true_links": best_f,
        "holdout_f05_vs_in_candidate_links": f05_at(scored, truth_incand, s1_ids, best_t),
        "by_country_vs_all_true_links": by_country,
        "pair_precision": tp / at.height if at.height else None,
        "pair_recall_in_candidates": tp / scored["label"].sum(),
        "feature_importance_gain": dict(zip(feats, [round(x, 1) for x in booster.feature_importance("gain")])),
    }, scored, best_t


def main() -> int:
    df = pl.concat([pl.read_parquet(TRAIN_OUT / f"labeled_{c}.parquet") for c in COUNTRIES], how="vertical")
    df = add_context_features(df)
    hold_mask = (pl.col("s1_gid") % 10) == 0
    fit_df, hold_df = df.filter(~hold_mask), df.filter(hold_mask)
    del df

    hold_s1 = pl.concat([dataset.sample_fit_s1_gids(c) for c in COUNTRIES]).filter(hold_mask)
    s1_ids = hold_s1["s1_gid"]
    truth_all = access.load_labels("FIT").select("s1_gid", "s23_gid").join(hold_s1, on="s1_gid", how="semi")
    truth_incand = hold_df.filter(pl.col("label") == 1).select("s1_gid", "s23_gid")

    base, _, _ = evaluate("base_9", fit_df, hold_df, T.FEATURE_COLS, truth_all, truth_incand, s1_ids)
    ctx, scored, t = evaluate("base_plus_context", fit_df, hold_df, T.FEATURE_COLS + CONTEXT_COLS,
                              truth_all, truth_incand, s1_ids)

    hold_feat = hold_df.select("s1_gid", "s23_gid", "name_jaccard", "addr_token_jaccard", "house_number_match",
                               "blocking_rank")
    prof = scored.join(hold_feat, on=["s1_gid", "s23_gid"])
    fn = prof.filter((pl.col("label") == 1) & (pl.col("prob") < t))
    fp = prof.filter((pl.col("label") == 0) & (pl.col("prob") >= t))

    def profile(d: pl.DataFrame) -> dict:
        return {"n": d.height,
                "name_jaccard_median": d["name_jaccard"].median(),
                "name_jaccard_eq_1_share": float((d["name_jaccard"] == 1.0).mean()) if d.height else None,
                "addr_jaccard_median": d["addr_token_jaccard"].median(),
                "house_match_share": float(d["house_number_match"].mean()) if d.height else None,
                "blocking_rank_median": d["blocking_rank"].median(),
                "by_country": dict(d.group_by("country").len().iter_rows())}

    report = {"n_holdout_s1": s1_ids.len(), "n_holdout_true_links": truth_all.height,
              "n_holdout_in_candidate_true": truth_incand.height,
              "base": base, "context": ctx,
              "context_model_false_negatives": profile(fn), "context_model_false_positives": profile(fp),
              "peak_ram_mb": peak_ram_mb()}
    (OUT_DIR / "model_miss_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
