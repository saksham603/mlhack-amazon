"""Error analysis for lgbm_v3 on FIT S1s held out from training (FIT labels only; VALIDATION untouched).
Splits the loss into: true links missing from the candidates (blocking), true links scored below the
threshold (model FN), and accepted wrong links (model FP), with feature summaries of each group.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.error_v3
"""
import glob
import json

import lightgbm as lgb
import polars as pl

from amlc.eval import metric
from amlc.foundation import access
from amlc.model.predict import predictions_at_threshold
from amlc.pipeline.run_v3 import FEATURES, MODEL_PATH, OUT, SEED, to_x

LOOK = ["n_tset", "n_jw_ns", "a_tset", "reg_eq", "reg_conflict", "hn_conflict", "legal_conflict", "v1_rank", "v3_rank",
        "s1_share_1", "ctx_c_rank"]


def main() -> int:
    thr = json.loads((OUT / "report_v3.json").read_text(encoding="utf-8"))["threshold"]
    files = sorted(glob.glob(str(OUT / "fit" / "*.parquet")))
    s1 = pl.concat([pl.read_parquet(f, columns=["s1_gid"]).unique() for f in files]).unique().sort("s1_gid")
    ho = s1.sample(fraction=0.1, seed=SEED)
    df = pl.concat([pl.read_parquet(f).join(ho, on="s1_gid", how="semi") for f in files])
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = df.with_columns(pl.Series("prob", booster.predict(to_x(df))))
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").join(ho, on="s1_gid", how="semi")
    pred = predictions_at_threshold(df.select("s1_gid", "s23_gid", "prob"), thr)
    per = metric.per_s1_f05(pred, truth, ho["s1_gid"])
    cand = df.select("s1_gid", "s23_gid")
    tp = pred.join(truth, on=["s1_gid", "s23_gid"], how="semi").height
    miss_block = truth.join(cand, on=["s1_gid", "s23_gid"], how="anti").height
    fn = df.filter(pl.col("label") == 1).join(pred, on=["s1_gid", "s23_gid"], how="anti")
    fp = df.filter(pl.col("label") == 0).join(pred, on=["s1_gid", "s23_gid"], how="semi")
    tpr = df.filter(pl.col("label") == 1).join(pred, on=["s1_gid", "s23_gid"], how="semi")
    summ = lambda d: {c: round(float(d[c].cast(pl.Float64).mean()), 3) for c in LOOK} if d.height else {}  # noqa: E731
    per_c = per.join(df.select("s1_gid", "country").unique(), on="s1_gid", how="left")
    rep = {
        "threshold": thr, "holdout_s1": ho.height, "true_links": truth.height,
        "f05": metric.macro(per), "by_country": {c: float(g["f05"].mean()) for (c,), g in per_c.group_by("country")},
        "pred_empty_share": float((per["n_pred"] == 0).mean()),
        "true_positives": tp, "missed_by_blocking": miss_block, "model_fn": fn.height, "model_fp": fp.height,
        "loss_share_s1": {
            "s1_all_links_missed_by_blocking": int(per.join(truth.join(cand, on=["s1_gid", "s23_gid"], how="semi")
                                                               .select("s1_gid").unique(), on="s1_gid", how="anti")
                                                   .filter(pl.col("n_true") > 0).height)},
        "fn_means": summ(fn), "fp_means": summ(fp), "tp_means": summ(tpr),
        "fn_prob_hist": fn.select(pl.col("prob").cut([0.1, 0.3, 0.5, 0.6]).value_counts()).unnest("prob").sort("prob").to_dicts()
        if fn.height else [],
    }
    (OUT / "error_v3.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
