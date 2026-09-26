"""Sprint step 4: train LightGBM on FIT (India+US), tune threshold on VALIDATION half A, report
on half B. Run leakscan before trusting any reported F0.5 (sprint sec7.3).

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.run_step4
"""
import json
import time

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.foundation import access
from amlc.model import train as T
from amlc.model import validate as V

TRAIN_OUT = access.DATA / "_v1" / "training"
VAL_OUT = access.DATA / "_v1" / "validation"
MODEL_OUT = access.ROOT / "models"
COUNTRIES = ("India", "US")
THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 50)]  # 0.02 .. 0.98


def load_labeled() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(TRAIN_OUT / f"labeled_{c}.parquet") for c in COUNTRIES], how="vertical")


def load_val_scored_input() -> pl.DataFrame:
    parts = []
    for c in COUNTRIES:
        cand = pl.concat([pl.read_parquet(f) for f in sorted((VAL_OUT / "candidates" / c).glob("candidates_*.parquet"))],
                         how="vertical")
        feat = pl.concat([pl.read_parquet(f) for f in sorted((VAL_OUT / "features" / c).glob("features_*.parquet"))],
                         how="vertical")
        parts.append(cand.join(feat, on=["s1_gid", "s23_gid"], how="inner", validate="1:1"))
    return pl.concat(parts, how="vertical")


def main() -> int:
    t0 = time.time()
    train_df = load_labeled()
    booster = T.train(train_df)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_OUT / "lgbm_v1.txt"))

    val_input = load_val_scored_input()
    probs = T.predict_proba(booster, val_input)
    scored = val_input.select("s1_gid", "s23_gid").with_columns(pl.Series("prob", probs))

    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored, a, THRESHOLDS)
    b_report = V.report_on_b(scored, b, tuned["best_threshold"])

    summary = {
        "n_train_rows": train_df.height, "n_train_positive": int(train_df["label"].sum()),
        "n_val_candidates": val_input.height,
        "best_threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
        "threshold_curve": tuned["curve"], "val_b": b_report,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step4_model_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
