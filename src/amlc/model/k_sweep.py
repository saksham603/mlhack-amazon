"""Candidate-budget sweep on VALIDATION half B: keep each S1's candidates with blocking_rank <= k,
apply lgbm_v1 at the step-4 threshold, report VAL-B F0.5 (overall + by country) and candidates/S1.
Probabilities are computed once; restricting to rank <= k does not change any kept row's features.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.k_sweep
"""
import json
import time

import lightgbm as lgb
import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.model import train as T
from amlc.model import validate as V
from amlc.model.run_step4 import MODEL_OUT, load_val_scored_input

KS = (5, 10, 20, 30, 50, 150)


def restrict(scored: pl.DataFrame, k: int) -> pl.DataFrame:
    return scored.filter(pl.col("blocking_rank") <= k)


def main() -> int:
    t0 = time.time()
    threshold = json.loads((OUT_DIR / "step4_model_report.json").read_text(encoding="utf-8"))["best_threshold"]
    booster = lgb.Booster(model_file=str(MODEL_OUT / "lgbm_v1.txt"))
    val_input = load_val_scored_input()
    probs = T.predict_proba(booster, val_input)
    scored = val_input.select("s1_gid", "s23_gid", "blocking_rank").with_columns(pl.Series("prob", probs))
    del val_input
    _, b = V.sample_validation_halves()
    b_df = pl.DataFrame({"s1_gid": b})
    rows = []
    for k in KS:
        sk = restrict(scored, k)
        rep = V.report_on_b(sk.select("s1_gid", "s23_gid", "prob"), b, threshold)
        n_cand_b = sk.join(b_df, on="s1_gid", how="semi").height
        rows.append({"k": k, "f05": rep["f05"], "by_country": rep["by_country"],
                     "pair_recall": rep["pair_recall"], "pair_precision": rep["pair_precision"],
                     "cand_per_s1_b": n_cand_b / b_df.height})
        print(json.dumps(rows[-1], default=str), flush=True)
    base = next(r for r in rows if r["k"] == 150)["f05"]
    chosen = min(r["k"] for r in rows if r["f05"] >= base - 0.002)
    out = {"threshold": threshold, "rows": rows, "chosen_k": chosen, "runtime_s": round(time.time() - t0, 1)}
    (OUT_DIR / "k_sweep.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("chosen_k", chosen, "runtime_s", out["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
