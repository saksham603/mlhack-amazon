"""Real FIT/VAL test of the ctx_c_adj_rank/gap fix (src/amlc/features/ctx2.py, delivered by the
second machine) -- Variant C (additive): keep v3.2's original ctx_c_rank/gap untouched, add two new
columns alongside them, retrain, compare VAL-B against v3.2 properly. Purely additive: never reads
or writes anything under data/_v32 except a new model file and its own report; never touches
amlc/features/w3.py or the live lgbm_v32.txt.

Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.run_v32_ctxfix
"""
import json
import time

import lightgbm as lgb
import polars as pl

from amlc.features.ctx2 import combo_mirror_name, compute_ctx
from amlc.foundation import access
from amlc.model import validate as V
from amlc.model.lowmem_train import fit_lowmem
from amlc.pipeline import run_v3 as R
from amlc.pipeline import run_v32 as R32
from amlc.pipeline.mem import atomic_write_parquet, free_gb

MODEL_PATH = access.ROOT / "models" / "lgbm_v32_ctxfix.txt"
OUT_DIR = access.DATA / "_helper" / "exp_ctxfix"
FEATS_CTX = R32.FEATS + ["ctx_c_adj_rank", "ctx_c_adj_gap"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[free RAM {free_gb()} GB]", flush=True)


def with_ctx_adj(f, sub: str, cols: list) -> pl.DataFrame:
    """v3.2's full feature set for file f (via run_v32._with_x, untouched) plus the two new
    additive columns computed from combo_mirror_name over the SAME candidate set."""
    base_cols = [c for c in cols if c not in ("ctx_c_adj_rank", "ctx_c_adj_gap")]
    need = list(dict.fromkeys(base_cols + ["s1_gid", "n_tset", "a_tset", "a_len_1", "a_len_2"]))
    d = R32._with_x(f, sub, need)
    adj = compute_ctx(d, combo_mirror_name(d), "ctx_c_adj")
    return adj.select(cols)


def train(skip_fit: bool = False) -> dict:
    t0 = time.time()
    files = sorted(R32.SRC["fit"].glob("*.parquet"))
    if not skip_fit:
        fit_lowmem(len(files), lambda i, cols: with_ctx_adj(files[i], "fit", cols), FEATS_CTX, R.subsample,
                   R.PARAMS, R.SEED, MODEL_PATH, log, sub_cols=R.SUB_COLS)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    val = pl.concat([with_ctx_adj(f, "val", ["s1_gid", "s23_gid", "country", *FEATS_CTX])
                     for f in sorted(R32.SRC["val"].glob("*.parquet"))])
    X = val.select([pl.col(f).cast(pl.Float32) for f in FEATS_CTX]).to_numpy()
    scored = val.select("s1_gid", "s23_gid", "country").with_columns(pl.Series("prob", booster.predict(X)))
    atomic_write_parquet(scored, OUT_DIR / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored.select("s1_gid", "s23_gid", "prob"), a, R.THRESHOLDS)
    rep = {"threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
           "val_b": V.report_on_b(scored.select("s1_gid", "s23_gid", "prob"), b, tuned["best_threshold"]),
           "best_iteration": booster.current_iteration(),
           "top_gain": sorted(zip(FEATS_CTX, booster.feature_importance("gain").round().tolist()), key=lambda z: -z[1])[:20],
           "minutes": round((time.time() - t0) / 60, 1),
           "reference_v32_val_b_f05": json.loads((access.DATA / "_v32" / "report_v32.json").read_text())["val_b"]["f05"]}
    (OUT_DIR / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    vb = rep["val_b"]
    log(f"CTXFIX VAL-B F0.5 {vb['f05']:.5f} (reference v3.2: {rep['reference_v32_val_b_f05']:.5f}) "
        f"thr {rep['threshold']} rounds {rep['best_iteration']} minutes {rep['minutes']}")
    return rep


if __name__ == "__main__":
    import sys
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train(skip_fit="--eval-only" in sys.argv)
