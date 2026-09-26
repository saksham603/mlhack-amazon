"""Day-2 Phase A driver (master prompt A2-A4): v3 features for VALIDATION and FIT candidates,
LightGBM v2, threshold tuned on VAL-A, report on VAL-B. VALIDATION labels only ever reach
amlc.eval.scorer (through amlc.model.validate); FIT labels come from the v1 labeled files.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.run_v2 {feats-val|feats-train|train} [max_extra_files]
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from amlc.features import w3
from amlc.foundation import access
from amlc.model import validate as V

V1 = access.DATA / "_v1"
V2 = access.DATA / "_v2"
VAL_FEATS = V2 / "val_feats"
TRAIN_FEATS = V2 / "train_feats"
REPORTS = V2 / "reports"
MODEL_PATH = access.ROOT / "models" / "lgbm_v2.txt"
COUNTRIES = ("India", "US")
SEED = 20260927
THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 50)]
EASY_KEEP = 0.2  # share of easy negatives kept (weight 1 / EASY_KEEP)
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 255, "seed": SEED, "deterministic": True, "force_row_wise": True, "verbosity": -1}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _s1_chunks(df: pl.DataFrame, n: int):
    gids = df.select("s1_gid").unique().sort("s1_gid")["s1_gid"]
    size = (gids.len() + n - 1) // n
    for i in range(n):
        part = gids.slice(i * size, size)
        if part.len():
            yield i, df.filter(pl.col("s1_gid").is_in(part.implode()))


def feats_val(r1, r23) -> None:
    VAL_FEATS.mkdir(parents=True, exist_ok=True)
    for c in COUNTRIES:
        cand = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(V1 / "validation" / "candidates" / c / "*.parquet")))])
        f1 = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(V1 / "validation" / "features" / c / "*.parquet")))])
        pairs = cand.join(f1, on=["s1_gid", "s23_gid"], how="inner", validate="1:1").with_columns(pl.lit(c).alias("country"))
        if pairs.height != cand.height:
            raise AssertionError(f"VAL {c}: candidates/features mismatch")
        out = pl.concat([w3.pair_features(p, r1, r23) for _, p in _s1_chunks(pairs, 4)])
        out.write_parquet(VAL_FEATS / f"{c}.parquet")
        log(f"val feats {c}: {out.height:,} rows")


def feats_train(r1, r23, max_extra: int) -> None:
    TRAIN_FEATS.mkdir(parents=True, exist_ok=True)
    for c in COUNTRIES:
        dst = TRAIN_FEATS / f"v1_{c}_0.parquet"
        if not dst.exists():
            df = pl.read_parquet(V1 / "training" / f"labeled_{c}.parquet")
            for i, p in _s1_chunks(df, 8):
                w3.pair_features(p, r1, r23).write_parquet(TRAIN_FEATS / f"v1_{c}_{i}.parquet")
            log(f"train feats v1 sample {c}: {df.height:,} rows")
    for i in range(max_extra):
        for c in COUNTRIES:
            src = V1 / "training_extra" / "labeled" / c / f"labeled_{i:05d}.parquet"
            dst = TRAIN_FEATS / f"extra_{c}_{i:05d}.parquet"
            if src.exists() and not dst.exists():
                w3.pair_features(pl.read_parquet(src), r1, r23).write_parquet(dst)
        log(f"train feats extra file {i} done")


def _subsample(df: pl.DataFrame) -> pl.DataFrame:
    hard = ((pl.col("label") == 1) | (pl.col("blocking_rank") <= 30) | (pl.col("n_tset") >= 70) | (pl.col("a_tset") >= 70))
    keep_easy = (pl.struct("s1_gid", "s23_gid").hash(seed=SEED) % 1000) < int(EASY_KEEP * 1000)
    return (df.filter(hard | keep_easy)
            .with_columns(pl.when(hard).then(1.0).otherwise(1.0 / EASY_KEEP).cast(pl.Float32).alias("w")))


def load_train() -> pl.DataFrame:
    files = sorted(glob.glob(str(TRAIN_FEATS / "*.parquet")))
    parts = [_subsample(pl.read_parquet(f, columns=["s1_gid", "s23_gid", "label", *w3.FEATURES])) for f in files]
    log(f"train files {len(files)}")
    return pl.concat(parts)


def to_x(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(f).cast(pl.Float32) for f in w3.FEATURES]).to_numpy()


def train() -> dict:
    t0 = time.time()
    df = load_train()
    s1 = df.select("s1_gid").unique().sort("s1_gid")
    ho = s1.sample(fraction=0.1, seed=SEED)
    tr, va = df.join(ho, on="s1_gid", how="anti"), df.join(ho, on="s1_gid", how="semi")
    info = {"train_rows": tr.height, "train_pos": int(tr["label"].sum()), "holdout_rows": va.height}
    log(f"train rows {tr.height:,} (pos {info['train_pos']:,}), holdout {va.height:,}")
    dtr = lgb.Dataset(to_x(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), feature_name=w3.FEATURES,
                      free_raw_data=True)
    dva = lgb.Dataset(to_x(va), label=va["label"].to_numpy(), weight=va["w"].to_numpy(), reference=dtr,
                      feature_name=w3.FEATURES)
    del df, tr, va
    booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
    imp = sorted(zip(w3.FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
    info.update({"best_iteration": booster.best_iteration, "train_minutes": round((time.time() - t0) / 60, 1),
                 "top_gain": [(f, round(float(g))) for f, g in imp[:15]]})
    return info


def evaluate(info: dict) -> dict:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    val = pl.concat([pl.read_parquet(VAL_FEATS / f"{c}.parquet") for c in COUNTRIES])
    scored = val.select("s1_gid", "s23_gid", "blocking_rank", "country").with_columns(pl.Series("prob", booster.predict(to_x(val))))
    del val
    REPORTS.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(REPORTS / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored.select("s1_gid", "s23_gid", "prob"), a, THRESHOLDS)
    t = tuned["best_threshold"]
    rep = {"threshold": t, "val_a_f05": tuned["best_f05"], "val_b_k150": V.report_on_b(scored.select("s1_gid", "s23_gid", "prob"), b, t)}
    for k in (30, 50):
        rep[f"val_b_k{k}"] = V.report_on_b(scored.filter(pl.col("blocking_rank") <= k).select("s1_gid", "s23_gid", "prob"), b, t)
    rep.update(info)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase_a_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return rep


def main() -> int:
    mode = sys.argv[1]
    if mode in ("feats-val", "feats-train"):
        r1, r23 = w3.load_records("train", "s1"), w3.load_records("train", "s23")
        log(f"records loaded: s1 {r1.height:,}, s23 {r23.height:,}")
        if mode == "feats-val":
            feats_val(r1, r23)
        else:
            feats_train(r1, r23, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "train":
        info = train()
        rep = evaluate(info)
        b = rep["val_b_k150"]
        log(f"VAL-B k150 F0.5 {b['f05']:.4f}  by country {b.get('by_country')}  threshold {rep['threshold']}")
        for k in (30, 50):
            log(f"VAL-B k{k} F0.5 {rep[f'val_b_k{k}']['f05']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
