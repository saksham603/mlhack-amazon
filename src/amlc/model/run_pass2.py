"""Second-pass driver on top of pipeline v3 / v3.1 (same OUT dir and features as amlc.pipeline.run_v3;
set AMLC_KA the same way). Steps, each resumable (skips outputs that already exist):
  folds    : two first-pass models, each trained on one half of FIT S1 (hash split) and scoring the other
             half -> out-of-fold p1 for every FIT candidate; fold models saved next to the main model.
  feats    : pass-2 features for FIT (from OOF p1), VALIDATION and TEST (p1 = mean of the two fold models)
  train    : pass-2 LightGBM on FIT (base + pass-2 features), threshold on VAL-A, report VAL-B.
  test     : pass-2 probabilities for TEST (prob2), written next to the pass-2 features.
Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.run_pass2 folds|feats|train|test
"""
import glob
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from amlc.model import validate as V
from amlc.model.pass2 import P2_FEATURES, add_pass2
from amlc.pipeline import run_v3 as R

FOLD_MODELS = [R.MODEL_PATH.with_name(R.MODEL_PATH.stem + f"_fold{k}.txt") for k in (0, 1)]
P2_MODEL = R.MODEL_PATH.with_name(R.MODEL_PATH.stem + "_pass2.txt")
FEATS2 = R.FEATURES + P2_FEATURES
THRESHOLDS = R.THRESHOLDS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fold_of(df: pl.DataFrame) -> pl.Series:
    return (df["s1_gid"].hash(seed=R.SEED) % 2).cast(pl.Int8)


def _subsample(d: pl.DataFrame) -> pl.DataFrame:
    hard = ((pl.col("label") == 1) | (pl.col("v1_rank") <= 10) | (pl.col("v3_rank") <= 10)
            | ((pl.col("va_rank") <= 5) if R.KA else pl.lit(False)) | (pl.col("n_tset") >= 70) | (pl.col("a_tset") >= 70))
    keep = (pl.struct("s1_gid", "s23_gid").hash(seed=R.SEED) % 1000) < int(R.EASY_KEEP * 1000)
    return d.filter(hard | keep).with_columns(pl.when(hard).then(1.0).otherwise(1.0 / R.EASY_KEEP).cast(pl.Float32).alias("w"))


def _fit(df: pl.DataFrame, feats: list, path: Path) -> None:
    ho = df.select("s1_gid").unique().sort("s1_gid").sample(fraction=0.1, seed=R.SEED)
    tr, va = df.join(ho, on="s1_gid", how="anti"), df.join(ho, on="s1_gid", how="semi")
    x = lambda d: d.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()  # noqa: E731
    dtr = lgb.Dataset(x(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(x(va), label=va["label"].to_numpy(), weight=va["w"].to_numpy(), reference=dtr, feature_name=feats)
    b = lgb.train(R.PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])
    b.save_model(str(path), num_iteration=b.best_iteration)
    log(f"saved {path.name}: {b.best_iteration} rounds, {tr.height:,} rows")


def folds() -> None:
    files = sorted(glob.glob(str(R.OUT / "fit" / "*.parquet")))
    for k in (0, 1):
        if FOLD_MODELS[k].exists():
            continue
        parts = []
        for f in files:
            d = pl.read_parquet(f, columns=["s1_gid", "s23_gid", "label", *R.FEATURES])
            parts.append(_subsample(d.filter(fold_of(d) != k)))  # train on the OTHER half
        _fit(pl.concat(parts), R.FEATURES, FOLD_MODELS[k])


def _p1(d: pl.DataFrame, boosters: list, oof: bool) -> np.ndarray:
    x = R.to_x(d)
    if not oof:
        return np.mean([b.predict(x) for b in boosters], axis=0)
    fo = fold_of(d).to_numpy()
    p = np.empty(d.height)
    for k in (0, 1):
        m = fo == k
        if m.any():
            p[m] = boosters[k].predict(x[m])  # model k never saw fold k
    return p


def feats() -> None:
    boosters = [lgb.Booster(model_file=str(p)) for p in FOLD_MODELS]
    jobs = [("fit", "train", True), ("val", "train", False), (R.TEST_SCORED.name, "test", False)]
    for sub, dataset, oof in jobs:
        src = sorted((R.OUT / sub).glob("*.parquet"))
        dst_dir = R.OUT / (sub + "_p2")
        dst_dir.mkdir(parents=True, exist_ok=True)
        r23 = pl.concat([pl.read_parquet(R.REC / f"{dataset}_s{s}.parquet", columns=["gid", "core", "addr"]) for s in (2, 3)])
        n = 0
        for f in src:
            dst = dst_dir / Path(f).name
            if dst.exists():
                continue
            d = pl.read_parquet(f)
            out = add_pass2(d.select("s1_gid", "s23_gid"), _p1(d, boosters, oof), r23)
            if out.height != d.height:
                raise AssertionError(f"{f}: pass-2 rows {out.height} != {d.height}. STOP.")
            out.select("s1_gid", "s23_gid", *P2_FEATURES).write_parquet(dst)
            n += 1
        log(f"pass-2 features {sub}: {n} new files of {len(src)}")


def _load(sub: str, cols: list) -> pl.DataFrame:
    parts = []
    for f in sorted((R.OUT / sub).glob("*.parquet")):
        d = pl.read_parquet(f, columns=cols)
        p2 = pl.read_parquet(R.OUT / (sub + "_p2") / Path(f).name)
        parts.append(d.join(p2, on=["s1_gid", "s23_gid"], how="inner", validate="1:1"))
    return pl.concat(parts)


def train() -> dict:
    parts = []
    for f in sorted((R.OUT / "fit").glob("*.parquet")):
        d = pl.read_parquet(f, columns=["s1_gid", "s23_gid", "label", *R.FEATURES])
        p2 = pl.read_parquet(R.OUT / "fit_p2" / Path(f).name)
        parts.append(_subsample(d.join(p2, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")))
    _fit(pl.concat(parts), FEATS2, P2_MODEL)
    del parts
    b = lgb.Booster(model_file=str(P2_MODEL))
    val = _load("val", ["s1_gid", "s23_gid", *R.FEATURES])
    x = val.select([pl.col(f).cast(pl.Float32) for f in FEATS2]).to_numpy()
    scored = val.select("s1_gid", "s23_gid").with_columns(pl.Series("prob", b.predict(x)))
    a, bb = V.sample_validation_halves()
    tuned = V.tune_threshold(scored, a, THRESHOLDS)
    rep = {"threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
           "val_b": V.report_on_b(scored, bb, tuned["best_threshold"])}
    (R.OUT / "report_pass2.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    vb = rep["val_b"]
    log(f"PASS-2 VAL-B F0.5 {vb['f05']:.4f} {({k: round(v['f05'], 4) for k, v in vb['by_country'].items()})} "
        f"thr {rep['threshold']} P {vb['pair_precision']:.4f} R {vb['pair_recall']:.4f} empty {vb['pred_empty_share']:.4f}")
    return rep


def test() -> None:
    b = lgb.Booster(model_file=str(P2_MODEL))
    sub = R.TEST_SCORED.name
    for f in sorted((R.OUT / sub).glob("*.parquet")):
        dst = R.OUT / (sub + "_p2") / Path(f).name
        p2 = pl.read_parquet(dst)
        if "prob2" in p2.columns:
            continue
        d = pl.read_parquet(f, columns=["s1_gid", "s23_gid", *R.FEATURES]).join(p2, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
        x = d.select([pl.col(c).cast(pl.Float32) for c in FEATS2]).to_numpy()
        p2.join(d.select("s1_gid", "s23_gid").with_columns(pl.Series("prob2", b.predict(x))), on=["s1_gid", "s23_gid"],
                how="left", validate="1:1").write_parquet(dst)
    log("pass-2 test probabilities written")


if __name__ == "__main__":
    {"folds": folds, "feats": feats, "train": train, "test": test}[sys.argv[1]]()
