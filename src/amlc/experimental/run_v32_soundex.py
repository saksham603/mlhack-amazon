"""Real FIT/VAL test of the soundex phonetic features (src/amlc/features/soundex_feat.py, delivered
by the second machine): v3.2's full feature set unchanged, snd_eq/snd_jacc added alongside (computed
from `core` name text via a join to records), retrain, compare VAL-B against v3.2 properly. Purely
additive: never writes into data/_v31 or data/_v32, never touches lgbm_v32.txt.

Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.run_v32_soundex [--eval-only]
"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl

from amlc.features.soundex_feat import soundex_features
from amlc.foundation import access
from amlc.model import validate as V
from amlc.model.lowmem_train import fit_lowmem
from amlc.pipeline import run_v3 as R
from amlc.pipeline import run_v32 as R32
from amlc.pipeline.mem import atomic_write_parquet, free_gb

MODEL_PATH = access.ROOT / "models" / "lgbm_v32_soundex.txt"
OUT_DIR = access.DATA / "_helper" / "exp_soundex"
FEATS_SND = R32.FEATS + ["snd_eq", "snd_jacc"]
_CORE_CACHE: dict = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[free RAM {free_gb()} GB]", flush=True)


def _core(dataset: str, s: int) -> pl.DataFrame:
    key = (dataset, s)
    if key not in _CORE_CACHE:
        _CORE_CACHE[key] = pl.read_parquet(R.REC / f"{dataset}_s{s}.parquet", columns=["gid", "core"])
    return _CORE_CACHE[key]


def with_soundex(f, sub: str, cols: list) -> pl.DataFrame:
    """v3.2's full feature set for file f (via run_v32._with_x, untouched) plus snd_eq/snd_jacc
    computed from a join of core name text (s1 vs s23), keyed the same way _with_x's DATASET map does."""
    base_cols = [c for c in cols if c not in ("snd_eq", "snd_jacc")]
    need = list(dict.fromkeys(base_cols + ["s1_gid", "s23_gid"]))
    d = R32._with_x(f, sub, need)
    ds = R32.DATASET[sub]
    core1 = _core(ds, 1).rename({"gid": "s1_gid", "core": "core_1"})
    core23 = pl.concat([_core(ds, s).rename({"gid": "s23_gid", "core": "core_23"}) for s in (2, 3)]).unique("s23_gid")
    orig_height = d.height
    d = d.join(core1, on="s1_gid", how="left").join(core23, on="s23_gid", how="left")
    if d.height != orig_height:
        raise AssertionError(f"{f.name}: core join changed row count. STOP.")
    d = soundex_features(d, "core_1", "core_23").drop("core_1", "core_23")
    return d.select(cols)


def train(skip_fit: bool = False) -> dict:
    t0 = time.time()
    files = sorted(R32.SRC["fit"].glob("*.parquet"))
    if not skip_fit:
        fit_lowmem(len(files), lambda i, cols: with_soundex(files[i], "fit", cols), FEATS_SND, R.subsample,
                   R.PARAMS, R.SEED, MODEL_PATH, log, sub_cols=R.SUB_COLS)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    val = pl.concat([with_soundex(f, "val", ["s1_gid", "s23_gid", "country", *FEATS_SND])
                     for f in sorted(R32.SRC["val"].glob("*.parquet"))])
    X = val.select([pl.col(f).cast(pl.Float32) for f in FEATS_SND]).to_numpy()
    scored = val.select("s1_gid", "s23_gid", "country").with_columns(pl.Series("prob", booster.predict(X)))
    atomic_write_parquet(scored, OUT_DIR / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored.select("s1_gid", "s23_gid", "prob"), a, R.THRESHOLDS)
    rep = {"threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
           "val_b": V.report_on_b(scored.select("s1_gid", "s23_gid", "prob"), b, tuned["best_threshold"]),
           "best_iteration": booster.current_iteration(),
           "top_gain": sorted(zip(FEATS_SND, booster.feature_importance("gain").round().tolist()), key=lambda z: -z[1])[:20],
           "minutes": round((time.time() - t0) / 60, 1),
           "reference_v32_val_b_f05": json.loads((access.DATA / "_v32" / "report_v32.json").read_text())["val_b"]["f05"]}
    (OUT_DIR / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    vb = rep["val_b"]
    log(f"SOUNDEX VAL-B F0.5 {vb['f05']:.5f} (reference v3.2: {rep['reference_v32_val_b_f05']:.5f}) "
        f"thr {rep['threshold']} rounds {rep['best_iteration']} minutes {rep['minutes']}")
    return rep


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train(skip_fit="--eval-only" in sys.argv)
