"""Pipeline v3.2 (2026-09-27 08:05): the v3.1 candidates and features plus the number-noise features
(amlc.features.numnoise), model retrained. Why: FIT-holdout error analysis (data/_v31/exp) showed the
largest group of missed links is same name + same address with a perturbed number; the A/B there gave
+0.0144 for a fresh model with these features. Pass-2 (+0.0009 in the same A/B) was skipped for this.

Steps (resumable: finished files are skipped; every file written atomically):
  x fit|val|test  number-noise features for every candidate pair of data/_v31/{fit,val,test_feats_scored}
                  -> data/_v32/x_<sub>/<same file name>
  train           LightGBM on FIT (v3.1 features + X), low-memory; threshold on VAL-A, report VAL-B
  test            TEST probabilities -> data/_v32/test_scored/<file> (s1_gid, s23_gid, prob)
  assemble <name> output/<name>/ (candidates = exactly the v3.1 candidate set)
Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.pipeline.run_v32 <step> [arg]   (AMLC_LOWPRI=1: below-normal priority)
"""
import ctypes
import json
import os
import sys
import time
from itertools import groupby

import lightgbm as lgb
import polars as pl

from amlc.features.numnoise import X_FEATURES, pair_features
from amlc.foundation import access
from amlc.model import validate as V
from amlc.model.assemble_output import build_id_maps, group_ids, links_to_tsv, write_tsv
from amlc.model.lowmem_train import fit_lowmem
from amlc.model.predict import predictions_at_threshold
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet, free_gb

if not R.KA:
    raise SystemExit("run_v32 builds on the v3.1 files: set AMLC_KA=10")
OUT = access.DATA / "_v32"
MODEL_PATH = R.MODEL_PATH.with_name("lgbm_v32.txt")
FEATS = R.FEATURES + X_FEATURES
KEYS = ["s1_gid", "s23_gid"]
SRC = {"fit": R.OUT / "fit", "val": R.OUT / "val", "test": R.TEST_SCORED}
DATASET = {"fit": "train", "val": "train", "test": "test"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[free RAM {free_gb()} GB]", flush=True)


def _addr(dataset: str, s: int, country: str) -> pl.DataFrame:
    return pl.read_parquet(R.REC / f"{dataset}_s{s}.parquet", columns=["gid", "country", "addr"]).filter(
        pl.col("country") == country).select("gid", "addr")


def xfeats(sub: str) -> None:
    src = sorted(SRC[sub].glob("*.parquet"))
    dst_dir = OUT / f"x_{sub}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in src if not (dst_dir / f.name).exists()]
    log(f"x {sub}: {len(todo)} of {len(src)} files to do")
    for country, files in groupby(todo, key=lambda f: f.name.split("_")[0]):
        files = list(files)
        t0 = time.time()
        rec1 = _addr(DATASET[sub], 1, country)
        rec23 = pl.concat([_addr(DATASET[sub], s, country) for s in (2, 3)])
        for f in files:
            pairs = pl.read_parquet(f, columns=KEYS)
            x = pair_features(pairs, rec1, rec23)
            if x.height != pairs.height or x.select(X_FEATURES).null_count().sum_horizontal().item():
                raise AssertionError(f"{f.name}: x features rows {x.height} vs {pairs.height} or nulls. STOP.")
            atomic_write_parquet(x, dst_dir / f.name)
        log(f"x {sub}/{country}: {len(files)} files ({time.time() - t0:.0f}s)")
        del rec1, rec23


def _with_x(f, sub: str, cols: list) -> pl.DataFrame:
    """Columns `cols` of v3.1 file f joined 1:1 with its number-noise features (row order of f kept)."""
    base = [c for c in dict.fromkeys(KEYS + cols) if c not in X_FEATURES]
    d = pl.read_parquet(f, columns=base)
    xc = [c for c in cols if c in X_FEATURES]
    if not xc:
        return d
    x = pl.read_parquet(OUT / f"x_{sub}" / f.name, columns=KEYS + xc)
    out = d.join(x, on=KEYS, how="left", validate="1:1", maintain_order="left")
    if out.height != d.height or out.select(xc).null_count().sum_horizontal().item():
        raise AssertionError(f"{f.name}: x join lost rows or left nulls. STOP.")
    return out


def to_x(d: pl.DataFrame):
    return d.select([pl.col(f).cast(pl.Float32) for f in FEATS]).to_numpy()


def train() -> dict:
    files = sorted(SRC["fit"].glob("*.parquet"))
    fit_lowmem(len(files), lambda i, cols: _with_x(files[i], "fit", cols), FEATS, R.subsample, R.PARAMS, R.SEED,
               MODEL_PATH, log, sub_cols=R.SUB_COLS)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    val = pl.concat([_with_x(f, "val", ["country", *FEATS]) for f in sorted(SRC["val"].glob("*.parquet"))])
    scored = val.select(*KEYS, "country").with_columns(pl.Series("prob", booster.predict(to_x(val))))
    atomic_write_parquet(scored, OUT / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored.select(*KEYS, "prob"), a, R.THRESHOLDS)
    rep = {"threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
           "val_b": V.report_on_b(scored.select(*KEYS, "prob"), b, tuned["best_threshold"]),
           "best_iteration": booster.current_iteration(),
           "top_gain": sorted(zip(FEATS, booster.feature_importance("gain").round().tolist()), key=lambda z: -z[1])[:20]}
    (OUT / "report_v32.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    vb = rep["val_b"]
    log(f"VAL-B F0.5 {vb['f05']:.4f} {({k: round(v['f05'], 4) for k, v in vb['by_country'].items()})} "
        f"thr {rep['threshold']} P {vb['pair_precision']:.4f} R {vb['pair_recall']:.4f} empty {vb['pred_empty_share']:.4f}")
    return rep


def test() -> None:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    dst = OUT / "test_scored"
    dst.mkdir(parents=True, exist_ok=True)
    src = sorted(SRC["test"].glob("*.parquet"))
    n = 0
    for j, f in enumerate(src):
        if (dst / f.name).exists():
            continue
        d = _with_x(f, "test", FEATS)
        atomic_write_parquet(d.select(KEYS).with_columns(pl.Series("prob", booster.predict(to_x(d)))), dst / f.name)
        n += 1
        if n % 25 == 1:
            log(f"test: {j + 1}/{len(src)} files")
    log(f"test: {n} new files, {len(src)} total")


def assemble(name: str) -> dict:
    thr = json.loads((OUT / "report_v32.json").read_text(encoding="utf-8"))["threshold"]
    src = sorted(SRC["test"].glob("*.parquet"))
    scored = [OUT / "test_scored" / f.name for f in src]
    missing = [p.name for p in scored if not p.exists()]
    if missing:
        raise AssertionError(f"{len(missing)} test files not scored yet, e.g. {missing[:3]}. STOP.")
    pred = predictions_at_threshold(pl.concat([pl.scan_parquet(p).filter(pl.col("prob") >= thr).collect() for p in scored]), thr)
    s1_map, s23_map = build_id_maps()
    all_s1 = s1_map.select("s1_id")
    dst = R.OUTPUT / name
    write_tsv(links_to_tsv(pred, s1_map, s23_map, all_s1, "matched_id"), "matched_id", "matched_entity_ids", dst / "matching_results.tsv")
    n_pairs, written = 0, set()
    with open(dst / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for f in src:  # candidates = the v3.1 candidate set (the model's input), same rows as test_scored
            b = pl.read_parquet(f, columns=KEYS)
            n_pairs += b.height
            rows = group_ids(b, s1_map, s23_map)
            written.update(rows["s1_id"].to_list())
            fh.write("".join(f"{x}\t{y}\n" for x, y in rows.iter_rows()))
        empty = set(all_s1["s1_id"].to_list()) - written
        fh.write("".join(f"{x}\t\n" for x in sorted(empty)))
    rep = {"name": name, "model": MODEL_PATH.name, "threshold": thr, "n_predictions": pred.height, "candidate_pairs": n_pairs,
           "s1_with_candidates": len(written), "s1_without_candidates": len(empty),
           "s1_with_predictions": pred["s1_gid"].n_unique(), "n_test_s1": all_s1.height}
    (dst / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    if os.environ.get("AMLC_LOWPRI") == "1" and os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    step = sys.argv[1]
    if step == "x":
        xfeats(sys.argv[2])
    elif step == "train":
        train()
    elif step == "test":
        test()
    elif step == "assemble":
        print(json.dumps(assemble(sys.argv[2]), indent=1))
    else:
        raise SystemExit(f"unknown step {step}")
