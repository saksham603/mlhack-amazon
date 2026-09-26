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
from amlc.model.lowmem_train import fit_lowmem
from amlc.model.pass2 import P2_FEATURES, add_pass2
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet

FOLD_MODELS = [R.MODEL_PATH.with_name(R.MODEL_PATH.stem + f"_fold{k}.txt") for k in (0, 1)]
P2_MODEL = R.MODEL_PATH.with_name(R.MODEL_PATH.stem + "_pass2.txt")
FEATS2 = R.FEATURES + P2_FEATURES
THRESHOLDS = R.THRESHOLDS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fold_of(df: pl.DataFrame) -> pl.Series:
    return (df["s1_gid"].hash(seed=R.SEED) % 2).cast(pl.Int8)


def folds() -> None:
    files = sorted(glob.glob(str(R.OUT / "fit" / "*.parquet")))
    for k in (0, 1):
        if FOLD_MODELS[k].exists():
            continue
        def load(i, cols, k=k):  # train on the OTHER half of FIT S1s
            d = pl.read_parquet(files[i], columns=cols)
            return d.filter(fold_of(d) != k)
        fit_lowmem(len(files), load, R.FEATURES, R.subsample, R.PARAMS, R.SEED, FOLD_MODELS[k], log, sub_cols=R.SUB_COLS)


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
            atomic_write_parquet(out.select("s1_gid", "s23_gid", *P2_FEATURES), dst)
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
    files = sorted((R.OUT / "fit").glob("*.parquet"))
    def load(i, cols):
        d = pl.read_parquet(files[i], columns=[c for c in cols if c not in P2_FEATURES])
        p2 = pl.read_parquet(R.OUT / "fit_p2" / files[i].name, columns=["s1_gid", "s23_gid", *[c for c in cols if c in P2_FEATURES]])
        out = d.join(p2, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
        if out.height != d.height:
            raise AssertionError(f"{files[i].name}: pass-2 join lost rows. STOP.")
        return out
    fit_lowmem(len(files), load, FEATS2, R.subsample, R.PARAMS, R.SEED, P2_MODEL, log, sub_cols=R.SUB_COLS)
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
        atomic_write_parquet(p2.join(d.select("s1_gid", "s23_gid").with_columns(pl.Series("prob2", b.predict(x))),
                                     on=["s1_gid", "s23_gid"], how="left", validate="1:1"), dst)
    log("pass-2 test probabilities written")


def assemble(name: str) -> dict:
    """output/<name>/ from pass-2 probabilities; candidates = exactly the pass-1 candidate set."""
    from amlc.model.assemble_output import build_id_maps, group_ids, links_to_tsv, write_tsv
    from amlc.model.predict import predictions_at_threshold
    thr = json.loads((R.OUT / "report_pass2.json").read_text(encoding="utf-8"))["threshold"]
    base = sorted(R.TEST_SCORED.glob("*.parquet"))
    p2dir = R.OUT / (R.TEST_SCORED.name + "_p2")
    pred = predictions_at_threshold(pl.concat([pl.read_parquet(p2dir / f.name, columns=["s1_gid", "s23_gid", "prob2"])
                                               .filter(pl.col("prob2") >= thr).rename({"prob2": "prob"}) for f in base]), thr)
    s1_map, s23_map = build_id_maps()
    all_s1 = s1_map.select("s1_id")
    dst = R.OUTPUT / name
    write_tsv(links_to_tsv(pred, s1_map, s23_map, all_s1, "matched_id"), "matched_id", "matched_entity_ids", dst / "matching_results.tsv")
    n_pairs, written = 0, set()
    with open(dst / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for f in base:
            b = pl.read_parquet(f, columns=["s1_gid", "s23_gid"])
            n_pairs += b.height
            rows = group_ids(b, s1_map, s23_map)
            written.update(rows["s1_id"].to_list())
            fh.write("".join(f"{x}\t{y}\n" for x, y in rows.iter_rows()))
        empty = set(all_s1["s1_id"].to_list()) - written
        fh.write("".join(f"{x}\t\n" for x in sorted(empty)))
    rep = {"name": name, "threshold": thr, "n_predictions": pred.height, "candidate_pairs": n_pairs,
           "s1_with_predictions": pred["s1_gid"].n_unique(), "n_test_s1": all_s1.height}
    (dst / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    if sys.argv[1] == "assemble":
        print(json.dumps(assemble(sys.argv[2]), indent=1))
    else:
        {"folds": folds, "feats": feats, "train": train, "test": test}[sys.argv[1]]()
