"""Day-2 pipeline v3: candidate set = v1 top-30 (existing budget-150 files, rank <= 30) UNION blocking-v3
top-20 (name x address / region|name keys). Measured on the FIT v1 sample: recall India 0.936 at 44.7
candidates per S1, US 0.971 at 43.6 (v1 top-150: 0.795 at 104, 0.939 at 137).

Features: v1_score, v1_rank, v3_score, v3_rank (missing -> 0 / 999) + w3.NEW_FEATURES computed over
each S1's union candidate list. candidate_pairs.tsv = exactly this union (the model's input).

Run (PYTHONPATH=src):  python -m amlc.pipeline.run_v3 fit | val | train | test | assemble <name> [threshold]
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from amlc.blocking import v3
from amlc.features import w3
from amlc.foundation import access
from amlc.model import validate as V
from amlc.model.assemble_output import build_id_maps, group_ids, links_to_tsv, write_tsv
from amlc.model.predict import predictions_at_threshold

K1, K3 = 30, 20
V1 = access.DATA / "_v1"
OUT = access.DATA / "_v3"
MODEL_PATH = access.ROOT / "models" / "lgbm_v3.txt"
TEST_SCORED = OUT / "test_feats_scored"
OUTPUT = access.ROOT / "output"
REC = access.DATA / "_v2" / "records"
BLOCK_FEATURES = ["v1_score", "v1_rank", "v3_score", "v3_rank"]
FEATURES = BLOCK_FEATURES + w3.NEW_FEATURES
SEED = 20260927
BATCH = 5000
EASY_KEEP = 0.2
THRESHOLDS = [round(0.02 * i, 2) for i in range(1, 50)]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 255, "seed": SEED, "deterministic": True, "force_row_wise": True, "verbosity": -1}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def records(dataset: str, country: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["gid", "country", *w3.FIELDS]
    r1 = pl.read_parquet(REC / f"{dataset}_s1.parquet", columns=cols).filter(pl.col("country") == country)
    r23 = pl.concat([pl.read_parquet(REC / f"{dataset}_s{s}.parquet", columns=cols) for s in (2, 3)]).filter(pl.col("country") == country)
    return r1, r23


def v1_top(files: list) -> pl.DataFrame:
    return pl.concat([pl.scan_parquet(f).filter(pl.col("blocking_rank") <= K1)
                      .select("s1_gid", "s23_gid", pl.col("blocking_score").alias("v1_score"),
                              pl.col("blocking_rank").cast(pl.UInt32).alias("v1_rank")).collect() for f in files])


def union(v1c: pl.DataFrame, v3c: pl.DataFrame) -> pl.DataFrame:
    b = v3c.filter(pl.col("v3_rank") <= K3).select("s1_gid", "s23_gid", "v3_score", pl.col("v3_rank").cast(pl.UInt32))
    return (v1c.join(b, on=["s1_gid", "s23_gid"], how="full", coalesce=True)
            .with_columns(pl.col("v1_score", "v3_score").fill_null(0.0), pl.col("v1_rank", "v3_rank").fill_null(999)))


def build(dataset: str, country: str, s1_gids: pl.Series, v1c: pl.DataFrame, out_dir, labels=None, booster=None) -> dict:
    """Candidates + features (+ label, or + prob) for s1_gids, in batches written to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r1, r23 = records(dataset, country)
    k23, rare = v3.build_index(r23.select("gid", "core", "addr", "region", "ns", "skel"))
    log(f"{dataset}/{country}: index {k23.height:,} keys ({time.time() - t0:.0f}s)")
    r1s = r1.join(pl.DataFrame({"gid": s1_gids}), on="gid", how="semi").sort("gid")
    n_rows = 0
    for i in range(0, r1s.height, BATCH):
        dst = out_dir / f"{country}_{i // BATCH:05d}.parquet"
        if dst.exists():
            continue
        rb = r1s.slice(i, BATCH)
        c3 = v3.candidates(rb.select("gid", "core", "addr", "region", "ns", "skel"), k23, rare, K3)
        c1 = v1c.join(rb.select(pl.col("gid").alias("s1_gid")), on="s1_gid", how="semi")
        u = union(c1, c3)
        if u.height == 0:
            continue
        x = w3.pair_features(u, r1, r23)
        if labels is not None:
            x = x.join(labels.with_columns(pl.lit(1, pl.Int32).alias("label")), on=["s1_gid", "s23_gid"], how="left") \
                 .with_columns(pl.col("label").fill_null(0))
        x = x.with_columns(pl.lit(country).alias("country"))
        if booster is not None:
            prob = booster.predict(to_x(x))
            x = x.with_columns(pl.Series("prob", prob))  # keep features: later model-only changes just rescore
        x.write_parquet(dst)
        n_rows += x.height
    log(f"{dataset}/{country}: {r1s.height:,} S1, {n_rows:,} new rows ({time.time() - t0:.0f}s)")
    return {"s1": r1s.height, "rows": n_rows}


def to_x(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(f).cast(pl.Float32) for f in FEATURES]).to_numpy()


def run_fit() -> None:
    labels = access.load_labels("FIT").select("s1_gid", "s23_gid")
    for c in ("India", "US"):
        files = [V1 / "training" / f"labeled_{c}.parquet"] + sorted((V1 / "training_extra" / "labeled" / c).glob("*.parquet"))
        v1c = v1_top(files)
        build("train", c, v1c["s1_gid"].unique(), v1c, OUT / "fit", labels=labels)


def run_val() -> None:
    for c in ("India", "US"):
        files = sorted((V1 / "validation" / "candidates" / c).glob("*.parquet"))
        v1c = v1_top(files)
        build("train", c, v1c["s1_gid"].unique(), v1c, OUT / "val")


def train() -> dict:
    t0 = time.time()
    parts = []
    for f in sorted(glob.glob(str(OUT / "fit" / "*.parquet"))):
        d = pl.read_parquet(f, columns=["s1_gid", "s23_gid", "label", *FEATURES])
        hard = ((pl.col("label") == 1) | (pl.col("v1_rank") <= 10) | (pl.col("v3_rank") <= 10)
                | (pl.col("n_tset") >= 70) | (pl.col("a_tset") >= 70))
        keep = (pl.struct("s1_gid", "s23_gid").hash(seed=SEED) % 1000) < int(EASY_KEEP * 1000)
        parts.append(d.filter(hard | keep).with_columns(pl.when(hard).then(1.0).otherwise(1.0 / EASY_KEEP).cast(pl.Float32).alias("w")))
    df = pl.concat(parts)
    ho = df.select("s1_gid").unique().sort("s1_gid").sample(fraction=0.1, seed=SEED)
    tr, va = df.join(ho, on="s1_gid", how="anti"), df.join(ho, on="s1_gid", how="semi")
    info = {"train_rows": tr.height, "train_pos": int(tr["label"].sum()), "holdout_rows": va.height}
    log(f"train rows {tr.height:,} (pos {info['train_pos']:,}), holdout {va.height:,}")
    dtr = lgb.Dataset(to_x(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(to_x(va), label=va["label"].to_numpy(), weight=va["w"].to_numpy(), reference=dtr, feature_name=FEATURES)
    del df, tr, va, parts
    booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
    info.update({"best_iteration": booster.best_iteration, "train_minutes": round((time.time() - t0) / 60, 1),
                 "top_gain": [(f, round(float(g))) for f, g in imp[:20]]})
    # evaluate
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    val = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(OUT / "val" / "*.parquet")))])
    scored = val.select("s1_gid", "s23_gid", "country").with_columns(pl.Series("prob", booster.predict(to_x(val))))
    scored.write_parquet(OUT / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = V.tune_threshold(scored.select("s1_gid", "s23_gid", "prob"), a, THRESHOLDS)
    rep = {"threshold": tuned["best_threshold"], "val_a_f05": tuned["best_f05"],
           "val_b": V.report_on_b(scored.select("s1_gid", "s23_gid", "prob"), b, tuned["best_threshold"]),
           "val_candidates_per_s1": val.height / val["s1_gid"].n_unique(), **info}
    (OUT / "report_v3.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    vb = rep["val_b"]
    log(f"VAL-B F0.5 {vb['f05']:.4f} {({k: round(v['f05'], 4) for k, v in vb['by_country'].items()})} "
        f"thr {rep['threshold']} P {vb['pair_precision']:.4f} R {vb['pair_recall']:.4f} empty {vb['pred_empty_share']:.4f}")
    return rep


def run_test() -> None:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    for c in sorted(p.name for p in (V1 / "test" / "candidates").iterdir() if p.is_dir()):
        v1c = v1_top(sorted((V1 / "test" / "candidates" / c).glob("*.parquet")))
        s1 = pl.read_parquet(REC / "test_s1.parquet", columns=["gid", "country"]).filter(pl.col("country") == c)["gid"]
        build("test", c, s1, v1c, TEST_SCORED, booster=booster)


def assemble(name: str, threshold: float) -> dict:
    files = sorted(TEST_SCORED.glob("*.parquet"))
    pred = predictions_at_threshold(pl.concat([pl.scan_parquet(f).filter(pl.col("prob") >= threshold)
                                               .select("s1_gid", "s23_gid", "prob").collect() for f in files]), threshold)
    s1_map, s23_map = build_id_maps()
    all_s1 = s1_map.select("s1_id")
    dst = OUTPUT / name
    write_tsv(links_to_tsv(pred, s1_map, s23_map, all_s1, "matched_id"), "matched_id", "matched_entity_ids",
              dst / "matching_results.tsv")
    n_pairs, written = 0, set()
    with open(dst / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for f in files:
            b = pl.read_parquet(f, columns=["s1_gid", "s23_gid"])
            n_pairs += b.height
            rows = group_ids(b, s1_map, s23_map)
            written.update(rows["s1_id"].to_list())
            fh.write("".join(f"{x}\t{y}\n" for x, y in rows.iter_rows()))
        empty = set(all_s1["s1_id"].to_list()) - written
        fh.write("".join(f"{x}\t\n" for x in sorted(empty)))
    rep = {"name": name, "threshold": threshold, "n_predictions": pred.height, "candidate_pairs": n_pairs,
           "s1_with_candidates": len(written), "s1_without_candidates": len(empty),
           "s1_with_predictions": pred["s1_gid"].n_unique(), "n_test_s1": all_s1.height}
    (dst / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> int:
    mode = sys.argv[1]
    if mode == "fit":
        run_fit()
    elif mode == "val":
        run_val()
    elif mode == "train":
        train()
    elif mode == "test":
        run_test()
    elif mode == "assemble":
        t = float(sys.argv[3]) if len(sys.argv) > 3 else json.loads((OUT / "report_v3.json").read_text())["threshold"]
        print(json.dumps(assemble(sys.argv[2], t), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
