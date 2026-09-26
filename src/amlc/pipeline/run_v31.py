"""Pipeline v3.1 = v3 candidates + blocking-v3.1 address candidates (top AMLC_KA by address-pair /
exact-name keys), memory-safe after two session crashes on 2026-09-27 (name + address index together
= 122M keys, paging). Rules: name and address indexes are never in memory together; FIT/VALIDATION
reuse the name candidates already stored in data/_v3; free RAM is logged per batch; every step is
resumable per batch file and checks its row counts.

Measured on the FIT v1 sample (address top-10 added): recall India 0.936 -> 0.966, US 0.971 -> 0.981,
candidates per S1 44.7 -> 50.7 / 43.6 -> 47.0.

Run with AMLC_KA=10 (PYTHONPATH=src):  python -m amlc.pipeline.run_v31 fit | val | test
Then:  python -m amlc.pipeline.run_v3 train   and   python -m amlc.pipeline.run_v3 assemble <name>
(same AMLC_KA=10, so run_v3 reads/writes data/_v31 and models/lgbm_v31.txt).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from amlc.blocking import v3
from amlc.features import w3
from amlc.foundation import access
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet, free_gb

if R.KA <= 0:
    raise SystemExit("set AMLC_KA (e.g. 10) so run_v3 uses data/_v31 and the address features")
V30 = access.DATA / "_v3"
KEY_COLS = ["gid", "core", "addr", "region", "ns", "skel"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[free RAM {free_gb()} GB]", flush=True)


def _finish(u: pl.DataFrame, r1, r23, country, labels, booster, dst: Path) -> int:
    x = w3.pair_features(u, r1, r23).with_columns(pl.lit(country).alias("country"))
    if x.height != u.height:
        raise AssertionError(f"{dst.name}: feature rows {x.height} != candidates {u.height}. STOP.")
    if labels is not None:
        x = (x.join(labels.with_columns(pl.lit(1, pl.Int32).alias("label")), on=["s1_gid", "s23_gid"], how="left")
             .with_columns(pl.col("label").fill_null(0)))
    if booster is not None:
        x = x.with_columns(pl.Series("prob", booster.predict(R.to_x(x))))
    atomic_write_parquet(x, dst)
    return x.height


def from_v30(sub: str, labels=None) -> None:
    """FIT / VALIDATION: name candidates from data/_v3/<sub> + address candidates -> data/_v31/<sub>."""
    out = R.OUT / sub
    out.mkdir(parents=True, exist_ok=True)
    for c in ("India", "US"):
        files = sorted((V30 / sub).glob(f"{c}_*.parquet"))
        todo = [f for f in files if not (out / f.name).exists()]
        if not todo:
            log(f"{sub}/{c}: all {len(files)} files present")
            continue
        t0 = time.time()
        r1, r23 = R.records("train", c)
        k23, rare = v3.build_index(r23.select(KEY_COLS), v3.ADDR_KINDS)
        log(f"{sub}/{c}: address index {k23.height:,} keys ({time.time() - t0:.0f}s), {len(todo)} files to do")
        n = 0
        for i, f in enumerate(todo):
            old = pl.read_parquet(f, columns=["s1_gid", "s23_gid", "v1_score", "v1_rank", "v3_score", "v3_rank"])
            c1 = old.filter(pl.col("v1_rank") <= R.K1).select("s1_gid", "s23_gid", "v1_score", "v1_rank")
            c3 = old.filter(pl.col("v3_rank") <= R.K3).select("s1_gid", "s23_gid", "v3_score", "v3_rank")
            s1 = old.select(pl.col("s1_gid").alias("gid")).unique()
            ca = v3.candidates_addr(r1.select(KEY_COLS).join(s1, on="gid", how="semi"), k23, rare, R.KA)
            n += _finish(R.union(c1, c3, ca), r1, r23, c, labels, None, out / f.name)
            if i % 10 == 0:
                log(f"{sub}/{c}: {i + 1}/{len(todo)} files, {n:,} rows")
        log(f"{sub}/{c}: done, {n:,} rows ({time.time() - t0:.0f}s)")
        del k23, rare, r1, r23


def test() -> None:
    booster = lgb.Booster(model_file=str(R.MODEL_PATH))
    c3_dir, out = R.OUT / "test_c3", R.TEST_SCORED
    c3_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for c in sorted(p.name for p in (R.V1 / "test" / "candidates").iterdir() if p.is_dir()):
        t0 = time.time()
        r1, r23 = R.records("test", c)
        r1s = r1.sort("gid")
        batches = [(i // R.BATCH, r1s.slice(i, R.BATCH)) for i in range(0, r1s.height, R.BATCH)]
        need3 = [(b, rb) for b, rb in batches if not (c3_dir / f"{c}_{b:05d}.parquet").exists()]
        if need3:  # pass 1: name index only
            k23, rare = v3.build_index(r23.select(KEY_COLS), v3.NAME_KINDS)
            log(f"test/{c}: name index {k23.height:,} keys, {len(need3)} batches")
            for b, rb in need3:
                atomic_write_parquet(v3.candidates(rb.select(KEY_COLS), k23, rare, R.K3), c3_dir / f"{c}_{b:05d}.parquet")
            del k23, rare
        need = [(b, rb) for b, rb in batches if not (out / f"{c}_{b:05d}.parquet").exists()]
        if not need:
            log(f"test/{c}: all {len(batches)} batches present")
            continue
        v1c = R.v1_top(sorted((R.V1 / "test" / "candidates" / c).glob("*.parquet")))
        k23, rare = v3.build_index(r23.select(KEY_COLS), v3.ADDR_KINDS)  # pass 2: address index only
        log(f"test/{c}: address index {k23.height:,} keys, {len(need)} batches to score")
        n = 0
        for j, (b, rb) in enumerate(need):
            c3 = pl.read_parquet(c3_dir / f"{c}_{b:05d}.parquet")
            ca = v3.candidates_addr(rb.select(KEY_COLS), k23, rare, R.KA)
            c1 = v1c.join(rb.select(pl.col("gid").alias("s1_gid")), on="s1_gid", how="semi")
            u = R.union(c1, c3, ca)
            if u.height:
                n += _finish(u, r1, r23, c, None, booster, out / f"{c}_{b:05d}.parquet")
            if j % 10 == 0:
                log(f"test/{c}: {j + 1}/{len(need)} batches, {n:,} rows")
        log(f"test/{c}: done, {n:,} rows ({time.time() - t0:.0f}s)")
        del k23, rare, v1c, r1, r23


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        from_v30("fit", access.load_labels("FIT").select("s1_gid", "s23_gid"))
    elif mode == "val":
        from_v30("val")
    elif mode == "test":
        test()
