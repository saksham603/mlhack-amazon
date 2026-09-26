"""Extra FIT training data: N more FIT S1 per country, disjoint from the v1 50k sample, through the
same fused batched pipeline (resumable: run_batch skips written batches, labeling skips written
labeled files). Labels are FIT-only and joined per batch so RAM stays bounded by one batch.
Builds data only; no training.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.build_fit_extra US 3000
"""
import json
import sys
import time

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import BYTES_PER_JOIN_ROW
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access
from amlc.model import dataset
from amlc.pipeline import fused

EXTRA_OUT = access.DATA / "_v1" / "training_extra"
LOG_DIR = access.DATA / "_v1" / "logs"
N_EXTRA = 250_000
SEED_EXTRA = dataset.SEED + 100
RAM_CEILING_MB = 10_752


def _log(path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg} (RSS {peak_ram_mb()} MB)"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sample_extra_fit_s1_gids(country: str, n: int = N_EXTRA) -> pl.DataFrame:
    split = access.load_split().filter((pl.col("split") == "FIT") & (pl.col("country") == country))
    pool = split.join(dataset.sample_fit_s1_gids(country), on="s1_gid", how="anti").sort("s1_gid")
    return pool.sample(n=min(n, pool.height), seed=SEED_EXTRA, shuffle=True, with_replacement=False).select("s1_gid")


def label_each_batch(cand_dir, feat_dir, labeled_dir, country: str) -> dict:
    labeled_dir.mkdir(parents=True, exist_ok=True)
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").with_columns(pl.lit(1, pl.Int32).alias("label"))
    n_rows = n_pos = 0
    for cp in sorted(cand_dir.glob("candidates_*.parquet")):
        idx = cp.stem.split("_")[1]
        out = labeled_dir / f"labeled_{idx}.parquet"
        if not out.exists():
            cand = pl.read_parquet(cp)
            feat = pl.read_parquet(feat_dir / f"features_{idx}.parquet")
            merged = cand.join(feat, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
            if merged.height != cand.height or feat.height != cand.height:
                raise RuntimeError(f"batch {idx}: {cand.height} candidates, {feat.height} features, "
                                   f"{merged.height} joined. STOP.")
            (merged.join(truth, on=["s1_gid", "s23_gid"], how="left")
             .with_columns(pl.col("label").fill_null(0), pl.lit(country).alias("country"))
             .write_parquet(out))
        stats = pl.scan_parquet(out).select(pl.len(), pl.col("label").sum()).collect().row(0)
        n_rows += stats[0]
        n_pos += int(stats[1])
    return {"n_labeled_rows": n_rows, "n_positive": n_pos}


def build_country(country: str, batch_size: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"build_fit_extra_{country}.log"
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    _log(log_path, f"start {country}, n={N_EXTRA:,}, batch_size={batch_size}, free RAM {free_ram_mb()} MB")

    gids = sample_extra_fit_s1_gids(country)
    overlap = gids.join(dataset.sample_fit_s1_gids(country), on="s1_gid", how="semi").height
    if overlap:
        raise RuntimeError(f"{overlap} extra S1 overlap the v1 FIT sample. STOP.")
    s1 = dataset.load_s1_rows(country, gids)
    s23 = dataset.load_s23(country)
    _log(log_path, f"loaded {s1.height:,} extra FIT S1 (0 overlap with v1 sample), {s23.height:,} S2/S3 pool")
    k23, rare = fused.build_country_index(s23)
    _log(log_path, "S2/S3 index built")
    if peak_ram_mb() > RAM_CEILING_MB:
        raise RuntimeError(f"peak RAM {peak_ram_mb()} MB > {RAM_CEILING_MB} MB. STOP.")

    cand_dir, feat_dir = EXTRA_OUT / "candidates" / country, EXTRA_OUT / "features" / country
    rep = fused.run_country_with_index(s1, s23, k23, rare, budget_rows, batch_size, cand_dir, feat_dir)
    _log(log_path, f"candidates+features done: {rep['n_batches']} batches, {rep['n_candidates_total']:,} candidates")
    del s23, k23, rare, s1

    lab = label_each_batch(cand_dir, feat_dir, EXTRA_OUT / "labeled" / country, country)
    _log(log_path, f"labeled: {lab['n_labeled_rows']:,} rows, {lab['n_positive']:,} positive")

    summary = {"country": country, "n_s1": rep["n_s1"], "batch_size": batch_size, "n_batches": rep["n_batches"],
               "n_candidates": rep["n_candidates_total"], **lab, "peak_ram_mb": peak_ram_mb()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"build_fit_extra_{country}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(log_path, "country complete")
    return summary


def main() -> int:
    country = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    print(json.dumps(build_country(country, batch_size), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
