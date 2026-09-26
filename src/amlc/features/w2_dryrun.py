"""W2 dry run (user go, 2026-09-26, triage step 4): measure peak RAM extracting the minimal
feature set over a real candidate slice, using the BATCHED writer (write_features_batched) so
peak RAM stays bounded by batch_size, not by total candidate-pair count.

Reuses the candidate pairs already persisted by oracle_ceiling.py (D_plus_budget_150, the locked
budget=150 blocking set) for one country's frozen T7 1% S1 sample. Blocking is NOT re-run in this
process (sequencing: candidate generation and feature extraction never share a process/peak, per
user decision 2026-09-26).

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.features.w2_dryrun US [batch_size]
"""
import json
import shutil
import sys

import polars as pl

from amlc.blocking.candidates import SILVER_DIR
from amlc.blocking.dryrun_w1 import OUT_DIR, free_ram_mb
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.features.w2 import REQUIRED_COLS, write_features_batched


def load_sides_for_batch(batch_candidates: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Country is irrelevant here: candidates already came from a country-restricted blocking run, so
    # the gids alone determine which rows are needed. Lazy scan + column selection + semi-join lets
    # Polars push the filter down instead of materialising full unfiltered multi-million-row tables.
    needed = ["gid", *REQUIRED_COLS]
    s1_gids = batch_candidates.lazy().select(pl.col("s1_gid").alias("gid")).unique()
    s23_gids = batch_candidates.lazy().select(pl.col("s23_gid").alias("gid")).unique()

    s1 = (pl.scan_parquet(SILVER_DIR / "train_source1.parquet").select(needed)
          .join(s1_gids, on="gid", how="semi").rename({"gid": "s1_gid"}).collect())
    s23 = pl.concat(
        [pl.scan_parquet(SILVER_DIR / f"train_source{s}.parquet").select(needed).join(s23_gids, on="gid", how="semi")
         for s in (2, 3)], how="vertical"
    ).rename({"gid": "s23_gid"}).collect()
    return s1, s23


def run(candidates_path, tag: str, batch_size: int) -> dict:
    print(f"=== {tag} W2 batched dry run, batch_size={batch_size} (free RAM {free_ram_mb()} MB)", flush=True)
    candidates = pl.read_parquet(candidates_path)

    out_dir = OUT_DIR / f"w2_features_dryrun_{tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    batch_rep = write_features_batched(candidates, load_sides_for_batch, out_dir, batch_size=batch_size)
    peak = peak_ram_mb()

    rep = {
        "tag": tag,
        "slice_n_candidates": candidates.height,
        **batch_rep,
        "peak_ram_mb": peak,
        "ram_ceiling_mb": 12288,
        "within_12gb_ceiling": peak < 12288,
    }
    assert batch_rep["feature_rows_total"] == candidates.height, "batched writer dropped or duplicated rows"
    return rep


def main() -> int:
    tag = sys.argv[1]                      # e.g. "US" or "US_frac0.1"; also used for output naming
    candidates_path = OUT_DIR / sys.argv[2] if len(sys.argv) > 2 else OUT_DIR / f"oracle_final_candidates_{tag}.parquet"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    rep = run(candidates_path, tag, batch_size)
    (OUT_DIR / f"w2_dryrun_{tag}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
