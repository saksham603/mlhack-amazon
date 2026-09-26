"""Sprint step 5: full TEST-set candidates + predictions for one country (India, US or France).
Reuses the fused batched pipeline (resumable) against the TEST silver pool, then scores every
candidate with the trained model and writes (s1_gid, s23_gid, prob) alongside the raw candidates.

One country per process. Progress logged to data/_v1/logs/test_<country>.log.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.run_test_country US 3000
"""
import json
import sys
import time

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import BYTES_PER_JOIN_ROW
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access
from amlc.model import train as T
from amlc.pipeline import fused

TEST_OUT = access.DATA / "_v1" / "test"
LOG_DIR = access.DATA / "_v1" / "logs"
MODEL_PATH = access.ROOT / "models" / "lgbm_v1.txt"


def _log(path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg} (RSS {peak_ram_mb()} MB)"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_test_s1(country: str) -> pl.DataFrame:
    silver = C.load_silver_with_country("test", 1).filter(pl.col("country") == country)
    return silver.rename({"gid": "s1_gid"}).select("s1_gid", *fused.S1_COLS)


def load_test_s23(country: str) -> pl.DataFrame:
    parts = [C.load_silver_with_country("test", s).filter(pl.col("country") == country) for s in (2, 3)]
    s23 = pl.concat(parts, how="vertical").rename({"gid": "s23_gid"})
    return s23.select("s23_gid", *fused.S1_COLS)


def score_batches(cand_dir, feat_dir, scored_dir, booster) -> int:
    """Scores one candidate/feature batch pair at a time (peak RAM bounded by one batch, not by the
    whole country) and writes scored_<i>.parquet. Resumable: an existing scored batch is skipped.
    Raises if a batch's candidate and feature files disagree -- the symptom of the concurrent-writer
    corruption found 2026-09-26 21:10 -- instead of letting an inner join drop rows silently."""
    scored_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for cand_path in sorted(cand_dir.glob("candidates_*.parquet")):
        idx = cand_path.stem.split("_")[1]
        out_path = scored_dir / f"scored_{idx}.parquet"
        if out_path.exists():
            total += pl.scan_parquet(out_path).select(pl.len()).collect().item()
            continue
        cand = pl.read_parquet(cand_path)
        feat = pl.read_parquet(feat_dir / f"features_{idx}.parquet")
        if cand.height != feat.height:
            raise RuntimeError(f"batch {idx}: {cand.height} candidates vs {feat.height} feature rows. STOP.")
        merged = cand.join(feat, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
        if merged.height != cand.height:
            raise RuntimeError(f"batch {idx}: join kept {merged.height} of {cand.height} rows. STOP.")
        probs = T.predict_proba(booster, merged)
        (merged.select("s1_gid", "s23_gid", "blocking_score", "blocking_rank")
         .with_columns(pl.Series("prob", probs)).write_parquet(out_path))
        total += merged.height
    return total


def main() -> int:
    import lightgbm as lgb

    country = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"test_{country}.log"
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    _log(log_path, f"start TEST {country}, batch_size={batch_size}, free RAM {free_ram_mb()} MB")

    s1 = load_test_s1(country)
    s23 = load_test_s23(country)
    _log(log_path, f"loaded: {s1.height:,} test S1, {s23.height:,} test S2/S3 pool")

    k23, rare = fused.build_country_index(s23)
    _log(log_path, "S2/S3 index built")

    cand_dir = TEST_OUT / "candidates" / country
    feat_dir = TEST_OUT / "features" / country
    rep = fused.run_country_with_index(s1, s23, k23, rare, budget_rows, batch_size, cand_dir, feat_dir)
    _log(log_path, f"candidates+features done: {rep['n_batches']} batches, {rep['n_candidates_total']:,} candidates")

    del s1, s23, k23, rare
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    scored_dir = TEST_OUT / "scored" / country
    n_scored = score_batches(cand_dir, feat_dir, scored_dir, booster)
    _log(log_path, f"scored {n_scored:,} candidates in batches, written to {scored_dir}")

    summary = {"country": country, "n_s1": rep["n_s1"], "n_candidates": rep["n_candidates_total"],
               "n_scored": n_scored, "n_batches": rep["n_batches"], "batch_size": batch_size,
               "peak_ram_mb": peak_ram_mb()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"test_{country}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(log_path, "country complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
