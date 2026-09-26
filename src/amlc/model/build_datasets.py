"""Sprint step 3: builds the FIT training set and VALIDATION candidate/feature set for one
country, reusing ONE S2/S3 key index for both (user go, 2026-09-26 18:30). One country per
process, resumable (fused.run_batch skips already-written batches).

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.build_datasets US 3000
"""
import json
import sys
import time

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import BYTES_PER_JOIN_ROW
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access
from amlc.model import dataset, validate
from amlc.pipeline import fused

TRAIN_OUT = access.DATA / "_v1" / "training"
VAL_OUT = access.DATA / "_v1" / "validation"
LOG_DIR = access.DATA / "_v1" / "logs"


def _log(path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg} (RSS {peak_ram_mb()} MB)"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_country(country: str, batch_size: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"build_datasets_{country}.log"
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    _log(log_path, f"start {country}, batch_size={batch_size}, free RAM {free_ram_mb()} MB")

    s23 = dataset.load_s23(country)
    k23, rare = fused.build_country_index(s23)
    _log(log_path, "S2/S3 index built (reused for FIT + VALIDATION)")

    fit_gids = dataset.sample_fit_s1_gids(country)
    fit_s1 = dataset.load_s1_rows(country, fit_gids)
    fit_cand_dir, fit_feat_dir = TRAIN_OUT / "candidates" / country, TRAIN_OUT / "features" / country
    fit_rep = fused.run_country_with_index(fit_s1, s23, k23, rare, budget_rows, batch_size, fit_cand_dir, fit_feat_dir)
    _log(log_path, f"FIT candidates+features done: {fit_rep['n_batches']} batches, "
                    f"{fit_rep['n_candidates_total']:,} candidates")

    labeled = dataset.label_batches(fit_cand_dir, fit_feat_dir, country)
    labeled_path = TRAIN_OUT / f"labeled_{country}.parquet"
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(labeled_path)
    _log(log_path, f"FIT labeled: {labeled.height:,} rows, {int(labeled['label'].sum()):,} positive")

    a, b = validate.sample_validation_halves()
    val_gids = pl.concat([a.to_frame(), b.to_frame()], how="vertical").unique()
    val_s1 = dataset.load_s1_rows(country, val_gids)
    val_cand_dir, val_feat_dir = VAL_OUT / "candidates" / country, VAL_OUT / "features" / country
    val_rep = fused.run_country_with_index(val_s1, s23, k23, rare, budget_rows, batch_size, val_cand_dir, val_feat_dir)
    _log(log_path, f"VALIDATION candidates+features done: {val_rep['n_batches']} batches, "
                    f"{val_rep['n_candidates_total']:,} candidates")

    summary = {"country": country, "batch_size": batch_size, "fit": fit_rep, "validation": val_rep,
               "n_positive_labels": int(labeled["label"].sum()), "positive_rate": float(labeled["label"].mean()),
               "peak_ram_mb": peak_ram_mb()}
    (OUT_DIR / f"build_datasets_{country}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(log_path, "country complete")
    return summary


def main() -> int:
    country = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    print(json.dumps(build_country(country, batch_size), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
