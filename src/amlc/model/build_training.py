"""D-S6 CLI: build the training set for one country (~50,000 FIT S1s, budget=150 candidates,
W2 features, FIT-only labels) via the fused pipeline. One country per process.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.build_training US <batch_size>
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

TRAIN_OUT = access.DATA / "_v1" / "training"


def main() -> int:
    country = sys.argv[1]
    batch_size = int(sys.argv[2])
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    print(f"=== training set {country}, batch_size={batch_size} (free RAM {free_ram_mb()} MB)", flush=True)

    t0 = time.time()
    gids = dataset.sample_fit_s1_gids(country)
    s1 = dataset.load_s1_rows(country, gids)
    s23 = dataset.load_s23(country)
    cand_dir = TRAIN_OUT / "candidates" / country
    feat_dir = TRAIN_OUT / "features" / country
    rep = fused.run_country(s1, s23, budget_rows, batch_size, cand_dir, feat_dir)

    labeled = dataset.label_batches(cand_dir, feat_dir, country)
    out_path = TRAIN_OUT / f"labeled_{country}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.write_parquet(out_path)

    summary = {
        "country": country, "batch_size": batch_size, "n_s1": rep["n_s1"], "n_batches": rep["n_batches"],
        "n_candidates": rep["n_candidates_total"], "n_positive_labels": int(labeled["label"].sum()),
        "positive_rate": float(labeled["label"].mean()), "runtime_s": round(time.time() - t0, 1),
        "peak_ram_mb": peak_ram_mb(), "out_path": str(out_path),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"training_{country}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
