"""CLI entry for the fused batched pipeline (D-S1), one country per process.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.pipeline.run_country <country> <batch_size> \
       [--frac F] [--dataset train|test] [--out-tag TAG]

--frac draws a fresh (unfrozen) S1 sample of that fraction via candidates.sample_fit_s1 -- for
scale/RAM testing only. Omit --frac to use ALL S1 rows for that country/dataset (a real run).
"""
import argparse
import json
import time

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import BYTES_PER_JOIN_ROW
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access
from amlc.pipeline import fused

PIPELINE_OUT = access.DATA / "_v1"


def load_s1(country: str, dataset: str, frac: float | None) -> pl.DataFrame:
    if frac is not None:
        s1 = C.sample_fit_s1(country, fraction=frac)  # train FIT only; frac is for scale-testing
    else:
        silver = C.load_silver_with_country(dataset, 1).filter(pl.col("country") == country)
        s1 = silver.rename({"gid": "s1_gid"})
    return s1.select("s1_gid", *fused.S1_COLS)


def load_s23(country: str, dataset: str) -> pl.DataFrame:
    parts = [C.load_silver_with_country(dataset, s).filter(pl.col("country") == country) for s in (2, 3)]
    s23 = pl.concat(parts, how="vertical").rename({"gid": "s23_gid"})
    return s23.select("s23_gid", *fused.S1_COLS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("country")
    ap.add_argument("batch_size", type=int)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--dataset", default="train", choices=["train", "test"])
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    tag = args.out_tag or (f"{args.dataset}_{args.country}" + (f"_frac{args.frac:g}" if args.frac else ""))
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    print(f"=== {tag} fused pipeline, batch_size={args.batch_size} (free RAM {free_ram_mb()} MB)", flush=True)

    t0 = time.time()
    s1 = load_s1(args.country, args.dataset, args.frac)
    s23 = load_s23(args.country, args.dataset)
    t_load = time.time() - t0

    candidates_dir = PIPELINE_OUT / "candidates" / tag
    features_dir = PIPELINE_OUT / "features" / tag
    t1 = time.time()
    rep = fused.run_country(s1, s23, budget_rows, args.batch_size, candidates_dir, features_dir)
    t_run = time.time() - t1

    out = {
        "tag": tag, "country": args.country, "dataset": args.dataset, "frac": args.frac,
        "batch_size": args.batch_size, "n_s1": rep["n_s1"], "n_batches": rep["n_batches"],
        "n_candidates_total": rep["n_candidates_total"],
        "candidates_per_s1": rep["n_candidates_total"] / rep["n_s1"] if rep["n_s1"] else None,
        "load_s_s": round(t_load, 1), "run_s": round(t_run, 1), "total_s": round(time.time() - t0, 1),
        "s1_per_sec": rep["n_s1"] / t_run if t_run else None,
        "peak_ram_mb": peak_ram_mb(), "ram_ceiling_mb": 12288,
    }
    out["within_ceiling"] = out["peak_ram_mb"] < 10.5 * 1024
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"pipeline_{tag}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
