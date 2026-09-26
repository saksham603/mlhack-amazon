"""Generate the locked D_plus_budget_150 candidate set at an arbitrary S1 sample fraction, for
scale/RAM testing (user go, 2026-09-26). NOT the frozen T7 1% evaluation sample -- this draws a
fresh, unfrozen sample and computes no recall/F0.5 (no ground truth needed), it only produces
candidate pairs at a chosen scale so W2 feature extraction can be measured against a realistic
larger slice in a SEPARATE process (sequencing: this process exits and frees its RAM before any
W2 process starts).

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.blocking.gen_candidates_scale US 0.10
"""
import json
import sys

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking import v2
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import CAP, COLS, BYTES_PER_JOIN_ROW, load_pool
from amlc.cleaning.c1_run import peak_ram_mb


def build(country: str, fraction: float) -> tuple[pl.DataFrame, int]:
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    s1 = C.sample_fit_s1(country, fraction=fraction).select(COLS[1:] + ["s1_gid"])
    s23 = load_pool(country)

    kinds = v2.KINDS_NAME + v2.KINDS_ADDR
    k1 = v2.build_keys(s1.select("s1_gid", *COLS[1:]), "s1_gid", kinds)
    k23 = v2.build_keys(s23, "s23_gid", kinds)
    rare = v2.idf_table(k23, s23.height, CAP)
    del s23

    name_all = v2.score(k1, k23, rare, ("tok", "pair", "nospace"), budget_rows)
    check_ram_budget()
    addr = v2.score(k1, k23, rare, ("house", "zip"), budget_rows)
    final = v2.union_candidates(v2.topk(name_all, 100), v2.topk(addr, 50))
    return final.select("s1_gid", "s23_gid"), s1.height


def main() -> int:
    country = sys.argv[1]
    fraction = float(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== {country} candidates @ fraction={fraction} (free RAM {free_ram_mb()} MB)", flush=True)
    final, n_s1 = build(country, fraction)
    tag = f"{country}_frac{fraction:g}"
    final.write_parquet(OUT_DIR / f"candidates_scale_{tag}.parquet")
    rep = {"country": country, "fraction": fraction, "n_s1": n_s1, "n_candidates": final.height,
           "candidates_per_s1": final.height / n_s1, "peak_ram_mb": peak_ram_mb()}
    (OUT_DIR / f"candidates_scale_{tag}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
