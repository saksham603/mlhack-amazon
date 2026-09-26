"""Blocking-only oracle ceiling (user go, 2026-09-26, triage step 3): the max macro F_0.5 any
model could reach with the locked D_plus_budget_150 candidate set, if it predicted exactly the
true pairs present among each S1's candidates and nothing else.

Same T7 sample, same ground truth, same budget=150 candidate construction as eval_w1_v2.py
(reused directly, not reimplemented). Per S1: oracle predicts exactly its true links that are
present in its own candidate set (precision=1 by construction on those); F_0.5 uses the SAME
per_s1_f05/macro functions as the real W0 scorer, so the singleton rule (empty truth -> f05=1.0
for an empty prediction) is identical.

One country per process (RAM discipline, same as eval_w1_v2.py):
  python -m amlc.blocking.oracle_ceiling India
  python -m amlc.blocking.oracle_ceiling US
"""
import json
import sys

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking import v2
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.blocking.eval_w1_v2 import CAP, COLS, EXPECTED, BYTES_PER_JOIN_ROW, frozen_sample, load_pool
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.eval import metric
from amlc.foundation import access


def oracle_final(country: str) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Returns (final_candidates, truth, n_s1) for the locked D_plus_budget_150 variant."""
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)
    s1 = frozen_sample(country)
    truth = (access.load_labels("FIT").select("s1_gid", "s23_gid")
             .join(s1.select("s1_gid"), on="s1_gid", how="semi"))
    if (s1.height, truth.height) != EXPECTED[country]:
        raise RuntimeError(f"{country}: population {(s1.height, truth.height)} != {EXPECTED[country]}. STOP.")
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
    return final.select("s1_gid", "s23_gid"), truth, s1.height


def oracle_score(final: pl.DataFrame, truth: pl.DataFrame, s1_gids: pl.Series) -> dict:
    """Oracle prediction per S1: exactly its true links present in its candidate set."""
    oracle_pred = truth.join(final, on=["s1_gid", "s23_gid"], how="semi")
    per = metric.per_s1_f05(oracle_pred, truth, s1_gids)
    return {
        "f05_macro": metric.macro(per),
        "n_scored": per.height,
        "n_true_links": truth.height,
        "n_recallable": oracle_pred.height,
        "recall_ceiling": oracle_pred.height / truth.height,
    }


def main() -> int:
    country = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== {country} oracle ceiling (free RAM {free_ram_mb()} MB)", flush=True)
    final, truth, n_s1 = oracle_final(country)
    s1 = frozen_sample(country)
    overall = oracle_score(final, truth, s1["s1_gid"])
    final.write_parquet(OUT_DIR / f"oracle_final_candidates_{country}.parquet")
    rep = {"country": country, "n_s1": n_s1, **overall, "peak_ram_mb": peak_ram_mb()}
    (OUT_DIR / f"oracle_ceiling_{country}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
