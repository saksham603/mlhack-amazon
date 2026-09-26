"""Fused batched blocking + W2 feature pipeline (D-S1, v1 sprint prompt, 2026-09-26 approved).

Per country: build the S2/S3 key index (k23) and IDF table (rare) ONCE. Then, for each batch of
S1s: build S1 keys, score against k23/rare (name kinds + address kinds separately, Fix 2), union
top-100 name / top-50 addr into the budget=150 candidate set (D-S3: with blocking_score and
blocking_rank 1..150 columns), write that candidate batch to Parquet, compute W2 features for the
same batch (reusing the s23 pool already resident in memory -- no re-scan from disk), write the
feature batch to Parquet, then drop both before the next batch.

candidate_pairs.tsv (final submission format TBD) is derived from the candidate Parquet output.
"""
from pathlib import Path

import polars as pl

from amlc.blocking import v2
from amlc.features.w2 import REQUIRED_COLS as W2_COLS, pair_features

NAME_KINDS = v2.KINDS_NAME
ADDR_KINDS = v2.KINDS_ADDR
K_NAME = 100
K_ADDR = 50
CAP = 5000  # rarity cap, matches candidates.py / eval_w1_v2.py
# union of what build_keys needs (name_tokens, addr_components, addr_house_number, addr_zip) and
# what pair_features needs (adds name_legal_form); name_clean is not used by either but kept out.
S1_COLS = sorted(set(["name_tokens", "addr_components", "addr_house_number", "addr_zip"]) | set(W2_COLS))


def score_and_rank_batch(k1_batch: pl.DataFrame, k23: pl.DataFrame, rare: pl.DataFrame,
                          budget_rows: int, k_name: int = K_NAME, k_addr: int = K_ADDR) -> pl.DataFrame:
    """Returns (s1_gid, s23_gid, blocking_score, blocking_rank) for the unioned candidate set."""
    name_scored = v2.score(k1_batch, k23, rare, NAME_KINDS, budget_rows)
    addr_scored = v2.score(k1_batch, k23, rare, ADDR_KINDS, budget_rows)
    name_top = v2.topk(name_scored, k_name).join(name_scored, on=["s1_gid", "s23_gid"], how="left")
    addr_top = v2.topk(addr_scored, k_addr).join(addr_scored, on=["s1_gid", "s23_gid"], how="left")

    merged = (name_top.rename({"score": "name_score"})
              .join(addr_top.rename({"score": "addr_score"}), on=["s1_gid", "s23_gid"], how="full", coalesce=True)
              .with_columns(pl.col("name_score").fill_null(0.0), pl.col("addr_score").fill_null(0.0))
              .with_columns((pl.col("name_score") + pl.col("addr_score")).alias("blocking_score")))
    ranked = (merged.sort(["s1_gid", "blocking_score", "s23_gid"], descending=[False, True, False])
              .with_columns(pl.col("s23_gid").cum_count().over("s1_gid").alias("blocking_rank")))
    return ranked.select("s1_gid", "s23_gid", "blocking_score", "blocking_rank")


def build_country_index(s23: pl.DataFrame, cap: int = CAP) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds (k23, rare) ONCE for a country's S2/S3 pool. Reuse the result across every S1 sample
    drawn from that same (dataset, country) pool (e.g. FIT and VALIDATION both draw from the same
    TRAIN pool) instead of rebuilding this ~20min step per sample (user go, 2026-09-26 18:30)."""
    kinds = NAME_KINDS + ADDR_KINDS
    k23 = v2.build_keys(s23.select("s23_gid", *[c for c in S1_COLS if c != "name_legal_form"]), "s23_gid", kinds)
    rare = v2.idf_table(k23, s23.height, cap)
    return k23, rare


def run_batch(s1_batch: pl.DataFrame, s23: pl.DataFrame, k23: pl.DataFrame, rare: pl.DataFrame,
              budget_rows: int, candidates_dir: Path, features_dir: Path, batch_idx: int) -> dict:
    """s1_batch: (s1_gid, *S1_COLS). s23: (s23_gid, *S1_COLS) -- the full country pool, resident once.
    Resumable (sprint 6a): if this batch's candidate+feature parquet already exist, skip recomputing.
    """
    cand_path = candidates_dir / f"candidates_{batch_idx:05d}.parquet"
    feat_path = features_dir / f"features_{batch_idx:05d}.parquet"
    if cand_path.exists() and feat_path.exists():
        return {"batch_idx": batch_idx, "n_s1": s1_batch.height,
                "n_candidates": pl.scan_parquet(cand_path).select(pl.len()).collect().item(), "skipped": True}

    kinds = NAME_KINDS + ADDR_KINDS
    k1_batch = v2.build_keys(s1_batch.select("s1_gid", *[c for c in S1_COLS if c != "name_legal_form"]),
                              "s1_gid", kinds)
    candidates = score_and_rank_batch(k1_batch, k23, rare, budget_rows)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(cand_path)

    s23_needed = s23.join(candidates.select("s23_gid").unique(), on="s23_gid", how="semi")
    feats = pair_features(candidates.select("s1_gid", "s23_gid"), s1_batch, s23_needed)
    features_dir.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(feat_path)

    n = candidates.height
    del k1_batch, candidates, s23_needed, feats
    return {"batch_idx": batch_idx, "n_s1": s1_batch.height, "n_candidates": n, "skipped": False}


def run_country_with_index(s1_full: pl.DataFrame, s23: pl.DataFrame, k23: pl.DataFrame, rare: pl.DataFrame,
                            budget_rows: int, batch_size: int, candidates_dir: Path, features_dir: Path) -> dict:
    """Same as run_country, but takes an already-built (k23, rare) index (build_country_index)."""
    n_s1 = s1_full.height
    n_batches = (n_s1 + batch_size - 1) // batch_size
    batch_reports = []
    for bi in range(n_batches):
        batch = s1_full.slice(bi * batch_size, batch_size)
        batch_reports.append(run_batch(batch, s23, k23, rare, budget_rows, candidates_dir, features_dir, bi))
    return {"n_s1": n_s1, "n_batches": n_batches, "batch_size": batch_size,
            "n_candidates_total": sum(b["n_candidates"] for b in batch_reports), "batches": batch_reports}


def run_country(s1_full: pl.DataFrame, s23: pl.DataFrame, budget_rows: int, batch_size: int,
                 candidates_dir: Path, features_dir: Path) -> dict:
    """s1_full: ALL S1 rows for this country (s1_gid, *S1_COLS), not sampled. Builds k23/rare once."""
    k23, rare = build_country_index(s23)
    return run_country_with_index(s1_full, s23, k23, rare, budget_rows, batch_size, candidates_dir, features_dir)
