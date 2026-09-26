"""W2 minimal feature set (user go, 2026-09-26, triage step 4): pairwise features for candidate
(s1_gid, s23_gid) links produced by blocking. Label-free: only reads the same silver columns
blocking already reads. No numpy/sklearn/lightgbm dependency -- pure Polars.

Features (one row per candidate pair):
  name_jaccard          Jaccard of name_tokens (List[str]) sets.
  name_overlap_n        |intersection| of name_tokens (raw count, not normalised).
  addr_token_jaccard     Jaccard of addr_components tokens (each component split on whitespace).
  house_number_match    addr_house_number equal, both non-empty.
  zip_match              addr_zip equal, both non-empty. Known low-value: addr_zip is frequently
                          just addr_house_number duplicated (STAGE1_ZIP_GATE_DECISION.md) --
                          kept as a feature since a model can learn to discount it, not dropped.
  legal_form_match      name_legal_form equal, both non-empty.
  legal_form_both_present  both sides have a non-empty legal form (regardless of match).

REQUIRED_COLS lists the silver columns each side (s1 / s23) must carry.
"""
from pathlib import Path

import polars as pl

REQUIRED_COLS = ["name_tokens", "addr_components", "addr_house_number", "addr_zip", "name_legal_form"]


def _addr_word_tokens(col: str) -> pl.Expr:
    # addr_components is List[str] of comma-separated pieces; join then re-split into flat words.
    return (pl.col(col).list.join(" ").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")))


def pair_features(candidates: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """candidates: (s1_gid, s23_gid). s1: (s1_gid, *REQUIRED_COLS). s23: (s23_gid, *REQUIRED_COLS)."""
    left = s1.select("s1_gid", *[pl.col(c).name.suffix("_1") for c in REQUIRED_COLS])
    right = s23.select("s23_gid", *[pl.col(c).name.suffix("_2") for c in REQUIRED_COLS])
    df = (candidates.join(left, on="s1_gid", how="left", validate="m:1")
          .join(right, on="s23_gid", how="left", validate="m:1"))

    name_inter = pl.col("name_tokens_1").list.set_intersection(pl.col("name_tokens_2"))
    name_union = pl.col("name_tokens_1").list.set_union(pl.col("name_tokens_2"))
    addr_w1 = _addr_word_tokens("addr_components_1")
    addr_w2 = _addr_word_tokens("addr_components_2")
    addr_inter = addr_w1.list.set_intersection(addr_w2)
    addr_union = addr_w1.list.set_union(addr_w2)

    return df.select(
        "s1_gid", "s23_gid",
        pl.when(name_union.list.len() > 0).then(name_inter.list.len() / name_union.list.len())
          .otherwise(0.0).alias("name_jaccard"),
        name_inter.list.len().alias("name_overlap_n"),
        pl.when(addr_union.list.len() > 0).then(addr_inter.list.len() / addr_union.list.len())
          .otherwise(0.0).alias("addr_token_jaccard"),
        ((pl.col("addr_house_number_1") != "") & (pl.col("addr_house_number_2") != "")
         & (pl.col("addr_house_number_1") == pl.col("addr_house_number_2"))).alias("house_number_match"),
        ((pl.col("addr_zip_1") != "") & (pl.col("addr_zip_2") != "")
         & (pl.col("addr_zip_1") == pl.col("addr_zip_2"))).alias("zip_match"),
        ((pl.col("name_legal_form_1") != "") & (pl.col("name_legal_form_2") != "")
         & (pl.col("name_legal_form_1") == pl.col("name_legal_form_2"))).alias("legal_form_match"),
        ((pl.col("name_legal_form_1") != "") & (pl.col("name_legal_form_2") != "")).alias("legal_form_both_present"),
    )


def write_features_batched(candidates: pl.DataFrame, load_sides_fn, out_dir: Path, batch_size: int = 10_000) -> dict:
    """Process S1s in batches of `batch_size`, writing each batch's features to its own parquet file
    and dropping it before the next batch, so peak RAM stays proportional to batch_size, not to the
    total candidate-pair count. `load_sides_fn(batch_candidates) -> (s1_batch, s23_batch)` supplies
    the silver rows needed for exactly that batch (caller decides how, e.g. lazy scan + semi-join).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_gids = candidates["s1_gid"].unique().sort()
    n_s1 = s1_gids.len()
    n_batches = (n_s1 + batch_size - 1) // batch_size
    total_rows = 0
    for bi in range(n_batches):
        batch_gids = s1_gids.slice(bi * batch_size, batch_size).rename("s1_gid").to_frame()
        batch_cand = candidates.join(batch_gids, on="s1_gid", how="semi")
        s1, s23 = load_sides_fn(batch_cand)
        feats = pair_features(batch_cand, s1, s23)
        feats.write_parquet(out_dir / f"features_batch_{bi:05d}.parquet")
        total_rows += feats.height
        del s1, s23, feats, batch_cand
    return {"n_s1_batches": n_batches, "batch_size": batch_size, "n_s1_total": n_s1, "feature_rows_total": total_rows}
