"""Number-noise features (2026-09-27 07:50). Error analysis on the FIT holdout: the largest group of
missed true links (25%) has the same name and the same address except for a perturbed number
("11424" vs "11426", "28" vs "29", "1 29" vs "129", "05005" vs "5005") or an injected house-number token
("hn 603 unit no 72 ..."). The v3 features only say whether house numbers are equal/conflicting, so the
model reads every perturbation as a different business. These features say HOW close the numbers are,
and how similar the addresses are once numbers are ignored. FIT-holdout A/B (07:55, small models):
+0.0144 F0.5 for a model trained from scratch, +0.0070 as an add-on to the v3.1 probabilities.

Per pair (addresses a1 = S1, a2 = S2/S3, from the cleanup-v2 records' `addr`):
  x_a_tset_nod  token_set_ratio of the addresses with numeric tokens removed
  x_a_ratio_nod ratio (order-sensitive) of the same
  x_num_tset    token_set_ratio of the numeric tokens only (leading zeros stripped)
  x_num_lev     Levenshtein distance of all digits concatenated (capped at 9)
  x_num_jacc    Jaccard of the numeric-token sets (-1 if both have none)
  x_num_extra2  numeric tokens in a2 not in a1 (injected numbers), x_num_miss2: in a1 not in a2
  x_hn_absdiff  |first number a1 - first number a2| (capped at 1000; -1 if either has none)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

X_FEATURES = ["x_a_tset_nod", "x_a_ratio_nod", "x_num_tset", "x_num_lev", "x_num_jacc", "x_num_extra2",
              "x_num_miss2", "x_hn_absdiff"]


def _nod(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(r"\b\w*\d\w*\b", " ").str.replace_all(r"\s+", " ").str.strip_chars()


def _nums(col: str) -> pl.Expr:
    return pl.col(col).str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0").replace("", "0"))


def pair_features(pairs: pl.DataFrame, rec1: pl.DataFrame, rec23: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """pairs: s1_gid, s23_gid. rec1 / rec23: (gid, addr) of the S1 / S2-S3 records. Returns
    (s1_gid, s23_gid, *X_FEATURES) in the order of `pairs`."""
    d = (pairs.select("s1_gid", "s23_gid")
         .join(rec1.select(pl.col("gid").alias("s1_gid"), pl.col("addr").alias("a1")), on="s1_gid", how="left", maintain_order="left")
         .join(rec23.select(pl.col("gid").alias("s23_gid"), pl.col("addr").alias("a2")), on="s23_gid", how="left", maintain_order="left")
         .with_columns(pl.col("a1", "a2").fill_null(""))
         .with_columns(_nod("a1").alias("nod1"), _nod("a2").alias("nod2"), _nums("a1").alias("nums1"), _nums("a2").alias("nums2")))
    d = d.with_columns(
        pl.col("nums1").list.join(" ").alias("ns1"), pl.col("nums2").list.join(" ").alias("ns2"),
        pl.col("nums1").list.join("").alias("dg1"), pl.col("nums2").list.join("").alias("dg2"),
        pl.col("nums1").list.set_intersection("nums2").list.len().alias("_i"),
        pl.col("nums1").list.set_union("nums2").list.len().alias("_u"),
        pl.col("nums2").list.set_difference("nums1").list.len().cast(pl.Float32).alias("x_num_extra2"),
        pl.col("nums1").list.set_difference("nums2").list.len().cast(pl.Float32).alias("x_num_miss2"),
        (pl.col("nums1").list.first().str.slice(0, 7).cast(pl.Int64, strict=False)
         - pl.col("nums2").list.first().str.slice(0, 7).cast(pl.Int64, strict=False)).abs().clip(0, 1000)
        .fill_null(-1).cast(pl.Float32).alias("x_hn_absdiff"))
    f32 = np.float32
    u, i = d["_u"].to_numpy(), d["_i"].to_numpy()
    out = d.select("s1_gid", "s23_gid", "x_num_extra2", "x_num_miss2", "x_hn_absdiff").with_columns(
        pl.Series("x_a_tset_nod", process.cpdist(d["nod1"].to_list(), d["nod2"].to_list(), scorer=fuzz.token_set_ratio, workers=workers, dtype=f32)),
        pl.Series("x_a_ratio_nod", process.cpdist(d["nod1"].to_list(), d["nod2"].to_list(), scorer=fuzz.ratio, workers=workers, dtype=f32)),
        pl.Series("x_num_tset", process.cpdist(d["ns1"].to_list(), d["ns2"].to_list(), scorer=fuzz.token_set_ratio, workers=workers, dtype=f32)),
        pl.Series("x_num_lev", np.minimum(process.cpdist(d["dg1"].to_list(), d["dg2"].to_list(), scorer=Levenshtein.distance, workers=workers), 9).astype(f32)),
        pl.Series("x_num_jacc", np.where(u == 0, -1.0, i / np.maximum(u, 1)).astype(f32)))
    return out.select("s1_gid", "s23_gid", *X_FEATURES)
