"""Second pass (day-2 plan Phase D): collective features from first-pass probabilities.

All variants of one business resemble each other, so a candidate that is weak against the S1 but
very similar to the S1's other confident matches is probably a match too (and vice versa). Features
per candidate (s1, s23) given first-pass probability p1:
  p1, p1_rank (within the S1), p1_max_other (best other candidate of the S1), p1_n_conf_other
  (other candidates with p1 >= CONF), sib_name_max / sib_addr_max / sib_both_max (max fuzzy similarity
  between this S2/S3 record and the S1's OTHER confident candidates).
Training uses out-of-fold p1 on FIT (2 folds by S1 hash), so the second model never sees scores from a
model trained on the same rows; VALIDATION/TEST use the average of the two fold models.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

CONF = 0.5
P2_FEATURES = ["p1", "p1_rank", "p1_max_other", "p1_n_conf_other", "sib_name_max", "sib_addr_max", "sib_both_max"]


def add_pass2(df: pl.DataFrame, p1: np.ndarray, r23: pl.DataFrame) -> pl.DataFrame:
    """df: one batch holding every candidate of its S1s (s1_gid, s23_gid, ...). r23: records (gid, core, addr)."""
    d = df.with_columns(pl.Series("p1", p1, dtype=pl.Float64))
    top = pl.col("p1").max().over("s1_gid")
    second = pl.col("p1").sort(descending=True).slice(1, 1).first().over("s1_gid")
    d = d.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1_gid").alias("p1_rank"),
        pl.when(pl.col("p1") >= top).then(second.fill_null(0.0)).otherwise(top).alias("p1_max_other"),
        ((pl.col("p1") >= CONF).sum().over("s1_gid") - (pl.col("p1") >= CONF).cast(pl.UInt32)).alias("p1_n_conf_other"))
    conf = d.filter(pl.col("p1") >= CONF).select("s1_gid", pl.col("s23_gid").alias("sib"))
    pairs = d.select("s1_gid", "s23_gid").join(conf, on="s1_gid").filter(pl.col("s23_gid") != pl.col("sib"))
    if pairs.height:
        rs = r23.select("gid", "core", "addr")
        pairs = (pairs.join(rs.rename({"gid": "s23_gid", "core": "c_a", "addr": "a_a"}), on="s23_gid", how="left")
                 .join(rs.rename({"gid": "sib", "core": "c_b", "addr": "a_b"}), on="sib", how="left")
                 .with_columns(pl.col("c_a", "a_a", "c_b", "a_b").fill_null(""))) 
        ns = process.cpdist(pairs["c_a"].to_list(), pairs["c_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        as_ = process.cpdist(pairs["a_a"].to_list(), pairs["a_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        sims = (pairs.select("s1_gid", "s23_gid").with_columns(pl.Series("n", ns), pl.Series("a", as_))
                .group_by("s1_gid", "s23_gid").agg(pl.col("n").max().alias("sib_name_max"), pl.col("a").max().alias("sib_addr_max"),
                                                   ((pl.col("n") + pl.col("a")) / 2).max().alias("sib_both_max")))
        d = d.join(sims, on=["s1_gid", "s23_gid"], how="left")
    else:
        d = d.with_columns(pl.lit(None, pl.Float32).alias(c) for c in ("sib_name_max", "sib_addr_max", "sib_both_max"))
    return d.with_columns(pl.col("sib_name_max", "sib_addr_max", "sib_both_max").fill_null(-1.0))
