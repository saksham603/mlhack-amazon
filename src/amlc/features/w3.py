"""Day-2 pairwise features v3 (master prompt Phase A2): fuzzy name and address similarity on the
cleanup-v2 records (data/_v2/records), plus per-S1 context. Label-free. Keeps the 9 v1 features
(passed through from the candidate/feature files) and adds ~46 new ones.

String similarities use rapidfuzz.process.cpdist (element-wise, multithreaded C++).
Context features are computed over each S1's full candidate list in the frame passed in, so a
frame must always hold every candidate of the S1s it contains (all v1 batch files do).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from amlc.foundation import access

REC_DIR = access.DATA / "_v2" / "records"
FIELDS = ["core", "ns", "skel", "ini", "first", "last", "ntok", "legal", "indic", "addr", "street",
          "region", "lastc", "hn", "hn_digits", "zip", "s1_share", "s23_share"]
V1_FEATURES = ["blocking_score", "blocking_rank", "name_jaccard", "name_overlap_n", "addr_token_jaccard",
               "house_number_match", "zip_match", "legal_form_match", "legal_form_both_present"]
NEW_FEATURES = [
    "n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw_ns", "n_jw_skel", "n_acr", "n_ini_prefix", "n_ini_eq",
    "n_contain", "n_first_eq", "n_last_eq", "n_len_ratio", "ntok_1", "ntok_2", "legal_eq", "legal_conflict",
    "indic_2", "s1_share_1", "s23_share_1", "s1_share_2",
    "a_tset", "a_tsort", "a_street_tset", "reg_eq", "reg_conflict", "reg_missing", "reg_jw",
    "hn_eq", "hn_conflict", "hn_missing", "hn_dig_eq", "zip_eq", "zip_conflict", "zip_missing", "zip_p3",
    "zip_lev1", "a_len_1", "a_len_2",
    "ctx_n_rank", "ctx_n_gap", "ctx_a_rank", "ctx_a_gap", "ctx_c_rank", "ctx_c_gap", "ctx_n_hi", "ctx_n_cand"]
FEATURES = V1_FEATURES + NEW_FEATURES


def load_records(dataset: str, side: str) -> pl.DataFrame:
    """side 's1' -> source 1; 's23' -> sources 2 and 3 stacked."""
    srcs = (1,) if side == "s1" else (2, 3)
    return pl.concat([pl.read_parquet(REC_DIR / f"{dataset}_s{s}.parquet", columns=["gid", *FIELDS]) for s in srcs],
                     how="vertical")


def _cp(df: pl.DataFrame, a: str, b: str, scorer) -> np.ndarray:
    return process.cpdist(df[a].to_list(), df[b].to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def pair_features(pairs: pl.DataFrame, r1: pl.DataFrame, r23: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_gid, s23_gid (+ any passthrough columns). r1 / r23: records (gid, *FIELDS)."""
    left = r1.join(pairs.select(pl.col("s1_gid").alias("gid")).unique(), on="gid", how="semi")
    right = r23.join(pairs.select(pl.col("s23_gid").alias("gid")).unique(), on="gid", how="semi")
    df = (pairs.join(left.select(pl.col("gid").alias("s1_gid"), *[pl.col(c).alias(c + "_1") for c in FIELDS]),
                     on="s1_gid", how="left")
          .join(right.select(pl.col("gid").alias("s23_gid"), *[pl.col(c).alias(c + "_2") for c in FIELDS]),
                on="s23_gid", how="left"))
    str_cols = [f"{c}_{k}" for c in FIELDS if c not in ("ntok", "indic", "s1_share", "s23_share") for k in (1, 2)]
    df = df.with_columns(pl.col(str_cols).fill_null(""))
    df = df.with_columns(
        pl.Series("n_ratio", _cp(df, "core_1", "core_2", fuzz.ratio)),
        pl.Series("n_partial", _cp(df, "core_1", "core_2", fuzz.partial_ratio)),
        pl.Series("n_tsort", _cp(df, "core_1", "core_2", fuzz.token_sort_ratio)),
        pl.Series("n_tset", _cp(df, "core_1", "core_2", fuzz.token_set_ratio)),
        pl.Series("n_jw_ns", _cp(df, "ns_1", "ns_2", JaroWinkler.normalized_similarity)),
        pl.Series("n_jw_skel", _cp(df, "skel_1", "skel_2", JaroWinkler.normalized_similarity)),
        pl.Series("a_tset", _cp(df, "addr_1", "addr_2", fuzz.token_set_ratio)),
        pl.Series("a_tsort", _cp(df, "addr_1", "addr_2", fuzz.token_sort_ratio)),
        pl.Series("a_street_tset", _cp(df, "street_1", "street_2", fuzz.token_set_ratio)),
        pl.Series("reg_jw", _cp(df, "lastc_1", "lastc_2", JaroWinkler.normalized_similarity)),
        pl.Series("zip_dist", _cp(df, "zip_1", "zip_2", Levenshtein.distance)),
    )
    ne = lambda a, b: (pl.col(a) != "") & (pl.col(b) != "")  # noqa: E731  both present
    df = df.with_columns(
        (((pl.col("ini_1") == pl.col("ns_2")) & (pl.col("ntok_1") >= 2))
         | ((pl.col("ini_2") == pl.col("ns_1")) & (pl.col("ntok_2") >= 2))).alias("n_acr"),
        (((pl.col("ntok_1") >= 2) & pl.col("ns_2").str.starts_with(pl.col("ini_1")))
         | ((pl.col("ntok_2") >= 2) & pl.col("ns_1").str.starts_with(pl.col("ini_2")))).alias("n_ini_prefix"),
        ((pl.col("ini_1") == pl.col("ini_2")) & (pl.col("ntok_1") >= 2)).alias("n_ini_eq"),
        ((pl.col("ns_1").str.len_chars() >= 4) & (pl.col("ns_2").str.len_chars() >= 4)
         & (pl.col("ns_2").str.contains(pl.col("ns_1"), literal=True)
            | pl.col("ns_1").str.contains(pl.col("ns_2"), literal=True))).alias("n_contain"),
        (ne("first_1", "first_2") & (pl.col("first_1") == pl.col("first_2"))).alias("n_first_eq"),
        (ne("last_1", "last_2") & (pl.col("last_1") == pl.col("last_2"))).alias("n_last_eq"),
        (pl.min_horizontal(pl.col("ns_1").str.len_chars(), pl.col("ns_2").str.len_chars())
         / pl.max_horizontal(pl.col("ns_1").str.len_chars(), pl.col("ns_2").str.len_chars(), pl.lit(1))).alias("n_len_ratio"),
        (ne("legal_1", "legal_2") & (pl.col("legal_1") == pl.col("legal_2"))).alias("legal_eq"),
        (ne("legal_1", "legal_2") & (pl.col("legal_1") != pl.col("legal_2"))).alias("legal_conflict"),
        pl.col("indic_2").fill_null(False),
        pl.col("s1_share_1").fill_null(0), pl.col("s23_share_1").fill_null(0), pl.col("s1_share_2").fill_null(0),
        (ne("region_1", "region_2") & (pl.col("region_1") == pl.col("region_2"))).alias("reg_eq"),
        (ne("region_1", "region_2") & (pl.col("region_1") != pl.col("region_2"))).alias("reg_conflict"),
        ((pl.col("region_1") == "") | (pl.col("region_2") == "")).alias("reg_missing"),
        (ne("hn_1", "hn_2") & (pl.col("hn_1") == pl.col("hn_2"))).alias("hn_eq"),
        (ne("hn_1", "hn_2") & (pl.col("hn_1") != pl.col("hn_2"))).alias("hn_conflict"),
        ((pl.col("hn_1") == "") | (pl.col("hn_2") == "")).alias("hn_missing"),
        (ne("hn_digits_1", "hn_digits_2") & (pl.col("hn_digits_1") == pl.col("hn_digits_2"))).alias("hn_dig_eq"),
        (ne("zip_1", "zip_2") & (pl.col("zip_1") == pl.col("zip_2"))).alias("zip_eq"),
        (ne("zip_1", "zip_2") & (pl.col("zip_1") != pl.col("zip_2"))).alias("zip_conflict"),
        ((pl.col("zip_1") == "") | (pl.col("zip_2") == "")).alias("zip_missing"),
        (ne("zip_1", "zip_2") & (pl.col("zip_1").str.slice(0, 3) == pl.col("zip_2").str.slice(0, 3))).alias("zip_p3"),
        (ne("zip_1", "zip_2") & (pl.col("zip_dist") <= 1)).alias("zip_lev1"),
        pl.col("addr_1").str.len_chars().alias("a_len_1"),
        pl.col("addr_2").str.len_chars().alias("a_len_2"),
        pl.col("ntok_1").fill_null(0), pl.col("ntok_2").fill_null(0),
    )
    combo = pl.col("n_tset") + pl.col("a_tset")
    df = df.with_columns(
        pl.col("n_tset").rank("min", descending=True).over("s1_gid").alias("ctx_n_rank"),
        (pl.col("n_tset").max().over("s1_gid") - pl.col("n_tset")).alias("ctx_n_gap"),
        pl.col("a_tset").rank("min", descending=True).over("s1_gid").alias("ctx_a_rank"),
        (pl.col("a_tset").max().over("s1_gid") - pl.col("a_tset")).alias("ctx_a_gap"),
        combo.rank("min", descending=True).over("s1_gid").alias("ctx_c_rank"),
        (combo.max().over("s1_gid") - combo).alias("ctx_c_gap"),
        (pl.col("n_tset") >= 95).sum().over("s1_gid").alias("ctx_n_hi"),
        pl.len().over("s1_gid").alias("ctx_n_cand"),
    )
    keep = [c for c in pairs.columns] + [f for f in NEW_FEATURES if f not in pairs.columns]
    out = df.select(keep)
    return out.with_columns(pl.col(pl.Boolean).cast(pl.UInt8))
