"""Reworked per-S1 context features, isolated from w3.py (never edited in place -- it's a live
file the laptop's pipeline depends on). `full_context_features` reproduces w3.py's CURRENT
ctx_n_*/ctx_a_*/ctx_c_*/ctx_n_hi/ctx_n_cand formula exactly (regression baseline, verified against
real holdout data). `combo_mirror_name` is the fix under test for a confirmed root cause: combo =
n_tset + a_tset is a plain unweighted sum, so a candidate whose address is genuinely EMPTY (a_tset
collapses to ~0) looks the same to `combo` as one whose address is present and a real mismatch --
both drag ctx_c_rank/ctx_c_gap down, even when the missing-address candidate has a perfect name
match. When address is missing on either side, mirror_name substitutes n_tset for a_tset in the sum
instead: absence of address evidence isn't evidence against a match. Label-free."""
import polars as pl


def compute_ctx(df: pl.DataFrame, combo: pl.Expr, prefix: str) -> pl.DataFrame:
    """Adds `{prefix}_rank` and `{prefix}_gap`, ranked/gapped over each s1_gid group by `combo`."""
    return df.with_columns(
        combo.rank("min", descending=True).over("s1_gid").alias(f"{prefix}_rank"),
        (combo.max().over("s1_gid") - combo).alias(f"{prefix}_gap"),
    )


def full_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Reproduces w3.py's exact current ctx_n_*, ctx_a_*, ctx_c_*, ctx_n_hi, ctx_n_cand -- the
    regression baseline this module is built against before any fix is trusted."""
    combo = pl.col("n_tset") + pl.col("a_tset")
    out = compute_ctx(df, pl.col("n_tset"), "ctx_n")
    out = compute_ctx(out, pl.col("a_tset"), "ctx_a")
    out = compute_ctx(out, combo, "ctx_c")
    return out.with_columns(
        (pl.col("n_tset") >= 95).sum().over("s1_gid").alias("ctx_n_hi"),
        pl.len().over("s1_gid").alias("ctx_n_cand"),
    )


def _addr_missing(df: pl.DataFrame) -> pl.Expr:
    return (pl.col("a_len_1") == 0) | (pl.col("a_len_2") == 0)


def combo_mirror_name(df: pl.DataFrame) -> pl.Expr:
    """Variant A: when address is missing on either side, treat the address term as if it mirrored
    n_tset (neutral) instead of counting the near-zero a_tset that string similarity produces for
    an empty string. Only applies when address is genuinely ABSENT (a_len_1==0 or a_len_2==0) --
    a real conflict between two present-but-different addresses is untouched and still penalized."""
    return pl.col("n_tset") + pl.when(_addr_missing(df)).then(pl.col("n_tset")).otherwise(pl.col("a_tset"))
