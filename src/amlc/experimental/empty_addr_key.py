"""Experimental, isolated candidate key: S2/S3 records with no address, matched by exact no-space
name within that small sub-pool. Never imported by amlc.blocking.v3 or any run_v3*/run_v32 module --
see docs/superpowers/plans/2026-09-27-empty-address-exact-name-key.md for the measurement this feeds.
MIN_NS_LEN mirrors amlc.blocking.v3's existing `cns` key floor (v3.py:79), not a new threshold.
"""
import polars as pl

MIN_NS_LEN = 4


def empty_addr_keys(r23: pl.DataFrame) -> pl.DataFrame:
    """r23: gid, addr, ns -> (s23_gid, ns) for records with addr == "" and len(ns) >= MIN_NS_LEN."""
    return (r23.filter((pl.col("addr") == "") & (pl.col("ns").str.len_chars() >= MIN_NS_LEN))
            .select(pl.col("gid").alias("s23_gid"), "ns"))


def empty_addr_candidates(r1: pl.DataFrame, r23_keys: pl.DataFrame) -> pl.DataFrame:
    """r1: gid, ns. r23_keys: output of empty_addr_keys. -> (s1_gid, s23_gid) for shared ns."""
    return (r1.filter(pl.col("ns").str.len_chars() >= MIN_NS_LEN)
            .rename({"gid": "s1_gid"})
            .join(r23_keys, on="ns", how="inner")
            .select("s1_gid", "s23_gid"))
