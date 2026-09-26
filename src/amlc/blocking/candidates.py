"""W1 blocking: candidate generation (design and measurement only per the overnight master prompt
T7). Reads ONLY published silver/v1 (CG-26/G-P16). Per country (R7): no cross-country true pairs
(0 of 69,157 sampled, Stage 0 F6). Three key types, all Polars joins (no char-3gram pass tonight:
scikit-learn is not installed):
  1. rare core-name tokens (name_tokens; already excludes name_legal_form, since C3 extracts it out)
  2. house number + a rare address token (a proxy for "house number + rare street token")
  3. ZIP/PIN (addr_zip)
A key is usable only if its document frequency on the FULL S2/S3 pool of that country is <= a cap
(rarity). Candidates are ranked by an IDF-weighted count of shared keys (label-free, cheap) and
capped at top-k. Never truncated in file order (M10).
"""
import math

import polars as pl

from amlc.foundation import access

SILVER_DIR = access.DATA / "silver" / "v1"
SEED = 20260925
SAMPLE_FRACTION = 0.01
CAPS = (100, 500, 1000, 5000)
KS = (10, 20, 30, 50, 100)


def load_silver_with_country(dataset: str, src: int) -> pl.DataFrame:
    df = pl.read_parquet(SILVER_DIR / f"{dataset}_source{src}.parquet")
    meta = access.load_bronze(dataset, src, columns=["gid", "country"])
    return df.join(meta, on="gid", how="left", validate="1:1")


def load_s23_pool(country: str) -> pl.DataFrame:
    parts = [load_silver_with_country("train", s).filter(pl.col("country") == country) for s in (2, 3)]
    return pl.concat(parts, how="vertical")


def sample_fit_s1(country: str, fraction: float = SAMPLE_FRACTION, seed: int = SEED) -> pl.DataFrame:
    """1% of FIT S1 entities in this country, seed 20260925. Samples S1 only, never the S2/S3 pool
    (sampling both sides breaks true pairs and hides candidate density -- T7 spec).
    """
    split = access.load_split().filter(pl.col("split") == "FIT")
    silver = load_silver_with_country("train", 1)
    pool = split.join(silver, left_on="s1_gid", right_on="gid", how="inner", validate="1:1")
    pool = pool.filter(pl.col("country") == country)
    n = max(1, round(pool.height * fraction))
    return pool.sample(n=n, seed=seed, shuffle=True, with_replacement=False)


def _name_tokens(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return (df.select(id_col, "name_tokens").explode("name_tokens", empty_as_null=True)
            .drop_nulls("name_tokens").filter(pl.col("name_tokens") != "").rename({"name_tokens": "key"}))


def _addr_tokens(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return (df.select(id_col, "addr_components").explode("addr_components", empty_as_null=True)
            .drop_nulls("addr_components")
            .with_columns(pl.col("addr_components").str.split(" ")).explode("addr_components", empty_as_null=True)
            .drop_nulls("addr_components").filter(pl.col("addr_components") != "").rename({"addr_components": "tok"}))


def house_street_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """House number + a co-occurring address token, as a composite key (T7's proxy for "house
    number + rare street token"; no separate street-word table lookup here, that's C4's job).
    """
    toks = _addr_tokens(df, id_col)
    hn = df.select(id_col, "addr_house_number").filter(pl.col("addr_house_number") != "")
    joined = toks.join(hn, on=id_col, how="inner")
    joined = joined.filter(pl.col("tok") != pl.col("addr_house_number"))
    return joined.with_columns((pl.col("addr_house_number") + "|" + pl.col("tok")).alias("key")).select(id_col, "key")


def zip_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return df.select(id_col, "addr_zip").filter(pl.col("addr_zip") != "").rename({"addr_zip": "key"})


def document_frequency(keys: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """DF of each key value = number of DISTINCT records (id_col) it appears on."""
    return keys.unique([id_col, "key"]).group_by("key").agg(pl.len().alias("df"))


def estimate_pair_count(s1_keys: pl.DataFrame, s23_keys: pl.DataFrame, rare: pl.DataFrame) -> int:
    """R6: sum over keys of (S1 rows with that key x S2/S3 rows with that key), computed from
    group_by().len() on each side -- BEFORE materialising the join.
    """
    s1_counts = s1_keys.join(rare.select("key"), on="key", how="semi").group_by("key").agg(pl.len().alias("n1"))
    s23_counts = s23_keys.join(rare.select("key"), on="key", how="semi").group_by("key").agg(pl.len().alias("n23"))
    both = s1_counts.join(s23_counts, on="key", how="inner")
    return int((both["n1"] * both["n23"]).sum()) if both.height else 0


def candidates_for_cap(s1: pl.DataFrame, s23: pl.DataFrame, cap: int, ram_budget_bytes: int) -> pl.DataFrame:
    """One country's candidate set at one rarity cap, across all 3 key types, scored by IDF-weighted
    shared-key count. Returns (s1_gid, s23_gid, score), NOT yet capped to top-k (candidates_topk does
    that). Raises if the estimated pair count would exceed the RAM budget (R6), rather than trying it.
    """
    s1_keys = pl.concat([
        _name_tokens(s1, "s1_gid"),
        house_street_keys(s1, "s1_gid"),
        zip_keys(s1, "s1_gid"),
    ], how="vertical")
    s23_keys = pl.concat([
        _name_tokens(s23, "s23_gid"),
        house_street_keys(s23, "s23_gid"),
        zip_keys(s23, "s23_gid"),
    ], how="vertical")

    s23_df = document_frequency(s23_keys, "s23_gid")
    rare = s23_df.filter(pl.col("df") <= cap)
    n23 = s23.height
    idf = rare.with_columns((((n23 + 1) / (pl.col("df") + 1)).log() + 1).alias("idf"))

    est_pairs = estimate_pair_count(s1_keys, s23_keys, rare)
    est_bytes = est_pairs * 24  # two UInt32 gids + a float32 score, generous
    if est_bytes > ram_budget_bytes:
        raise MemoryError(f"estimated {est_pairs:,} candidate pairs (~{est_bytes/2**20:.0f} MB) "
                          f"exceeds the {ram_budget_bytes/2**20:.0f} MB budget at cap={cap}; chunk by key hash")

    s1_rare = s1_keys.join(idf.select("key", "idf"), on="key", how="inner")
    s23_rare = s23_keys.join(idf.select("key"), on="key", how="semi")
    pairs = s1_rare.join(s23_rare, on="key", how="inner", validate="m:m")
    scored = pairs.group_by("s1_gid", "s23_gid").agg(pl.col("idf").sum().alias("score"))
    return scored


def candidates_topk(scored: pl.DataFrame, k: int) -> pl.DataFrame:
    """Top-k S2/S3 candidates per S1, by score, ranked (never truncated in file order, M10)."""
    return (scored.sort(["s1_gid", "score"], descending=[False, True])
            .group_by("s1_gid", maintain_order=True).head(k))


def recall_at_k(topk: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """truth: (s1_gid, s23_gid) true links restricted to the sampled S1s. recall = share of true
    links whose s23_gid is among that S1's top-k candidates.
    """
    if truth.height == 0:
        return {"recall": None, "n_true_links": 0}
    hit = truth.join(topk.select("s1_gid", "s23_gid"), on=["s1_gid", "s23_gid"], how="semi")
    return {"recall": hit.height / truth.height, "n_true_links": truth.height}
