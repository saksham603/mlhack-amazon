"""Blocking v3 (day-2 plan Phase C): extra candidate keys on the cleanup-v2 records, added to v1's
budget-150 candidates. Measured motivation (FIT v1 sample): of the true links v1 missed, 47% (India)
/ 27% (US) have identical cleaned names and 92% / 79% have address token_set >= 80 -- they were lost
because their names are common, not because they differ. So v3 keys pair a name word with an
address word, and pin whole names to the state:

  na   : core-name token x address token (tokens >= 3 chars, generic street words excluded)
  rns  : region | no-space name        rsk : region | skeleton name

Keys are hashed to UInt64 (kind prefix, so kinds never collide); rarity cap, IDF and top-k reuse
blocking.v2. Label-free: only the evaluation helper reads FIT labels.
"""
import polars as pl

from amlc.blocking import v2

KIND = pl.Enum(["na", "rns", "rsk"])
MAX_NAME_TOKENS = 5
MAX_ADDR_TOKENS = 10
MIN_TOKEN_LEN = 3
GENERIC_ADDR = ["rd", "st", "ave", "blvd", "dr", "ln", "ct", "pl", "ste", "apt", "fl", "hwy", "pkwy", "cir", "ter",
                "ngr", "clny", "sec", "ph", "blk", "opp", "nr", "bldg", "no", "rue", "ch", "rte", "imp", "all",
                "fbg", "sq", "mt", "ctr", "near", "road", "street", "the", "and", "des", "les", "floor", "plot",
                "shop", "house", "main", "cross", "india", "saint"]
HASH_SEED = 20260927
CAP_NA = 1000
CAP_WHOLE = 5000


def _h(df: pl.DataFrame, kind: str) -> pl.DataFrame:
    return df.select("gid", (pl.lit(kind + ":") + pl.col("key")).hash(seed=HASH_SEED).alias("key"),
                     pl.lit(kind, dtype=KIND).alias("kind"))


def keys(rec: pl.DataFrame) -> pl.DataFrame:
    """rec: gid, core, addr, region, ns, skel -> (gid, key, kind), unique."""
    nt = (rec.select("gid", pl.col("core").str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() >= MIN_TOKEN_LEN)).list.unique(maintain_order=True)
            .list.head(MAX_NAME_TOKENS).alias("nt"))
          .explode("nt", empty_as_null=True).drop_nulls("nt"))
    at = (rec.select("gid", pl.col("addr").str.split(" ").list.eval(
            pl.element().filter((pl.element().str.len_chars() >= MIN_TOKEN_LEN) & ~pl.element().is_in(GENERIC_ADDR)))
            .list.unique(maintain_order=True).list.head(MAX_ADDR_TOKENS).alias("at"))
          .explode("at", empty_as_null=True).drop_nulls("at"))
    na = nt.join(at, on="gid").select("gid", (pl.col("nt") + "|" + pl.col("at")).alias("key"))
    reg = rec.filter(pl.col("region") != "")
    rns = reg.filter(pl.col("ns").str.len_chars() >= 4).select("gid", (pl.col("region") + "|" + pl.col("ns")).alias("key"))
    rsk = reg.filter(pl.col("skel").str.len_chars() >= 3).select("gid", (pl.col("region") + "|" + pl.col("skel")).alias("key"))
    return pl.concat([_h(na, "na"), _h(rns, "rns"), _h(rsk, "rsk")]).unique(["gid", "key"])


def build_index(r23: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(k23, rare) for one country's S2/S3 pool; rarity caps per kind."""
    k23 = keys(r23).rename({"gid": "s23_gid"})
    n = r23.height
    rare_na = v2.idf_table(k23.filter(pl.col("kind") == "na"), n, CAP_NA)
    rare_whole = v2.idf_table(k23.filter(pl.col("kind") != "na"), n, CAP_WHOLE)
    rare = pl.concat([rare_na, rare_whole]).with_columns(pl.col("kind").cast(pl.String))
    return k23.with_columns(pl.col("kind").cast(pl.String)), rare


def candidates(r1_batch: pl.DataFrame, k23: pl.DataFrame, rare: pl.DataFrame, k: int,
               max_rows: int = 150_000_000) -> pl.DataFrame:
    """(s1_gid, s23_gid, v3_score, v3_rank) top-k by IDF-weighted shared v3 keys (rounded, so ties are
    broken the same way on every run)."""
    k1 = keys(r1_batch).rename({"gid": "s1_gid"}).with_columns(pl.col("kind").cast(pl.String))
    scored = v2.score(k1, k23, rare, ("na", "rns", "rsk"), max_rows).with_columns(pl.col("score").round(6))
    top = v2.topk(scored, k).join(scored, on=["s1_gid", "s23_gid"], how="left")
    return (top.sort(["s1_gid", "score", "s23_gid"], descending=[False, True, False])
            .with_columns(pl.col("s23_gid").cum_count().over("s1_gid").alias("v3_rank"))
            .rename({"score": "v3_score"}))
