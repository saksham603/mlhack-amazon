"""Blocking v2 (user go, 2026-09-26 15:00): three fixes on top of the existing strategy.

  Fix 1  word-pair name keys: every unordered pair of distinct name tokens ("consultants|shiva"),
         so two common words can still form a rare key. Added, never replacing single tokens.
  Fix 2  separate ranking: a name score (name keys only) and an address score (house|token and
         ZIP keys only) rank candidates independently; the final set is the union of the top-kn by
         name and the top-ka by address, so crowded addresses cannot push out a strong name match.
  Fix 3  no-space name key: the name with its spaces removed, after dropping a leading "www" /
         "http(s)" and trailing domain tokens ("com", "in", ...), so "servicesraajinfra com" and
         "services raaj infra" share the key "servicesraajinfra". Also acts as a full-name key.

Every key is hashed to UInt64 with its kind as a prefix (kinds never collide with each other) to cut
memory; a 64-bit collision only adds a spurious candidate, never removes one. Rarity cap, IDF and
top-k follow candidates.py. Label-free: nothing here reads labels.
"""
import polars as pl

KINDS_NAME = ("tok", "pair", "nospace")
KINDS_ADDR = ("house", "zip")
KIND = pl.Enum(["tok", "pair", "nospace", "house", "zip"])
MAX_PAIR_TOKENS = 8          # pairs are formed among the first 8 distinct tokens (C(8,2) = 28 max)
MIN_NOSPACE_LEN = 6          # shorter concatenations are too generic to be a key
LEADING_WEB = ("www", "http", "https")
DOMAIN_SUFFIXES = ("com", "in", "net", "org", "co", "biz", "info")
STRIP_ROUNDS = 3             # vectorised stripping handles up to 3 leading/trailing web tokens
HASH_SEED = 20260926


def _hashed(df: pl.DataFrame, id_col: str, kind: str) -> pl.DataFrame:
    return df.select(
        pl.col(id_col),
        (pl.lit(kind + ":") + pl.col("key")).hash(seed=HASH_SEED).alias("key"),
        pl.lit(kind, dtype=KIND).alias("kind"))


def token_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    t = (df.select(id_col, pl.col("name_tokens").list.unique(maintain_order=True))
         .explode("name_tokens", empty_as_null=True).drop_nulls("name_tokens")
         .filter(pl.col("name_tokens") != "").rename({"name_tokens": "key"}))
    return _hashed(t, id_col, "tok")


def _distinct_tokens(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    return df.select(id_col, pl.col("name_tokens").list.eval(pl.element().filter(pl.element() != ""))
                     .list.unique(maintain_order=True).alias("tok"))


def pair_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Fix 1: unordered pairs of distinct tokens (sorted within the pair, so word order is ignored).
    Built by direct list indexing (at most C(8,2) = 28 passes, each hashed at once) instead of a
    self-join, which would materialise ~n^2 rows per record before filtering."""
    t = _distinct_tokens(df, id_col).with_columns(pl.col("tok").list.head(MAX_PAIR_TOKENS))
    t = t.filter(pl.col("tok").list.len() >= 2)
    parts = []
    for i in range(MAX_PAIR_TOKENS):
        for j in range(i + 1, MAX_PAIR_TOKENS):
            a = pl.col("tok").list.get(i, null_on_oob=True)
            b = pl.col("tok").list.get(j, null_on_oob=True)
            p = (t.filter(pl.col("tok").list.len() > j)
                 .select(id_col, pl.when(a < b).then(a + "|" + b).otherwise(b + "|" + a).alias("key")))
            if p.height:
                parts.append(_hashed(p, id_col, "pair"))
    if not parts:
        return _hashed(pl.DataFrame(schema={id_col: df.schema[id_col], "key": pl.String}), id_col, "pair")
    return pl.concat(parts, how="vertical")


def nospace_value(tokens: list[str]) -> str:
    """Fix 3 reference definition (tests and the vectorised version must agree with this)."""
    toks = [t for t in tokens if t]
    while toks and toks[0] in LEADING_WEB:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in DOMAIN_SUFFIXES:
        toks = toks[:-1]
    s = "".join(toks)
    return s if len(s) >= MIN_NOSPACE_LEN else ""


def nospace_expr() -> pl.Expr:
    """Vectorised nospace_value (the reference above). Strips up to STRIP_ROUNDS leading web tokens
    and trailing domain tokens; tests check it equals the reference on real names."""
    t = pl.col("name_tokens").list.eval(pl.element().filter(pl.element() != ""))
    for _ in range(STRIP_ROUNDS):
        t = pl.when(t.list.first().is_in(LEADING_WEB)).then(t.list.slice(1)).otherwise(t)
    for _ in range(STRIP_ROUNDS):
        t = (pl.when((t.list.len() > 1) & t.list.last().is_in(DOMAIN_SUFFIXES))
             .then(t.list.slice(0, t.list.len() - 1)).otherwise(t))
    s = t.list.join("")
    return pl.when(s.str.len_chars() >= MIN_NOSPACE_LEN).then(s).otherwise(pl.lit(""))


def nospace_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    vals = df.select(id_col, nospace_expr().alias("key"))
    return _hashed(vals.filter(pl.col("key").is_not_null() & (pl.col("key") != "")), id_col, "nospace")


def address_keys(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    from amlc.blocking import candidates as C
    return pl.concat([_hashed(C.house_street_keys(df, id_col), id_col, "house"),
                      _hashed(C.zip_keys(df, id_col), id_col, "zip")], how="vertical")


def build_keys(df: pl.DataFrame, id_col: str, kinds: tuple[str, ...]) -> pl.DataFrame:
    parts = []
    if "tok" in kinds:
        parts.append(token_keys(df, id_col))
    if "pair" in kinds:
        parts.append(pair_keys(df, id_col))
    if "nospace" in kinds:
        parts.append(nospace_keys(df, id_col))
    if "house" in kinds or "zip" in kinds:
        a = address_keys(df, id_col)
        parts.append(a.filter(pl.col("kind").is_in(list(kinds))))
    return pl.concat(parts, how="vertical").unique([id_col, "key"])


def idf_table(s23_keys: pl.DataFrame, n_s23: int, cap: int) -> pl.DataFrame:
    """Rare keys only (df <= cap on the full S2/S3 pool), with the same IDF formula as candidates.py."""
    df = s23_keys.group_by("key").agg(pl.len().alias("df"), pl.col("kind").first())
    return (df.filter(pl.col("df") <= cap)
            .with_columns((((n_s23 + 1) / (pl.col("df") + 1)).log() + 1).alias("idf")))


def estimate_join_rows(s1_keys: pl.DataFrame, s23_keys: pl.DataFrame, rare: pl.DataFrame) -> int:
    """R6: sum over rare keys of (S1 rows x S2/S3 rows), before materialising the join."""
    a = s1_keys.join(rare.select("key"), on="key", how="semi").group_by("key").len("n1")
    b = s23_keys.join(rare.select("key"), on="key", how="semi").group_by("key").len("n2")
    j = a.join(b, on="key", how="inner")
    return int((j["n1"].cast(pl.UInt64) * j["n2"].cast(pl.UInt64)).sum()) if j.height else 0


def score(s1_keys: pl.DataFrame, s23_keys: pl.DataFrame, rare: pl.DataFrame, kinds: tuple[str, ...],
          max_rows: int) -> pl.DataFrame:
    """(s1_gid, s23_gid, score): sum of IDF over the shared rare keys of the given kinds."""
    r = rare.filter(pl.col("kind").is_in(list(kinds)))
    a = s1_keys.filter(pl.col("kind").is_in(list(kinds)))
    b = s23_keys.filter(pl.col("kind").is_in(list(kinds)))
    est = estimate_join_rows(a, b, r)
    if est > max_rows:
        raise MemoryError(f"{est:,} join rows for kinds {kinds} exceed the {max_rows:,}-row budget; chunk first")
    a = a.join(r.select("key", "idf"), on="key", how="inner")
    return (a.join(b.select("s23_gid", "key"), on="key", how="inner")
            .group_by("s1_gid", "s23_gid").agg(pl.col("idf").sum().alias("score")))


def topk(scored: pl.DataFrame, k: int) -> pl.DataFrame:
    """Deterministic top-k (ties broken by s23_gid, for reproducibility only)."""
    if k <= 0:
        return scored.clear().select("s1_gid", "s23_gid")
    return (scored.sort(["s1_gid", "score", "s23_gid"], descending=[False, True, False])
            .group_by("s1_gid", maintain_order=True).head(k).select("s1_gid", "s23_gid"))


def union_candidates(*sets: pl.DataFrame) -> pl.DataFrame:
    """Fix 2: combine independently ranked sets only after ranking; dedupe the union."""
    return pl.concat([s.select("s1_gid", "s23_gid") for s in sets], how="vertical").unique()
