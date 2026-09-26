"""C5: document frequency / IDF per (dataset, country) (Stage 1 prompt rev 5, §3 C5).

Label-free: computed from name_clean + addr_clean tokens of that dataset's own pool (train stats
from train pool, test stats from test pool), matching inference exactly. Token-level processing is
required, not optional (§3 C5, CG-8): clean each distinct token once, join back.
"""
import math

import polars as pl


def _tokens(name_col: str, addr_col: str) -> pl.Expr:
    return (pl.col(name_col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.concat(pl.col(addr_col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))))


def partial_counts(df: pl.DataFrame, name_col: str, addr_col: str,
                   group_cols: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """R5 (additive C5): this table's own (group_cols, n_docs) and (group_cols, token, df). Holding
    only this one table in memory at a time, never all of a dataset's tables together.
    """
    n_docs = df.group_by(group_cols).agg(pl.len().alias("n_docs"))
    exploded = (df.with_columns(_tokens(name_col, addr_col).list.unique().alias("_tok"))
                .select(*group_cols, "_tok").explode("_tok", empty_as_null=True).drop_nulls("_tok")
                .filter(pl.col("_tok") != ""))
    dfreq = exploded.group_by(*group_cols, "_tok").agg(pl.len().alias("df")).rename({"_tok": "token"})
    return n_docs, dfreq


def finalize_idf(n_docs_parts: list[pl.DataFrame], dfreq_parts: list[pl.DataFrame],
                 group_cols: list[str]) -> pl.DataFrame:
    """R5: sum partial counts across tables of the same dataset, THEN compute idf once. df and n_docs
    are both additive across disjoint row sets (each row is exactly one document, tables don't overlap).
    """
    n_docs = (pl.concat(n_docs_parts, how="vertical").group_by(group_cols)
              .agg(pl.col("n_docs").sum()))
    dfreq = (pl.concat(dfreq_parts, how="vertical").group_by(*group_cols, "token")
             .agg(pl.col("df").sum()))
    out = dfreq.join(n_docs, on=group_cols, how="left", validate="m:1")
    return out.with_columns((((pl.col("n_docs") + 1) / (pl.col("df") + 1)).log() + 1).alias("idf")).drop("n_docs")


def document_frequency(df: pl.DataFrame, name_col: str, addr_col: str, group_cols: list[str]) -> pl.DataFrame:
    """One row per (group_cols..., token): df (document count), idf. 'Document' = one row of df."""
    n_docs = df.group_by(group_cols).agg(pl.len().alias("n_docs"))
    exploded = (df.with_columns(_tokens(name_col, addr_col).list.unique().alias("_tok"))
                .select(*group_cols, "_tok").explode("_tok", empty_as_null=True).drop_nulls("_tok")
                .filter(pl.col("_tok") != ""))
    dfreq = exploded.group_by(*group_cols, "_tok").agg(pl.len().alias("df")).rename({"_tok": "token"})
    out = dfreq.join(n_docs, on=group_cols, how="left", validate="m:1")
    return out.with_columns(
        (((pl.col("n_docs") + 1) / (pl.col("df") + 1)).log() + 1).alias("idf")
    ).drop("n_docs")


def document_frequency_row_level(df: pl.DataFrame, name_col: str, addr_col: str,
                                  group_cols: list[str]) -> pl.DataFrame:
    """Reference (slow) implementation for the token-level-vs-row-level equivalence gate (§3 C5)."""
    rows = []
    for g_key, group in df.group_by(group_cols):
        n_docs = group.height
        counts: dict[str, int] = {}
        for name, addr in zip(group[name_col].to_list(), group[addr_col].to_list()):
            toks = set((name or "").split(" ")) | set((addr or "").split(" "))
            toks.discard("")
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        for tok, dfr in counts.items():
            idf = math.log((n_docs + 1) / (dfr + 1)) + 1
            rows.append({**dict(zip(group_cols, g_key)), "token": tok, "df": dfr, "idf": idf})
    schema = {c: pl.String for c in group_cols} | {"token": pl.String, "df": pl.UInt32, "idf": pl.Float64}
    return pl.DataFrame(rows, schema=schema).with_columns(pl.col("df").cast(pl.UInt32))
