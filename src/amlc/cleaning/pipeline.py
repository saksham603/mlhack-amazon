"""Orchestrates C2 -> C3 -> C4 -> C5 on C1's output (Stage 1 prompt rev 5). C1 is done separately
(c1_run.py) and published at data/interim/c1/v2; this module never rewrites it (G-P16: resume from
published, hash-checked output only).
"""
import json
from pathlib import Path

import polars as pl

from amlc.cleaning import addresses, names, script, stats
from amlc.foundation import access
from amlc.foundation.hashing import file_sha256

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
C1_DIR = ROOT / "data" / "interim" / "c1" / "v2"
C1_MANIFEST = C1_DIR / "MANIFEST_c1_v2.json"
SILVER_DIR = ROOT / "data" / "silver" / "v1"
DICT_DIR = ROOT / "data" / "dictionaries" / "v1"

TABLES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]

INDIC_RANGE = r"[\u0900-\u0D7F]"


def _c1_manifest() -> dict:
    m = json.loads(C1_MANIFEST.read_text(encoding="utf-8"))
    if not m.get("all_gates_passed"):
        raise RuntimeError("C1 v2 manifest records failed gates. STOP.")
    return m


def load_c1_output(dataset: str, src: int) -> pl.DataFrame:
    """C1's published output, hash-checked against MANIFEST_c1_v2.json (G-P16)."""
    name = f"{dataset}_source{src}"
    m = _c1_manifest()["outputs"][name]
    path = C1_DIR / f"{name}.parquet"
    actual = file_sha256(path)
    if actual != m["sha256"]:
        raise RuntimeError(f"{path} hash mismatch: expected {m['sha256']}, got {actual}. STOP.")
    return pl.read_parquet(path)


def _with_country(dataset: str, src: int, c1: pl.DataFrame) -> pl.DataFrame:
    meta = access.load_bronze(dataset, src, columns=["gid", "country"])
    out = c1.join(meta, on="gid", how="left", validate="1:1")
    if out["country"].null_count():
        raise AssertionError(f"{dataset}_source{src}: a row failed to match a bronze country")
    return out


def build_fit_dictionary_pairs() -> pl.DataFrame:
    """FIT-only (G-L1) India-candidate pairs for C2 dictionary construction: s1_gid, s1_name,
    s23_name, country. Filtered in Polars (cheap) before the Python-side alignment loop.
    """
    labels = access.load_labels("FIT").select("s1_gid", "s23_gid", "s23_src")
    split = access.load_split().filter(pl.col("split") == "FIT").select(
        pl.col("s1_gid"), pl.col("country"))
    s1 = load_c1_output("train", 1).select(
        pl.col("gid").alias("s1_gid"), pl.col("name_clean").alias("s1_name"))
    s23_parts = []
    for src in (2, 3):
        t = load_c1_output("train", src).select(
            pl.col("gid").alias("s23_gid"), pl.col("name_clean").alias("s23_name"))
        s23_parts.append(labels.filter(pl.col("s23_src") == src).join(t, on="s23_gid", how="inner", validate="m:1"))
    pairs = pl.concat(s23_parts, how="vertical")
    pairs = (pairs.join(s1, on="s1_gid", how="inner", validate="m:1")
                  .join(split, on="s1_gid", how="inner", validate="m:1"))
    pairs = pairs.filter(pl.col("s23_name").str.contains(INDIC_RANGE))
    pairs = pairs.filter(
        pl.col("s1_name").str.split(" ").list.len() == pl.col("s23_name").str.split(" ").list.len())
    return pairs.select("s1_gid", "s1_name", "s23_name", "country")


def clean_table(dataset: str, src: int, dictionary: pl.DataFrame, c1: pl.DataFrame | None = None) -> pl.DataFrame:
    """Runs C2 (apply) -> C3 -> C4 -> C5-ready columns on one table. Row count/gid preserved (CG-1).
    c1: pre-loaded (and optionally sliced, for a dry run) C1 output; defaults to the full published
    table via load_c1_output(). Passing it explicitly avoids monkeypatching the loader for slicing.
    """
    c1 = (c1 if c1 is not None else load_c1_output(dataset, src)).rename(
        {"name_clean": "_name_c1", "addr_clean": "_addr_c1"})
    df = _with_country(dataset, src, c1)

    # C2: per-token script flags on C1-cleaned name (before translation, per spec), then translate.
    script_flags = pl.Series("name_script_flags", [script.classify_tokens(n) for n in df["_name_c1"]])
    c2 = script.apply_dictionary(df["_name_c1"], dictionary)
    df = df.with_columns(script_flags, c2["name_translit"], c2["name_indic_untranslated"],
                         c2["name_token_variants"])

    # C3: country-aware, row-wise. Two with_columns calls, not one: every expression in a single
    # with_columns() call sees only the DataFrame as it was BEFORE that call, so a name_tokens
    # expression referencing name_clean in the same call as name_clean is added would fail to find
    # it (the bug this fixes: 129 unit tests passed, none of them exercised this two-column path).
    c3 = names.clean_names(df["name_translit"], df["country"])
    df = df.with_columns(c3["name_clean"], c3["name_legal_form"])
    df = df.with_columns(pl.col("name_clean").str.split(" ").alias("name_tokens"))

    # C4. with_columns doesn't accept a DataFrame directly; unpack its Series (a DataFrame literal
    # would otherwise be coerced into a single Object-dtype column, the CG-27 test caught this).
    c4 = addresses.clean_addresses(df["_addr_c1"], df["country"])
    df = df.with_columns(*c4.get_columns(), pl.col("_addr_c1").alias("addr_clean"))

    keep = ["gid", "name_clean", "name_legal_form", "name_tokens", "name_token_variants",
            "name_script_flags", "name_indic_untranslated", "addr_clean", "addr_components",
            "addr_house_number", "addr_house_number_raw", "addr_zip", "addr_missing",
            "addr_null_parts_removed", "addr_parts_emptied"]
    out = df.select(keep)
    if out.height != c1.height or set(out["gid"]) != set(c1["gid"]):
        raise AssertionError(f"{dataset}_source{src}: row count or gid set changed (CG-1)")
    return out


def idf_partial_counts(dataset: str, src: int, cleaned: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """R5: this ONE table's contribution to C5 IDF (n_docs, df), computed without holding any other
    table of the dataset in memory. Caller sums these across a dataset's 3 tables (finalize_idf).
    """
    meta = access.load_bronze(dataset, src, columns=["gid", "country"])
    pool = cleaned.join(meta, on="gid", how="left", validate="1:1").select("country", "name_clean", "addr_clean")
    return stats.partial_counts(pool, "name_clean", "addr_clean", ["country"])
