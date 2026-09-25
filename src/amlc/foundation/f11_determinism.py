"""F11: rebuild bronze from raw into a temp dir and prove content hashes match v1 exactly."""
import hashlib
import shutil
import sys
from pathlib import Path
import polars as pl

RAW = Path(r"C:\Users\suremdra singh\amlc2026\data\raw\v1")
BRONZE_V1 = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
LABELS_V1 = Path(r"C:\Users\suremdra singh\amlc2026\data\labels\v1")
TMP = Path(r"C:\Users\SUREMD~1\AppData\Local\Temp\claude\C--Users-suremdra-singh\c01f1928-96ac-4a7d-be72-6b1e3f8f43f8\scratchpad\f11_rebuild")

SOURCE_FILES = [
    ("train", 1, "train_source1.tsv"), ("train", 2, "train_source2.tsv"), ("train", 3, "train_source3.tsv"),
    ("test", 1, "test_source1.tsv"), ("test", 2, "test_source2.tsv"), ("test", 3, "test_source3.tsv"),
]
SCHEMA = {"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}


def content_hash(df: pl.DataFrame, cols: list[str], sort_col: str) -> str:
    h = hashlib.sha256()
    df = df.sort(sort_col)
    for row in df.select(cols).iter_rows():
        line = "\t".join(str(x) if x is not None else "" for x in row) + "\n"
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def rebuild():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    gid = 0
    for dataset, src, fname in SOURCE_FILES:
        df = pl.read_csv(
            RAW / fname, separator="\t", has_header=True, schema_overrides=SCHEMA,
            infer_schema_length=0, try_parse_dates=False, null_values=[],
            quote_char=None, encoding="utf8", empty_string_is_null=False,
        )
        n = df.height
        df = df.with_columns([
            pl.lit(dataset).alias("dataset"), pl.lit(src, dtype=pl.UInt8).alias("src"),
            (pl.int_range(2, n + 2)).cast(pl.UInt32).alias("line_no"),
            (pl.int_range(gid, gid + n)).cast(pl.UInt32).alias("gid"),
        ])
        gid += n
        df.write_parquet(TMP / f"{dataset}_source{src}.parquet", compression="zstd")

    gt = pl.read_csv(
        RAW / "train_ground_truth.tsv", separator="\t", has_header=True,
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        infer_schema_length=0, try_parse_dates=False, null_values=[],
        quote_char=None, encoding="utf8", empty_string_is_null=False,
    )
    n = gt.height
    gt = gt.with_columns((pl.int_range(2, n + 2)).cast(pl.UInt32).alias("line_no"))
    gt.write_parquet(TMP / "gt_raw.parquet", compression="zstd")


def main():
    print("Rebuilding into temp dir (this reprocesses ~2.5GB from raw)...")
    rebuild()

    mismatches = 0
    for dataset, src, fname in SOURCE_FILES:
        v1_df = pl.read_parquet(BRONZE_V1 / f"{dataset}_source{src}.parquet")
        v2_df = pl.read_parquet(TMP / f"{dataset}_source{src}.parquet")
        cols = ["entity_id", "business_name", "business_address", "country"]
        h1 = content_hash(v1_df, cols, "gid")
        h2 = content_hash(v2_df, cols, "gid")
        status = "MATCH" if h1 == h2 else "MISMATCH"
        if h1 != h2:
            mismatches += 1
        print(f"[{status}] {dataset}_source{src}: v1={h1[:16]}... v2={h2[:16]}...")

    v1_gt = pl.read_parquet(LABELS_V1 / "gt_raw.parquet")
    v2_gt = pl.read_parquet(TMP / "gt_raw.parquet")
    h1 = content_hash(v1_gt, ["source1_entity_id", "matched_entity_ids"], "line_no")
    h2 = content_hash(v2_gt, ["source1_entity_id", "matched_entity_ids"], "line_no")
    status = "MATCH" if h1 == h2 else "MISMATCH"
    if h1 != h2:
        mismatches += 1
    print(f"[{status}] gt_raw: v1={h1[:16]}... v2={h2[:16]}...")

    print()
    print("F11 OVERALL:", "PASS — rebuild is fully deterministic" if mismatches == 0 else f"FAIL ({mismatches} mismatches)")

    if mismatches == 0:
        shutil.rmtree(TMP)
        print("[OK] temp rebuild directory cleaned up.")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
