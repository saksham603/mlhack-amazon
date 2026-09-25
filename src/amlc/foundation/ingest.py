"""F3: lossless bronze ingest. Every column stays pl.Utf8. No interpretation."""
import polars as pl
from pathlib import Path

RAW = Path(r"C:\Users\suremdra singh\amlc2026\data\raw\v1")
BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
BRONZE.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = [
    ("train", 1, "train_source1.tsv"),
    ("train", 2, "train_source2.tsv"),
    ("train", 3, "train_source3.tsv"),
    ("test", 1, "test_source1.tsv"),
    ("test", 2, "test_source2.tsv"),
    ("test", 3, "test_source3.tsv"),
]

SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def ingest_one(dataset, src, fname, gid_start):
    path = RAW / fname
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        schema_overrides=SCHEMA,
        infer_schema_length=0,
        try_parse_dates=False,
        null_values=[],
        quote_char=None,
        encoding="utf8",
        empty_string_is_null=False,
    )
    n = df.height
    # null count assertion — empty fields must be empty strings, never null
    null_counts = df.null_count()
    for col in df.columns:
        nc = null_counts[col][0]
        if nc != 0:
            raise AssertionError(f"{fname} column {col} has {nc} nulls (must be 0, empty string expected)")

    df = df.with_columns([
        pl.lit(dataset).alias("dataset"),
        pl.lit(src, dtype=pl.UInt8).alias("src"),
        (pl.int_range(2, n + 2)).cast(pl.UInt32).alias("line_no"),  # header=1, first data row=2
        (pl.int_range(gid_start, gid_start + n)).cast(pl.UInt32).alias("gid"),
    ])
    out_path = BRONZE / f"{dataset}_source{src}.parquet"
    df.write_parquet(out_path, compression="zstd", row_group_size=500_000)
    return n, gid_start + n, out_path


def ingest_ground_truth():
    path = RAW / "train_ground_truth.tsv"
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        infer_schema_length=0,
        try_parse_dates=False,
        null_values=[],
        quote_char=None,
        encoding="utf8",
        empty_string_is_null=False,
    )
    null_counts = df.null_count()
    for col in df.columns:
        nc = null_counts[col][0]
        if nc != 0:
            raise AssertionError(f"train_ground_truth.tsv column {col} has {nc} nulls (must be 0)")
    n = df.height
    df = df.with_columns([
        (pl.int_range(2, n + 2)).cast(pl.UInt32).alias("line_no"),
    ])
    out_path = BRONZE.parent.parent / "labels" / "v1" / "gt_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd", row_group_size=500_000)
    return n, out_path


def main():
    gid = 0
    report = []
    for dataset, src, fname in SOURCE_FILES:
        n, gid, out_path = ingest_one(dataset, src, fname, gid)
        report.append((dataset, src, fname, n, out_path))
        print(f"[OK] {fname} -> {out_path.name}: {n:,} rows, gid now at {gid:,}")
    print(f"\nTotal gid range: 0..{gid-1} ({gid:,} rows)")

    gt_n, gt_path = ingest_ground_truth()
    print(f"[OK] train_ground_truth.tsv -> {gt_path.name}: {gt_n:,} rows")
    return report


if __name__ == "__main__":
    main()
