"""F5: key and field integrity checks across all bronze tables."""
import sys
from pathlib import Path
import polars as pl

BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")

FILES = [
    ("train", 1), ("train", 2), ("train", 3),
    ("test", 1), ("test", 2), ("test", 3),
]


def main():
    errors = []
    all_ids = {}
    tables = {}
    for dataset, src in FILES:
        df = pl.read_parquet(BRONZE / f"{dataset}_source{src}.parquet")
        tables[(dataset, src)] = df

        # uniqueness within file
        dup = df.height - df.select("entity_id").unique().height
        if dup != 0:
            errors.append(f"{dataset} S{src}: {dup} duplicate entity_id within file")

        # ID format: ^S{src}-\d+$
        fmt_ok = df.select(
            pl.col("entity_id").str.contains(rf"^S{src}-\d+$").alias("ok")
        )["ok"]
        bad_fmt = (~fmt_ok).sum()
        if bad_fmt != 0:
            errors.append(f"{dataset} S{src}: {bad_fmt} IDs don't match ^S{src}-\\d+$")

        # empty country
        empty_country = (df["country"] == "").sum()
        if empty_country != 0:
            errors.append(f"{dataset} S{src}: {empty_country} rows with empty country")

        # empty name (informational, expected 0 but recorded either way)
        empty_name = (df["business_name"] == "").sum()

        ids_set = set(df["entity_id"].to_list())
        key = f"{dataset}_S{src}"
        all_ids[key] = ids_set
        print(f"[info] {dataset} S{src}: n={df.height:,} dup_ids={dup} bad_format={bad_fmt} "
              f"empty_country={empty_country} empty_name={empty_name} "
              f"countries={sorted(df['country'].unique().to_list())}")

    # cross-file uniqueness: no ID string in more than one of the 6 files
    seen = {}
    cross_dup = 0
    for key, ids in all_ids.items():
        for i in ids:
            if i in seen:
                cross_dup += 1
            else:
                seen[i] = key
    if cross_dup != 0:
        errors.append(f"{cross_dup} entity_id strings appear in more than one file")

    # train vs test overlap (should be 0 per source)
    for src in [1, 2, 3]:
        ov = len(all_ids[f"train_S{src}"] & all_ids[f"test_S{src}"])
        print(f"[info] S{src} train/test ID overlap: {ov}")
        if ov != 0:
            errors.append(f"S{src}: {ov} IDs shared between train and test")

    print()
    if errors:
        print("F5 FAILURES:")
        for e in errors:
            print(f"  - {e}")
    print("F5 OVERALL:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
