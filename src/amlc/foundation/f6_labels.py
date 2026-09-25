"""F6: build labels tables (gt_links, s1_match_counts) with full integrity checks."""
import sys
from pathlib import Path
import polars as pl

BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
LABELS = Path(r"C:\Users\suremdra singh\amlc2026\data\labels\v1")


def main():
    errors = []

    s1 = pl.read_parquet(BRONZE / "train_source1.parquet").select(["gid", "entity_id", "country"])
    s2 = pl.read_parquet(BRONZE / "train_source2.parquet").select(["gid", "entity_id", "country"])
    s3 = pl.read_parquet(BRONZE / "train_source3.parquet").select(["gid", "entity_id", "country"])
    gt = pl.read_parquet(LABELS / "gt_raw.parquet")

    # every train S1 appears exactly once in gt
    s1_ids = set(s1["entity_id"].to_list())
    gt_ids = set(gt["source1_entity_id"].to_list())
    missing_from_gt = s1_ids - gt_ids
    extra_in_gt = gt_ids - s1_ids
    dup_gt_s1 = gt.height - gt.select("source1_entity_id").unique().height
    if missing_from_gt:
        errors.append(f"{len(missing_from_gt)} train S1 IDs missing from ground truth")
    if extra_in_gt:
        errors.append(f"{len(extra_in_gt)} ground-truth S1 IDs not in train_source1")
    if dup_gt_s1 != 0:
        errors.append(f"{dup_gt_s1} duplicate source1_entity_id rows in ground truth")

    # explode links
    gt_exp = gt.select(["source1_entity_id", "matched_entity_ids"]).with_columns(
        pl.when(pl.col("matched_entity_ids") == "")
          .then(pl.lit([], dtype=pl.List(pl.Utf8)))
          .otherwise(pl.col("matched_entity_ids").str.split(","))
          .alias("ids_list")
    )

    # intra-list duplicate check (before explode)
    intra_dup = gt_exp.select(
        (pl.col("ids_list").list.len() != pl.col("ids_list").list.unique().list.len()).alias("has_dup")
    )["has_dup"].sum()
    if intra_dup != 0:
        errors.append(f"{intra_dup} rows have duplicate IDs within their own list")

    # empty token check (e.g. "A,,B")
    empty_tok = gt_exp.select(
        pl.col("ids_list").list.eval(pl.element() == "").list.any().alias("has_empty")
    )["has_empty"].sum()
    if empty_tok != 0:
        errors.append(f"{empty_tok} rows contain an empty token inside a non-empty list")

    links = gt_exp.explode("ids_list").filter(
        pl.col("ids_list").is_not_null() & (pl.col("ids_list") != "")
    ).rename({"ids_list": "s23_id"})
    n_links = links.height
    print(f"[info] total links exploded: {n_links:,}")

    # resolve against S2/S3 pool
    s23 = pl.concat([
        s2.rename({"gid": "s23_gid", "entity_id": "s23_id", "country": "s23_country"}).with_columns(pl.lit(2, dtype=pl.UInt8).alias("s23_src")),
        s3.rename({"gid": "s23_gid", "entity_id": "s23_id", "country": "s23_country"}).with_columns(pl.lit(3, dtype=pl.UInt8).alias("s23_src")),
    ])

    links_r = links.join(s23, on="s23_id", how="left")
    unresolved = links_r.filter(pl.col("s23_gid").is_null()).height
    if unresolved != 0:
        errors.append(f"{unresolved} links reference an S2/S3 ID not found in train")

    # no S2/S3 record linked more than once
    dup_target = links_r.height - links_r.select("s23_id").unique().height
    if dup_target != 0:
        errors.append(f"{dup_target} S2/S3 records are linked to more than one S1 (partition violated)")

    # no cross-country links
    s1_small = s1.rename({"gid": "s1_gid", "entity_id": "s1_id", "country": "s1_country"})
    links_full = links_r.join(
        s1_small, left_on="source1_entity_id", right_on="s1_id", how="left"
    )
    cross = links_full.filter(pl.col("s1_country") != pl.col("s23_country")).height
    if cross != 0:
        errors.append(f"{cross} links cross countries")

    if n_links != 7_638_365:
        errors.append(f"link count {n_links:,} != expected 7,638,365 (recomputed check)")

    # write gt_links table
    gt_links = links_full.select([
        pl.col("s1_gid"), pl.col("source1_entity_id").alias("s1_id"),
        pl.col("s23_gid"), pl.col("s23_id"), pl.col("s23_src"),
    ])
    gt_links.write_parquet(LABELS / "gt_links.parquet", compression="zstd")
    print(f"[OK] gt_links.parquet written: {gt_links.height:,} rows")

    # s1_match_counts
    counts = gt_links.group_by(["s1_gid", "s1_id"]).agg([
        (pl.col("s23_src") == 2).sum().alias("n_s2"),
        (pl.col("s23_src") == 3).sum().alias("n_s3"),
        pl.len().alias("n_total"),
    ])
    # join back to get ALL s1 (including singletons with 0 rows in gt_links)
    full_counts = s1_small.select(["s1_gid", "s1_id", "s1_country"]).join(
        counts, on=["s1_gid", "s1_id"], how="left"
    ).with_columns([
        pl.col("n_s2").fill_null(0), pl.col("n_s3").fill_null(0), pl.col("n_total").fill_null(0),
    ]).with_columns(
        (pl.col("n_total") == 0).alias("is_singleton")
    ).rename({"s1_country": "country"})

    if full_counts.height != 2_206_821:
        errors.append(f"s1_match_counts has {full_counts.height:,} rows, expected 2,206,821")

    full_counts.write_parquet(LABELS / "s1_match_counts.parquet", compression="zstd")
    print(f"[OK] s1_match_counts.parquet written: {full_counts.height:,} rows")

    n_singleton = full_counts["is_singleton"].sum()
    print(f"[info] singletons: {n_singleton:,} ({100*n_singleton/full_counts.height:.2f}%)")
    print(f"[info] country distribution:", full_counts.group_by("country").len().to_dicts())

    print()
    if errors:
        print("F6 FAILURES:")
        for e in errors:
            print(f"  - {e}")
    print("F6 OVERALL:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
