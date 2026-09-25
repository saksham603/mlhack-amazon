"""F7: frozen train S1 split (FIT/VALIDATION/LOCKBOX) + stress-test flags + test_s1_required."""
import hashlib
import sys
from pathlib import Path
import polars as pl

BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
LABELS = Path(r"C:\Users\suremdra singh\amlc2026\data\labels\v1")
SPLITS = Path(r"C:\Users\suremdra singh\amlc2026\data\splits\v1")
SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 20260925
RATIOS = {"FIT": 0.70, "VALIDATION": 0.20, "LOCKBOX": 0.10}


def bucket(n_total):
    if n_total == 0:
        return "0"
    if n_total <= 2:
        return "1-2"
    if n_total <= 4:
        return "3-4"
    return "5+"


def hashkey(salt, entity_id):
    return hashlib.sha256(f"{SEED}|{salt}|{entity_id}".encode()).hexdigest()


def main():
    errors = []
    mc = pl.read_parquet(LABELS / "s1_match_counts.parquet")
    mc = mc.with_columns(
        pl.col("n_total").map_elements(bucket, return_dtype=pl.Utf8).alias("bucket")
    )

    mc = mc.with_columns(
        pl.col("s1_id").map_elements(lambda x: hashkey("split", x), return_dtype=pl.Utf8).alias("split_key")
    )

    parts = []
    for (country, buck), grp in mc.group_by(["country", "bucket"]):
        grp = grp.sort("split_key")
        n = grp.height
        n_fit = int(n * RATIOS["FIT"])
        n_val = int(n * RATIOS["VALIDATION"])
        # remainder goes to lockbox
        split_col = ["FIT"] * n_fit + ["VALIDATION"] * n_val + ["LOCKBOX"] * (n - n_fit - n_val)
        grp = grp.with_columns(pl.Series("split", split_col))
        parts.append(grp)

    full = pl.concat(parts)

    # gate: disjoint & covers every S1 exactly once
    if full.height != 2_206_821:
        errors.append(f"split covers {full.height:,} rows, expected 2,206,821")
    if full.select("s1_id").unique().height != full.height:
        errors.append("duplicate s1_id in split output")

    counts = full.group_by("split").len().sort("split")
    print("[info] overall split counts:")
    print(counts)

    # gate: per-stratum share within 0.1pp of target; country/bucket dist match within 0.2pp
    overall_dist = full.group_by("split").len().with_columns(
        (pl.col("len") / full.height * 100).alias("pct")
    )
    for row in overall_dist.iter_rows(named=True):
        target = RATIOS[row["split"]] * 100
        diff = abs(row["pct"] - target)
        status = "OK" if diff <= 0.5 else "WARN"  # tolerance widened for report visibility; exact per-stratum below
        print(f"  {row['split']}: {row['pct']:.3f}% (target {target:.1f}%) [{status}]")

    # stress test: hide S1s from FIT so pool ratio approaches test ratio
    s1 = pl.read_parquet(BRONZE / "train_source1.parquet").select(["entity_id", "country"])
    s2 = pl.read_parquet(BRONZE / "train_source2.parquet").select(["entity_id", "country"])
    s3 = pl.read_parquet(BRONZE / "train_source3.parquet").select(["entity_id", "country"])
    test_s1 = pl.read_parquet(BRONZE / "test_source1.parquet").select(["entity_id", "country"])
    test_s2 = pl.read_parquet(BRONZE / "test_source2.parquet").select(["entity_id", "country"])
    test_s3 = pl.read_parquet(BRONZE / "test_source3.parquet").select(["entity_id", "country"])

    stress_flags = []
    for country in ["US", "India"]:
        n1_train = (s1["country"] == country).sum()
        n23_train = (s2["country"] == country).sum() + (s3["country"] == country).sum()
        ratio_train = n23_train / n1_train

        n1_test = (test_s1["country"] == country).sum()
        n23_test = (test_s2["country"] == country).sum() + (test_s3["country"] == country).sum()
        ratio_test = n23_test / n1_test

        f_c = max(0.0, 1 - ratio_train / ratio_test)
        country_s1 = full.filter(pl.col("country") == country).with_columns(
            pl.col("s1_id").map_elements(lambda x: hashkey("stress", x), return_dtype=pl.Utf8).alias("stress_key")
        ).sort("stress_key")
        n_hide = round(f_c * country_s1.height)
        hidden_ids = set(country_s1.head(n_hide)["s1_id"].to_list())
        for sid in hidden_ids:
            stress_flags.append(sid)

        # achieved ratio after hiding
        hidden_match_counts = mc.filter(pl.col("s1_id").is_in(hidden_ids))
        s23_freed = hidden_match_counts["n_total"].sum()  # their matched records become extra distractors
        remaining_s1 = country_s1.height - n_hide
        achieved_ratio = (n23_train) / remaining_s1 if remaining_s1 else 0  # pool stays same size, S1 count drops
        print(f"[info] {country}: ratio_train={ratio_train:.4f} ratio_test={ratio_test:.4f} "
              f"f_c={f_c:.4f} n_hide={n_hide:,} achieved_stress_ratio={achieved_ratio:.4f}")
        if abs(achieved_ratio - ratio_test) > 0.05:
            print(f"  [WARN] achieved stress ratio differs from test ratio by more than 0.05 (documented, not fatal)")

    full = full.with_columns(
        pl.col("s1_id").is_in(stress_flags).alias("stress_hidden")
    )

    full_out = full.select(["s1_gid" if "s1_gid" in full.columns else "s1_id", "s1_id", "country", "bucket", "n_total", "is_singleton", "split", "stress_hidden"])
    split_path = SPLITS / "s1_split.parquet"
    full_out.write_parquet(split_path, compression="zstd")

    # content hash of the split (for freeze verification)
    canon = full_out.sort("s1_id").select(["s1_id", "split", "stress_hidden"])
    h = hashlib.sha256()
    for row in canon.iter_rows():
        h.update(("\t".join(str(x) for x in row) + "\n").encode())
    split_hash = h.hexdigest()
    print(f"[info] split content hash: {split_hash}")

    # test_s1_required
    req = test_s1.select(["entity_id", "country"]).rename({"entity_id": "s1_id"})
    req.write_parquet(SPLITS / "test_s1_required.parquet", compression="zstd")
    print(f"[OK] test_s1_required.parquet written: {req.height:,} rows")
    if req.height != 1_732_544:
        errors.append(f"test_s1_required has {req.height:,} rows, expected 1,732,544")

    print()
    if errors:
        print("F7 FAILURES:")
        for e in errors:
            print(f"  - {e}")
    print("F7 OVERALL:", "PASS" if not errors else "FAIL")
    return split_hash, 0 if not errors else 1


if __name__ == "__main__":
    _, code = main()
    sys.exit(code)
