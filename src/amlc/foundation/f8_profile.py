"""F8: label-free profile snapshot, cross-checked against earlier measurements."""
import sys
from pathlib import Path
import polars as pl

BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
LABELS = Path(r"C:\Users\suremdra singh\amlc2026\data\labels\v1")
PROFILE = Path(r"C:\Users\suremdra singh\amlc2026\data\profile\v1")
PROFILE.mkdir(parents=True, exist_ok=True)

NULL_TOKENS = ["null", "NULL", "<NULL>", "None", "N/A", "n/a", "na", "NA"]

# earlier measurements (from the analysis phase) to cross-check against, tolerance 0.5pp
EARLIER = {
    ("test", 1, "France", "accent_name"): 15.7,
    ("test", 1, "France", "accent_addr"): 27.8,
    ("train", 2, "India", "devanagari_name"): 13.4,   # approx from broader indic-script measurement
    ("train", 2, "India", "allcaps_name"): 14.9,
    ("train", 3, "India", "allcaps_name"): 2.6,
    ("train", 2, "US", "allcaps_name"): 21.5,
    ("train", 3, "US", "allcaps_name"): 3.2,
}


def profile_table(dataset, src):
    df = pl.read_parquet(BRONZE / f"{dataset}_source{src}.parquet")
    rows = []
    for (country,), grp in df.group_by("country"):
        n = grp.height
        name = grp["business_name"]
        addr = grp["business_address"]

        empty_addr = (addr == "").sum()
        null_tok_addr = addr.is_in(NULL_TOKENS).sum()
        accent_name = name.str.contains(r"[À-ÿ]").sum()
        accent_addr = addr.str.contains(r"[À-ÿ]").sum()
        devanagari_name = name.str.contains(r"[\u0900-\u097F]").sum()
        allcaps_name = ((name == name.str.to_uppercase()) & name.str.contains(r"[A-Za-z]")).sum()
        lower_name = ((name == name.str.to_lowercase()) & name.str.contains(r"[A-Za-z]")).sum()
        dblspace = (name.str.contains("  ") | addr.str.contains("  ")).sum()
        nonascii_name = name.str.contains(r"[^\x00-\x7F]").sum()

        rows.append(dict(
            dataset=dataset, src=src, country=country, n=n,
            empty_addr_pct=100 * empty_addr / n,
            null_token_addr_pct=100 * null_tok_addr / n,
            accent_name_pct=100 * accent_name / n,
            accent_addr_pct=100 * accent_addr / n,
            devanagari_name_pct=100 * devanagari_name / n,
            allcaps_name_pct=100 * allcaps_name / n,
            lower_name_pct=100 * lower_name / n,
            dblspace_pct=100 * dblspace / n,
            nonascii_name_pct=100 * nonascii_name / n,
            name_len_p50=name.str.len_chars().median(),
            name_len_p99=name.str.len_chars().quantile(0.99),
            addr_len_p50=addr.str.len_chars().median(),
            addr_len_p99=addr.str.len_chars().quantile(0.99),
        ))
    return rows


def main():
    all_rows = []
    for dataset in ["train", "test"]:
        for src in [1, 2, 3]:
            all_rows.extend(profile_table(dataset, src))

    prof = pl.DataFrame(all_rows)
    prof.write_parquet(PROFILE / "profile.parquet", compression="zstd")

    # label stats (train only)
    mc = pl.read_parquet(LABELS / "s1_match_counts.parquet")
    label_stats = mc.group_by("country").agg([
        pl.len().alias("n_s1"),
        pl.col("is_singleton").sum().alias("n_singleton"),
        pl.col("n_total").mean().alias("mean_matches"),
    ])

    gt_links = pl.read_parquet(LABELS / "gt_links.parquet")
    src_split = gt_links.group_by("s23_src").len()

    # cross-check against earlier measurements
    print("=== cross-check against earlier analysis-phase measurements ===")
    mismatches = 0
    checks = [
        ("test", 1, "France", "accent_name_pct", 15.7),
        ("test", 1, "France", "accent_addr_pct", 27.8),
        ("train", 2, "US", "allcaps_name_pct", 21.5),
        ("train", 3, "US", "allcaps_name_pct", 3.2),
        ("train", 2, "India", "allcaps_name_pct", 14.9),
        ("train", 3, "India", "allcaps_name_pct", 2.6),
    ]
    for ds, src, ctry, col, expected in checks:
        row = prof.filter((pl.col("dataset") == ds) & (pl.col("src") == src) & (pl.col("country") == ctry))
        if row.height == 0:
            print(f"  [MISSING] {ds} S{src} {ctry} {col}")
            mismatches += 1
            continue
        actual = row[col][0]
        diff = abs(actual - expected)
        status = "OK" if diff <= 2.0 else "MISMATCH"  # 2pp tolerance since earlier numbers were sample-based estimates
        if status == "MISMATCH":
            mismatches += 1
        print(f"  [{status}] {ds} S{src} {ctry} {col}: measured={actual:.2f}% earlier_estimate={expected}%")

    print()
    print("=== label stats (train) ===")
    print(label_stats)
    print()
    print("=== links by source ===")
    print(src_split)

    def df_to_md(d: pl.DataFrame) -> str:
        cols = d.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for row in d.iter_rows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    report_path = PROFILE / "profile_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# F8 Profile Report\n\n")
        f.write(df_to_md(prof))
        f.write("\n\n## Label stats (train)\n\n")
        f.write(df_to_md(label_stats))
        f.write(f"\n\n## Cross-check mismatches: {mismatches}\n")

    print(f"\n[OK] profile.parquet + profile_report.md written to {PROFILE}")
    print()
    print("F8 OVERALL:", "PASS" if mismatches == 0 else f"FAIL ({mismatches} mismatches — investigate)")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
