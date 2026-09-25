"""Foundation v2 build: fixes the Stage 0 stress-flag and label-exposure errors.

Changes vs v1 (v1 stays on disk, read-only, untouched):
  - stress_hidden drawn from FIT only (v1 wrongly drew from all three splits);
  - label-derived columns (n_total, is_singleton, bucket) removed from the open split file and
    moved behind access.load_match_counts();
  - manifest numbers and gate results computed during the run (v1 typed some by hand);
  - rebuild check extended to the label tables and the split;
  - strict tolerances restored (strata 0.1pp, profile cross-checks 0.5pp);
  - profile records the component-level missing-address rate.

Nothing is written unless every pre-write gate passes. The manifest is written only if the
post-write rebuild gates also pass; without a manifest, access.py refuses to serve v2.
"""
import importlib.metadata as md
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from amlc.foundation import hashing, split_v2
from amlc.foundation.f6_labels import build_label_tables, load_inputs
from amlc.foundation.f7_split import RATIOS, SEED
from amlc.foundation.f8_profile import profile_table

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
DATA = ROOT / "data"
RAW, BRONZE = DATA / "raw" / "v1", DATA / "bronze" / "v1"
LABELS_V1, LABELS_V2 = DATA / "labels" / "v1", DATA / "labels" / "v2"
SPLITS_V1, SPLITS_V2 = DATA / "splits" / "v1", DATA / "splits" / "v2"
PROFILE_V2 = DATA / "profile" / "v2"
MANIFEST_V1, MANIFEST_V2 = DATA / "MANIFEST_v1.json", DATA / "MANIFEST_v2.json"

REASON = (
    "v1 drew stress_hidden from all three splits (84,699 VALIDATION and 42,222 LOCKBOX S1 were "
    "flagged; spec said FIT only) and its stress check was tautological; v1's open split file "
    "exposed label-derived columns (n_total, is_singleton, bucket) for VALIDATION/LOCKBOX; v1 "
    "manifest contained hand-typed numbers; v1 F11 did not cover label tables or the split; two "
    "tolerances were loosened without notice. FIT/VALIDATION/LOCKBOX membership is unchanged."
)

NULL_TOKENS = ["null", "NULL", "<NULL>", "None", "N/A", "n/a", "na", "NA", ""]
PROFILE_CROSSCHECKS = [  # (dataset, src, country, column, earlier measurement), tolerance 0.5pp
    ("test", 1, "France", "accent_name_pct", 15.7),
    ("test", 1, "France", "accent_addr_pct", 27.8),
    ("train", 2, "US", "allcaps_name_pct", 21.5),
    ("train", 3, "US", "allcaps_name_pct", 3.2),
    ("train", 2, "India", "allcaps_name_pct", 14.9),
    ("train", 3, "India", "allcaps_name_pct", 2.6),
]
PROFILE_TOL_PP = 0.5
NULL_COMPONENT_EXPECTED = {("train", 2): 6.850, ("train", 3): 6.632, ("test", 2): 5.532, ("test", 3): 5.447}
NULL_COMPONENT_TOL_PP = 0.1

GATES: list[dict] = []


def gate(gid: str, desc: str, passed: bool, measured) -> bool:
    GATES.append({"gate": gid, "description": desc, "passed": bool(passed), "measured": measured})
    print(f"[{'PASS' if passed else 'FAIL'}] {gid}: {desc} | measured={measured}")
    return passed


def all_passed() -> bool:
    return all(g["passed"] for g in GATES)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    if hasattr(x, "item"):
        return x.item()
    return x


def df_to_md(d: pl.DataFrame) -> str:
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for row in d.iter_rows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def null_component_flags(addr: pl.Series) -> pl.Series:
    return addr.str.split(",").list.eval(pl.element().str.strip_chars().is_in(NULL_TOKENS)).list.any()


def main() -> int:
    for p in (SPLITS_V2, LABELS_V2, PROFILE_V2, MANIFEST_V2):
        if p.exists():
            print(f"REFUSING: {p} already exists. v2 is never overwritten; a rebuild must become v3.")
            return 1

    v1_manifest = json.loads(MANIFEST_V1.read_text(encoding="utf-8"))

    # ---- U: artifacts that v2 must NOT change are byte-identical to the v1 manifest ----
    unchanged = {
        "raw_file_hashes": {f.name: hashing.file_sha256(f) for f in sorted(RAW.glob("*.tsv"))},
        "bronze_file_hashes": {f.name: hashing.file_sha256(f) for f in sorted(BRONZE.glob("*.parquet"))},
        "label_file_hashes": {n: hashing.file_sha256(LABELS_V1 / n) for n in ("gt_raw.parquet", "gt_links.parquet")},
    }
    for key, now in unchanged.items():
        expected = {k: v for k, v in v1_manifest[key].items() if k in now}
        gate(f"U-{key}", f"{key} identical to MANIFEST_v1", now == expected and len(now) == len(expected),
             f"{sum(now[k] == expected.get(k) for k in now)}/{len(now)} match")

    # ---- A1: assignment identical to v1 ----
    mc = pl.read_parquet(LABELS_V1 / "s1_match_counts.parquet")
    assign = split_v2.build_assignment(mc)
    v1_split = pl.read_parquet(SPLITS_V1 / "s1_split.parquet", columns=["s1_id", "split"])
    diffs = split_v2.assignment_diffs(assign, v1_split)
    gate("A1", "FIT/VALIDATION/LOCKBOX assignment identical to v1", diffs == 0 and assign.height == v1_split.height,
         {"differences": diffs, "rows_v2": assign.height, "rows_v1": v1_split.height})
    first_id, first_split = assign["s1_id"][0], assign["split"][0]
    corrupt = assign.with_columns(
        pl.when(pl.col("s1_id") == first_id)
          .then(pl.lit("VALIDATION" if first_split != "VALIDATION" else "FIT"))
          .otherwise(pl.col("split")).alias("split")
    )
    neg = split_v2.assignment_diffs(corrupt, v1_split)
    gate("A1-neg", "comparison detects 1 deliberately changed assignment", neg == 1, {"detected_differences": neg})

    # ---- A2: stress parameters computed from bronze; stress drawn from FIT only ----
    train_n1, train_pool = split_v2.country_counts(BRONZE, "train")
    test_n1, test_pool = split_v2.country_counts(BRONZE, "test")
    params, notes = split_v2.stress_params(train_n1, train_pool, test_n1, test_pool)
    for n in notes:
        print(f"[note] {n}")
    gate("A2", "stress targets computed for every train country present in test",
         len(params) == len([c for c in train_n1 if c in test_n1]), params)
    split_full = split_v2.build_stress(assign, params)

    # ---- A3: independent recount + negative tests ----
    errs, meas = split_v2.check_stress(split_full, params)
    gate("A3", "stress: 0 hidden outside FIT, hidden count = target, achieved ratio within 0.01 of test",
         not errs, {"errors": errs, **meas})
    val_id = split_full.filter(pl.col("split") == "VALIDATION")["s1_id"][0]
    bad_a = split_full.with_columns(
        pl.when(pl.col("s1_id") == val_id).then(True).otherwise(pl.col("stress_hidden")).alias("stress_hidden"))
    errs_a, _ = split_v2.check_stress(bad_a, params)
    gate("A3-neg-a", "check fails when one VALIDATION S1 is flagged hidden", bool(errs_a), errs_a[:2])
    hid_id = split_full.filter(pl.col("stress_hidden"))["s1_id"][0]
    bad_b = split_full.with_columns(
        pl.when(pl.col("s1_id") == hid_id).then(False).otherwise(pl.col("stress_hidden")).alias("stress_hidden"))
    errs_b, _ = split_v2.check_stress(bad_b, params)
    gate("A3-neg-b", "check fails when one hidden FIT S1 is un-flagged", bool(errs_b), errs_b[:2])

    # ---- A4: strict strata balance ----
    errs, max_dev = split_v2.check_strata(split_full)
    gate("A4", f"every (country,bucket) stratum within {split_v2.STRATUM_TOL_PP}pp of 70/20/10",
         not errs, {"max_deviation_pp": max_dev, "errors": errs})

    # ---- B1/B2: open split without label-derived columns; counts moved to labels/v2 ----
    split_out = split_full.select(split_v2.SPLIT_V2_COLUMNS)
    leaked = set(split_out.columns) & split_v2.LABEL_DERIVED_COLUMNS
    gate("B1", "open split has exactly the 5 non-label columns", split_out.columns == split_v2.SPLIT_V2_COLUMNS and not leaked,
         {"columns": split_out.columns, "label_derived_present": sorted(leaked)})
    counts_out = mc.join(split_full.select(["s1_id", "bucket"]), on="s1_id", how="left")
    bucket_ok = counts_out.filter(
        pl.col("bucket") != pl.col("n_total").map_elements(split_v2.bucket, return_dtype=pl.Utf8)).height == 0
    singleton_ok = counts_out.filter(pl.col("is_singleton") != (pl.col("n_total") == 0)).height == 0
    gate("B2", "match counts carry bucket, consistent with n_total; row count = train S1",
         bucket_ok and singleton_ok and counts_out.height == mc.height and counts_out["bucket"].null_count() == 0,
         {"rows": counts_out.height, "bucket_consistent": bucket_ok, "singleton_consistent": singleton_ok})

    # ---- T1: required test S1 list carried over unchanged ----
    req_v1 = SPLITS_V1 / "test_s1_required.parquet"
    req_df = pl.read_parquet(req_v1)
    gate("T1", "test_s1_required row count = test S1", req_df.height == sum(test_n1.values()),
         {"rows": req_df.height, "test_s1": sum(test_n1.values())})

    # ---- C3/C4: profile v2 with strict cross-checks and component-level missing addresses ----
    rows, comp_tables = [], {}
    for ds in ("train", "test"):
        for src in (1, 2, 3):
            base = profile_table(ds, src)
            df = pl.read_parquet(BRONZE / f"{ds}_source{src}.parquet", columns=["business_address", "country"])
            flags = df.with_columns(null_component_flags(df["business_address"]).alias("f"))
            comp_tables[(ds, src)] = 100 * flags["f"].sum() / flags.height
            by_c = {(c,): 100 * g["f"].sum() / g.height for (c,), g in flags.group_by("country")}
            for r in base:
                r["addr_null_component_pct"] = by_c[(r["country"],)]
            rows.extend(base)
    prof = pl.DataFrame(rows)
    for ds, src, ctry, col, expected in PROFILE_CROSSCHECKS:
        v = prof.filter((pl.col("dataset") == ds) & (pl.col("src") == src) & (pl.col("country") == ctry))[col][0]
        gate(f"C3-{ds}S{src}-{ctry}-{col}", f"profile cross-check within {PROFILE_TOL_PP}pp",
             abs(v - expected) <= PROFILE_TOL_PP, {"measured": v, "earlier": expected})
    for (ds, src), expected in NULL_COMPONENT_EXPECTED.items():
        v = comp_tables[(ds, src)]
        gate(f"C4-{ds}S{src}-null-component", f"component-level missing-address rate within {NULL_COMPONENT_TOL_PP}pp",
             abs(v - expected) <= NULL_COMPONENT_TOL_PP, {"measured": v, "stage1_spec": expected})
    for ds in ("train", "test"):
        v = comp_tables[(ds, 1)]
        gate(f"C4-{ds}S1-null-component", "S1 has no missing address components", v == 0.0, {"measured": v})

    if not all_passed():
        print("\nPRE-WRITE GATES FAILED: nothing written.")
        return 1

    # ---- write v2 artifacts ----
    SPLITS_V2.mkdir(parents=True)
    LABELS_V2.mkdir(parents=True)
    PROFILE_V2.mkdir(parents=True)
    split_out.write_parquet(SPLITS_V2 / "s1_split.parquet", compression="zstd")
    (SPLITS_V2 / "test_s1_required.parquet").write_bytes(req_v1.read_bytes())
    counts_out.write_parquet(LABELS_V2 / "s1_match_counts.parquet", compression="zstd")
    prof.write_parquet(PROFILE_V2 / "profile.parquet", compression="zstd")
    (PROFILE_V2 / "profile_report.md").write_text(
        "# Profile v2\n\n`addr_null_component_pct`: % of rows with at least one comma-separated address "
        f"component equal to one of {NULL_TOKENS} after stripping.\n\n" + df_to_md(prof) + "\n", encoding="utf-8")

    # ---- F11-v2: rebuild from inputs a second time and compare content with what was written ----
    SPLIT_COLS = split_v2.SPLIT_V2_COLUMNS
    COUNT_COLS = counts_out.columns
    LINK_COLS = ["s1_gid", "s1_id", "s23_gid", "s23_id", "s23_src"]
    V1_COUNT_COLS = ["s1_gid", "s1_id", "country", "n_s2", "n_s3", "n_total", "is_singleton"]

    written_split = pl.read_parquet(SPLITS_V2 / "s1_split.parquet")
    written_counts = pl.read_parquet(LABELS_V2 / "s1_match_counts.parquet")
    split_hash = hashing.content_hash(written_split, SPLIT_COLS, "s1_id")
    counts_hash = hashing.content_hash(written_counts, COUNT_COLS, "s1_id")

    assign_b = split_v2.build_assignment(mc)
    split_b = split_v2.build_stress(assign_b, params)
    gate("F11-split", "split rebuilt from scratch is content-identical to written split v2",
         hashing.content_hash(split_b.select(SPLIT_COLS), SPLIT_COLS, "s1_id") == split_hash, {"hash": split_hash})
    counts_b = mc.join(split_b.select(["s1_id", "bucket"]), on="s1_id", how="left")
    gate("F11-counts-v2", "match counts rebuilt are content-identical to written labels v2",
         hashing.content_hash(counts_b, COUNT_COLS, "s1_id") == counts_hash, {"hash": counts_hash})
    gate("F11-required", "test_s1_required v2 is byte-identical to v1",
         hashing.file_sha256(SPLITS_V2 / "test_s1_required.parquet") == hashing.file_sha256(req_v1), {})

    s1, s2, s3, gt = load_inputs(BRONZE, LABELS_V1)
    links_b, counts_v1_b, label_errs, n_links = build_label_tables(s1, s2, s3, gt)
    gate("F11-label-integrity", "label rebuild integrity checks", not label_errs, {"errors": label_errs, "links": n_links})
    gl_v1 = pl.read_parquet(LABELS_V1 / "gt_links.parquet")
    gate("F11-gt_links", "gt_links rebuilt from bronze is content-identical to labels v1",
         hashing.content_hash(links_b, LINK_COLS, "s23_gid") == hashing.content_hash(gl_v1, LINK_COLS, "s23_gid"),
         {"rows_rebuilt": links_b.height, "rows_v1": gl_v1.height})
    mc_v1_hash = hashing.content_hash(mc, V1_COUNT_COLS, "s1_id")
    gate("F11-counts-v1", "s1_match_counts rebuilt from bronze is content-identical to labels v1",
         hashing.content_hash(counts_v1_b, V1_COUNT_COLS, "s1_id") == mc_v1_hash, {"hash": mc_v1_hash})

    if not all_passed():
        print("\nPOST-WRITE GATES FAILED: v2 files exist but NO manifest was written, so access.py will refuse them.")
        return 1

    # ---- manifest: every number computed in this run ----
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "src", "tests"], text=True).strip()
    per_split = split_out.group_by(["split", "country"]).len().sort(["split", "country"]).to_dicts()
    manifest = {
        "version": "v2",
        "reason": REASON,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_command": "python src/amlc/foundation/build_v2.py",
        "git_commit": commit,
        "code_uncommitted_changes": bool(dirty),
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {p: md.version(p) for p in ("polars", "pyarrow", "duckdb", "pytest")},
        "v1_manifest_sha256": hashing.file_sha256(MANIFEST_V1),
        "unchanged_artifacts": unchanged,
        "new_artifacts": {
            "split_file": {"path": "data/splits/v2/s1_split.parquet", "sha256": hashing.file_sha256(SPLITS_V2 / "s1_split.parquet")},
            "split_content_hash": split_hash,
            "split_content_hash_columns": SPLIT_COLS,
            "test_s1_required": {"path": "data/splits/v2/test_s1_required.parquet", "sha256": hashing.file_sha256(SPLITS_V2 / "test_s1_required.parquet")},
            "match_counts": {"path": "data/labels/v2/s1_match_counts.parquet", "sha256": hashing.file_sha256(LABELS_V2 / "s1_match_counts.parquet"), "content_hash": counts_hash},
            "profile": {f.name: hashing.file_sha256(f) for f in sorted(PROFILE_V2.iterdir())},
        },
        "split_parameters": {
            "seed": SEED, "ratios": RATIOS, "strata": "country x match-count bucket (0 / 1-2 / 3-4 / 5+)",
            "stratum_tolerance_pp": split_v2.STRATUM_TOL_PP,
        },
        "computed_counts": {
            "train_s1_by_country": train_n1, "train_pool_by_country": train_pool,
            "test_s1_by_country": test_n1, "test_pool_by_country": test_pool,
            "split_by_country": per_split,
            "gt_links_rows": n_links,
        },
        "stress_test": {"parameters": params, "notes": notes, "drawn_from": "FIT only"},
        "profile_tolerances": {"crosscheck_pp": PROFILE_TOL_PP, "null_component_pp": NULL_COMPONENT_TOL_PP},
        "gates": GATES,
        "all_gates_passed": all_passed(),
    }
    MANIFEST_V2.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    for d in (SPLITS_V2, LABELS_V2, PROFILE_V2):
        for f in d.iterdir():
            os.chmod(f, stat.S_IREAD)
    os.chmod(MANIFEST_V2, stat.S_IREAD)
    print(f"\n[OK] {len(GATES)} gates passed. v2 written and locked read-only. Manifest: {MANIFEST_V2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
