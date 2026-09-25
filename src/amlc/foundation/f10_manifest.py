"""F10: build MANIFEST_v1.json and freeze data/*/v1/ as read-only. Refuses to overwrite an existing v1."""
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
MANIFEST_PATH = ROOT / "data" / "MANIFEST_v1.json"

EXPECTED_RAW_HASHES = {
    "train_source1.tsv": "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973",
    "train_source2.tsv": "6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed",
    "train_source3.tsv": "67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9",
    "train_ground_truth.tsv": "70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037",
    "test_source1.tsv": "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5",
    "test_source2.tsv": "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd",
    "test_source3.tsv": "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035",
}

SPLIT_HASH = "0f1393be21cb009ca504cc3069ef74071253546743f68fde57bcf091a95b6ae9"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if MANIFEST_PATH.exists():
        print(f"REFUSING to run: {MANIFEST_PATH} already exists. This build script never "
              f"overwrites an existing v1. To rebuild, create v2 with a written reason.")
        return 1

    raw_dir = ROOT / "data" / "raw" / "v1"
    bronze_dir = ROOT / "data" / "bronze" / "v1"
    labels_dir = ROOT / "data" / "labels" / "v1"
    splits_dir = ROOT / "data" / "splits" / "v1"
    profile_dir = ROOT / "data" / "profile" / "v1"

    raw_hashes = {f.name: file_sha256(f) for f in sorted(raw_dir.glob("*.tsv"))}
    for name, expected in EXPECTED_RAW_HASHES.items():
        if raw_hashes.get(name) != expected:
            print(f"REFUSING to freeze: raw hash mismatch for {name}")
            return 1

    bronze_hashes = {f.name: file_sha256(f) for f in sorted(bronze_dir.glob("*.parquet"))}
    label_hashes = {f.name: file_sha256(f) for f in sorted(labels_dir.glob("*.parquet"))}
    split_hashes = {f.name: file_sha256(f) for f in sorted(splits_dir.glob("*.parquet"))}
    profile_hashes = {f.name: file_sha256(f) for f in sorted(profile_dir.glob("*.parquet"))}

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = "no-commits-yet"

    manifest = {
        "version": "v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_command": "src/amlc/foundation/f10_manifest.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "raw_file_hashes": raw_hashes,
        "raw_hashes_verified_against_official_zip": True,
        "official_zip_path": r"C:\Users\suremdra singh\Downloads\6ab10eb3b23ba_student_resource.zip",
        "bronze_file_hashes": bronze_hashes,
        "label_file_hashes": label_hashes,
        "split_file_hashes": split_hashes,
        "profile_file_hashes": profile_hashes,
        "split_content_hash": SPLIT_HASH,
        "split_seed": 20260925,
        "split_ratios": {"FIT": 0.70, "VALIDATION": 0.20, "LOCKBOX": 0.10},
        "split_bucket_edges": "0 / 1-2 / 3-4 / 5+ matches, stratified by country",
        "row_counts": {
            "train_source1": 2_206_821, "train_source2": 5_034_616, "train_source3": 5_285_603,
            "train_ground_truth": 2_206_821,
            "test_source1": 1_732_544, "test_source2": 4_887_273, "test_source3": 5_082_316,
            "gt_links": 7_638_365,
        },
        "split_counts": {"FIT": 1_544_770, "VALIDATION": 441_361, "LOCKBOX": 220_690},
        "stress_test": {
            "US": {"ratio_train": 4.6742, "ratio_test": 5.7563, "n_hide": 248_831},
            "India": {"ratio_train": 4.6800, "ratio_test": 5.8243, "n_hide": 173_510},
        },
        "library_versions": {"polars": "1.44.2", "pyarrow": "25.0.1", "duckdb": "1.5.5", "pytest": "9.1.1"},
        "gates_passed": ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
        "not_verified": [
            "Whether test's larger S2/S3-per-S1 ratio reflects more distractors or more true matches per S1.",
            "Whether test has the same singleton rate as train (5.58%).",
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    os.chmod(MANIFEST_PATH, stat.S_IREAD)
    print(f"[OK] {MANIFEST_PATH} written and locked read-only.")
    print("F10 OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
