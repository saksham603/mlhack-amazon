"""F4: prove bronze Parquet round-trips to byte-identical TSV."""
import hashlib
import sys
from pathlib import Path
import polars as pl

BRONZE = Path(r"C:\Users\suremdra singh\amlc2026\data\bronze\v1")
LABELS = Path(r"C:\Users\suremdra singh\amlc2026\data\labels\v1")

EXPECTED_HASHES = {
    "train_source1.tsv": "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973",
    "train_source2.tsv": "6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed",
    "train_source3.tsv": "67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9",
    "train_ground_truth.tsv": "70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037",
    "test_source1.tsv": "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5",
    "test_source2.tsv": "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd",
    "test_source3.tsv": "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035",
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def hash_of_tsv_bytes(header_cols, rows_iter):
    h = hashlib.sha256()
    h.update(("\t".join(header_cols) + "\n").encode("utf-8"))
    for row in rows_iter:
        h.update(("\t".join(row) + "\n").encode("utf-8"))
    return h.hexdigest()


def check_source(dataset, src, fname):
    df = pl.read_parquet(BRONZE / f"{dataset}_source{src}.parquet")
    df = df.sort("gid")
    h = hashlib.sha256()
    h.update(("\t".join(SOURCE_COLS) + "\n").encode("utf-8"))
    for row in df.select(SOURCE_COLS).iter_rows():
        h.update(("\t".join(row) + "\n").encode("utf-8"))
    actual = h.hexdigest()
    expected = EXPECTED_HASHES[fname]
    return actual == expected, actual, expected


def check_gt():
    df = pl.read_parquet(LABELS / "gt_raw.parquet").sort("line_no")
    h = hashlib.sha256()
    h.update(("\t".join(GT_COLS) + "\n").encode("utf-8"))
    for row in df.select(GT_COLS).iter_rows():
        h.update(("\t".join(row) + "\n").encode("utf-8"))
    actual = h.hexdigest()
    expected = EXPECTED_HASHES["train_ground_truth.tsv"]
    return actual == expected, actual, expected


def main():
    all_ok = True
    for dataset, src, fname in [
        ("train", 1, "train_source1.tsv"), ("train", 2, "train_source2.tsv"), ("train", 3, "train_source3.tsv"),
        ("test", 1, "test_source1.tsv"), ("test", 2, "test_source2.tsv"), ("test", 3, "test_source3.tsv"),
    ]:
        ok, actual, expected = check_source(dataset, src, fname)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"[{status}] {fname}: roundtrip={actual}")
        if not ok:
            print(f"         expected={expected}")

    ok, actual, expected = check_gt()
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"[{status}] train_ground_truth.tsv: roundtrip={actual}")
    if not ok:
        print(f"         expected={expected}")

    print()
    print("F4 OVERALL:", "PASS — bronze is byte-for-byte lossless" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
