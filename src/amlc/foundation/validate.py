"""F2: structural validation of raw TSV files, streaming over raw bytes only."""
import sys

EXPECTED_HEADERS = {
    "train_source1.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "train_source2.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "train_source3.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "test_source1.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "test_source2.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "test_source3.tsv": b"entity_id\tbusiness_name\tbusiness_address\tcountry",
    "train_ground_truth.tsv": b"source1_entity_id\tmatched_entity_ids",
}
EXPECTED_TABS = {
    "train_source1.tsv": 3, "train_source2.tsv": 3, "train_source3.tsv": 3,
    "test_source1.tsv": 3, "test_source2.tsv": 3, "test_source3.tsv": 3,
    "train_ground_truth.tsv": 1,
}
EXPECTED_LINES = {
    "train_source1.tsv": 2_206_822, "train_source2.tsv": 5_034_617, "train_source3.tsv": 5_285_604,
    "train_ground_truth.tsv": 2_206_822,
    "test_source1.tsv": 1_732_545, "test_source2.tsv": 4_887_274, "test_source3.tsv": 5_082_317,
}


def validate_file(path):
    import os
    name = os.path.basename(path)
    errors = []
    stats = dict(
        lines=0, crlf=0, blank=0, wrong_tabs=0, nul=0,
        leading_or_trailing_space_fields=0, quote_chars=0, maxlen=0,
    )
    header = None
    last_byte = b""
    with open(path, "rb") as f:
        for i, raw in enumerate(f, start=1):
            stats["lines"] += 1
            stats["maxlen"] = max(stats["maxlen"], len(raw))
            if raw.endswith(b"\r\n"):
                stats["crlf"] += 1
            stripped = raw.rstrip(b"\n").rstrip(b"\r")
            if stripped == b"":
                stats["blank"] += 1
            if b"\x00" in raw:
                stats["nul"] += raw.count(b"\x00")
            tabs = raw.count(b"\t")
            if i == 1:
                header = stripped
            else:
                if tabs != EXPECTED_TABS[name]:
                    stats["wrong_tabs"] += 1
            fields = stripped.split(b"\t")
            for fld in fields:
                if fld != fld.strip():
                    stats["leading_or_trailing_space_fields"] += 1
                if b'"' in fld:
                    stats["quote_chars"] += 1
            last_byte = raw

    if header != EXPECTED_HEADERS[name]:
        errors.append(f"header mismatch: got {header!r}")
    if stats["crlf"] != 0:
        errors.append(f"CRLF lines found: {stats['crlf']}")
    if stats["blank"] != 0:
        errors.append(f"blank lines found: {stats['blank']}")
    if stats["wrong_tabs"] != 0:
        errors.append(f"lines with wrong tab count: {stats['wrong_tabs']}")
    if stats["nul"] != 0:
        errors.append(f"NUL bytes found: {stats['nul']}")
    if not last_byte.endswith(b"\n"):
        errors.append("file does not end with a newline")
    if stats["lines"] != EXPECTED_LINES[name]:
        errors.append(f"line count mismatch: got {stats['lines']}, expected {EXPECTED_LINES[name]}")

    # strict UTF-8 check
    utf8_ok = True
    utf8_err = None
    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            for _ in f:
                pass
    except UnicodeDecodeError as e:
        utf8_ok = False
        utf8_err = str(e)
        errors.append(f"UTF-8 decode failure: {e}")

    return name, errors, stats, utf8_ok, utf8_err


def main():
    import glob
    files = sorted(glob.glob(r"C:\Users\suremdra singh\amlc2026\data\raw\v1\*.tsv"))
    all_ok = True
    for path in files:
        name, errors, stats, utf8_ok, utf8_err = validate_file(path)
        status = "PASS" if not errors else "FAIL"
        if errors:
            all_ok = False
        print(f"[{status}] {name}: lines={stats['lines']:,} maxlen={stats['maxlen']} "
              f"leading/trailing-space-fields={stats['leading_or_trailing_space_fields']} "
              f"quote-chars={stats['quote_chars']} utf8={'ok' if utf8_ok else 'FAIL'}")
        for e in errors:
            print(f"    ERROR: {e}")
    print()
    print("F2 OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
