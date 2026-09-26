"""Streaming check of output/<name>/candidate_pairs.tsv (the official validator holds every id in memory,
which ran out of RAM on 206.7M pairs): header, one row per test S1, no duplicate ids in a row, only S2/S3
ids, and every matched id inside that S1's candidates. Prints CANDIDATES OK on success.

Run: python -m amlc.pipeline.check_candidates <output dir>
"""
import sys
from pathlib import Path

import polars as pl

TEST1 = Path(r"C:\Users\suremdra singh\Desktop\dataset_student_resource\student_resource\dataset\test\test_source1.tsv")


def main() -> int:
    out = Path(sys.argv[1])
    s1 = set(pl.read_csv(TEST1, separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])["entity_id"].to_list())
    match = {}
    with open(out / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if b:
                match[a] = b.split(",")
    errors, seen, pairs = [], set(), 0
    with open(out / "candidate_pairs.tsv", encoding="utf-8") as f:
        if next(f).rstrip("\n") != "source1_entity_id\tcandidate_entity_ids":
            errors.append("bad header")
        for n, line in enumerate(f, start=2):
            a, _, b = line.rstrip("\n").partition("\t")
            ids = b.split(",") if b else []
            pairs += len(ids)
            if a in seen:
                errors.append(f"line {n}: duplicate S1 {a}")
            seen.add(a)
            if len(set(ids)) != len(ids):
                errors.append(f"line {n}: duplicate candidate ids")
            if any(not (i.startswith("S2-") or i.startswith("S3-")) for i in ids):
                errors.append(f"line {n}: non S2/S3 id")
            if a in match and not set(match[a]) <= set(ids):
                errors.append(f"line {n}: matched ids outside candidates for {a}")
            if len(errors) > 20:
                break
    if seen != s1:
        errors.append(f"S1 rows: {len(seen)} vs {len(s1)} test S1 (missing {len(s1 - seen)}, extra {len(seen - s1)})")
    print(f"candidate pairs {pairs:,}, S1 rows {len(seen):,}, S1 with matches {len(match):,}")
    for e in errors[:20]:
        print("ERROR", e)
    print("CANDIDATES OK" if not errors else "CANDIDATES FAILED")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
