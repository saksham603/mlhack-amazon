"""Sprint sec5 "validation before commit" checks, run on our own output files before the official
validator. Read-only: never touches ground truth or a gate.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.validate_output
"""
import sys

import polars as pl

from amlc.foundation import access

OUTPUT_DIR = access.ROOT / "output"


def read_submission_tsv(path, id_col: str) -> pl.DataFrame:
    """Parses a matching_results.tsv / candidate_pairs.tsv style file the same way the ground-truth
    loader does (f6_labels.build_label_tables): comma-split the second column, empty string -> [].
    Returns exploded (source1_entity_id, <id_col>) rows, one per link (S1s with no links absent)."""
    df = pl.read_csv(path, separator="\t", schema_overrides={"source1_entity_id": pl.String})
    cols = df.columns
    df = df.rename({cols[1]: id_col})
    return (df.with_columns(
        pl.when(pl.col(id_col) == "").then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col(id_col).str.split(",")).alias(id_col))
        .explode(id_col, empty_as_null=True).drop_nulls(id_col))


def check_matching_and_candidates(matching_path, candidate_path, test_s1_ids: set[str]) -> list[str]:
    problems = []

    matching_raw = pl.read_csv(matching_path, separator="\t", schema_overrides={"source1_entity_id": pl.String})
    if matching_raw.height != len(test_s1_ids):
        problems.append(f"matching_results.tsv has {matching_raw.height} rows, expected {len(test_s1_ids)}")
    if matching_raw["source1_entity_id"].is_duplicated().any():
        problems.append("matching_results.tsv has duplicate source1_entity_id rows")
    missing = test_s1_ids - set(matching_raw["source1_entity_id"].to_list())
    if missing:
        problems.append(f"matching_results.tsv missing {len(missing)} required S1 ids, e.g. {list(missing)[:5]}")

    matched = read_submission_tsv(matching_path, "matched_id")
    self_matches = matched.filter(pl.col("source1_entity_id") == pl.col("matched_id"))
    if self_matches.height:
        problems.append(f"{self_matches.height} self-match rows (S1 id in its own match list)")
    bad_prefix = matched.filter(~pl.col("matched_id").str.starts_with("S2-") & ~pl.col("matched_id").str.starts_with("S3-"))
    if bad_prefix.height:
        problems.append(f"{bad_prefix.height} matched ids without an S2-/S3- prefix")

    candidate_raw = pl.read_csv(candidate_path, separator="\t", schema_overrides={"source1_entity_id": pl.String})
    if candidate_raw.height != len(test_s1_ids):
        problems.append(f"candidate_pairs.tsv has {candidate_raw.height} rows, expected {len(test_s1_ids)}")
    if candidate_raw["source1_entity_id"].is_duplicated().any():
        problems.append("candidate_pairs.tsv has duplicate source1_entity_id rows")

    candidates = read_submission_tsv(candidate_path, "candidate_id")
    sub = (matched.join(candidates, left_on=["source1_entity_id", "matched_id"],
                        right_on=["source1_entity_id", "candidate_id"], how="anti"))
    if sub.height:
        problems.append(f"{sub.height} matched pairs are NOT present in candidate_pairs.tsv "
                        f"(predictions must be subset of candidates)")

    return problems


def main() -> int:
    test_s1 = pl.read_csv(
        access.ROOT.parent / "Desktop" / "dataset_student_resource" / "student_resource" / "dataset" / "test"
        / "test_source1.tsv", separator="\t", columns=["entity_id"])
    test_s1_ids = set(test_s1["entity_id"].to_list())

    problems = check_matching_and_candidates(OUTPUT_DIR / "matching_results.tsv",
                                              OUTPUT_DIR / "candidate_pairs.tsv", test_s1_ids)
    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS: all local checks clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
