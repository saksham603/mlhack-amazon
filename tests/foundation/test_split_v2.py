"""Split v2 properties, measured from the written files (tests may read files directly; leakscan covers src/)."""
import json
from pathlib import Path

import polars as pl

from amlc.foundation import split_v2

DATA = Path(r"C:\Users\suremdra singh\amlc2026\data")
V2 = pl.read_parquet(DATA / "splits" / "v2" / "s1_split.parquet")
V1 = pl.read_parquet(DATA / "splits" / "v1" / "s1_split.parquet")
MANIFEST = json.loads((DATA / "MANIFEST_v2.json").read_text(encoding="utf-8"))
PARAMS = MANIFEST["stress_test"]["parameters"]


def test_assignment_identical_to_v1():
    assert split_v2.assignment_diffs(V2, V1) == 0


def test_stress_hidden_only_in_fit():
    assert V2.filter(pl.col("stress_hidden") & (pl.col("split") != "FIT")).height == 0


def test_stress_check_passes_on_real_split():
    errs, _ = split_v2.check_stress(V2, PARAMS)
    assert errs == []


def test_stress_check_fails_when_validation_flagged():
    vid = V2.filter(pl.col("split") == "VALIDATION")["s1_id"][0]
    bad = V2.with_columns(pl.when(pl.col("s1_id") == vid).then(True).otherwise(pl.col("stress_hidden")).alias("stress_hidden"))
    errs, _ = split_v2.check_stress(bad, PARAMS)
    assert errs


def test_stress_check_fails_when_count_wrong():
    hid = V2.filter(pl.col("stress_hidden"))["s1_id"][0]
    bad = V2.with_columns(pl.when(pl.col("s1_id") == hid).then(False).otherwise(pl.col("stress_hidden")).alias("stress_hidden"))
    errs, _ = split_v2.check_stress(bad, PARAMS)
    assert errs


def test_strata_within_strict_tolerance():
    counts = pl.read_parquet(DATA / "labels" / "v2" / "s1_match_counts.parquet", columns=["s1_id", "bucket"])
    errs, max_dev = split_v2.check_strata(V2.join(counts, on="s1_id"))
    assert errs == [] and max_dev <= split_v2.STRATUM_TOL_PP


def test_assignment_diff_detects_a_change():
    first = V2["s1_id"][0]
    bad = V2.with_columns(pl.when(pl.col("s1_id") == first).then(pl.lit("LOCKBOX" if V2["split"][0] != "LOCKBOX" else "FIT")).otherwise(pl.col("split")).alias("split"))
    assert split_v2.assignment_diffs(bad, V1) == 1


def test_manifest_records_all_gates_passed():
    assert MANIFEST["all_gates_passed"] is True
    assert all(g["passed"] for g in MANIFEST["gates"])
    assert MANIFEST["stress_test"]["drawn_from"] == "FIT only"
