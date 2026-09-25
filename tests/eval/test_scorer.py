"""VALIDATION scorer: checks that need no label access outside amlc.eval.scorer."""
import polars as pl
import pytest

from amlc.eval import scorer

EMPTY = pl.DataFrame(schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})


def test_all_empty_scores_exactly_the_singleton_share():
    # Spec: an empty prediction scores 1.0 on a singleton and 0.0 otherwise, so the macro mean of the
    # all-empty baseline must equal the singleton share (computed from match counts, a separate file).
    r = scorer.score(EMPTY)
    assert r["f05"] == pytest.approx(r["singleton_share"], abs=1e-12)
    assert r["pred_empty_share"] == 1.0
    assert r["n_scored"] == 441_361  # VALIDATION size from load_split: 176,636 India + 264,725 US


def test_one_record_linked_to_two_s1_is_rejected():
    bad = pl.DataFrame({"s1_gid": [1, 2], "s23_gid": [7, 7]}, schema=EMPTY.schema)
    with pytest.raises(ValueError, match="G-M3"):
        scorer.score(bad)
