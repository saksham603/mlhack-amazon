"""W0 metric tests. Every expected value is derived from the README spec by hand (CG-17), not from code output.

Spec: per S1, F_0.5 = 1.25*P*R / (0.25*P + R); an S1 with no true match scores 1.0 for an empty
prediction and 0.0 for any prediction; an S1 with true matches scores 0.0 for an empty prediction;
the score is the mean over S1 (macro).
"""
import random

import polars as pl
import pytest

from amlc.eval import metric


def _links(pairs):
    return pl.DataFrame({"s1_gid": [a for a, _ in pairs], "s23_gid": [b for _, b in pairs]},
                        schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})


def _score(pred, truth, s1):
    return metric.per_s1_f05(_links(pred), _links(truth), pl.Series("s1_gid", s1, dtype=pl.UInt32))


def test_readme_example_is_five_sevenths():
    # README: predict [47, 193, 812], truth [47, 812] -> P = 2/3, R = 1 -> 1.25*(2/3) / (0.25*(2/3) + 1) = 5/7
    out = _score([(1, 47), (1, 193), (1, 812)], [(1, 47), (1, 812)], [1])
    assert out["f05"].to_list() == [pytest.approx(5 / 7, abs=1e-12)]
    assert round(out["f05"][0], 3) == 0.714


def test_singleton_empty_prediction_scores_one():
    assert _score([], [], [1])["f05"].to_list() == [1.0]


def test_singleton_any_prediction_scores_zero():
    assert _score([(1, 5)], [], [1])["f05"].to_list() == [0.0]


def test_true_matches_empty_prediction_scores_zero():
    assert _score([], [(1, 5)], [1])["f05"].to_list() == [0.0]


def test_all_wrong_prediction_scores_zero():
    assert _score([(1, 6)], [(1, 5)], [1])["f05"].to_list() == [0.0]


def test_perfect_prediction_scores_one():
    assert _score([(1, 5), (1, 6)], [(1, 5), (1, 6)], [1])["f05"].to_list() == [1.0]


def test_recall_half_precision_one():
    # P = 1, R = 1/2 -> 1.25*0.5 / (0.25 + 0.5) = 0.625/0.75 = 5/6
    assert _score([(1, 5)], [(1, 5), (1, 6)], [1])["f05"].to_list() == [pytest.approx(5 / 6, abs=1e-12)]


def test_macro_mean_and_unscored_s1_ignored():
    # S1 1 perfect (1.0), S1 2 singleton predicted empty (1.0), S1 3 missed (0.0); S1 9 not in the scored set
    pred = [(1, 5), (9, 7)]
    truth = [(1, 5), (3, 8), (9, 7)]
    out = _score(pred, truth, [1, 2, 3])
    assert out.sort("s1_gid")["f05"].to_list() == [1.0, 1.0, 0.0]
    assert metric.macro(out) == pytest.approx(2 / 3, abs=1e-12)


def test_duplicate_prediction_rows_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _score([(1, 5), (1, 5)], [(1, 5)], [1])


def test_vectorised_equals_scalar_reference():
    rng = random.Random(20260926)
    pred, truth, s1 = [], [], list(range(300))
    for s in s1:
        t = set(rng.sample(range(20), rng.randint(0, 4)))
        p = set(rng.sample(range(20), rng.randint(0, 4)))
        truth += [(s, x) for x in t]
        pred += [(s, x) for x in p]
    out = dict(zip(*_score(pred, truth, s1).select("s1_gid", "f05").to_dict(as_series=False).values()))
    for s in s1:
        t = {x for a, x in truth if a == s}
        p = {x for a, x in pred if a == s}
        assert out[s] == pytest.approx(metric.f05_scalar(p, t), abs=1e-12)


def test_scalar_reference_matches_spec_cases():
    assert metric.f05_scalar(set(), set()) == 1.0
    assert metric.f05_scalar({1}, set()) == 0.0
    assert metric.f05_scalar(set(), {1}) == 0.0
    assert metric.f05_scalar({47, 193, 812}, {47, 812}) == pytest.approx(5 / 7, abs=1e-12)
