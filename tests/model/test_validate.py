import polars as pl

from amlc.eval import scorer
from amlc.model import validate as V


def _fake_validation(monkeypatch):
    counts = pl.DataFrame({
        "s1_gid": [1, 2, 3, 4], "country": ["US", "US", "India", "India"],
        "bucket": ["1", "1", "1", "1"], "is_singleton": [False, False, False, False],
    }, schema_overrides={"s1_gid": pl.UInt32})
    truth = pl.DataFrame({"s1_gid": [1, 2, 3, 4], "s23_gid": [10, 20, 30, 40]},
                         schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    monkeypatch.setattr(scorer, "_load_validation_match_counts", lambda: counts)
    monkeypatch.setattr(scorer, "_load_validation_labels", lambda: truth)


def test_tune_threshold_picks_the_threshold_that_recovers_true_pairs(monkeypatch):
    _fake_validation(monkeypatch)
    # s1=1 -> s23=10 (true) at prob 0.9; s1=2 -> s23=20 (true) at prob 0.4
    scored = pl.DataFrame({"s1_gid": [1, 1, 2], "s23_gid": [10, 11, 20], "prob": [0.9, 0.2, 0.4]},
                          schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32, "prob": pl.Float64})
    a = pl.Series("s1_gid", [1, 2], dtype=pl.UInt32)
    out = V.tune_threshold(scored, a, thresholds=[0.1, 0.5, 0.95])
    # at 0.1: both predicted correctly + s1=1's false candidate also predicted (lower precision)
    # at 0.5: only s1=1's true candidate predicted -> s1=2 gets nothing (n_true=1, tp=0 -> f05=0)
    # at 0.95: nothing predicted for either -> both f05=0
    assert out["best_threshold"] == 0.1
    assert out["best_f05"] > 0


def test_report_on_b_scores_only_the_b_subset(monkeypatch):
    _fake_validation(monkeypatch)
    scored = pl.DataFrame({"s1_gid": [3, 4], "s23_gid": [30, 41], "prob": [0.9, 0.9]},
                          schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32, "prob": pl.Float64})
    b = pl.Series("s1_gid", [3, 4], dtype=pl.UInt32)
    out = V.report_on_b(scored, b, threshold=0.5)
    assert out["n_scored"] == 2
    assert "India" in out["by_country"]
