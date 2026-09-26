import polars as pl

from amlc.blocking.oracle_ceiling import oracle_score


def test_oracle_scores_empty_truth_as_perfect_and_full_recall_as_perfect():
    truth = pl.DataFrame({"s1_gid": [1, 2], "s23_gid": [10, 20]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    final = pl.DataFrame({"s1_gid": [1, 2, 2], "s23_gid": [10, 20, 21]},
                          schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    s1_gids = pl.Series("s1_gid", [1, 2, 3], dtype=pl.UInt32)  # s1=3 has no true link at all
    out = oracle_score(final, truth, s1_gids)
    assert out["n_recallable"] == 2
    assert out["recall_ceiling"] == 1.0
    assert out["n_scored"] == 3
    assert out["f05_macro"] == 1.0  # every S1 fully recallable (s1=3 correctly predicts nothing)


def test_oracle_scores_partial_recall_below_one():
    truth = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 11]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    final = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    s1_gids = pl.Series("s1_gid", [1], dtype=pl.UInt32)
    out = oracle_score(final, truth, s1_gids)
    assert out["n_recallable"] == 1
    assert out["recall_ceiling"] == 0.5
    assert 0.0 < out["f05_macro"] < 1.0
