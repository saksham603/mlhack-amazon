import polars as pl

from amlc.model.model_miss_analysis import CONTEXT_COLS, add_context_features


def test_context_features_are_computed_within_each_s1():
    df = pl.DataFrame({
        "s1_gid": [1, 1, 1, 2],
        "s23_gid": [10, 11, 12, 10],
        "blocking_score": [4.0, 2.0, 1.0, 3.0],
        "name_jaccard": [1.0, 0.5, 0.0, 0.0],
        "addr_token_jaccard": [0.2, 0.9, 0.1, 0.0],
    })
    out = add_context_features(df).sort(["s1_gid", "s23_gid"])
    assert set(CONTEXT_COLS) <= set(out.columns)
    s1 = out.filter(pl.col("s1_gid") == 1)
    assert s1["n_cand_s1"].to_list() == [3, 3, 3]
    assert s1["score_rel_max"].to_list() == [1.0, 0.5, 0.25]
    assert s1["name_jac_rank"].to_list() == [1, 2, 3]
    assert s1["n_exact_name_s1"].to_list() == [1, 1, 1]
    assert s1["addr_jac_rank"].to_list() == [2, 1, 3]
    # s23=10 appears under two S1s
    assert out.filter(pl.col("s23_gid") == 10)["n_s1_sharing_s23"].to_list() == [2, 2]
    # an S1 whose best name_jaccard is 0 must not produce NaN
    assert out.filter(pl.col("s1_gid") == 2)["name_jac_rel_max"].item() == 0.0
