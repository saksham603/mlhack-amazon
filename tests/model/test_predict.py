import polars as pl

from amlc.model.predict import enforce_g_m3, predictions_at_threshold


def test_enforce_g_m3_keeps_highest_probability_s1_per_s23():
    scored = pl.DataFrame({
        "s1_gid": [1, 2, 3], "s23_gid": [100, 100, 200], "prob": [0.6, 0.9, 0.5],
    })
    out = enforce_g_m3(scored).sort("s23_gid")
    assert out["s23_gid"].to_list() == [100, 200]
    assert out.filter(pl.col("s23_gid") == 100)["s1_gid"].item() == 2  # 0.9 > 0.6


def test_enforce_g_m3_breaks_ties_by_lowest_s1_gid():
    scored = pl.DataFrame({"s1_gid": [5, 3, 4], "s23_gid": [100, 100, 100], "prob": [0.7, 0.7, 0.7]})
    out = enforce_g_m3(scored)
    assert out.height == 1
    assert out["s1_gid"].item() == 3


def test_predictions_at_threshold_filters_then_deduplicates():
    scored = pl.DataFrame({
        "s1_gid": [1, 2], "s23_gid": [100, 100], "prob": [0.3, 0.9],
    })
    out = predictions_at_threshold(scored, 0.5)
    assert out.height == 1
    assert out["s1_gid"].item() == 2
    assert set(out.columns) == {"s1_gid", "s23_gid"}
