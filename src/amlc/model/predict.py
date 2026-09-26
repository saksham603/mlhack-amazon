"""D-S9 prediction rule + G-M3 enforcement: predict every candidate with p >= threshold, then keep
each S2/S3 record on only its highest-probability S1 (an S23 may match at most one S1 in the
ground-truth partition; amlc.eval.scorer.score() raises ValueError on a violation)."""
import polars as pl


def enforce_g_m3(scored: pl.DataFrame) -> pl.DataFrame:
    """scored: (s1_gid, s23_gid, prob, ...). Keeps one row per s23_gid: the highest prob, ties
    broken by s1_gid ascending for determinism."""
    return (scored.sort(["s23_gid", "prob", "s1_gid"], descending=[False, True, False])
            .group_by("s23_gid", maintain_order=True).head(1))


def predictions_at_threshold(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """scored: (s1_gid, s23_gid, prob). Returns (s1_gid, s23_gid) predictions: threshold filter,
    then G-M3 (each S23 kept only on its best S1)."""
    pred = scored.filter(pl.col("prob") >= threshold)
    return enforce_g_m3(pred).select("s1_gid", "s23_gid")
