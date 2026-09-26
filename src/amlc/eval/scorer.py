"""VALIDATION scorer. Public API: score() only. Raw VALIDATION labels never leave this module.

The private loaders exist so the access gate can be exercised; code outside amlc.eval calling them
is flagged by leakscan (R2).
"""
import polars as pl

from amlc.eval import metric
from amlc.foundation import access

__all__ = ["score"]


def _load_validation_labels():
    return access.load_labels("VALIDATION")


def _load_validation_match_counts():
    return access.load_match_counts("VALIDATION")


def _group_means(per: pl.DataFrame, col: str) -> dict:
    g = per.group_by(col).agg(pl.col("f05").mean(), pl.len().alias("n")).sort(col)
    return {k: {"f05": f, "n": n} for k, f, n in g.iter_rows()}


def score(predictions: pl.DataFrame, s1_subset: pl.Series | None = None) -> dict:
    """Macro F_0.5 on VALIDATION S1s, plus slices. Returns metrics only, never labels.

    predictions: (s1_gid, s23_gid) links for ALL train S1s that competed (G-S2); only VALIDATION S1s
    are scored. Each s23_gid may be linked to at most one S1 (G-M3).
    s1_subset: optional -- restricts scoring to these VALIDATION s1_gids (e.g. one half of a fixed
    A/B split), so a caller can tune on one half and report on the other without ever reading raw
    VALIDATION labels itself (sprint D-S7). None (default, unchanged) scores all VALIDATION S1s.
    """
    pred = predictions.select(pl.col("s1_gid").cast(pl.UInt32, strict=True),
                              pl.col("s23_gid").cast(pl.UInt32, strict=True))
    if pred["s23_gid"].is_duplicated().any():
        raise ValueError("an S2/S3 record is linked to more than one S1 (G-M3)")
    counts = _load_validation_match_counts()
    if s1_subset is not None:
        subset = pl.DataFrame({"s1_gid": s1_subset.cast(pl.UInt32, strict=True)}).unique()
        counts = counts.join(subset, on="s1_gid", how="semi")
    truth = _load_validation_labels().select("s1_gid", "s23_gid")
    val_pred = pred.join(counts.select("s1_gid"), on="s1_gid", how="semi")
    per = metric.per_s1_f05(val_pred, truth, counts["s1_gid"]).join(
        counts.select("s1_gid", "country", "bucket", "is_singleton"), on="s1_gid", how="left", validate="1:1")
    tp, n_pred, n_true = (int(per[c].sum()) for c in ("tp", "n_pred", "n_true"))
    return {
        "f05": metric.macro(per),
        "n_scored": per.height,
        "singleton_share": float(per["is_singleton"].mean()),
        "pred_empty_share": float((per["n_pred"] == 0).mean()),
        "pair_precision": tp / n_pred if n_pred else None,
        "pair_recall": tp / n_true if n_true else None,
        "by_country": _group_means(per, "country"),
        "by_bucket": _group_means(per, "bucket"),
    }
