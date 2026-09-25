"""Competition metric: macro F_0.5 over S1 entities (README). Pure functions; reads no data.

Per S1: an S1 with no true match scores 1.0 for an empty prediction and 0.0 for any prediction; an S1
with true matches scores 0.0 for an empty prediction; otherwise F_0.5 = 1.25*P*R / (0.25*P + R).
"""
import polars as pl

BETA2 = 0.25


def f05_scalar(pred: set, true: set) -> float:
    """Reference implementation for one S1 (used by tests to check the vectorised version)."""
    if not true:
        return 0.0 if pred else 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def _count(links: pl.DataFrame, name: str) -> pl.DataFrame:
    return links.group_by("s1_gid").len(name=name)


def per_s1_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.Series) -> pl.DataFrame:
    """One row per S1 in s1: n_pred, n_true, tp, f05. pred/truth are (s1_gid, s23_gid) link tables."""
    for name, df in (("prediction", pred), ("truth", truth)):
        if df.select("s1_gid", "s23_gid").is_duplicated().any():
            raise ValueError(f"{name} links contain duplicate (s1_gid, s23_gid) rows")
    base = pl.DataFrame({"s1_gid": s1.cast(pl.UInt32, strict=True)})
    if base["s1_gid"].is_duplicated().any():
        raise ValueError("scored S1 set contains duplicates")
    tp = pred.join(truth.select("s1_gid", "s23_gid"), on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
    out = (base
           .join(_count(pred, "n_pred"), on="s1_gid", how="left", validate="1:1")
           .join(_count(truth, "n_true"), on="s1_gid", how="left", validate="1:1")
           .join(_count(tp, "tp"), on="s1_gid", how="left", validate="1:1")
           .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0)))
    if out.height != base.height:
        raise AssertionError("per_s1_f05 changed the number of scored S1s")
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (1 + BETA2) * p * r / (BETA2 * p + r)
    return out.with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.when(pl.col("n_pred") == 0).then(1.0).otherwise(0.0))
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(f).alias("f05"))


def macro(per_s1: pl.DataFrame) -> float:
    return float(per_s1["f05"].mean())
