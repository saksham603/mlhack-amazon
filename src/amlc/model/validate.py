"""D-S7: VALIDATION threshold tuning (half A) and reporting (half B), via amlc.eval.scorer.score()'s
s1_subset parameter -- raw VALIDATION labels never leave amlc.eval.scorer; this module only ever
sees aggregate metrics back. VALIDATION s1_gid identity (no labels) comes from access.load_split(),
which is unrestricted (not a labels loader).
"""
import polars as pl

from amlc.eval import scorer
from amlc.foundation import access
from amlc.model.predict import predictions_at_threshold

N_PER_COUNTRY = 20_000
SEED = 20260926
COUNTRIES = ("India", "US")


def sample_validation_halves(n_per_country: int = N_PER_COUNTRY, seed: int = SEED) -> tuple[pl.Series, pl.Series]:
    """Combined-country VALIDATION sample, split into two disjoint halves A/B by a fixed seed."""
    split = access.load_split().filter(pl.col("split") == "VALIDATION")
    parts = []
    for country in COUNTRIES:
        pool = split.filter(pl.col("country") == country)
        n = min(n_per_country, pool.height)
        parts.append(pool.sample(n=n, seed=seed, shuffle=True, with_replacement=False).select("s1_gid"))
    combined = pl.concat(parts, how="vertical").sample(fraction=1.0, seed=seed + 1, shuffle=True)
    half = combined.height // 2
    return combined.slice(0, half)["s1_gid"], combined.slice(half)["s1_gid"]  # A, B


def tune_threshold(scored: pl.DataFrame, s1_subset_a: pl.Series, thresholds: list[float]) -> dict:
    """scored: (s1_gid, s23_gid, prob) for half A's S1s. Sweeps thresholds via scorer.score(); G-M3
    (each S23 kept only on its highest-probability S1) is enforced before every scored call, since
    scorer.score() raises on a duplicated s23_gid."""
    curve = []
    for t in thresholds:
        pred = predictions_at_threshold(scored, t)
        r = scorer.score(pred, s1_subset=s1_subset_a)
        curve.append({"threshold": t, "f05": r["f05"]})
    best = max(curve, key=lambda r: r["f05"])
    return {"best_threshold": best["threshold"], "best_f05": best["f05"], "curve": curve}


def report_on_b(scored: pl.DataFrame, s1_subset_b: pl.Series, threshold: float) -> dict:
    """scored: (s1_gid, s23_gid, prob) for half B's S1s. Full scorer.score() report at `threshold`
    (G-M3 enforced first, same as tune_threshold)."""
    pred = predictions_at_threshold(scored, threshold)
    return scorer.score(pred, s1_subset=s1_subset_b)
