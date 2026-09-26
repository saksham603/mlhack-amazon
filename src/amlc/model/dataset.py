"""D-S6 training set: a random sample of ~N FIT S1s per country, with all their budget=150
candidates (via the fused pipeline) and labels from FIT ground truth ONLY (never VALIDATION or
LOCKBOX -- access.load_labels("FIT") has no restriction per amlc.foundation.access._authorize,
consistent with the leak-prevention gate).
"""
import polars as pl

from amlc.blocking import candidates as C
from amlc.foundation import access
from amlc.pipeline import fused

N_PER_COUNTRY = 50_000
SEED = 20260926


def sample_fit_s1_gids(country: str, n: int = N_PER_COUNTRY, seed: int = SEED) -> pl.DataFrame:
    split = access.load_split().filter((pl.col("split") == "FIT") & (pl.col("country") == country))
    n = min(n, split.height)
    return split.sample(n=n, seed=seed, shuffle=True, with_replacement=False).select("s1_gid")


def load_s1_rows(country: str, gids: pl.DataFrame) -> pl.DataFrame:
    silver = C.load_silver_with_country("train", 1).filter(pl.col("country") == country)
    s1 = silver.rename({"gid": "s1_gid"}).join(gids, on="s1_gid", how="semi")
    return s1.select("s1_gid", *fused.S1_COLS)


def load_s23(country: str) -> pl.DataFrame:
    parts = [C.load_silver_with_country("train", s).filter(pl.col("country") == country) for s in (2, 3)]
    s23 = pl.concat(parts, how="vertical").rename({"gid": "s23_gid"})
    return s23.select("s23_gid", *fused.S1_COLS)


def label_batches(candidates_dir, features_dir, country: str) -> pl.DataFrame:
    """Joins every batch's candidates+features with FIT ground truth (s1_gid, s23_gid) -> label
    (1 = true match, 0 = not, i.e. a blocking false positive). FIT-only; never touches VALIDATION.
    """
    cand = pl.concat([pl.read_parquet(f) for f in sorted(candidates_dir.glob("candidates_*.parquet"))],
                     how="vertical")
    feats = pl.concat([pl.read_parquet(f) for f in sorted(features_dir.glob("features_*.parquet"))],
                      how="vertical")
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").with_columns(pl.lit(1).alias("label"))
    out = (cand.join(feats, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
           .join(truth, on=["s1_gid", "s23_gid"], how="left")
           .with_columns(pl.col("label").fill_null(0), pl.lit(country).alias("country")))
    return out
