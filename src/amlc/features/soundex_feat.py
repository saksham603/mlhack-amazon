"""Pairwise soundex features over `core` name text: snd_eq (whole no-space-name soundex match) and
snd_jacc (Jaccard of per-token soundex codes -- word-level phonetic overlap regardless of word order
or count, same spirit as n_tset's token_set_ratio but sound-alike rather than exact-character).
Label-free."""
import polars as pl

from amlc.features.soundex import soundex


def _snd_eq(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return int(soundex(a.replace(" ", "")) == soundex(b.replace(" ", "")))


def _snd_jacc(a: str, b: str) -> float:
    sa = {soundex(t) for t in a.split() if t}
    sb = {soundex(t) for t in b.split() if t}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def soundex_features(d: pl.DataFrame, col1: str = "core_1", col2: str = "core_2") -> pl.DataFrame:
    a, b = d[col1].to_list(), d[col2].to_list()
    return d.with_columns(
        pl.Series("snd_eq", [_snd_eq(x, y) for x, y in zip(a, b)], dtype=pl.UInt8),
        pl.Series("snd_jacc", [_snd_jacc(x, y) for x, y in zip(a, b)], dtype=pl.Float32),
    )
