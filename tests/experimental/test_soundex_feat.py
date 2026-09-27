"""Tests for the pairwise soundex features: snd_eq (whole no-space-name soundex match) and snd_jacc
(Jaccard of per-token soundex codes, catching word-level phonetic overlap even with different word
order/count -- same spirit as n_tset's token_set_ratio, but sound-alike instead of exact-char)."""
import polars as pl

from amlc.features.soundex_feat import soundex_features


def test_snd_eq_one_for_soundex_equal_names():
    d = pl.DataFrame({"core_1": ["acme cabil traders"], "core_2": ["acme cavil traders"]})  # b/v swap
    out = soundex_features(d)
    assert out["snd_eq"][0] == 1


def test_snd_eq_zero_for_different_first_letters():
    d = pl.DataFrame({"core_1": ["acme traders"], "core_2": ["axe traders"]})
    out = soundex_features(d)
    assert out["snd_eq"][0] == 0


def test_snd_eq_zero_when_either_side_empty():
    d = pl.DataFrame({"core_1": [""], "core_2": ["acme traders"]})
    out = soundex_features(d)
    assert out["snd_eq"][0] == 0


def test_snd_jacc_full_overlap_for_identical_tokens():
    d = pl.DataFrame({"core_1": ["acme cabil traders"], "core_2": ["acme cabil traders"]})
    out = soundex_features(d)
    assert out["snd_jacc"][0] == 1.0


def test_snd_jacc_partial_overlap_reordered_tokens():
    d = pl.DataFrame({"core_1": ["acme trading company"], "core_2": ["company acme services"]})
    out = soundex_features(d)
    assert 0.0 < out["snd_jacc"][0] < 1.0  # "acme"/"company" overlap, "trading"/"services" don't


def test_snd_jacc_zero_for_empty_names_no_crash():
    d = pl.DataFrame({"core_1": [""], "core_2": [""]})
    out = soundex_features(d)
    assert out["snd_jacc"][0] == 0.0


def test_no_nulls_multiple_rows():
    d = pl.DataFrame({"core_1": ["acme traders", "", "beta co"], "core_2": ["acme trader", "gamma inc", ""]})
    out = soundex_features(d)
    assert out.select("snd_eq", "snd_jacc").null_count().sum_horizontal().sum() == 0
