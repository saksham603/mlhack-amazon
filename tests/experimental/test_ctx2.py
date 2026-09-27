"""Tests for the reworked ctx_c_rank/ctx_c_gap computation. Root cause (confirmed against real
data, S1 59762): combo = n_tset + a_tset is a plain unweighted sum, so a candidate whose address is
genuinely EMPTY (a_tset -> ~0) looks identical to one whose address is present and a bad mismatch --
both drag combo down, even when the missing-address candidate has a perfect name match. Real
example: 4 true-match siblings with full address data get combo=200, rank 1, gap 0; a 5th true
match with the same perfect name but an empty address gets combo=100, rank 31 of 49, gap 100 (the
worst possible) -- buried behind several FALSE candidates with merely mediocre non-empty
addresses. `full_context_features` must reproduce the CURRENT (buggy) w3.py behavior exactly first
(regression baseline); `combo_mirror_name` is the fix under test."""
import polars as pl

from amlc.features.ctx2 import combo_mirror_name, compute_ctx, full_context_features


def test_full_context_features_reproduces_w3_formula():
    # Hand-computed expected values from w3.py's exact formula: combo = n_tset + a_tset,
    # rank("min", descending=True).over("s1_gid"), gap = max(combo).over("s1_gid") - combo.
    d = pl.DataFrame({
        "s1_gid": [1, 1, 1],
        "n_tset": [100.0, 80.0, 100.0],
        "a_tset": [100.0, 80.0, 0.0],
    })
    result = full_context_features(d)
    # combos: 200, 160, 100 -> ranks 1, 2, 3; gaps 0, 40, 100
    assert result["ctx_c_rank"].to_list() == [1, 2, 3]
    assert result["ctx_c_gap"].to_list() == [0.0, 40.0, 100.0]
    assert result["ctx_n_rank"].to_list() == [1, 3, 1]  # n_tset: 100,80,100 -> tied rank 1, then 3
    assert result["ctx_n_hi"].to_list() == [2, 2, 2]  # two rows with n_tset>=95
    assert result["ctx_n_cand"].to_list() == [3, 3, 3]


def test_full_context_features_matches_real_holdout_example():
    # The exact S1 59762 numbers from holdout_p1.parquet (n_tset, a_tset only -- s23_gid omitted,
    # order doesn't matter since ctx is computed per s1_gid group, not per row identity).
    d = pl.DataFrame({
        "s1_gid": [59762] * 7,
        "n_tset": [100.0, 100.0, 100.0, 84.61538696289062, 73.68421173095703, 100.0, 100.0],
        "a_tset": [100.0, 100.0, 100.0, 100.0, 66.66666412353516, 34.04255294799805, 0.0],
    })
    result = full_context_features(d)
    # last row is the known problem case: n_tset=100, a_tset=0 -> combo=100
    assert result["ctx_c_rank"].to_list()[-1] == 7  # last of 7 by rank (real data: 31 of 49 candidates)
    assert abs(result["ctx_c_gap"].to_list()[-1] - 100.0) < 1e-6


def test_combo_mirror_name_fixes_missing_address_true_match():
    # Same shape as the real S1 59762 case: three candidates with perfect name+address (combo=200),
    # one with a perfect name but genuinely empty address should be treated as combo=200 too (not
    # 100), matching its siblings -- because "no address recorded" isn't evidence against a match.
    d = pl.DataFrame({
        "s1_gid": [1, 1, 1, 1],
        "n_tset": [100.0, 100.0, 100.0, 100.0],
        "a_tset": [100.0, 100.0, 100.0, 0.0],
        "a_len_1": [40, 40, 40, 40],
        "a_len_2": [38, 39, 41, 0],  # last row: S23 address is empty
    })
    combo = combo_mirror_name(d)
    result = compute_ctx(d, combo, "ctx_c")
    assert result["ctx_c_rank"].to_list() == [1, 1, 1, 1]
    assert result["ctx_c_gap"].to_list() == [0.0, 0.0, 0.0, 0.0]


def test_combo_mirror_name_does_not_reward_weak_name_with_missing_address():
    d = pl.DataFrame({
        "s1_gid": [1, 1],
        "n_tset": [100.0, 30.0],
        "a_tset": [100.0, 20.0],
        "a_len_1": [40, 40],
        "a_len_2": [38, 0],  # second row: address missing, but name is also weak
    })
    combo = combo_mirror_name(d)
    result = compute_ctx(d, combo, "ctx_c")
    # weak name (30) + missing address (mirrored to 30) = combo 60, still ranks below the strong match
    assert result["ctx_c_rank"].to_list() == [1, 2]
    assert result["ctx_c_gap"].to_list()[1] > 0


def test_combo_mirror_name_only_applies_when_address_genuinely_absent():
    # Both addresses present but genuinely different (a_tset low, not because of absence) --
    # mirror_name must NOT rescue this; a real conflict should still be penalized normally.
    d = pl.DataFrame({
        "s1_gid": [1, 1],
        "n_tset": [100.0, 100.0],
        "a_tset": [100.0, 10.0],
        "a_len_1": [40, 40],
        "a_len_2": [38, 35],  # both present, just dissimilar
    })
    combo = combo_mirror_name(d)
    result = compute_ctx(d, combo, "ctx_c")
    assert result["ctx_c_rank"].to_list() == [1, 2]
    assert result["ctx_c_gap"].to_list()[1] == 90.0  # unchanged from the raw a_tset=10 case


def test_variant_c_keeps_original_alongside_adjusted():
    d = pl.DataFrame({
        "s1_gid": [1, 1, 1, 1],
        "n_tset": [100.0, 100.0, 100.0, 100.0],
        "a_tset": [100.0, 100.0, 100.0, 0.0],
        "a_len_1": [40, 40, 40, 40],
        "a_len_2": [38, 39, 41, 0],
    })
    base = full_context_features(d)
    adjusted = compute_ctx(d, combo_mirror_name(d), "ctx_c_adj")
    out = base.with_columns(adjusted.select("ctx_c_adj_rank", "ctx_c_adj_gap"))
    # original (buggy) ctx_c_rank still buries the last row; adjusted rescues it
    assert out["ctx_c_rank"].to_list()[-1] == 4
    assert out["ctx_c_adj_rank"].to_list()[-1] == 1
