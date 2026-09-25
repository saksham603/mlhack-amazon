"""C1 tests. Every expected value is derived by hand from the Stage 1 prompt's rules (CG-17), never
copied from the code's output. Rows are loaded live from bronze by gid (CG-7)."""
import unicodedata

import polars as pl
import pytest

from amlc.cleaning import text
from tests.cleaning.fixtures import (CONTROL, EMPTIED, GARBLED, INDIC_NAMES, QUOTED_NULL, S1_NULL_GIDS,
                                     VARIANT_NULL, row)


def c1_address(raw: str) -> dict:
    return text.clean_addresses(pl.Series([raw])).row(0, named=True)


def addr(clean: str, removed: int = 0, emptied: int = 0, missing: bool = False) -> dict:
    return {"addr_clean": clean, "addr_missing": missing, "addr_null_parts_removed": removed, "addr_parts_emptied": emptied}


@pytest.mark.parametrize("key", list(QUOTED_NULL))
def test_quoted_null_examples_exact(key):
    expected, removed = QUOTED_NULL[key]
    assert c1_address(row(*key)["business_address"]) == addr(expected, removed)


@pytest.mark.parametrize("key", list(VARIANT_NULL))
def test_case_variant_null_examples_exact(key):
    expected, removed = VARIANT_NULL[key]
    assert c1_address(row(*key)["business_address"]) == addr(expected, removed)


@pytest.mark.parametrize("ds", ["train", "test"])
def test_every_s1_null_row_loses_its_null_parts_and_nothing_else(ds):
    for gid in S1_NULL_GIDS[ds]:
        raw = row(ds, 1, gid)["business_address"]
        out = c1_address(raw)
        real_parts = [p for p in raw.split(",") if p.strip() and p.strip().lower() not in text.MISSING_WORDS]
        assert out["addr_null_parts_removed"] >= 1
        assert out["addr_null_parts_removed"] == len(raw.split(",")) - len(real_parts)
        assert out["addr_clean"] == ", ".join(text.clean_text(p.strip()) for p in real_parts)


@pytest.mark.parametrize("key", list(GARBLED))
def test_garbled_text_and_ligature_examples_exact(key):
    ds, src, gid, col = key
    raw = row(ds, src, gid)[col]
    got = c1_address(raw)["addr_clean"] if col == "business_address" else text.clean_text(raw)
    assert got == GARBLED[key]


@pytest.mark.parametrize("key", list(CONTROL))
def test_ascii_control_examples_exact(key):
    ds, src, gid, col = key
    raw = row(ds, src, gid)[col]
    got = c1_address(raw)["addr_clean"] if col == "business_address" else text.clean_text(raw)
    assert got == CONTROL[key]


@pytest.mark.parametrize("key", list(EMPTIED))
def test_parts_emptied_by_cleaning_are_dropped(key):
    expected, emptied = EMPTIED[key]
    assert c1_address(row(*key)["business_address"]) == addr(expected, 0, emptied)


@pytest.mark.parametrize("key", INDIC_NAMES)
def test_indic_names_only_lose_cf_characters(key):
    raw = row(*key)["business_name"]
    expected = " ".join(unicodedata.normalize("NFC", "".join(c for c in raw if unicodedata.category(c) != "Cf")).split())
    assert text.clean_text(raw) == expected


def test_constructed_na_traders_is_not_a_null_part():
    # constructed edge case (prompt C1 tests): 'NA' inside a larger part must never be removed
    assert c1_address("NA Traders, 5 Main St") == addr("na traders, 5 main st")


def test_constructed_all_parts_missing_sets_addr_missing():
    assert c1_address(" NULL ,, n/a") == addr("", removed=3, missing=True)
    assert c1_address("") == addr("", removed=1, missing=True)


def test_constructed_all_parts_emptied_sets_addr_missing():
    # D-C1f: two parts that are only a SUB control survive step 1 and are emptied by cleaning
    assert c1_address("\x1a, \x1a") == addr("", emptied=2, missing=True)


def test_constructed_ascii_controls_become_spaces():
    # D-C1e: SUB between two words becomes a space and never glues them; DEL likewise
    assert text.clean_text("Medchal\x1aMalkajgiri") == "medchal malkajgiri"
    assert text.clean_text("a\x7fb\x00c") == "a b c"
    assert text.clean_text("Président\x1aSARL") == "president sarl"  # non-ASCII path too


def test_constructed_accents_and_stacked_marks():
    assert text.clean_text("Président") == text.clean_text("President") == "president"
    assert text.clean_text("ẹ́") == "e"  # two stacked marks, both removed
    assert text.clean_text("Straße") == "strasse"


def test_constructed_space_before_comma_removed():
    # rev 4 step 6: "x - , y" -> "x -, y"
    assert text.clean_text("Jaipur - , Jaipur") == "jaipur -, jaipur"


def test_constructed_indic_marks_are_never_removed():
    # Devanagari nukta letter U+0958 is a composition exclusion: NFKC/NFC keep it decomposed, same glyph
    assert text.clean_text("क़") == "क़"
    s = "लक्ष्मी"  # vowel signs and virama sit outside U+0300-036F
    assert text.clean_text(s) == s


def test_ascii_fast_path_equals_full_path():
    samples = ["  Hello   WORLD ", "A.B.C. Corp\tInc", "", "x\x1fy", "x\x1ay\x7fz", "NA Traders, 5 Main St",
               "#302-B/4", "a , b"]
    for s in samples:
        assert text.clean_text(s) == text.clean_text_full(s)


def test_idempotent_on_all_fixture_outputs():
    for key in list(QUOTED_NULL) + list(VARIANT_NULL) + list(EMPTIED):
        once = c1_address(row(*key)["business_address"])["addr_clean"]
        assert c1_address(once) == addr(once)
    for expected in list(GARBLED.values()) + list(CONTROL.values()):
        assert text.clean_text(expected) == expected


def test_already_clean_text_only_changes_case():
    for s in ["mulberry dr, buckeye, arizona", "oil trading private limited"]:
        assert text.clean_text(s) == s
