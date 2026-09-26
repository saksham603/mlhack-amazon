"""C4 tests, expected values written from Stage 1 prompt rev 5 §3 C4 before running the code (CG-17)."""
import polars as pl

from amlc.cleaning import addresses


def test_us_street_word_canonicalized():
    assert addresses.canonicalize_street_words("123 main st", "US") == "123 main street"
    assert addresses.canonicalize_street_words("5 oak dr", "US") == "5 oak drive"


def test_france_st_means_saint_never_street():
    # Direct fix for M7: France's st/ste is Saint/Sainte, not Street.
    assert addresses.canonicalize_street_words("12 rue st michel", "France") == "12 rue saint michel"
    assert addresses.canonicalize_street_words("ste anne", "France") == "sainte anne"


def test_india_street_words():
    assert addresses.canonicalize_street_words("nr temple", "India") == "near temple"
    assert addresses.canonicalize_street_words("opp station", "India") == "opposite station"


def test_unlisted_country_falls_through_unchanged():
    assert addresses.canonicalize_street_words("5 st example", "Germany") == "5 st example"


def test_house_number_strips_hash_and_hno_and_leading_zeros():
    h, raw = addresses.extract_house_number(["#007 main street"])
    assert h == "7" and raw == "007"
    h, raw = addresses.extract_house_number(["h.no 56-2-6/1"])
    assert h == "56-2-6/1" and raw == "56-2-6/1"


def test_house_number_is_the_first_numeric_leading_component_not_necessarily_component_zero():
    # "first numeric-leading token" (spec): scan components in order, take the first that starts
    # with a digit. Real addresses don't always put the house number in component 0.
    h, raw = addresses.extract_house_number(["main street", "42 oak ave"])
    assert h == "42" and raw == "42"


def test_no_numeric_leading_component_gives_empty_not_none():
    h, raw = addresses.extract_house_number(["main street", "some city"])
    assert h == "" and raw == ""


def test_zip5_extracted_and_kept():
    assert addresses.extract_zip(["123 main st", "austin tx 78701"]) == "78701"


def test_pin6_extracted_and_kept():
    assert addresses.extract_zip(["mg road", "bangalore 560001"]) == "560001"


def test_indic_marks_survive_house_number_strip_word_pattern():
    # CG-28: _STRIP_WORD's \b anchors sit only around ASCII strip-words; an adjacent Indic mark
    # sequence in the same component must come through unchanged.
    marks = "मार्केटिंग"
    assert addresses._STRIP_WORD.sub(" ", marks) == marks
    h, raw = addresses.extract_house_number([f"h.no 5 {marks}"])
    assert h == "5" and raw == "5"


def test_no_zip_returns_empty_not_none():
    assert addresses.extract_zip(["main street", "some city"]) == ""


def test_clean_addresses_null_component_examples_still_extract_around_removed_null():
    # post-C1, a NULL component is already removed; house number/zip must extract around the gap.
    out = addresses.clean_addresses(
        pl.Series(["067 production ct, independence, ky 12345"]), pl.Series(["US"]))
    assert out["addr_house_number"][0] == "67"
    assert out["addr_zip"][0] == "12345"
