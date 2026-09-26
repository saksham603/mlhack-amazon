"""Independent-recount tests (CG-17): expected values from the definition in zip_recount.py's
docstring, not copied from addresses.py."""
from amlc.cleaning import zip_recount as R


def test_finds_zip5_in_a_component():
    assert R.recount_one("123 main st, austin tx 78701") is True


def test_finds_pin6_in_a_component():
    assert R.recount_one("mg road, bangalore 560001") is True


def test_no_match_on_plain_address():
    assert R.recount_one("main street, some city") is False


def test_house_number_range_is_not_a_zip_like_token():
    # a hyphenated house-number range is one token ("11432-11436"), never exactly 5 or 6 digits
    assert R.recount_one("11432-11436 kristina cir, montgomery, tx") is False


def test_hash_prefixed_house_number_is_not_a_zip_like_token():
    assert R.recount_one("#16002 sycamore maple lane, tx") is False


def test_trailing_punctuation_on_a_house_number_is_not_a_zip_like_token():
    assert R.recount_one("12873. pyxis pl, tucson, az") is False


def test_recount_table_matches_manual_count():
    out = R.recount_table(["78701 austin", "no zip here", "560001 bangalore", ""])
    assert out == {"rows": 4, "zip_like_rows": 2, "rate": 0.5}


def test_recount_table_empty_list():
    assert R.recount_table([]) == {"rows": 0, "zip_like_rows": 0, "rate": None}
