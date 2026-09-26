"""C3 tests, expected values written from Stage 1 prompt rev 5 §3 C3 before running the code (CG-17)."""
import polars as pl

from amlc.cleaning import names


def test_dotted_abbreviations_collapse_longest_pattern_first():
    assert names.collapse_abbreviations("acme p.l.l.c.") == "acme pllc"
    assert names.collapse_abbreviations("acme p.c.") == "acme pc"
    assert names.collapse_abbreviations("acme l.l.c.") == "acme llc"
    assert names.collapse_abbreviations("acme ltd.") == "acme ltd"
    assert names.collapse_abbreviations("acme pvt.") == "acme pvt"


def test_bracket_contents_kept_brackets_removed():
    out, _ = names.clean_one_name("bay guild (inc)", "US")
    assert out == "bay guild"  # "inc" is then extracted as the legal form


def test_hyphen_between_letters_becomes_space():
    assert names.strip_punctuation("general-printing") == "general printing"


def test_indic_marks_survive_digit_run_and_leading_word_stripping():
    # CG-28: every ASCII-\b pattern in this module is proven not to touch Indic marks it sits next to.
    marks = "मार्केटिंग"  # contains virama U+094D and vowel signs U+093E/U+093F
    assert names.strip_appendages(f"{marks} 1234567") == f"{marks}  "
    assert names.strip_leading(f"the {marks}") == marks


def test_indic_combining_marks_survive_punctuation_stripping():
    # Regression: a "not \\w" regex strips Mn/Mc (vowel signs, virama) as if they were punctuation,
    # corrupting any untranslated Indic token that D6 requires to survive intact.
    assert names.strip_punctuation("मार्केटिंग") == "मार्केटिंग"
    assert names.strip_punctuation("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_seven_digit_run_stripped():
    out, _ = names.clean_one_name("acme traders 1234567", "US")
    assert out == "acme traders"


def test_id_tag_stripped():
    out, _ = names.clean_one_name("acme traders (id: 4821)", "US")
    assert out == "acme traders"


def test_leading_honorific_and_the_stripped():
    out, _ = names.clean_one_name("the mr acme traders", "US")
    assert out == "acme traders"


def test_us_legal_form_extracted_and_removed():
    clean, legal = names.clean_one_name("acme llc", "US")
    assert clean == "acme" and legal == "llc"


def test_india_pvt_ltd_phrase_extracted_before_single_tokens():
    clean, legal = names.clean_one_name("acme pvt ltd", "India")
    assert clean == "acme" and legal == "pvt ltd"


def test_india_spelling_variants_canonicalise():
    assert names.clean_one_name("acme private limited", "India")[1] == "pvt ltd"
    assert names.clean_one_name("acme pvt", "India")[1] == "pvt"
    assert names.clean_one_name("acme limited", "India")[1] == "ltd"


def test_france_legal_form_extracted():
    clean, legal = names.clean_one_name("acme sarl", "France")
    assert clean == "acme" and legal == "sarl"


def test_unlisted_country_falls_through_unchanged_never_dropped():
    clean, legal = names.clean_one_name("acme gmbh", "Germany")
    assert clean == "acme gmbh" and legal == ""


def test_all_legal_form_tokens_extracted_not_just_the_first():
    # Regression (found by the silver_run.py dry run, train S1 gid 56200): "Phylys M. Hillis, DDS,
    # M.D., P.C." has TWO US legal-form tokens (dds, pc). A single-match version left "pc" behind,
    # which broke idempotence when the output was cleaned a second time.
    clean, legal = names.clean_one_name("phylys m. hillis, dds, m.d., p.c.", "US")
    assert "dds" not in clean.split() and "pc" not in clean.split()
    assert legal == "dds pc"
    # and now genuinely idempotent:
    clean2, legal2 = names.clean_one_name(clean, "US")
    assert clean2 == clean and legal2 == ""


def test_leading_legal_form_exposes_a_new_leading_the_and_it_still_gets_stripped():
    # Regression (silver_run.py dry run, real train S2 name "Inc The-Anchor Beacon Quetta"):
    # extracting the LEADING "inc" exposes "the" as the new first word; strip_leading and
    # legal-form extraction must loop together, or a one-shot strip_leading (which already ran
    # before "inc" was removed) never re-checks it, breaking idempotence.
    clean, legal = names.clean_one_name("inc the-anchor beacon quetta", "US")
    assert clean == "anchor beacon quetta" and legal == "inc"
    clean2, legal2 = names.clean_one_name(clean, "US")
    assert clean2 == clean and legal2 == ""


def test_clean_names_is_row_wise_country_aware():
    out = names.clean_names(pl.Series(["acme llc", "acme pvt ltd"]), pl.Series(["US", "India"]))
    assert out["name_legal_form"].to_list() == ["llc", "pvt ltd"]
