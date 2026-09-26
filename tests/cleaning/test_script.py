"""C2 tests, written from Stage 1 prompt rev 5 §3 C2 before checking behaviour."""
import polars as pl

from amlc.cleaning import script


def test_accented_latin_classifies_as_latin_not_other():
    # M2 fix, tested directly per the prompt.
    assert script.classify_tokens("Président") == ["latin"]
    assert script.classify_tokens("President") == ["latin"]


def test_each_indic_block_classifies_correctly():
    assert script.classify_tokens("लक्ष्मी") == ["devanagari"]
    assert script.classify_tokens("ஸ்டோர்ஸ்") == ["tamil"]
    assert script.classify_tokens("ಪ್ರೈವೇಟ್") == ["kannada"]


def test_punctuation_only_token_is_not_indic():
    # Regression: all() over an empty "has an alpha char" generator is vacuously True.
    for tok in ["&", "-", "+", "***", "|", ">>", "...", "--", "123", ""]:
        assert not script._all_indic(tok), tok


def test_mixed_token_name_gets_one_flag_per_token():
    assert script.classify_tokens("gali no लिमिटेड") == ["latin", "latin", "devanagari"]


def _pairs(rows):
    return pl.DataFrame(rows, schema={"s1_gid": pl.UInt32, "s1_name": pl.String,
                                       "s23_name": pl.String, "country": pl.String})


def test_dictionary_needs_minimum_independent_support():
    # 4 distinct S1 entities map लि -> ltd: below MIN_SUPPORT=5, so it must be dropped.
    rows = [{"s1_gid": i, "s1_name": "acme ltd", "s23_name": "acme लि", "country": "India"}
            for i in range(4)]
    d = script.build_dictionary(_pairs(rows))
    assert d.height == 0


def test_dictionary_kept_at_minimum_support():
    rows = [{"s1_gid": i, "s1_name": "acme ltd", "s23_name": "acme लि", "country": "India"}
            for i in range(5)]
    d = script.build_dictionary(_pairs(rows))
    assert d.height == 1
    row = d.row(0, named=True)
    assert row["indic_token"] == "लि" and row["latin"] == ["ltd"] and row["total_support"] == 5


def test_repeated_pairs_from_the_same_s1_do_not_inflate_support():
    # support is independent S1 entities, not raw row count
    rows = [{"s1_gid": 1, "s1_name": "acme ltd", "s23_name": "acme लि", "country": "India"}] * 10
    d = script.build_dictionary(_pairs(rows))
    assert d.height == 0  # only 1 independent entity, below MIN_SUPPORT


def test_variant_conflict_keeps_both_above_the_10pct_floor():
    # 60 laxmi, 40 lakshmi (each S1 distinct): both >= 10% share, both kept, majority first.
    rows = ([{"s1_gid": i, "s1_name": "laxmi traders", "s23_name": "लक्ष्मी traders", "country": "India"}
             for i in range(60)]
            + [{"s1_gid": 1000 + i, "s1_name": "lakshmi traders", "s23_name": "लक्ष्मी traders", "country": "India"}
               for i in range(40)])
    d = script.build_dictionary(_pairs(rows))
    row = d.filter(pl.col("indic_token") == "लक्ष्मी").row(0, named=True)
    assert row["latin"] == ["laxmi", "lakshmi"] and not row["is_legal_form"]


def test_variant_below_floor_is_dropped():
    rows = ([{"s1_gid": i, "s1_name": "laxmi traders", "s23_name": "लक्ष्मी traders", "country": "India"}
             for i in range(95)]
            + [{"s1_gid": 1000 + i, "s1_name": "weird traders", "s23_name": "लक्ष्मी traders", "country": "India"}
               for i in range(5)])
    d = script.build_dictionary(_pairs(rows))
    row = d.filter(pl.col("indic_token") == "लक्ष्मी").row(0, named=True)
    assert row["latin"] == ["laxmi"]  # 5% share dropped


def test_legal_form_conflict_maps_to_canonical_not_variant_set():
    rows = ([{"s1_gid": i, "s1_name": "acme ltd", "s23_name": "acme लि", "country": "India"}
             for i in range(84)]
            + [{"s1_gid": 1000 + i, "s1_name": "acme limited", "s23_name": "acme लि", "country": "India"}
               for i in range(16)])
    d = script.build_dictionary(_pairs(rows))
    row = d.filter(pl.col("indic_token") == "लि").row(0, named=True)
    assert row["is_legal_form"] and row["latin"] == ["ltd"]  # canonical form, majority


def test_apply_dictionary_translates_and_flags_unmapped():
    dictionary = pl.DataFrame({"indic_token": ["लिमिटेड"], "latin": [["limited"]],
                               "is_legal_form": [False], "total_support": pl.Series([20], dtype=pl.UInt32)})
    names = pl.Series(["acme लिमिटेड", "acme अज्ञात"])
    out = script.apply_dictionary(names, dictionary)
    assert out["name_translit"].to_list() == ["acme limited", "acme अज्ञात"]
    assert out["name_indic_untranslated"].to_list() == [False, True]


def test_apply_dictionary_records_variant_set_on_the_row():
    dictionary = pl.DataFrame({"indic_token": ["लक्ष्मी"], "latin": [["laxmi", "lakshmi"]],
                               "is_legal_form": [False], "total_support": pl.Series([100], dtype=pl.UInt32)})
    out = script.apply_dictionary(pl.Series(["लक्ष्मी traders"]), dictionary)
    import json
    assert json.loads(out["name_token_variants"][0]) == [["लक्ष्मी", ["laxmi", "lakshmi"]]]


def test_coverage_counts_token_instances_not_distinct_tokens():
    dictionary = pl.DataFrame({"indic_token": ["लि"], "latin": [["ltd"]],
                               "is_legal_form": [True], "total_support": pl.Series([10], dtype=pl.UInt32)})
    names = pl.Series(["acme लि", "beta लि", "gamma अज्ञात"])
    cov = script.coverage(names, dictionary)
    assert cov == {"instances_total": 3, "instances_covered": 2, "coverage": 2 / 3}
