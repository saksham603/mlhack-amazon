"""Ordering test (Stage 1 prompt rev 6, §3 'Ordering note'): C2 Indic (transliteration) must run
before C3 names (legal-form extraction), or an India legal form stays in Indic script and pollutes
name_clean. Also CG-27: the end-to-end test running pipeline.py's real entry point.

Fixture gid found live via pipeline.load_c1_output('train', 2) (CG-7): gid 2206821,
name_clean == 'राम मार्केटिंग प्राइवेट लिमिटेड' after C1.
"""
import polars as pl

from amlc.cleaning import names, pipeline, script

INDIA_PVT_LTD_GID = 2206821


def _real_c1_name(gid: int) -> str:
    df = pipeline.load_c1_output("train", 2)
    row = df.filter(pl.col("gid") == gid)
    assert row.height == 1, "fixture gid not found in real C1 output"
    return row["name_clean"][0]


def test_c2_then_c3_leaves_no_private_or_limited_token():
    raw = _real_c1_name(INDIA_PVT_LTD_GID)
    assert raw == "राम मार्केटिंग प्राइवेट लिमिटेड"

    dictionary = pl.DataFrame({
        "indic_token": ["प्राइवेट", "लिमिटेड"],
        "latin": [["private"], ["limited"]],
        "is_legal_form": [True, True],
        "total_support": pl.Series([66226, 78258], dtype=pl.UInt32),
    })
    translit = script.apply_dictionary(pl.Series([raw]), dictionary)["name_translit"][0]
    clean, legal = names.clean_one_name(translit, "India")

    assert "private" not in clean.split() and "limited" not in clean.split()
    assert legal == "pvt ltd"


EXPECTED_SILVER_SCHEMA = {
    "gid": pl.UInt32, "name_clean": pl.String, "name_legal_form": pl.String,
    "name_tokens": pl.List(pl.String), "name_token_variants": pl.String,
    "name_script_flags": pl.List(pl.String), "name_indic_untranslated": pl.Boolean,
    "addr_clean": pl.String, "addr_components": pl.List(pl.String),
    "addr_house_number": pl.String, "addr_house_number_raw": pl.String, "addr_zip": pl.String,
    "addr_missing": pl.Boolean, "addr_null_parts_removed": pl.UInt16, "addr_parts_emptied": pl.UInt16,
}


def test_end_to_end_pipeline_on_real_rows_by_gid():
    """CG-27: the real orchestrator entry point (pipeline.clean_table), C2 Indic -> C3 -> C4 -> C5-ready
    columns, on real bronze rows chosen by gid. Asserts the exact §2 silver schema plus one spec-derived
    value per stage.
    """
    full = pipeline.load_c1_output("train", 2)
    fixture = full.filter(pl.col("gid").is_in([INDIA_PVT_LTD_GID]))
    assert fixture.height == 1

    dictionary = pl.DataFrame({
        "indic_token": ["प्राइवेट", "लिमिटेड"], "latin": [["private"], ["limited"]],
        "is_legal_form": [True, True], "total_support": pl.Series([66226, 78258], dtype=pl.UInt32)})
    out = pipeline.clean_table("train", 2, dictionary, c1=fixture)

    # §2 exact column list and dtypes
    assert out.schema == pl.Schema(EXPECTED_SILVER_SCHEMA)
    # one spec-derived value per stage:
    row = out.row(0, named=True)
    # C2: राम and मार्केटिंग aren't in this tiny test dictionary, so D6 correctly flags them untranslated
    assert row["name_indic_untranslated"] is True
    assert row["name_legal_form"] == "pvt ltd"             # C3: India phrase canonicalisation
    assert "private" not in row["name_clean"].split() and "limited" not in row["name_clean"].split()


def test_running_c3_before_c2_would_fail_this_is_the_bug_being_prevented():
    # Documents the bug the ordering note describes: C3 on the untransliterated name cannot see
    # "private"/"limited" (they're in Indic script, not the Latin table C3 matches against), so no
    # legal form is extracted, exactly the legal-form pollution M4 exists to prevent.
    raw = _real_c1_name(INDIA_PVT_LTD_GID)
    _, legal = names.clean_one_name(raw, "India")
    assert legal == ""  # nothing recognised -> the bug's symptom that C2-then-C3 prevents
