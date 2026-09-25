import polars as pl

from amlc.cleaning import recount, text


def test_recount_distinguishes_stage0_and_case_variants():
    # constructed cases
    r = recount.classify_address("A, NULL, B")
    assert (r["stage0"], r["variant"], r["c1"], r["parts_removed"]) == (True, False, True, 1)
    r = recount.classify_address("A, Na, B")
    assert (r["stage0"], r["variant"], r["c1"], r["parts_removed"]) == (False, True, True, 1)
    r = recount.classify_address("Na, null, B")
    assert (r["stage0"], r["variant"], r["c1"], r["parts_removed"]) == (True, True, True, 2)
    r = recount.classify_address("NA Traders, 5 Main St")
    assert (r["stage0"], r["variant"], r["c1"]) == (False, False, False)
    r = recount.classify_address("")
    assert (r["raw_empty"], r["all_missing"], r["parts_removed"], r["parts_emptied"]) == (True, True, 1, 0)


def test_recount_parts_emptied():
    # constructed: parts made only of characters C1 removes (D-C1f)
    r = recount.classify_address("A, \x1a, B")
    assert (r["parts_emptied"], r["all_missing"]) == (1, False)
    r = recount.classify_address("\x1a, ï¿½, ‌")
    assert (r["parts_emptied"], r["all_missing"]) == (3, True)
    r = recount.classify_address("A, Â\x80\x8b, B")  # garbled UTF-8 for ZWSP (Cf) -> removed
    assert r["parts_emptied"] == 1
    r = recount.classify_address("A, Â\x80\x93, B")  # garbled en dash -> '–' survives
    assert r["parts_emptied"] == 0


def test_recount_whitespace_matches_polars_strip():
    # recount writes out Unicode White_Space itself; it must agree with Polars' strip_chars()
    ws = recount.WHITE_SPACE
    s = pl.Series([ws + "x" + ws, "​x", "\x1cx\x1c", "\x1ax\x1a"])
    assert s.str.strip_chars().to_list() == [v.strip(ws) for v in s.to_list()]


def test_recount_agrees_with_text_on_constructed_mix():
    addrs = ["A, NULL, B", "A, Na, B", "Na, null, B", "", " , ", "NA Traders, 5 Main St", "x,N/a,<Null>",
             "A, \x1a, B", "\x1a, \x1a", "A, Â\x80\x8b, B", "Road\x1a, B"]
    out = text.clean_addresses(pl.Series(addrs))
    rc = recount.recount_addresses(addrs, ["X"] * len(addrs))["total"]
    assert int((out["addr_null_parts_removed"] > 0).sum()) == rc["c1"]
    assert int(out["addr_null_parts_removed"].sum()) == rc["parts_removed"]
    assert int(out["addr_missing"].sum()) == rc["all_missing"]
    assert int(out["addr_parts_emptied"].sum()) == rc["parts_emptied"]
    assert int((out["addr_parts_emptied"] > 0).sum()) == rc["rows_with_emptied"]
