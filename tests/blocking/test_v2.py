"""Blocking v2 tests. Expected values are written from the fix definitions (module docstring), CG-17."""
import polars as pl
import pytest

from amlc.blocking import candidates, v2


def _rows(names, gid0=1):
    return pl.DataFrame({
        "gid": pl.Series(range(gid0, gid0 + len(names)), dtype=pl.UInt32),
        "name_tokens": [n.split(" ") if n else [] for n in names],
        "addr_components": [[] for _ in names],
        "addr_house_number": ["" for _ in names],
        "addr_zip": ["" for _ in names]})


def _keyset(df):
    return sorted(df["key"].to_list())


# ---- Fix 1: word pairs --------------------------------------------------------------------------
def test_pair_keys_ignore_word_order():
    a = v2.pair_keys(_rows(["shiva consultants"]), "gid")
    b = v2.pair_keys(_rows(["consultants shiva"]), "gid")
    assert a.height == 1 and _keyset(a) == _keyset(b)


def test_pair_keys_count_is_n_choose_2_of_distinct_tokens():
    assert v2.pair_keys(_rows(["a b c"]), "gid").height == 3
    assert v2.pair_keys(_rows(["global global"]), "gid").height == 0   # one distinct token
    assert v2.pair_keys(_rows(["solo"]), "gid").height == 0


def test_pair_keys_capped_at_first_8_distinct_tokens():
    assert v2.pair_keys(_rows(["t1 t2 t3 t4 t5 t6 t7 t8 t9 t10"]), "gid").height == 28


def test_pair_key_differs_from_either_single_token_key():
    df = _rows(["shiva consultants"])
    assert set(_keyset(v2.pair_keys(df, "gid"))).isdisjoint(_keyset(v2.token_keys(df, "gid")))


# ---- Fix 3: no-space / domain -------------------------------------------------------------------
@pytest.mark.parametrize("tokens, expected", [
    (["servicesraajinfra", "com"], "servicesraajinfra"),     # real no_key example from the diagnosis
    (["services", "raaj", "infra"], "servicesraajinfra"),    # its S1 side
    (["ernakulambeverages", "com"], "ernakulambeverages"),
    (["www", "abccompany", "com"], "abccompany"),
    (["abc", "company"], "abccompany"),                       # "abc-company": C3 already made it "abc company"
    (["abccompany", "co", "in"], "abccompany"),
    (["made", "in"], ""),                                     # "made" is under 6 characters
    (["com"], ""),                                            # a lone suffix is never stripped to nothing
    (["www"], ""),
    ([], ""),
])
def test_nospace_value(tokens, expected):
    assert v2.nospace_value(tokens) == expected


def test_nospace_vectorised_equals_reference():
    cases = [["servicesraajinfra", "com"], ["services", "raaj", "infra"], ["www", "abccompany", "com"],
             ["http", "www", "acme", "traders", "com", "in"], ["made", "in"], ["com"], [], ["a", "", "bcdefg"],
             ["tirupaticonstructions", "com"], ["bangalore", "it"], ["plain", "business", "name"]]
    df = pl.DataFrame({"name_tokens": cases}, schema={"name_tokens": pl.List(pl.String)})
    got = df.select(v2.nospace_expr().alias("k"))["k"].to_list()
    assert got == [v2.nospace_value(c) for c in cases]


def test_nospace_matches_across_the_domain_form():
    s1 = v2.nospace_keys(_rows(["services raaj infra"]), "gid")
    s23 = v2.nospace_keys(_rows(["servicesraajinfra com"]), "gid")
    assert _keyset(s1) == _keyset(s23) and s1.height == 1


# ---- Fix 2: separate ranking ----------------------------------------------------------------------
def _rare(keys, idf):
    return pl.DataFrame({"key": keys, "idf": idf, "kind": ["tok"] * len(keys)},
                        schema={"key": pl.UInt64, "idf": pl.Float64, "kind": v2.KIND})


def test_score_sums_idf_of_shared_keys_of_the_requested_kinds_only():
    s1 = pl.DataFrame({"s1_gid": [1, 1], "key": [10, 20], "kind": ["tok", "house"]},
                      schema={"s1_gid": pl.UInt32, "key": pl.UInt64, "kind": v2.KIND})
    s23 = pl.DataFrame({"s23_gid": [7, 7], "key": [10, 20], "kind": ["tok", "house"]},
                       schema={"s23_gid": pl.UInt32, "key": pl.UInt64, "kind": v2.KIND})
    rare = pl.DataFrame({"key": [10, 20], "idf": [2.0, 5.0], "kind": ["tok", "house"]},
                        schema={"key": pl.UInt64, "idf": pl.Float64, "kind": v2.KIND})
    name = v2.score(s1, s23, rare, ("tok",), 10**6)
    addr = v2.score(s1, s23, rare, ("house",), 10**6)
    assert name.to_dicts() == [{"s1_gid": 1, "s23_gid": 7, "score": 2.0}]
    assert addr.to_dicts() == [{"s1_gid": 1, "s23_gid": 7, "score": 5.0}]


def test_score_refuses_an_over_budget_join():
    s1 = pl.DataFrame({"s1_gid": [1] * 3, "key": [10] * 3, "kind": ["tok"] * 3},
                      schema={"s1_gid": pl.UInt32, "key": pl.UInt64, "kind": v2.KIND})
    s23 = pl.DataFrame({"s23_gid": [5, 6, 7], "key": [10] * 3, "kind": ["tok"] * 3},
                       schema={"s23_gid": pl.UInt32, "key": pl.UInt64, "kind": v2.KIND})
    with pytest.raises(MemoryError, match="chunk first"):
        v2.score(s1, s23, _rare([10], [1.0]), ("tok",), max_rows=8)   # 3 x 3 = 9 > 8


def test_union_keeps_strong_candidates_from_either_ranking_and_dedupes():
    name = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 11]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    addr = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [11, 12]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    assert sorted(v2.union_candidates(name, addr)["s23_gid"].to_list()) == [10, 11, 12]


def test_crowded_address_cannot_push_out_an_identical_name():
    # S1 1 has an identical-name candidate 99 (name score 3) and 5 same-address neighbours whose
    # combined score beats it. Old single ranking at k=5 loses 99; separate ranking (kn=1, ka=4) keeps it.
    scored_old = pl.DataFrame({"s1_gid": [1] * 6, "s23_gid": [99, 1, 2, 3, 4, 5],
                               "score": [3.0, 4.0, 4.0, 4.0, 4.0, 4.0]},
                              schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32, "score": pl.Float64})
    assert 99 not in candidates.candidates_topk(scored_old, 5)["s23_gid"].to_list()
    name = scored_old.filter(pl.col("s23_gid") == 99)
    addr = scored_old.filter(pl.col("s23_gid") != 99)
    got = v2.union_candidates(v2.topk(name, 1), v2.topk(addr, 4))
    assert 99 in got["s23_gid"].to_list() and got.height == 5


# ---- determinism ------------------------------------------------------------------------------------
def test_topk_breaks_ties_by_s23_gid():
    scored = pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [30, 10, 20], "score": [1.0, 1.0, 1.0]},
                          schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32, "score": pl.Float64})
    assert v2.topk(scored, 2)["s23_gid"].to_list() == [10, 20]
    assert candidates.candidates_topk(scored, 2)["s23_gid"].to_list() == [10, 20]


def test_estimate_join_rows_is_sum_of_products():
    s1 = pl.DataFrame({"key": [1, 1, 2]}, schema={"key": pl.UInt64})
    s23 = pl.DataFrame({"key": [1, 1, 1, 2]}, schema={"key": pl.UInt64})
    assert v2.estimate_join_rows(s1, s23, pl.DataFrame({"key": [1, 2]}, schema={"key": pl.UInt64})) == 7
