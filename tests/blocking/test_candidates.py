"""T7 blocking tests, expected values written from the master prompt's T7 spec before running (CG-17)."""
import polars as pl
import pytest

from amlc.blocking import candidates


def _silverish(rows):
    """rows: list of dicts with gid, name_tokens, addr_components, addr_house_number, addr_zip."""
    return pl.DataFrame(rows, schema={
        "gid": pl.UInt32, "name_tokens": pl.List(pl.String), "addr_components": pl.List(pl.String),
        "addr_house_number": pl.String, "addr_zip": pl.String})


def test_name_tokens_explodes_and_drops_empty():
    df = _silverish([{"gid": 1, "name_tokens": ["acme", "traders"], "addr_components": [],
                      "addr_house_number": "", "addr_zip": ""},
                     {"gid": 2, "name_tokens": [""], "addr_components": [], "addr_house_number": "", "addr_zip": ""}])
    out = candidates._name_tokens(df, "gid")
    assert out.sort("gid", "key")["key"].to_list() == ["acme", "traders"]
    assert out["gid"].to_list() == [1, 1]


def test_zip_keys_drops_empty():
    df = _silverish([{"gid": 1, "name_tokens": [], "addr_components": [], "addr_house_number": "", "addr_zip": "78701"},
                     {"gid": 2, "name_tokens": [], "addr_components": [], "addr_house_number": "", "addr_zip": ""}])
    out = candidates.zip_keys(df, "gid")
    assert out.to_dicts() == [{"gid": 1, "key": "78701"}]


def test_house_street_keys_excludes_the_house_number_itself_as_a_token():
    df = _silverish([{"gid": 1, "name_tokens": [], "addr_components": ["42 oak ave", "austin tx"],
                      "addr_house_number": "42", "addr_zip": ""}])
    out = candidates.house_street_keys(df, "gid")
    keys = set(out["key"].to_list())
    assert "42|42" not in keys  # house number never paired with itself
    assert "42|oak" in keys and "42|ave" in keys


def test_document_frequency_counts_distinct_records_not_occurrences():
    keys = pl.DataFrame({"id": [1, 1, 2], "key": ["acme", "acme", "acme"]}, schema={"id": pl.UInt32, "key": pl.String})
    df = candidates.document_frequency(keys, "id")
    assert df.to_dicts() == [{"key": "acme", "df": 2}]  # 2 distinct records, not 3 occurrences


def test_estimate_pair_count_is_sum_of_products_not_the_materialised_join():
    s1 = pl.DataFrame({"key": ["a", "a", "b"]})
    s23 = pl.DataFrame({"key": ["a", "a", "a", "b"]})
    rare = pl.DataFrame({"key": ["a", "b"]})
    # key "a": 2 S1 rows x 3 S23 rows = 6; key "b": 1 x 1 = 1; total 7
    assert candidates.estimate_pair_count(s1, s23, rare) == 7


def test_candidates_topk_ranks_by_score_never_file_order():
    scored = pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [10, 20, 30], "score": [0.5, 9.0, 3.0]},
                          schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32, "score": pl.Float64})
    top2 = candidates.candidates_topk(scored, 2)
    assert top2["s23_gid"].to_list() == [20, 30]  # highest score first, not file order (10,20,30)


def test_recall_at_k_counts_true_links_found_in_topk():
    topk = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 20]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    truth = pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [10, 20, 30]}, schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    out = candidates.recall_at_k(topk, truth)
    assert out == {"recall": pytest.approx(2 / 3), "n_true_links": 3}


def test_recall_at_k_handles_no_true_links_without_dividing_by_zero():
    topk = pl.DataFrame(schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    truth = pl.DataFrame(schema={"s1_gid": pl.UInt32, "s23_gid": pl.UInt32})
    assert candidates.recall_at_k(topk, truth) == {"recall": None, "n_true_links": 0}


def test_candidates_for_cap_raises_before_materialising_an_over_budget_join():
    s1 = _silverish([{"gid": i, "name_tokens": ["acme"], "addr_components": [], "addr_house_number": "", "addr_zip": ""}
                     for i in range(200)]).rename({"gid": "s1_gid"})
    s23 = _silverish([{"gid": i, "name_tokens": ["acme"], "addr_components": [], "addr_house_number": "", "addr_zip": ""}
                      for i in range(200)]).rename({"gid": "s23_gid"})
    with pytest.raises(MemoryError, match="chunk by key hash"):
        candidates.candidates_for_cap(s1, s23, cap=1000, ram_budget_bytes=1)  # 200x200=40,000 pairs > 1 byte
