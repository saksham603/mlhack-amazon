import polars as pl

from amlc.model.assemble_output import links_to_tsv


def _maps():
    s1_map = pl.DataFrame({"s1_gid": [1, 2, 3], "s1_id": ["S1-1", "S1-2", "S1-3"]})
    s23_map = pl.DataFrame({"s23_gid": [10, 11, 20], "s23_id": ["S2-10", "S2-11", "S3-20"]})
    return s1_map, s23_map


def test_links_to_tsv_includes_every_s1_even_with_no_links():
    s1_map, s23_map = _maps()
    links = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 11]})
    all_ids = s1_map.select("s1_id")
    out = links_to_tsv(links, s1_map, s23_map, all_ids, "candidate_id").sort("s1_id")
    assert out["s1_id"].to_list() == ["S1-1", "S1-2", "S1-3"]
    assert out.filter(pl.col("s1_id") == "S1-1")["candidate_id"].item() == "S2-10,S2-11"
    assert out.filter(pl.col("s1_id") == "S1-2")["candidate_id"].item() == ""
    assert out.filter(pl.col("s1_id") == "S1-3")["candidate_id"].item() == ""


def test_links_to_tsv_ids_are_sorted_and_comma_joined_no_dupes():
    s1_map, s23_map = _maps()
    links = pl.DataFrame({"s1_gid": [2, 2], "s23_gid": [20, 10]})
    all_ids = s1_map.select("s1_id")
    out = links_to_tsv(links, s1_map, s23_map, all_ids, "matched_id")
    row = out.filter(pl.col("s1_id") == "S1-2")["matched_id"].item()
    assert row == "S2-10,S3-20"  # sorted lexically
