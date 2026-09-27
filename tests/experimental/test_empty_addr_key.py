import polars as pl

from amlc.experimental.empty_addr_key import empty_addr_candidates, empty_addr_keys


def test_short_ns_excluded():
    r23 = pl.DataFrame({"gid": [1, 2], "addr": ["", ""], "ns": ["ab", "widgetsinc"]})
    keys = empty_addr_keys(r23)
    assert keys["s23_gid"].to_list() == [2]


def test_nonempty_addr_excluded():
    r23 = pl.DataFrame({"gid": [1, 2], "addr": ["12 main st", ""], "ns": ["widgetsinc", "widgetsinc"]})
    keys = empty_addr_keys(r23)
    assert keys["s23_gid"].to_list() == [2]


def test_no_empty_address_records_returns_empty():
    r23 = pl.DataFrame({"gid": [1], "addr": ["12 main st"], "ns": ["widgetsinc"]})
    assert empty_addr_keys(r23).height == 0


def test_candidates_match_on_shared_ns():
    r1 = pl.DataFrame({"gid": [10, 11], "ns": ["widgetsinc", "zz"]})
    keys = pl.DataFrame({"s23_gid": [2], "ns": ["widgetsinc"]})
    cand = empty_addr_candidates(r1, keys)
    assert cand.select("s1_gid", "s23_gid").rows() == [(10, 2)]
