import polars as pl

from amlc.experimental.measure_empty_addr import new_pairs_only


def test_new_pairs_only_excludes_existing():
    existing = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [2, 3]})
    candidate = pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [2, 3, 4]})
    result = new_pairs_only(candidate, existing)
    assert result.select("s1_gid", "s23_gid").rows() == [(1, 4)]
