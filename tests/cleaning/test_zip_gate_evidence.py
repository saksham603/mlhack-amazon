"""Constructed-case tests for the CG-14 ZIP gate evidence script."""
import polars as pl

from amlc.cleaning import zip_gate_evidence as E


def _df(rows):
    return pl.DataFrame(rows, schema={"gid": pl.Int64, "addr_clean": pl.String,
                                       "addr_components": pl.List(pl.String),
                                       "addr_house_number": pl.String, "addr_zip": pl.String})


def test_collision_stats_counts_exact_string_equal_only():
    df = _df([
        {"gid": 1, "addr_clean": "78701 main st", "addr_components": ["78701 main st"],
         "addr_house_number": "78701", "addr_zip": "78701"},
        {"gid": 2, "addr_clean": "304 x, 75182, tx", "addr_components": ["304 x", "75182", "tx"],
         "addr_house_number": "304", "addr_zip": "75182"},
        {"gid": 3, "addr_clean": "no zip", "addr_components": ["no zip"],
         "addr_house_number": "12", "addr_zip": ""},
    ])
    out = E.collision_stats(df)
    assert out == {"rows": 3, "addr_zip_nonempty": 2, "collision_with_house_number": 1, "collision_rate": 0.5}


def test_zero_padding_samples_matches_int_equal_string_unequal():
    df = _df([
        {"gid": 1, "addr_clean": "03001 foo", "addr_components": ["03001 foo"],
         "addr_house_number": "3001", "addr_zip": "03001"},
        {"gid": 2, "addr_clean": "78701 main st", "addr_components": ["78701 main st"],
         "addr_house_number": "78701", "addr_zip": "78701"},
    ])
    out = E.zero_padding_samples(df)
    assert [r["gid"] for r in out] == [1]


def test_recoverable_trailing_zip_requires_state_and_distinct_5digit():
    df = _df([
        # collision row with a genuine trailing "state + distinct 5-digit" shape -> recoverable
        {"gid": 1, "addr_clean": "10075 x ave, 75127, tx",
         "addr_components": ["10075 x ave", "75127", "tx"],
         "addr_house_number": "10075", "addr_zip": "10075"},
        # collision row, last component not a real US state code -> not counted
        {"gid": 2, "addr_clean": "door no 17112, fn 107, indore",
         "addr_components": ["door no 17112", "fn 107", "indore"],
         "addr_house_number": "17112", "addr_zip": "17112"},
        # not a collision row at all -> excluded from the checked set
        {"gid": 3, "addr_clean": "304 x, 75182, tx", "addr_components": ["304 x", "75182", "tx"],
         "addr_house_number": "304", "addr_zip": "75182"},
    ])
    out = E.recoverable_trailing_zip(df)
    assert out["rows_checked"] == 2
    assert out["count"] == 1
    assert out["examples"][0]["gid"] == 1
