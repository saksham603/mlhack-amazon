import polars as pl

from amlc.features.w2 import pair_features, write_features_batched

SCHEMA_EXTRA = {"addr_house_number": pl.String, "addr_zip": pl.String, "name_legal_form": pl.String}


def _s1(rows):
    return pl.DataFrame(rows, schema={"s1_gid": pl.Int64, "name_tokens": pl.List(pl.String),
                                       "addr_components": pl.List(pl.String), **SCHEMA_EXTRA})


def _s23(rows):
    return pl.DataFrame(rows, schema={"s23_gid": pl.Int64, "name_tokens": pl.List(pl.String),
                                       "addr_components": pl.List(pl.String), **SCHEMA_EXTRA})


def test_name_jaccard_and_overlap():
    s1 = _s1([{"s1_gid": 1, "name_tokens": ["acme", "traders"], "addr_components": ["1 main st"],
               "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    s23 = _s23([{"s23_gid": 10, "name_tokens": ["acme", "traders", "llc"], "addr_components": ["1 main st"],
                 "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    cand = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    out = pair_features(cand, s1, s23).row(0, named=True)
    assert out["name_overlap_n"] == 2
    assert abs(out["name_jaccard"] - 2 / 3) < 1e-9
    assert out["house_number_match"] is True


def test_no_overlap_gives_zero_jaccard_not_nan():
    s1 = _s1([{"s1_gid": 1, "name_tokens": [], "addr_components": [],
               "addr_house_number": "", "addr_zip": "", "name_legal_form": ""}])
    s23 = _s23([{"s23_gid": 10, "name_tokens": [], "addr_components": [],
                 "addr_house_number": "", "addr_zip": "", "name_legal_form": ""}])
    cand = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    out = pair_features(cand, s1, s23).row(0, named=True)
    assert out["name_jaccard"] == 0.0
    assert out["addr_token_jaccard"] == 0.0
    assert out["house_number_match"] is False
    assert out["zip_match"] is False


def test_legal_form_match_requires_both_present():
    s1 = _s1([{"s1_gid": 1, "name_tokens": ["a"], "addr_components": ["x"],
               "addr_house_number": "1", "addr_zip": "", "name_legal_form": "llc"}])
    s23 = _s23([{"s23_gid": 10, "name_tokens": ["a"], "addr_components": ["x"],
                 "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    cand = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    out = pair_features(cand, s1, s23).row(0, named=True)
    assert out["legal_form_match"] is False
    assert out["legal_form_both_present"] is False


def test_addr_token_jaccard_splits_components_into_words():
    s1 = _s1([{"s1_gid": 1, "name_tokens": [], "addr_components": ["1 main st", "austin", "tx"],
               "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    s23 = _s23([{"s23_gid": 10, "name_tokens": [], "addr_components": ["1 main street", "austin", "tx"],
                 "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    cand = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    out = pair_features(cand, s1, s23).row(0, named=True)
    # shared words: 1, main, austin, tx (4); union adds st, street (6 total) -> 4/6
    assert abs(out["addr_token_jaccard"] - 4 / 6) < 1e-9


def test_write_features_batched_covers_all_s1_across_batches(tmp_path):
    n = 25
    s1 = _s1([{"s1_gid": i, "name_tokens": ["a"], "addr_components": ["1 x"],
               "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""} for i in range(n)])
    s23 = _s23([{"s23_gid": 100 + i, "name_tokens": ["a"], "addr_components": ["1 x"],
                 "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""} for i in range(n)])
    cand = pl.DataFrame({"s1_gid": list(range(n)), "s23_gid": [100 + i for i in range(n)]},
                        schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})

    calls = []

    def loader(batch_cand):
        calls.append(batch_cand.height)
        s1_gids = batch_cand.select(pl.col("s1_gid").alias("gid")).unique()
        s23_gids = batch_cand.select(pl.col("s23_gid").alias("gid")).unique()
        return (s1.rename({"s1_gid": "gid"}).join(s1_gids, on="gid").rename({"gid": "s1_gid"}),
                s23.rename({"s23_gid": "gid"}).join(s23_gids, on="gid").rename({"gid": "s23_gid"}))

    out = write_features_batched(cand, loader, tmp_path, batch_size=10)
    assert out == {"n_s1_batches": 3, "batch_size": 10, "n_s1_total": n, "feature_rows_total": n}
    assert calls == [10, 10, 5]
    written = sorted(tmp_path.glob("features_batch_*.parquet"))
    assert len(written) == 3
    total = sum(pl.read_parquet(f).height for f in written)
    assert total == n


def test_multiple_candidates_per_s1_are_independent():
    s1 = _s1([{"s1_gid": 1, "name_tokens": ["a", "b"], "addr_components": ["1 x"],
               "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""}])
    s23 = _s23([
        {"s23_gid": 10, "name_tokens": ["a", "b"], "addr_components": ["1 x"],
         "addr_house_number": "1", "addr_zip": "", "name_legal_form": ""},
        {"s23_gid": 11, "name_tokens": ["c"], "addr_components": ["2 y"],
         "addr_house_number": "2", "addr_zip": "", "name_legal_form": ""},
    ])
    cand = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 11]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    out = pair_features(cand, s1, s23).sort("s23_gid")
    assert out["name_jaccard"].to_list() == [1.0, 0.0]
    assert out["house_number_match"].to_list() == [True, False]
