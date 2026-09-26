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


def test_write_candidate_tsv_streaming_keeps_rank_le_k_and_writes_every_s1_once(tmp_path, monkeypatch):
    from amlc.model import assemble_output as A
    monkeypatch.setattr(A, "TEST_OUT", tmp_path)
    monkeypatch.setattr(A, "COUNTRIES", ("US", "France"))
    sdir = tmp_path / "scored" / "US"
    sdir.mkdir(parents=True)
    pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [10, 11], "blocking_score": [2.0, 1.0],
                  "blocking_rank": [1, 2], "prob": [0.9, 0.1]}).write_parquet(sdir / "scored_00000.parquet")
    pl.DataFrame({"s1_gid": [2], "s23_gid": [20], "blocking_score": [1.0],
                  "blocking_rank": [1], "prob": [0.5]}).write_parquet(tmp_path / "scored_France.parquet")
    s1_map, s23_map = _maps()
    out = tmp_path / "candidate_pairs.tsv"
    rep = A.write_candidate_tsv_streaming(out, s1_map, s23_map, {"S1-1", "S1-2", "S1-3"}, k=1)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    body = dict(line.split("\t") for line in lines[1:])
    assert body == {"S1-1": "S2-10", "S1-2": "S3-20", "S1-3": ""}
    assert rep == {"n_candidate_pairs": 2, "s1_with_candidates": 2, "s1_without_candidates": 1}


def test_load_predicted_applies_rank_and_threshold(tmp_path, monkeypatch):
    from amlc.model import assemble_output as A
    monkeypatch.setattr(A, "TEST_OUT", tmp_path)
    monkeypatch.setattr(A, "COUNTRIES", ("US",))
    sdir = tmp_path / "scored" / "US"
    sdir.mkdir(parents=True)
    pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [10, 11, 12], "blocking_score": [3.0, 2.0, 1.0],
                  "blocking_rank": [1, 2, 3], "prob": [0.9, 0.2, 0.95]}).write_parquet(sdir / "scored_00000.parquet")
    assert A.load_predicted(0.5, k=2)["s23_gid"].to_list() == [10]
