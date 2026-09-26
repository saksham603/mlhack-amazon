from amlc.model.validate_output import check_matching_and_candidates, read_submission_tsv


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_submission_tsv_splits_and_drops_empty(tmp_path):
    p = tmp_path / "m.tsv"
    _write(p, ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-10,S3-20", "S1-2\t"])
    out = read_submission_tsv(p, "matched_id").sort(["source1_entity_id", "matched_id"])
    assert out["source1_entity_id"].to_list() == ["S1-1", "S1-1"]
    assert out["matched_id"].to_list() == ["S2-10", "S3-20"]


def test_check_matching_and_candidates_clean_passes():
    test_ids = {"S1-1", "S1-2"}
    matching = {"S1-1": "S2-10", "S1-2": ""}
    cand = {"S1-1": "S2-10,S3-30", "S1-2": ""}

    def write_all(tmp_path):
        m = tmp_path / "matching_results.tsv"
        c = tmp_path / "candidate_pairs.tsv"
        _write(m, ["source1_entity_id\tmatched_entity_ids"] + [f"{k}\t{v}" for k, v in matching.items()])
        _write(c, ["source1_entity_id\tcandidate_entity_ids"] + [f"{k}\t{v}" for k, v in cand.items()])
        return m, c
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        m, c = write_all(Path(d))
        problems = check_matching_and_candidates(m, c, test_ids)
    assert problems == []


def test_check_matching_and_candidates_flags_prediction_not_in_candidates(tmp_path):
    m = tmp_path / "matching_results.tsv"
    c = tmp_path / "candidate_pairs.tsv"
    _write(m, ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-99"])
    _write(c, ["source1_entity_id\tcandidate_entity_ids", "S1-1\tS2-10"])
    problems = check_matching_and_candidates(m, c, {"S1-1"})
    assert any("NOT present in candidate_pairs" in p for p in problems)


def test_check_matching_and_candidates_flags_self_match(tmp_path):
    m = tmp_path / "matching_results.tsv"
    c = tmp_path / "candidate_pairs.tsv"
    _write(m, ["source1_entity_id\tmatched_entity_ids", "S1-1\tS1-1"])
    _write(c, ["source1_entity_id\tcandidate_entity_ids", "S1-1\tS1-1"])
    problems = check_matching_and_candidates(m, c, {"S1-1"})
    assert any("self-match" in p for p in problems)
