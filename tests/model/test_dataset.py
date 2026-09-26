import polars as pl

from amlc.model import dataset


def test_label_batches_marks_true_links_and_fills_negatives(tmp_path, monkeypatch):
    cand_dir, feat_dir = tmp_path / "candidates", tmp_path / "features"
    cand_dir.mkdir()
    feat_dir.mkdir()
    pl.DataFrame({"s1_gid": [1, 1, 2], "s23_gid": [10, 11, 20],
                  "blocking_score": [1.0, 0.5, 2.0], "blocking_rank": [1, 2, 1]}
                 ).write_parquet(cand_dir / "candidates_00000.parquet")
    pl.DataFrame({"s1_gid": [1, 1, 2], "s23_gid": [10, 11, 20], "name_jaccard": [0.9, 0.1, 0.8]}
                 ).write_parquet(feat_dir / "features_00000.parquet")

    truth = pl.DataFrame({"s1_gid": [1], "s23_gid": [10]}, schema={"s1_gid": pl.Int64, "s23_gid": pl.Int64})
    monkeypatch.setattr(dataset.access, "load_labels", lambda split: truth)

    out = dataset.label_batches(cand_dir, feat_dir, "US")
    out = out.sort(["s1_gid", "s23_gid"])
    assert out["label"].to_list() == [1, 0, 0]
    assert out["country"].unique().to_list() == ["US"]
    assert out.height == 3
