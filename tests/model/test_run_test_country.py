import numpy as np
import polars as pl
import pytest

from amlc.model import train as T
from amlc.model.run_test_country import score_batches


class _FakeBooster:
    best_iteration = 0

    def predict(self, X, num_iteration=None):
        return np.full(len(X), 0.5)


def _write_batch(cdir, fdir, idx, n_cand, n_feat):
    pl.DataFrame({"s1_gid": [1] * n_cand, "s23_gid": list(range(n_cand)),
                  "blocking_score": [1.0] * n_cand, "blocking_rank": list(range(1, n_cand + 1))}
                 ).write_parquet(cdir / f"candidates_{idx}.parquet")
    feats = {c: [0.0] * n_feat for c in T.FEATURE_COLS if c not in ("blocking_score", "blocking_rank")}
    pl.DataFrame({"s1_gid": [1] * n_feat, "s23_gid": list(range(n_feat)), **feats}
                 ).write_parquet(fdir / f"features_{idx}.parquet")


def test_score_batches_writes_one_scored_file_per_batch_and_resumes(tmp_path):
    cdir, fdir, sdir = tmp_path / "c", tmp_path / "f", tmp_path / "s"
    cdir.mkdir(); fdir.mkdir()
    _write_batch(cdir, fdir, "00000", 3, 3)
    _write_batch(cdir, fdir, "00001", 2, 2)
    assert score_batches(cdir, fdir, sdir, _FakeBooster()) == 5
    assert sorted(p.name for p in sdir.iterdir()) == ["scored_00000.parquet", "scored_00001.parquet"]
    assert score_batches(cdir, fdir, sdir, _FakeBooster()) == 5  # second call skips, same total


def test_score_batches_stops_on_candidate_feature_mismatch(tmp_path):
    cdir, fdir, sdir = tmp_path / "c", tmp_path / "f", tmp_path / "s"
    cdir.mkdir(); fdir.mkdir()
    _write_batch(cdir, fdir, "00000", 3, 2)
    with pytest.raises(RuntimeError, match="STOP"):
        score_batches(cdir, fdir, sdir, _FakeBooster())
