"""Access-control gate tests for foundation v2."""
import inspect
import os
import shutil
import stat

import polars as pl
import pytest

from amlc.eval import lockbox, scorer
from amlc.foundation import access
from amlc.foundation.split_v2 import LABEL_DERIVED_COLUMNS, SPLIT_V2_COLUMNS


@pytest.fixture(autouse=True)
def isolate_lockbox_log():
    backup = access.LOCKBOX_LOG.read_bytes() if access.LOCKBOX_LOG.exists() else None
    if backup is not None:
        access.LOCKBOX_LOG.unlink()
    yield
    if access.LOCKBOX_LOG.exists():
        access.LOCKBOX_LOG.unlink()
    if backup is not None:
        access.LOCKBOX_LOG.write_bytes(backup)


def _writable_copy(src, dst):
    shutil.copyfile(src, dst)
    os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)


def _flip_middle_byte(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


# ---- open split carries no label-derived data ----
def test_split_has_only_non_label_columns():
    df = access.load_split()
    assert df.columns == SPLIT_V2_COLUMNS
    assert not set(df.columns) & LABEL_DERIVED_COLUMNS
    assert df.height == 2_206_821


# ---- labels ----
def test_fit_labels_open():
    df = access.load_labels("FIT")
    fit_ids = set(access.load_split().filter(pl.col("split") == "FIT")["s1_id"].to_list())
    assert df.height > 0 and set(df["s1_id"].unique().to_list()) <= fit_ids


def test_validation_labels_refused_outside_scorer():
    with pytest.raises(access.LockboxAccessError):
        access.load_labels("VALIDATION")


def test_validation_labels_allowed_from_scorer():
    assert scorer._load_validation_labels().height > 0


def test_lockbox_labels_refused_outside_lockbox_module():
    with pytest.raises(access.LockboxAccessError):
        access.load_labels("LOCKBOX")
    assert not access.LOCKBOX_LOG.exists(), "a refused read must not consume the single lockbox read"


def test_lockbox_single_read_then_refused():
    assert lockbox._load_lockbox_labels().height > 0
    with pytest.raises(access.LockboxAccessError):
        lockbox._load_lockbox_labels()


# ---- match counts (label-derived) gated exactly like labels ----
def test_fit_match_counts_open():
    df = access.load_match_counts("FIT")
    assert {"n_total", "is_singleton", "bucket"} <= set(df.columns)


def test_validation_match_counts_refused_outside_scorer():
    with pytest.raises(access.LockboxAccessError):
        access.load_match_counts("VALIDATION")


def test_validation_match_counts_allowed_from_scorer():
    assert scorer._load_validation_match_counts().height > 0


def test_lockbox_match_counts_refused_outside_lockbox_module():
    with pytest.raises(access.LockboxAccessError):
        access.load_match_counts("LOCKBOX")


# ---- evaluation modules expose scores only, never raw labels ----
def test_scorer_public_api_is_score_only():
    public_funcs = [n for n, o in inspect.getmembers(scorer, inspect.isfunction)
                    if not n.startswith("_") and o.__module__ == scorer.__name__]
    assert public_funcs == ["score"] and scorer.__all__ == ["score"]


def test_lockbox_public_api_is_score_once_only():
    public_funcs = [n for n, o in inspect.getmembers(lockbox, inspect.isfunction)
                    if not n.startswith("_") and o.__module__ == lockbox.__name__]
    assert public_funcs == ["score_once"] and lockbox.__all__ == ["score_once"]


# ---- integrity: tampered files are refused ----
def test_tampered_split_refused(tmp_path, monkeypatch):
    _writable_copy(access.SPLITS / "s1_split.parquet", tmp_path / "s1_split.parquet")
    _flip_middle_byte(tmp_path / "s1_split.parquet")
    monkeypatch.setattr(access, "SPLITS", tmp_path)
    with pytest.raises(Exception):
        access.load_split()


def test_tampered_bronze_refused(tmp_path, monkeypatch):
    _writable_copy(access.BRONZE / "train_source1.parquet", tmp_path / "train_source1.parquet")
    _flip_middle_byte(tmp_path / "train_source1.parquet")
    monkeypatch.setattr(access, "BRONZE", tmp_path)
    with pytest.raises(access.FoundationIntegrityError):
        access.load_bronze("train", 1)


def test_real_bronze_verifies():
    assert access.load_bronze("test", 1, columns=["entity_id"]).height == 1_732_544
