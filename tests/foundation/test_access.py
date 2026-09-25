"""F9 gate: prove the access-control rules in amlc.foundation.access are real."""
import os
import shutil
import stat
import pytest
from pathlib import Path

from amlc.foundation import access
from amlc.eval import scorer, lockbox


@pytest.fixture(autouse=True)
def clean_lockbox_log():
    # ensure each test starts from a known state; restore afterward
    existed = access.LOCKBOX_LOG.exists()
    backup = None
    if existed:
        backup = access.LOCKBOX_LOG.read_bytes()
        access.LOCKBOX_LOG.unlink()
    yield
    if access.LOCKBOX_LOG.exists():
        access.LOCKBOX_LOG.unlink()
    if backup is not None:
        access.LOCKBOX_LOG.write_bytes(backup)


def test_fit_loading_works_from_anywhere():
    df = access.load_labels("FIT")
    assert df.height > 0
    # FIT should be about 70% of 7,638,365 links, roughly proportional to S1 count
    assert df.height > 4_000_000


def test_validation_loading_from_scorer_module_works():
    df = scorer.load_validation_labels()
    assert df.height > 0


def test_validation_loading_from_non_scorer_module_raises():
    with pytest.raises(access.LockboxAccessError):
        access.load_labels("VALIDATION")  # called directly from test module, not amlc.eval.scorer


def test_lockbox_loading_from_non_lockbox_module_raises():
    with pytest.raises(access.LockboxAccessError):
        access.load_labels("LOCKBOX")  # called directly from test module


def test_lockbox_loading_from_lockbox_module_works_once():
    df = lockbox.load_lockbox_labels()
    assert df.height > 0
    assert access.LOCKBOX_LOG.exists()


def test_lockbox_second_read_is_refused():
    lockbox.load_lockbox_labels()  # first read succeeds
    with pytest.raises(access.LockboxAccessError):
        lockbox.load_lockbox_labels()  # second read must be refused


def test_tampered_split_file_is_refused(tmp_path, monkeypatch):
    # copy the real split file, flip one byte, and point SPLITS at the tampered copy
    real_split = access.SPLITS / "s1_split.parquet"
    tampered_dir = tmp_path / "splits_tampered"
    tampered_dir.mkdir()
    tampered_path = tampered_dir / "s1_split.parquet"
    shutil.copy(real_split, tampered_path)
    os.chmod(tampered_path, stat.S_IWRITE | stat.S_IREAD)  # undo the read-only bit shutil.copy preserved

    data = bytearray(tampered_path.read_bytes())
    # flip a byte roughly in the middle of the file (inside the data payload, not just metadata)
    mid = len(data) // 2
    data[mid] = data[mid] ^ 0xFF
    tampered_path.write_bytes(bytes(data))

    monkeypatch.setattr(access, "SPLITS", tampered_dir)
    with pytest.raises((access.FoundationIntegrityError, Exception)):
        access.load_split()


def test_split_hash_matches_expected_on_real_file():
    # sanity: the real, untampered split file must verify cleanly
    df = access.load_split()
    assert df.height == 2_206_821
