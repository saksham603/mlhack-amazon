"""The ONLY sanctioned way later code may read foundation data (v2).

Rules enforced here:
  - Every file served is hash-verified against MANIFEST_v2.json on first load per process
    (v1 promised this for bronze but did not implement it).
  - load_split() returns only non-label columns; its content hash is checked on every call.
  - Label-derived data (links AND match counts) per split:
      FIT        -> open to anyone;
      VALIDATION -> only when called from amlc.eval.scorer;
      LOCKBOX    -> only when called from amlc.eval.lockbox, and only ONE label-derived read ever
                    (logged to data/lockbox_reads.log; any later read is refused).
Limit, stated honestly: Python cannot stop code from opening files directly. That is detected by
amlc.foundation.leakscan (run in the test suite), not prevented here.
"""
import datetime
import inspect
import json
from pathlib import Path

import polars as pl

from amlc.foundation.hashing import content_hash, file_sha256

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
DATA = ROOT / "data"
BRONZE = DATA / "bronze" / "v1"
LABELS_V1 = DATA / "labels" / "v1"
LABELS_V2 = DATA / "labels" / "v2"
SPLITS = DATA / "splits" / "v2"
MANIFEST = DATA / "MANIFEST_v2.json"
LOCKBOX_LOG = DATA / "lockbox_reads.log"

_BRONZE_FILES = {
    ("train", 1): "train_source1.parquet", ("train", 2): "train_source2.parquet", ("train", 3): "train_source3.parquet",
    ("test", 1): "test_source1.parquet", ("test", 2): "test_source2.parquet", ("test", 3): "test_source3.parquet",
}
_verified: set[str] = set()
_manifest_cache: dict | None = None


class FoundationIntegrityError(Exception):
    pass


class LockboxAccessError(Exception):
    pass


def _manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        if not MANIFEST.exists():
            raise FoundationIntegrityError(f"{MANIFEST} missing: foundation v2 was not completed. STOP.")
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if not m.get("all_gates_passed"):
            raise FoundationIntegrityError("MANIFEST_v2 records failed gates. STOP.")
        _manifest_cache = m
    return _manifest_cache


def _read_verified(path: Path, expected_sha256: str, columns=None) -> pl.DataFrame:
    key = str(path.resolve())
    if key not in _verified:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise FoundationIntegrityError(f"{path} hash mismatch (expected {expected_sha256}, got {actual}). STOP.")
        _verified.add(key)
    return pl.read_parquet(path, columns=columns)


def _caller_module() -> str:
    # frame 0 = this function, 1 = the public API function, 2+ = the caller
    for frame in inspect.stack()[2:]:
        mod = inspect.getmodule(frame[0])
        if mod is not None:
            return mod.__name__
    return "<unknown>"


def _authorize(split: str, caller: str, what: str) -> None:
    if split == "FIT":
        return
    if split == "VALIDATION":
        if caller != "amlc.eval.scorer":
            raise LockboxAccessError(
                f"{what}('VALIDATION') is only allowed from amlc.eval.scorer, not {caller!r}. "
                "Validation labels must never reach feature, cleaning or training code.")
        return
    if split == "LOCKBOX":
        if caller != "amlc.eval.lockbox":
            raise LockboxAccessError(f"{what}('LOCKBOX') is only allowed from amlc.eval.lockbox, not {caller!r}.")
        if LOCKBOX_LOG.exists() and LOCKBOX_LOG.stat().st_size > 0:
            raise LockboxAccessError(
                "LOCKBOX label data has already been read once (see data/lockbox_reads.log). It may be "
                "read a single time, ever. Only a human may override this, with a written reason.")
        LOCKBOX_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCKBOX_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\t{what}\tLOCKBOX\t{caller}\n")
        return
    raise ValueError(f"split must be FIT/VALIDATION/LOCKBOX, got {split!r}")


def load_bronze(dataset: str, src: int, columns: list[str] | None = None) -> pl.DataFrame:
    if (dataset, src) not in _BRONZE_FILES:
        raise ValueError(f"unknown (dataset, src) = ({dataset}, {src})")
    name = _BRONZE_FILES[(dataset, src)]
    expected = _manifest()["unchanged_artifacts"]["bronze_file_hashes"][name]
    return _read_verified(BRONZE / name, expected, columns)


def load_split() -> pl.DataFrame:
    """Train S1 split with non-label columns only: s1_gid, s1_id, country, split, stress_hidden."""
    m = _manifest()["new_artifacts"]
    df = pl.read_parquet(SPLITS / "s1_split.parquet")
    cols = m["split_content_hash_columns"]
    if list(df.columns) != cols or content_hash(df, cols, "s1_id") != m["split_content_hash"]:
        raise FoundationIntegrityError("Split v2 content does not match MANIFEST_v2 (tampered or regenerated). STOP.")
    return df


def load_test_required() -> pl.DataFrame:
    m = _manifest()["new_artifacts"]["test_s1_required"]
    return _read_verified(SPLITS / "test_s1_required.parquet", m["sha256"])


def _split_ids(split: str) -> pl.Series:
    return load_split().filter(pl.col("split") == split)["s1_id"]


def load_labels(split: str) -> pl.DataFrame:
    """Ground-truth links (s1_gid, s1_id, s23_gid, s23_id, s23_src) for one split's S1."""
    split = split.upper()
    _authorize(split, _caller_module(), "load_labels")
    expected = _manifest()["unchanged_artifacts"]["label_file_hashes"]["gt_links.parquet"]
    links = _read_verified(LABELS_V1 / "gt_links.parquet", expected)
    return links.filter(pl.col("s1_id").is_in(_split_ids(split)))


def load_match_counts(split: str) -> pl.DataFrame:
    """Label-derived per-S1 counts (n_s2, n_s3, n_total, is_singleton, bucket) for one split."""
    split = split.upper()
    _authorize(split, _caller_module(), "load_match_counts")
    m = _manifest()["new_artifacts"]["match_counts"]
    counts = _read_verified(LABELS_V2 / "s1_match_counts.parquet", m["sha256"])
    return counts.filter(pl.col("s1_id").is_in(_split_ids(split)))
