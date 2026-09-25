"""F9: the ONLY sanctioned way later code may read foundation data.

Guardrails enforced here (see amlc2026-workflow-plan.md and the Stage 0 spec):
  - G-S1/G-S2: split hash is checked on every load; refuses to run on a tampered split.
  - G-L1/G-L2/G-L3: VALIDATION labels only load from amlc.eval.scorer; LOCKBOX labels
    only load from amlc.eval.lockbox, and only once ever (logged to data/lockbox_reads.log).
  - FG-6: gid/line_no are provenance columns; callers must not use them as features
    (this module does not filter them out of bronze reads, but access to gid/line_no
    outside of joins should be treated as a code-review flag downstream).
"""
import datetime
import hashlib
import inspect
from pathlib import Path

import polars as pl

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
BRONZE = ROOT / "data" / "bronze" / "v1"
LABELS = ROOT / "data" / "labels" / "v1"
SPLITS = ROOT / "data" / "splits" / "v1"
LOCKBOX_LOG = ROOT / "data" / "lockbox_reads.log"

EXPECTED_SPLIT_HASH = "0f1393be21cb009ca504cc3069ef74071253546743f68fde57bcf091a95b6ae9"

_BRONZE_FILES = {
    ("train", 1): "train_source1.parquet", ("train", 2): "train_source2.parquet", ("train", 3): "train_source3.parquet",
    ("test", 1): "test_source1.parquet", ("test", 2): "test_source2.parquet", ("test", 3): "test_source3.parquet",
}


class FoundationIntegrityError(Exception):
    pass


class LockboxAccessError(Exception):
    pass


def _content_hash_of_split() -> str:
    df = pl.read_parquet(SPLITS / "s1_split.parquet").sort("s1_id").select(["s1_id", "split", "stress_hidden"])
    h = hashlib.sha256()
    for row in df.iter_rows():
        h.update(("\t".join(str(x) for x in row) + "\n").encode())
    return h.hexdigest()


def _verify_split_hash():
    actual = _content_hash_of_split()
    if actual != EXPECTED_SPLIT_HASH:
        raise FoundationIntegrityError(
            f"Split hash mismatch! expected={EXPECTED_SPLIT_HASH} actual={actual}. "
            f"The split file may have been tampered with or regenerated. STOP."
        )


def _caller_module() -> str:
    stack = inspect.stack()
    # frame 0 = this function, frame 1 = the public API function, frame 2 = the actual caller
    for frame in stack[2:]:
        mod = inspect.getmodule(frame[0])
        if mod is not None:
            return mod.__name__
    return "<unknown>"


def load_bronze(dataset: str, src: int, columns: list[str] | None = None) -> pl.DataFrame:
    """Load a bronze table. dataset in {'train','test'}, src in {1,2,3}."""
    if (dataset, src) not in _BRONZE_FILES:
        raise ValueError(f"unknown (dataset, src) = ({dataset}, {src})")
    path = BRONZE / _BRONZE_FILES[(dataset, src)]
    df = pl.read_parquet(path)
    if columns is not None:
        df = df.select(columns)
    return df


def load_split() -> pl.DataFrame:
    """Load the frozen train S1 split. Verifies content hash every call."""
    _verify_split_hash()
    return pl.read_parquet(SPLITS / "s1_split.parquet")


def load_test_required() -> pl.DataFrame:
    return pl.read_parquet(SPLITS / "test_s1_required.parquet")


def load_labels(split: str) -> pl.DataFrame:
    """Load ground-truth links for one split's S1 entities.

    split='FIT'        -> open to anyone (training data).
    split='VALIDATION'  -> only callable from amlc.eval.scorer.
    split='LOCKBOX'     -> only callable from amlc.eval.lockbox, and only ONCE ever
                            (subsequent calls raise, logged to data/lockbox_reads.log).
    """
    split = split.upper()
    if split not in ("FIT", "VALIDATION", "LOCKBOX"):
        raise ValueError(f"split must be FIT/VALIDATION/LOCKBOX, got {split!r}")

    caller = _caller_module()

    if split == "VALIDATION" and caller != "amlc.eval.scorer":
        raise LockboxAccessError(
            f"load_labels('VALIDATION') may only be called from amlc.eval.scorer, "
            f"not from {caller!r}. This guardrail exists so validation labels are never "
            f"read from feature-engineering or training code paths."
        )

    if split == "LOCKBOX":
        if caller != "amlc.eval.lockbox":
            raise LockboxAccessError(
                f"load_labels('LOCKBOX') may only be called from amlc.eval.lockbox, "
                f"not from {caller!r}."
            )
        if LOCKBOX_LOG.exists() and LOCKBOX_LOG.stat().st_size > 0:
            raise LockboxAccessError(
                "LOCKBOX has already been read once (see data/lockbox_reads.log). "
                "It may only be opened a single time, ever. If this is intentional "
                "(e.g. re-running after a genuine bug fix in the scoring code itself, "
                "not in the model), a human must delete the log file explicitly and "
                "document why in the experiment log first."
            )
        LOCKBOX_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCKBOX_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\tLOCKBOX read by {caller}\n")

    split_df = load_split()
    s1_ids = set(split_df.filter(pl.col("split") == split)["s1_id"].to_list())

    gt_links = pl.read_parquet(LABELS / "gt_links.parquet")
    return gt_links.filter(pl.col("s1_id").is_in(s1_ids))


def get_manifest_split_hash() -> str:
    return EXPECTED_SPLIT_HASH
