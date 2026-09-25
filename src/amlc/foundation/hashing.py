"""The single content-hash definition used by every foundation build and by access.py."""
import hashlib
from pathlib import Path

import polars as pl


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(df: pl.DataFrame, cols: list[str], sort_by: str | list[str]) -> str:
    """SHA-256 over tab-joined rows, sorted, so it is independent of row order and file encoding."""
    h = hashlib.sha256()
    for row in df.sort(sort_by).select(cols).iter_rows():
        h.update(("\t".join("" if x is None else str(x) for x in row) + "\n").encode("utf-8"))
    return h.hexdigest()
