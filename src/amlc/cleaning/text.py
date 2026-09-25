"""C1: general text cleanup (Stage 1 prompt rev 4; user decisions D-C1a..f, 2026-09-26).

Order is binding:
  1. address only, on the RAW string: remove whole comma-separated parts that are missing
  2. ASCII controls -> space (D-C1e), garbled-text repair (D-C1b), then remove Unicode Cf characters
  3. NFKC
  4. casefold, then NFKC again
  5. Latin-only accent removal + ligatures oe/ae (D-C1c), then NFC
  6. collapse whitespace; no space directly before a comma
  7. address only: split on ",", strip, drop parts that are now empty (D-C1f), rejoin with ", "
Symmetric by construction: no function here receives or reads a country, a source or a label.
"""
import re
import unicodedata
from collections import Counter

import polars as pl

# D-C1a: a part is missing when empty or when its lowercase form is one of these.
MISSING_WORDS = ["null", "<null>", "none", "n/a", "na"]
# D-C1e: every ASCII control becomes a space (SUB marks a lost character, like the garbled marker below)
ASCII_CONTROL = re.compile("[\x00-\x1f\x7f]")
ASCII_CONTROL_NON_WS = re.compile("[\x00-\x08\x0e-\x1f\x7f]")  # the ones step 6 would not already collapse
# D-C1b
GARBLED_LOST = re.compile("[ïÏ]¿½")                              # U+FFFD garbled twice: character lost
GARBLED_UTF8 = re.compile("[âÂ]([\u0080-¿])([\u0080-¿])")  # bytes E2 xx yy decoded as Latin-1
C1_CONTROL = re.compile("[\u0080-\u009f]")
# D-C1c
LIGATURES = {"œ": "oe", "æ": "ae"}
LATIN_RANGES = ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF))
ACCENT_MARKS = (0x300, 0x36F)
# Step 6 (rev 4, approved): a mid-string substitution can leave a space directly before a comma that
# step 1 used as a part separator; a second pass would drop it, so C1 would not be idempotent.
SPACE_BEFORE_COMMA = re.compile(r" +,")


def _is_latin(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in LATIN_RANGES)


def _decode_utf8(m: re.Match) -> str:
    try:
        return bytes([0xE2, ord(m[1]), ord(m[2])]).decode("utf-8")
    except UnicodeDecodeError:
        return m[0]


def _cp1252(m: re.Match) -> str:
    try:
        return bytes([ord(m[0])]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


def repair_garbled(s: str) -> str:
    s = GARBLED_LOST.sub(" ", s)
    s = GARBLED_UTF8.sub(_decode_utf8, s)
    return C1_CONTROL.sub(_cp1252, s)


def repair_counts(s: str) -> Counter:
    """What step 2 would do to s, by kind (for the report and the D-C1e gate)."""
    c = Counter()
    for m in ASCII_CONTROL_NON_WS.finditer(s):
        c[f"ascii_control U+{ord(m[0]):04X} -> space"] += 1
    s = ASCII_CONTROL.sub(" ", s)
    c["lost_char_to_space"] += len(GARBLED_LOST.findall(s))
    rest = GARBLED_LOST.sub(" ", s)
    for m in GARBLED_UTF8.finditer(rest):
        fixed = _decode_utf8(m)
        c[f"utf8 -> {fixed!r}" if fixed != m[0] else "utf8 undecodable (left)"] += 1
    rest = GARBLED_UTF8.sub(_decode_utf8, rest)
    for m in C1_CONTROL.finditer(rest):
        fixed = _cp1252(m)
        c[f"cp1252 U+{ord(m[0]):04X} -> {fixed!r}" if fixed else f"cp1252 U+{ord(m[0]):04X} removed"] += 1
    return +c


def fold_latin(s: str) -> str:
    out, base_is_latin = [], False
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch).startswith("M"):
            if base_is_latin and ACCENT_MARKS[0] <= ord(ch) <= ACCENT_MARKS[1]:
                continue
            out.append(ch)
            continue
        base_is_latin = _is_latin(ch)
        out.append(LIGATURES.get(ch, ch))
    return unicodedata.normalize("NFC", "".join(out))


def _finalize_whitespace(s: str) -> str:
    return SPACE_BEFORE_COMMA.sub(",", " ".join(s.split()))


def clean_text(s: str) -> str:
    """C1 steps 2-6 on one string."""
    if s.isascii():  # garbled repair, Cf, NFKC and accents are provably no-ops on ASCII; casefold == lower
        return _finalize_whitespace(ASCII_CONTROL.sub(" ", s).lower())
    return clean_text_full(s)


def clean_text_full(s: str) -> str:
    s = repair_garbled(ASCII_CONTROL.sub(" ", s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFKC", s.casefold())
    s = fold_latin(s)
    return _finalize_whitespace(s)


def map_distinct(values: pl.Series, fn) -> pl.Series:
    """Apply fn once per distinct value and join the result back (C5's token-level rule, applied to C1)."""
    if values.null_count():
        raise ValueError(f"{values.name}: bronze strings are never null; got {values.null_count()} nulls")
    uniq = values.unique()
    lookup = pl.DataFrame({"k": uniq, "v": pl.Series([fn(x) for x in uniq.to_list()], dtype=pl.String)})
    out = pl.DataFrame({"k": values}).join(lookup, on="k", how="left", maintain_order="left")["v"]
    if out.len() != values.len() or out.null_count():
        raise AssertionError("map_distinct lost or added rows")
    return out


def remove_missing_parts(addr: pl.Series) -> pl.DataFrame:
    """C1 step 1 on RAW addresses: (addr_joined, addr_null_parts_removed, addr_missing)."""
    parts = addr.str.split(",").list.eval(pl.element().str.strip_chars())
    missing = pl.element().eq("") | pl.element().str.to_lowercase().is_in(MISSING_WORDS)
    kept = parts.list.eval(pl.element().filter(~missing))
    return pl.DataFrame({
        "addr_joined": kept.list.join(", "),
        "addr_null_parts_removed": parts.list.eval(missing).list.sum().cast(pl.UInt16),
        "addr_missing": kept.list.len() == 0,
    })


def drop_emptied_parts(cleaned: pl.Series, missing_after_step1: pl.Series) -> pl.DataFrame:
    """C1 step 7 (D-C1f): parts that are empty after cleaning are dropped and counted."""
    df = pl.DataFrame({"c": cleaned, "m": missing_after_step1}).with_columns(
        pl.col("c").str.split(",").list.eval(pl.element().str.strip_chars()).alias("p"))
    df = df.with_columns(pl.col("p").list.eval(pl.element().filter(pl.element() != "")).alias("k"))
    return df.select(
        pl.when(pl.col("m")).then(pl.lit("")).otherwise(pl.col("k").list.join(", ")).alias("addr_clean"),
        pl.when(pl.col("m")).then(pl.lit(0)).otherwise(pl.col("p").list.len() - pl.col("k").list.len())
        .cast(pl.UInt16).alias("addr_parts_emptied"),
        (pl.col("m") | (pl.col("k").list.len() == 0)).alias("addr_missing"),
    )


def clean_names(names: pl.Series) -> pl.Series:
    return map_distinct(names, clean_text).alias("name_clean")


def clean_addresses(addr: pl.Series) -> pl.DataFrame:
    step1 = remove_missing_parts(addr)
    step7 = drop_emptied_parts(map_distinct(step1["addr_joined"], clean_text), step1["addr_missing"])
    return pl.DataFrame({
        "addr_clean": step7["addr_clean"],
        "addr_missing": step7["addr_missing"],
        "addr_null_parts_removed": step1["addr_null_parts_removed"],
        "addr_parts_emptied": step7["addr_parts_emptied"],
    })
