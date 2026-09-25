"""Independent recount of C1's gate numbers (CG-9). Imports nothing from amlc.cleaning.text.

Plain Python loops over raw strings, versus text.py's Polars list expressions. Whitespace for
stripping is the Unicode White_Space property, the same character set Polars' str.strip_chars()
strips (Rust char::is_whitespace), written out here instead of borrowed from either library.
"""
import unicodedata
from collections import Counter

WHITE_SPACE = "".join(map(chr, [
    *range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
    0x2028, 0x2029, 0x202F, 0x205F, 0x3000]))
STAGE0_TOKENS = frozenset(["null", "NULL", "<NULL>", "None", "N/A", "n/a", "na", "NA", ""])
LOWER_WORDS = frozenset(["null", "<null>", "none", "n/a", "na"])
LOST_MARKERS = ("ï¿½", "Ï¿½")


def _char_survives(ch: str) -> bool:
    """False if C1 deletes ch or turns it into whitespace (spec: D-C1b, D-C1e, step 2 Cf, step 6)."""
    cp = ord(ch)
    if cp <= 0x1F or cp == 0x7F or ch.isspace():
        return False
    if 0x80 <= cp <= 0x9F:  # Windows-1252 meaning if defined, else removed
        try:
            return not bytes([cp]).decode("cp1252").isspace()
        except UnicodeDecodeError:
            return False
    if unicodedata.category(ch) == "Cf":
        return False
    return not unicodedata.normalize("NFKC", ch).isspace()


def part_is_emptied(part: str) -> bool:
    """True if a step-1-surviving part contains nothing C1 keeps (D-C1f)."""
    for m in LOST_MARKERS:
        part = part.replace(m, " ")
    i, chars = 0, []
    while i < len(part):
        ch, nxt = part[i], part[i + 1:i + 3]
        if ch in "âÂ" and len(nxt) == 2 and all(0x80 <= ord(x) <= 0xBF for x in nxt):
            try:
                chars.append(bytes([0xE2, ord(nxt[0]), ord(nxt[1])]).decode("utf-8"))
                i += 3
                continue
            except UnicodeDecodeError:
                pass
        chars.append(ch)
        i += 1
    return not any(_char_survives(c) for c in chars)


def classify_address(addr: str) -> dict:
    parts = [p.strip(WHITE_SPACE) for p in addr.split(",")]
    stage0 = [p in STAGE0_TOKENS for p in parts]
    c1 = [p == "" or p.lower() in LOWER_WORDS for p in parts]
    variant = [b and not a for a, b in zip(stage0, c1)]
    emptied = [not m and part_is_emptied(p) for p, m in zip(parts, c1)]
    all_step1 = all(c1)
    return {
        "stage0": any(stage0),
        "variant": any(variant),
        "c1": any(c1),
        "parts_removed": sum(c1),
        "all_missing_step1": all_step1,
        "parts_emptied": 0 if all_step1 else sum(emptied),
        "all_missing": all(m or e for m, e in zip(c1, emptied)),
        "raw_empty": addr == "",
    }


def recount_addresses(addresses: list[str], countries: list[str]) -> dict:
    """Row counts per country and in total, for G-C1-2, G-C1-3 and the D-C1f gate."""
    per = Counter()
    flags = ("stage0", "variant", "c1", "all_missing_step1", "all_missing", "raw_empty")
    for a, c in zip(addresses, countries):
        r = classify_address(a)
        for k in flags:
            if r[k]:
                per[(c, k)] += 1
        if r["all_missing"] and not r["raw_empty"]:
            per[(c, "all_missing_not_empty")] += 1
        if r["parts_emptied"]:
            per[(c, "rows_with_emptied")] += 1
        per[(c, "parts_removed")] += r["parts_removed"]
        per[(c, "parts_emptied")] += r["parts_emptied"]
    keys = (*flags, "all_missing_not_empty", "rows_with_emptied", "parts_removed", "parts_emptied")
    out = {"by_country": {c: {k: per[(c, k)] for k in keys} for c in sorted(set(countries))}}
    out["total"] = {k: sum(v[k] for v in out["by_country"].values()) for k in keys}
    return out
