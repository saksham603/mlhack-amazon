"""C4: address cleaning (Stage 1 prompt rev 5, §3 C4). Runs on C1 output. Country-aware via open
tables (CG-4). City/state recognition is explicitly deferred to W7 (§6).
"""
import re

import polars as pl

from amlc.cleaning.tokens import HOUSE_NUMBER_STRIP_WORDS, STREET_WORDS

_ZIP5 = re.compile(r"^\d{5}$")
_PIN6 = re.compile(r"^\d{6}$")
_HASH = re.compile(r"#")
_STRIP_WORD = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in HOUSE_NUMBER_STRIP_WORDS) + r")\b",
                          re.IGNORECASE)
_WS = re.compile(r"\s+")


def split_components(addr: str) -> list[str]:
    """C1's addr_clean already removed null parts and collapsed whitespace; just split and strip."""
    return [p.strip() for p in addr.split(",")] if addr else []


def canonicalize_street_words(component: str, country: str) -> str:
    table = STREET_WORDS.get(country)
    if not table:
        return component  # CG-4: unlisted country falls through unchanged
    out = []
    for tok in component.split():
        key = tok.rstrip(".")
        out.append(table.get(key, tok))
    return " ".join(out)


def extract_house_number(components: list[str]) -> tuple[str, str]:
    """Returns (house_number, raw_digits) from the first numeric-leading component, else ("", "")."""
    for comp in components:
        stripped = _HASH.sub(" ", comp)
        stripped = _STRIP_WORD.sub(" ", stripped)
        stripped = _WS.sub(" ", stripped).strip()
        m = re.match(r"^(\d[\d/\-]*)\b", stripped)
        if m:
            raw = m.group(1)
            no_leading_zero = raw.lstrip("0") or "0"
            return no_leading_zero, raw
    return "", ""


def extract_zip(components: list[str]) -> str:
    for comp in components:
        for tok in comp.replace(",", " ").split():
            if _ZIP5.match(tok) or _PIN6.match(tok):
                return tok
    return ""


def clean_addresses(addr_clean: pl.Series, country: pl.Series) -> pl.DataFrame:
    if addr_clean.len() != country.len():
        raise ValueError("addr_clean and country must be the same length")
    comps_col, house_col, house_raw_col, zip_col = [], [], [], []
    for addr, c in zip(addr_clean.to_list(), country.to_list()):
        comps = split_components(addr)
        canon = [canonicalize_street_words(p, c) for p in comps]
        h, hraw = extract_house_number(canon)
        z = extract_zip(canon)
        comps_col.append(canon)
        house_col.append(h)
        house_raw_col.append(hraw)
        zip_col.append(z)
    return pl.DataFrame({
        "addr_components": pl.Series(comps_col, dtype=pl.List(pl.String)),
        "addr_house_number": pl.Series(house_col, dtype=pl.String),
        "addr_house_number_raw": pl.Series(house_raw_col, dtype=pl.String),
        "addr_zip": pl.Series(zip_col, dtype=pl.String),
    })
