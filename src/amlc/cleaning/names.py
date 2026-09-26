"""C3: name cleaning (Stage 1 prompt rev 5, §3 C3). Runs on C2 output (Indic legal words already
Latin). Country-aware via open tables (CG-4): an unlisted country falls through unchanged, flagged.
"""
import re
import unicodedata

import polars as pl

from amlc.cleaning.tokens import DOTTED_ABBREV, HONORIFICS, LEGAL_FORM_PHRASES, LEGAL_FORMS

_DOTTED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in DOTTED_ABBREV]
_DIGIT_RUN = re.compile(r"\b\d{7,}\b")
_ID_TAG = re.compile(r"\(\s*id\s*:\s*\d+\s*\)", re.IGNORECASE)

# Punctuation/symbol -> space, built once from Unicode category, NOT from \w's complement: \w (even
# with re.UNICODE) excludes combining marks (category Mn/Mc — Indic vowel signs and virama), so a
# "not \w" regex would silently mangle any untranslated Indic token that D6 requires to survive
# intact (found 2026-09-26: "मार्केटिंग" -> "म र क ट ग", combining marks stripped as if punctuation).
# Category P*/S* covers brackets too, so this also satisfies "brackets become spaces, contents kept"
# and "hyphens become spaces" without a separate rule: every hyphen is category Pd.
_PUNCT_TRANSLATE = {cp: " " for cp in range(0x110000) if unicodedata.category(chr(cp))[0] in ("P", "S")}
_HONORIFIC = re.compile(r"^(?:" + "|".join(re.escape(h) for h in HONORIFICS) + r")\b\.?\s*",
                        re.IGNORECASE)
_LEADING_THE = re.compile(r"^the\b\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")


def collapse_abbreviations(s: str) -> str:
    for pat, repl in _DOTTED:
        s = pat.sub(repl, s)
    return s


def strip_appendages(s: str) -> str:
    s = _ID_TAG.sub(" ", s)
    s = _DIGIT_RUN.sub(" ", s)
    return s


def strip_punctuation(s: str) -> str:
    return s.translate(_PUNCT_TRANSLATE)


def strip_leading(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _HONORIFIC.sub("", s)
        s = _LEADING_THE.sub("", s)
    return s


def _extract_legal_form(tokens: list[str], country: str) -> tuple[list[str], str]:
    """Extracts EVERY matching legal-form token/phrase, not just the first (CG-13): a name can carry
    more than one (e.g. a US professional's "DDS ... PC"). A single-match version left the second one
    sitting in name_clean, so re-running C3 on its own output kept finding and removing more — a real
    idempotence failure caught by the silver_run.py dry run, not a test artifact.
    """
    phrases = LEGAL_FORM_PHRASES.get(country, [])
    table = LEGAL_FORMS.get(country, {})
    found = []
    changed = True
    while changed:
        changed = False
        for words, canon in phrases:
            n = len(words)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i + n]) == words:
                    tokens = tokens[:i] + tokens[i + n:]
                    found.append(canon)
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        for i, tok in enumerate(tokens):
            if tok in table:
                tokens = tokens[:i] + tokens[i + 1:]
                found.append(table[tok])
                changed = True
                break
    # dedupe while keeping order (a name with "pvt ltd" then a stray "ltd" would otherwise report "ltd" twice)
    seen, uniq_found = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            uniq_found.append(f)
    return tokens, " ".join(uniq_found)


def clean_one_name(raw: str, country: str) -> tuple[str, str]:
    """Returns (name_clean, name_legal_form). raw is C2 output (already casefolded/transliterated)."""
    s = collapse_abbreviations(raw)
    s = strip_appendages(s)
    s = strip_punctuation(s)
    s = _WS.sub(" ", s).strip()
    # strip_leading and legal-form extraction must loop together, not run once each in sequence:
    # extracting a LEADING legal-form token (e.g. "Inc The-Anchor Beacon Quetta" -> extract "inc")
    # exposes a new leading "the" that a one-shot strip_leading (which already ran) never re-checks.
    # Found by the silver_run.py dry run as a real idempotence failure (CG-13), train S2 gid search.
    found_forms, prev = [], None
    while prev != s:
        prev = s
        s = strip_leading(s)
        tokens, legal = _extract_legal_form(s.split(), country)
        s = " ".join(tokens)
        if legal:
            found_forms.append(legal)
    # dedupe at the whole-chunk level (each chunk, e.g. "pvt ltd" or "dds pc", is already an intact
    # phrase from one _extract_legal_form call; splitting further would break "pvt ltd" into two)
    seen, uniq = set(), []
    for f in found_forms:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return s, " ".join(uniq)


def clean_names(names: pl.Series, country: pl.Series) -> pl.DataFrame:
    """Row-wise (country varies per row, so this cannot be a distinct-value join like C1/C2)."""
    if names.len() != country.len():
        raise ValueError("names and country must be the same length")
    clean, legal = [], []
    for n, c in zip(names.to_list(), country.to_list()):
        nc, lf = clean_one_name(n, c)
        clean.append(nc)
        legal.append(lf)
    return pl.DataFrame({"name_clean": pl.Series(clean, dtype=pl.String),
                         "name_legal_form": pl.Series(legal, dtype=pl.String)})
