"""The 2026-09-26 speed-ups must be exact: each shortcut is checked against the slow reference it replaces."""
import unicodedata

from amlc.cleaning import recount, text
from tests.cleaning.fixtures import GARBLED, INDIC_NAMES, row


def _is_latin_reference(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in text.LATIN_RANGES)


def _fold_latin_reference(s: str) -> str:
    """fold_latin exactly as committed in e197d8c, before the fast path."""
    out, base_is_latin = [], False
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch).startswith("M"):
            if base_is_latin and text.ACCENT_MARKS[0] <= ord(ch) <= text.ACCENT_MARKS[1]:
                continue
            out.append(ch)
            continue
        base_is_latin = _is_latin_reference(ch)
        out.append(text.LIGATURES.get(ch, ch))
    return unicodedata.normalize("NFC", "".join(out))


def test_latin_set_equals_range_check_for_every_code_point():
    assert all(text._is_latin(chr(cp)) == _is_latin_reference(chr(cp)) for cp in range(0x30000))


def test_fold_latin_fast_path_equals_reference():
    samples = ["Président", "ẹ́", "œuvre", "Cœur Æsir", "लक्ष्मी", "क़", "ପ୍ରାଇଭେଟ", "Straße",
               "ñandú", "ä́", "plain ascii", "", "́leading mark", "Ωmegá"]
    samples += [row(*k)["business_name"] for k in INDIC_NAMES]
    samples += [row(ds, src, gid)[col] for ds, src, gid, col in GARBLED]
    for s in samples:
        assert text.fold_latin(s) == _fold_latin_reference(s), repr(s)


def test_every_visible_ascii_character_survives_c1():
    # proof step for recount's shortcut: each of the 94 visible ASCII characters survives on its own
    for cp in range(0x21, 0x7F):
        assert recount._char_survives(chr(cp)), chr(cp)
        assert text.clean_text(chr(cp)) == chr(cp).lower()
