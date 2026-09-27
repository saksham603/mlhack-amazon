"""Standard American Soundex: groups similar-SOUNDING consonants (b/f/p/v, c/g/j/k/q/s/x/z, d/t,
l, m/n, r) into the same code, one level beyond the project's existing `skel` field (which only
strips vowels, keeping every consonant distinct). Untried today -- everything else has been exact
string similarity (rapidfuzz) or vowel-stripped skeletons, never sound-alike grouping, which is the
known gap for transliteration variants (a name spelled differently but pronounced the same)."""
import re

_CODE = {}
for letters, digit in (("bfpv", "1"), ("cgjkqsxz", "2"), ("dt", "3"), ("l", "4"), ("mn", "5"), ("r", "6")):
    for c in letters:
        _CODE[c] = digit


def _code(c: str) -> str | None:
    if c in _CODE:
        return _CODE[c]
    if c in "aeiouy":
        return "0"
    return None  # h, w: transparent -- do not break adjacency of the surrounding consonants


def soundex(s: str) -> str:
    letters = re.sub(r"[^a-z]", "", s.lower())
    if not letters:
        return ""
    first = letters[0].upper()
    digits = []
    last = _code(letters[0])
    for c in letters[1:]:
        cd = _code(c)
        if cd is None:
            continue
        if cd != "0" and cd != last:
            digits.append(cd)
        last = cd
    return (first + "".join(digits) + "000")[:4]
