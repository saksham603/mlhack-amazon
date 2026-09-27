"""Tests for a standard Soundex encoder: classic textbook cases, plus business-name edge cases
(empty string, digits, single letter) that a real feature pipeline will actually hit."""
import pytest

from amlc.features.soundex import soundex


@pytest.mark.parametrize("word,code", [
    ("Robert", "R163"),
    ("Rupert", "R163"),
    ("Ashcraft", "A261"),   # H/W are separators, not dropped consonants, per the classic rule
    ("Tymczak", "T522"),
    ("Pfister", "P236"),
    ("Honeyman", "H555"),
])
def test_classic_soundex_examples(word, code):
    assert soundex(word) == code


def test_empty_string_gives_empty_code():
    assert soundex("") == ""


def test_single_letter():
    assert soundex("a") == "A000"


def test_digits_and_spaces_are_ignored():
    assert soundex("ab 12 cd") == soundex("abcd")


def test_case_insensitive():
    assert soundex("Robert") == soundex("ROBERT") == soundex("robert")


def test_similar_sounding_different_spelling_matches():
    # b/v are one soundex group -- this is the whole point of using it over plain edit distance.
    # (Soundex always keeps the literal first letter, so this only shows up when it already matches.)
    assert soundex("Cabil") == soundex("Cavil")
