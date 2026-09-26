"""Diagnosis helpers; expected values from the definition (padded char-3grams, Jaccard), CG-17."""
import pytest

from amlc.blocking import diagnose_w1 as D


def test_char3_pads_both_ends():
    assert D.char3("ab") == {" ab", "ab "}


def test_jaccard_identical_and_disjoint():
    assert D.jaccard("laxmi", "laxmi") == 1.0
    assert D.jaccard("abc", "xyz") == 0.0


def test_jaccard_spelling_variant_is_partial():
    # " la","lax","axm","xmi","mi " vs " la","lak","aks","ksh","shm","hmi","mi " -> 2 shared of 10
    assert D.jaccard("laxmi", "lakshmi") == pytest.approx(2 / 10)


def test_jaccard_empty_strings_is_zero_not_error():
    assert D.jaccard("", "") == 0.0  # both pad to "  ", which has no 3-gram
