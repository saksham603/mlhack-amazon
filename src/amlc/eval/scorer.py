"""VALIDATION scorer. Public API: score() only. Raw VALIDATION labels never leave this module.

The macro F_0.5 logic is built in W0; until then score() raises. The private loaders exist so the
access gate can be exercised; code outside amlc.eval calling them is flagged by leakscan (R2).
"""
from amlc.foundation import access

__all__ = ["score"]


def _load_validation_labels():
    return access.load_labels("VALIDATION")


def _load_validation_match_counts():
    return access.load_match_counts("VALIDATION")


def score(predictions):
    raise NotImplementedError("W0 not built yet: score() must return metrics only, never raw labels.")
