"""Placeholder module: F9 tests need a real module at this exact dotted path
to prove the per-split gate checks caller identity correctly. The real F_0.5
scorer implementation belongs to a later stage (W0), not Stage 0.
"""
from amlc.foundation import access


def load_validation_labels():
    return access.load_labels("VALIDATION")
