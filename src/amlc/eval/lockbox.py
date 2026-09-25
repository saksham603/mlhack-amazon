"""Placeholder module: same reasoning as scorer.py, for the LOCKBOX gate."""
from amlc.foundation import access


def load_lockbox_labels():
    return access.load_labels("LOCKBOX")
