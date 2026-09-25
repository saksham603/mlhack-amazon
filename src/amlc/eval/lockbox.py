"""LOCKBOX scorer. Public API: score_once() only, returning one aggregate score, once, ever.

The single permitted LOCKBOX read is enforced in access.py. Singletons must be derived from the
open split plus the links (an S1 with no link is a singleton), so one read is sufficient.
"""
from amlc.foundation import access

__all__ = ["score_once"]


def _load_lockbox_labels():
    return access.load_labels("LOCKBOX")


def score_once(predictions):
    raise NotImplementedError("W9 not reached: the lockbox is opened exactly once, at the end.")
