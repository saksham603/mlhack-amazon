"""Independent recount of C4's ZIP gate (CG-9 style, mirrors recount.py). Imports nothing from
amlc.cleaning.addresses: a fresh, separately-written implementation of the SAME definition C4 uses
(component split on comma, then on whitespace, exact 5-digit or 6-digit token match), so this check
catches a divergence between the spec and the code, not a divergence between two different ideas of
what a ZIP is.

Decision (user, 2026-09-26, CG-14): replaces the old gate reference, the ad-hoc §1 whole-table
"ZIP-like token" count. That count could not be reproduced and, when the same digit-length categories
were investigated by hand on train S2/S3, turned out to be inflated by 5-digit US house numbers
(e.g. "10333- broadmoor ct", "#16002 sycamore maple lane") -- not missed ZIPs. See
STAGE1_ZIP_GATE_DECISION.md for the full evidence. The extractor itself was not changed.
"""
import re

WHITE_SPACE = "".join(map(chr, [
    *range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
    0x2028, 0x2029, 0x202F, 0x205F, 0x3000]))
_ZIP5 = re.compile(r"^\d{5}$")
_PIN6 = re.compile(r"^\d{6}$")


def recount_one(addr: str) -> bool:
    """True if the address has any comma-separated component containing a whitespace-separated
    token that is exactly 5 or 6 digits -- the same definition as addresses.extract_zip, written
    independently (plain string splits, no shared regex object, no shared helper).
    """
    for comp in addr.split(","):
        comp = comp.strip(WHITE_SPACE)
        for tok in comp.split():
            tok = tok.strip(WHITE_SPACE)
            if _ZIP5.match(tok) or _PIN6.match(tok):
                return True
    return False


def recount_table(addresses: list[str]) -> dict:
    n = len(addresses)
    hits = sum(recount_one(a or "") for a in addresses)
    return {"rows": n, "zip_like_rows": hits, "rate": hits / n if n else None}
