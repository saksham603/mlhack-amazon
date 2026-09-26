r"""Evidence for the CG-14 LLC gate decision (2026-09-26): count llc in name_clean OR
name_legal_form (post-C3), closed after validating it against a raw count.

The raw count applies collapse_abbreviations() (the same C3 function, imported directly since
this checks the SILVER OUTPUT against an intermediate step of its own pipeline, not an
independent reimplementation) to the untouched bronze business_name, then checks for a standalone
"llc" token. Collapsing first is required: it turns "p.l.l.c." into "pllc" (no internal dots), so
a naive \bllc\b on raw dotted text would otherwise match inside "p.l.l.c." and "l.c.s.w." (an
unrelated professional credential) and inflate the raw count with non-LLC matches.

Reads bronze (raw) and published silver/v1 (read-only, neither modified here).
Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.cleaning.llc_gate_evidence
"""
import polars as pl

from amlc.cleaning.names import collapse_abbreviations
from amlc.foundation import access

SILVER_DIR = access.DATA / "silver" / "v1"
TABLES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]


def table_counts(ds: str, src: int) -> dict:
    raw = access.load_bronze(ds, src, columns=["gid", "business_name"])
    collapsed = raw["business_name"].map_elements(collapse_abbreviations, return_dtype=pl.String)
    raw_llc = int(collapsed.str.contains(r"\bllc\b").sum())

    silver = pl.read_parquet(SILVER_DIR / f"{ds}_source{src}.parquet", columns=["name_clean", "name_legal_form"])
    combined = int((silver["name_clean"].str.contains(r"\bllc\b")
                    | silver["name_legal_form"].str.contains(r"\bllc\b")).sum())

    return {"table": f"{ds}_source{src}", "raw_llc": raw_llc, "combined_post_c3": combined, "gate_pass": raw_llc == combined}


def main() -> None:
    for ds, src in TABLES:
        r = table_counts(ds, src)
        print(f"{r['table']}: raw_llc(collapsed, excl. pllc)={r['raw_llc']:,}  "
              f"combined_post_c3={r['combined_post_c3']:,}  gate_pass={r['gate_pass']}")


if __name__ == "__main__":
    main()
