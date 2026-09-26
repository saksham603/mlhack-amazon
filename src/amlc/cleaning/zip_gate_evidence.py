r"""Evidence for the CG-14 ZIP gate decision (2026-09-26): addr_zip is not a postal code.

Reads ONLY published silver/v1 (read-only, never modified here). Produces, per table:
  (1) collision rate: non-empty addr_zip that is literally equal to addr_house_number.
  (2) zero-padding samples: non-empty addr_zip that is NOT equal to addr_house_number as a
      string, but IS equal once both are read as integers (i.e. the "zip" is just the house
      number with leading zeros, e.g. "00103 36th ave west" -> zip="00103", hn="103").
  (3) recoverable_trailing_zip: among the COLLISION rows only (addr_zip == addr_house_number,
      i.e. rows where addr_zip is currently useless), rows whose last addr_component is a
      two-letter US state postal abbreviation AND whose second-to-last component is a standalone
      5-digit token distinct from the house number -- the one shape that could hold a real,
      currently-unrecovered trailing ZIP. An earlier looser pattern (\b[a-z]{2},? \d{5}(-\d{4})?$
      run over addr_clean directly) reported 27 hits on train_source2; on inspection every one of
      those 27 was a false positive (the "state" was actually "no", "ug", "fn", ... -- Indian
      door/plot-number text, not a US state code) except one row that addr_zip already extracts
      correctly. That looser pattern is not used here; see STAGE1_ZIP_GATE_DECISION.md.

Writes nothing. Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.cleaning.zip_gate_evidence
"""
import json

import polars as pl

from amlc.foundation import access

SILVER_DIR = access.DATA / "silver" / "v1"
TABLES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]

# US state/territory 2-letter postal abbreviations (USPS), written here from scratch --
# not imported from any cleaning module.
US_STATE_ABBREVS = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "dc", "pr", "vi", "gu", "as", "mp",
}


def _load(ds: str, src: int) -> pl.DataFrame:
    return pl.read_parquet(
        SILVER_DIR / f"{ds}_source{src}.parquet",
        columns=["gid", "addr_clean", "addr_components", "addr_house_number", "addr_zip"],
    )


def collision_stats(df: pl.DataFrame) -> dict:
    nonempty = df.filter(pl.col("addr_zip") != "")
    n_nonempty = nonempty.height
    collide = (nonempty["addr_zip"] == nonempty["addr_house_number"]).sum()
    return {
        "rows": df.height,
        "addr_zip_nonempty": n_nonempty,
        "collision_with_house_number": int(collide),
        "collision_rate": collide / n_nonempty if n_nonempty else None,
    }


def zero_padding_samples(df: pl.DataFrame, n: int = 10) -> list[dict]:
    nonempty = df.filter(pl.col("addr_zip") != "").filter(pl.col("addr_zip") != pl.col("addr_house_number"))
    out = []
    for row in nonempty.iter_rows(named=True):
        z, h = row["addr_zip"], row["addr_house_number"]
        if z.isdigit() and h.isdigit() and int(z) == int(h) and z != h:
            out.append({"gid": row["gid"], "addr_clean": row["addr_clean"], "addr_zip": z, "addr_house_number": h})
            if len(out) >= n:
                break
    return out


def recoverable_trailing_zip(df: pl.DataFrame, n_examples: int = 10) -> dict:
    collisions = df.filter((pl.col("addr_zip") != "") & (pl.col("addr_zip") == pl.col("addr_house_number")))
    n = 0
    examples = []
    for row in collisions.iter_rows(named=True):
        comps = row["addr_components"]
        if len(comps) < 2:
            continue
        last = comps[-1].strip().lower()
        if last not in US_STATE_ABBREVS:
            continue
        zc = comps[-2].strip()
        if zc.isdigit() and len(zc) == 5 and zc != row["addr_house_number"]:
            n += 1
            if len(examples) < n_examples:
                examples.append({"gid": row["gid"], "addr_clean": row["addr_clean"]})
    return {"rows_checked": collisions.height, "count": n, "examples": examples}


def main() -> None:
    report = {}
    for ds, src in TABLES:
        df = _load(ds, src)
        entry = {
            "collision": collision_stats(df),
            "zero_padding_samples": zero_padding_samples(df),
            "recoverable_trailing_zip": recoverable_trailing_zip(df),
        }
        report[f"{ds}_source{src}"] = entry
        print(f"=== {ds}_source{src} ===")
        print(json.dumps(entry, indent=1))
    return report


if __name__ == "__main__":
    main()
