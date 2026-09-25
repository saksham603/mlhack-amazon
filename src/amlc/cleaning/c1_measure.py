"""C1-M: label-free measurements taken BEFORE C1 is written (Stage 1 prompt rev 3, C1-M).

Nothing here is a pass/fail gate. Every number is input for a decision the user makes.
Reads bronze only through access.load_bronze(). Writes data/interim/c1/measure/ (refuses overwrite).
"""
import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter

import polars as pl

from amlc.foundation.access import DATA, ROOT, load_bronze

# v1 = data/interim/c1/measure: 2 deprecated Polars calls, no ASCII-control count.
# v2: examples picked in non-deterministic group_by order; no commit recorded. Counts identical to v1.
MEASURE_VERSION = "measure_v3"
OUT = DATA / "interim" / "c1" / MEASURE_VERSION
TABLES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]
STAGE0_NULL_TOKENS = ["null", "NULL", "<NULL>", "None", "N/A", "n/a", "na", "NA", ""]
CASEFOLDED_NULL_WORDS = {"null", "<null>", "none", "n/a", "na"}
SPECIAL_LATIN = "œŒæÆøØßẞłŁđĐþÞ"
N_EXAMPLES = 20


def measure_table(ds: str, src: int) -> dict:
    df = load_bronze(ds, src, columns=["gid", "country", "business_name", "business_address"])
    out = {"table": f"{ds}_source{src}", "rows": df.height}

    # (a) null words in address components beyond Stage 0's case-sensitive list
    # str.split never yields an empty list ("" -> [""]), so empty_as_null has nothing to act on; passed explicitly (CG-18)
    comps = (df.select("gid", "country", pl.col("business_address").str.split(",").alias("c"))
             .explode("c", empty_as_null=True).with_columns(pl.col("c").str.strip_chars().alias("c")))
    variant = comps.filter(pl.col("c").str.to_lowercase().is_in(list(CASEFOLDED_NULL_WORDS))
                           & ~pl.col("c").is_in(STAGE0_NULL_TOKENS))
    out["a_case_variant_null_rows"] = variant["gid"].n_unique()
    out["a_case_variant_null_tokens"] = dict(Counter(variant["c"].to_list()).most_common())
    ex_gids = variant.select("gid").unique(maintain_order=True).head(N_EXAMPLES)
    out["a_examples"] = [dict(zip(["gid", "country", "business_address"], r)) for r in
                         df.join(ex_gids, on="gid", how="semi").select("gid", "country", "business_address").iter_rows()]
    # components that become a null word only after NFKC + casefold (e.g. full-width 'ＮＡ')
    nonascii_comps = comps.filter(pl.col("c").str.contains(r"[^\x00-\x7F]"))["c"].unique().sort().to_list()
    after_norm = [c for c in nonascii_comps
                  if unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", c).casefold()).strip() in CASEFOLDED_NULL_WORDS]
    out["a_null_after_nfkc_distinct"] = after_norm[:N_EXAMPLES]
    out["a_null_after_nfkc_rows"] = comps.filter(pl.col("c").is_in(after_norm))["gid"].n_unique() if after_norm else 0

    # (b) name field that is a null token or empty
    name = df["business_name"]
    out["b_name_exact_null_token_rows"] = dict(Counter(name.filter(name.is_in(STAGE0_NULL_TOKENS)).to_list()))
    stripped = name.str.strip_chars()
    out["b_name_stripped_null_or_empty_rows"] = int((stripped.is_in(STAGE0_NULL_TOKENS)
                                                     | stripped.str.to_lowercase().is_in(list(CASEFOLDED_NULL_WORDS))).sum())

    # (g) ASCII control characters other than whitespace (D-C1e; a blind spot in measure v1)
    for col, key in (("business_name", "name"), ("business_address", "addr")):
        ctrl = Counter()
        hits = df.filter(pl.col(col).str.contains("[\x00-\x08\x0e-\x1f\x7f]")).group_by(col).len().sort(col)
        for v, n in hits.iter_rows():
            for ch in v:
                if ord(ch) <= 0x08 or 0x0E <= ord(ch) <= 0x1F or ord(ch) == 0x7F:
                    ctrl[f"U+{ord(ch):04X}"] += n
        out[f"g_{key}_ascii_control_non_ws"] = dict(ctrl.most_common())
        out[f"g_{key}_ascii_control_rows"] = int(hits["len"].sum()) if hits.height else 0

    # (c)-(f) character-level: only non-ASCII strings can contain any of these (ASCII is NFKC-stable)
    for col, key in (("business_name", "name"), ("business_address", "addr")):
        # grouped by (value, country): a value seen in two countries is counted in each (v1/v2 gave all rows
        # to one arbitrary country via first()); sorted so example selection is reproducible
        vc = (df.filter(pl.col(col).str.contains(r"[^\x00-\x7F]"))
              .group_by(col, "country").agg(pl.len().alias("n")).sort(col, "country")
              .select(col, "n", "country"))
        cf, nfkc_changed, special, digits = Counter(), Counter(), Counter(), Counter()
        zwsp_between_letters = zwsp_other = 0
        nfkc_ex, zwsp_ex, digit_ex = {}, [], {}
        special_by_country = Counter()
        for s, n, ctry in vc.iter_rows():
            for i, ch in enumerate(s):
                cp = ord(ch)
                if unicodedata.category(ch) == "Cf" or 0x80 <= cp <= 0x9F:
                    cf[f"U+{cp:04X} {unicodedata.name(ch, '?')}"] += n
                    if cp == 0x200B:
                        prev_ok = i > 0 and s[i - 1].isalnum()
                        next_ok = i + 1 < len(s) and s[i + 1].isalnum()
                        if prev_ok and next_ok:
                            zwsp_between_letters += n
                            if len(zwsp_ex) < N_EXAMPLES:
                                zwsp_ex.append(s)
                        else:
                            zwsp_other += n
                if cp > 0x7F:
                    k = unicodedata.normalize("NFKC", ch)
                    if k != ch:
                        nfkc_changed[f"U+{cp:04X} {unicodedata.name(ch, '?')} -> {k!r}"] += n
                        nfkc_ex.setdefault(ch, s)
                if ch in SPECIAL_LATIN:
                    special[ch] += n
                    special_by_country[f"{ch} {ctry}"] += n
                if unicodedata.category(ch) == "Nd" and not ("0" <= ch <= "9"):
                    block = unicodedata.name(ch, "?").split(" DIGIT")[0]
                    digits[block] += n
                    digit_ex.setdefault(block, s)
        out[f"c_{key}_invisible_control"] = dict(cf.most_common())
        out[f"c_{key}_zwsp_between_alnum"] = zwsp_between_letters
        out[f"c_{key}_zwsp_other"] = zwsp_other
        out[f"c_{key}_zwsp_examples"] = zwsp_ex
        out[f"d_{key}_nfkc_changed_top50"] = dict(nfkc_changed.most_common(50))
        out[f"d_{key}_nfkc_examples"] = {f"U+{ord(c):04X}": s for c, s in list(nfkc_ex.items())[:50]}
        out[f"e_{key}_special_latin"] = dict(special.most_common())
        out[f"e_{key}_special_latin_by_country"] = dict(special_by_country.most_common())
        out[f"f_{key}_nonascii_digits"] = dict(digits.most_common())
        out[f"f_{key}_nonascii_digit_examples"] = digit_ex
    return out


def main() -> int:
    if OUT.exists():
        print(f"REFUSING: {OUT} exists. Measurements are never overwritten.")
        return 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        print(f"REFUSING: uncommitted changes in src/ or tests/ (G-P5). Commit first.\n{dirty}")
        return 1
    t0 = time.time()
    results = []
    for ds, src in TABLES:
        t = time.time()
        results.append(measure_table(ds, src))
        print(f"{ds} S{src}: {time.time() - t:.0f}s", flush=True)
    OUT.mkdir(parents=True)
    meta = {"version": MEASURE_VERSION, "git_commit": commit, "python": sys.version,
            "unicode": unicodedata.unidata_version, "polars": pl.__version__, "runtime_s": round(time.time() - t0, 1)}
    (OUT / "c1m_measurements.json").write_text(
        json.dumps({"meta": meta, "tables": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {OUT / 'c1m_measurements.json'} in {meta['runtime_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
