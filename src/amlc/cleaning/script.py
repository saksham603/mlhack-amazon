"""C2: per-token script classification and FIT-only Indic->Latin dictionary (Stage 1 prompt rev 5, §3 C2).

Runs on C1 output. Dictionary construction touches labels (FIT only, G-L1) — the only label-touching
step in this stage (CG-2). Everything else here is label-free.
"""
import json
from collections import Counter, defaultdict

import polars as pl

from amlc.cleaning.tokens import LATIN_ALPHA, LEGAL_FORMS, SCRIPT_RANGES

MIN_SUPPORT = 5          # D5
VARIANT_SHARE_FLOOR = 0.10  # D5a


def _is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in LATIN_ALPHA)


def _script_of_token(tok: str) -> str:
    for ch in tok:
        if ch.isalpha():
            if _is_latin_char(ch):
                return "latin"
            cp = ord(ch)
            for name, lo, hi in SCRIPT_RANGES:
                if lo <= cp <= hi:
                    return name
            return "other"
    return "other"  # no alphabetic character (pure digits/punct)


def classify_tokens(name: str) -> list[str]:
    """One script label per whitespace-split token."""
    return [_script_of_token(t) for t in name.split()]


def _legal_canon(country: str) -> dict:
    return LEGAL_FORMS.get(country, {})


def _all_indic(tok: str) -> bool:
    # a token with NO alphabetic character (e.g. "&", "***", "-") is not Indic: all() over an empty
    # generator is vacuously True, which silently counted punctuation as "Indic" until this fix.
    alpha = [ch for ch in tok if ch.isalpha()]
    return bool(alpha) and all(_script_of_token_char(ch) for ch in alpha)


def _script_of_token_char(ch: str) -> bool:
    if _is_latin_char(ch):
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for _, lo, hi in SCRIPT_RANGES)


def build_dictionary(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: one row per FIT true pair with equal token count, columns s1_gid, s1_name, s23_name,
    country (label-free after this call is made; the label linkage itself is what makes this FIT-only).
    Returns one row per Indic token: latin (list[str] candidates kept), support (dict candidate->support),
    total_support, is_legal_form_conflict.
    """
    # support[token][candidate] = set of s1_gid (independent entities), not raw pair count
    support: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for s1_gid, s1_name, s23_name, country in pairs.select(
            "s1_gid", "s1_name", "s23_name", "country").iter_rows():
        s1_tok = s1_name.split()
        s23_tok = s23_name.split()
        if len(s1_tok) != len(s23_tok):
            continue
        for a, b in zip(s23_tok, s1_tok):
            if _all_indic(a) and b:
                support[a][b].add(s1_gid)
    rows = []
    for indic, cands in support.items():
        counts = {c: len(s) for c, s in cands.items()}
        kept = {c: n for c, n in counts.items() if n >= MIN_SUPPORT}
        if not kept:
            continue
        total = sum(kept.values())
        rows.append({"indic_token": indic, "candidates": kept, "total_support": total})
    if not rows:
        return pl.DataFrame(schema={"indic_token": pl.String, "latin": pl.List(pl.String),
                                     "is_legal_form": pl.Boolean, "total_support": pl.UInt32})
    out = []
    legal_variants = set()  # every raw spelling that maps to a legal form (keys AND values)
    for forms in LEGAL_FORMS.values():
        legal_variants.update(forms.keys())
        legal_variants.update(forms.values())
    for r in rows:
        cands = r["candidates"]
        if len(cands) == 1:
            latin, is_legal = list(cands.keys()), False
        elif set(cands.keys()) <= legal_variants:
            # legal-form variants -> canonical form of the highest-support candidate
            winner = max(cands, key=cands.get)
            canon = next((forms[winner] for forms in LEGAL_FORMS.values() if winner in forms), winner)
            latin = [canon]
            is_legal = True
        else:
            floor = VARIANT_SHARE_FLOOR * r["total_support"]
            latin = sorted([c for c, n in cands.items() if n >= floor], key=lambda c: -cands[c])
            is_legal = False
        out.append({"indic_token": r["indic_token"], "latin": latin, "is_legal_form": is_legal,
                    "total_support": r["total_support"]})
    return pl.DataFrame(out, schema={"indic_token": pl.String, "latin": pl.List(pl.String),
                                      "is_legal_form": pl.Boolean, "total_support": pl.UInt32})


def _translate_name(name: str, lookup: dict) -> tuple[str, bool, list]:
    out_tok, untranslated, variant_rows = [], False, []
    for tok in name.split():
        if _all_indic(tok):
            hit = lookup.get(tok)
            if hit is None:
                out_tok.append(tok)
                untranslated = True
            else:
                latin, is_legal = hit
                out_tok.append(latin[0])
                if len(latin) > 1:
                    variant_rows.append((tok, latin))
        else:
            out_tok.append(tok)
    return " ".join(out_tok), untranslated, variant_rows


def apply_dictionary(names: pl.Series, dictionary: pl.DataFrame) -> pl.DataFrame:
    """Token-level (map_distinct-style) application: clean each distinct name once (§3 C5 rule)."""
    lookup = {r["indic_token"]: (r["latin"], r["is_legal_form"]) for r in dictionary.to_dicts()}
    uniq = names.unique().to_list()
    translated, untrans, variants = [], [], []
    for n in uniq:
        t, u, v = _translate_name(n, lookup)
        translated.append(t)
        untrans.append(u)
        # JSON string, not a Python object: pl.Object cannot be written to Parquet (found running the
        # dry run). [[indic_token, [latin_variants]], ...] or null when there are no variant tokens.
        variants.append(json.dumps(v, ensure_ascii=False) if v else None)
    tbl = pl.DataFrame({"k": pl.Series(uniq, dtype=pl.String), "name_translit": translated,
                        "name_indic_untranslated": untrans,
                        "name_token_variants": pl.Series(variants, dtype=pl.String)})
    out = pl.DataFrame({"k": names}).join(tbl, on="k", how="left", validate="m:1")
    if out.height != names.len() or out["name_translit"].null_count():
        raise AssertionError("apply_dictionary lost or added rows")
    return out.drop("k")


def coverage(names: pl.Series, dictionary: pl.DataFrame) -> dict:
    """Label-free: share of Indic token INSTANCES covered by the dictionary (§1's coverage metric)."""
    known = set(dictionary["indic_token"].to_list())
    total = covered = 0
    for n in names.to_list():
        for tok in n.split():
            if _all_indic(tok):
                total += 1
                if tok in known:
                    covered += 1
    return {"instances_total": total, "instances_covered": covered,
            "coverage": covered / total if total else None}
