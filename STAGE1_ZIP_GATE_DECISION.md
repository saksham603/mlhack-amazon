# Stage 1 ZIP gate — CG-14 decision log (2026-09-26)

## What the gate used to be

Stage 1 prompt §1 baseline: "ZIP-like (5-digit) token present: train S1 145,601 / S2 327,300 /
S3 343,630 ... PIN-like (6-digit): train S1 1,656 / S2 41,851 / S3 42,140", with the gate defined
as "addr_zip non-null rate must be within 0.5pp of §1's zip5+pin6 rates per table". That §1 count
could not be reproduced from scratch and, on inspection, was itself inflated by house-number
tokens, not missed ZIPs (see `src/amlc/cleaning/zip_recount.py`, CG-9 style independent recount:
it confirms `extract_zip()` matches its OWN definition exactly on all 6 tables — the code is
internally consistent with its spec).

## What that consistency check missed

The spec itself is defective: `extract_zip()` never excludes the token already claimed by
`extract_house_number()`. When an address has no genuine postal code anywhere in its free text —
common in this dataset, both US and India — the house-number token (frequently 5 digits) gets
returned as the "ZIP" purely by digit-length coincidence.

## Evidence (`src/amlc/cleaning/zip_gate_evidence.py`, committed alongside this doc)

**1. Collision rate** — non-empty `addr_zip` literally equal to `addr_house_number`:

| Table | rows | addr_zip non-empty | == house number | rate |
|---|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 146,134 | 144,138 | 98.63% |
| train_source2 | 5,034,616 | 317,095 | 212,203 | 66.92% |
| train_source3 | 5,285,603 | 333,305 | 227,191 | 68.16% |
| test_source1 | 1,732,544 | 74,672 | 72,539 | 97.14% |
| test_source2 | 4,887,273 | 210,262 | 134,555 | 63.99% |
| test_source3 | 5,082,316 | 219,498 | 143,810 | 65.52% |

**2. Zero-padding samples** — the remaining non-collision non-empty rows are overwhelmingly the
same phenomenon with a cosmetic difference: `addr_zip` is the house number read with leading
zeros, e.g. `"03001, amrapali grand, ..."` → zip="03001", hn="3001" (equal as integers, only
differ as strings). Confirmed present in every table (see script output for full samples).

**3. Recoverable trailing ZIP** — among the collision rows only, checked for the one shape that
could plausibly hide a real, currently-lost ZIP: last `addr_component` is a genuine 2-letter US
state abbreviation AND the second-to-last component is a standalone 5-digit token distinct from
the house number.

| Table | collision rows checked | recoverable |
|---|---:|---:|
| train_source1 | 144,138 | 1 (`"10073 manzanilla avenue, 75127, tx"`) |
| train_source2 | 212,203 | 1 (`"10075 manzanilla ave, 75127, tx"`) |
| train_source3 | 227,191 | 0 |
| test_source1 | 72,539 | 0 |
| test_source2 | 134,555 | 0 |
| test_source3 | 143,810 | 0 |

**Total: 2 recoverable rows out of ~24.2M rows across all 6 tables.**

An earlier, looser pattern (`\b[a-z]{2},? \d{5}(-\d{4})?$` run directly over `addr_clean`)
reported 27 hits on train_source2. On inspection, all 27 were false positives — the "state" match
was actually stray 2-letter tokens like "no", "ug", "fn" (Indian door/plot-number text), not a US
state code — except one row (`"289 morning fog ln, 75182"`) whose ZIP `addr_zip` already extracts
correctly. That pattern is not used in the committed evidence script; it is recorded here only so
the discrepancy is documented rather than silently dropped.

## Conclusion

`addr_zip` in silver v1 is **not a usable postal-code signal**. It is, in the overwhelming
majority of non-empty cases, a duplicate of `addr_house_number` (either literally or after
stripping leading zeros). Genuinely recoverable real ZIPs number 2 rows total across the entire
dataset — not worth a silver rebuild to fix.

## Decisions (user, 2026-09-26, CG-14)

1. **No silver v2. Silver v1 stays as published. `addresses.py` is not changed.** The extractor's
   behavior is now understood and documented, not treated as a bug to patch under deadline
   pressure.
2. **`addr_zip` is marked "not a postal code, do not use downstream"** wherever it appears in
   docs/manifests referencing it as such.
3. **The ZIP gate itself is replaced**, not loosened: the new gate reference is (a) the
   independent recount in `zip_recount.py` (confirms the extractor matches its own token
   definition exactly — passes on all 6 tables) plus (b) this evidence report (confirms that
   definition, while internally consistent, does not yield a usable ZIP signal). The old §1
   baseline reference is retired.
4. **Blocking impact — no re-run.** `candidates.py`'s ZIP key and `v2.py`'s `address_keys()`
   "zip" kind both consume `addr_zip` as-is; since it is effectively a second house-number key,
   not a postal-code key, budget=150 blocking was evaluated *with* this key already active. The
   measured recall ceiling (India 79.15%, US 94.01%, from the earlier `D_plus_budget_150` run) is
   locked as-is; no separate with/without-zip-key comparison was run, per triage decision under
   the 2026-09-27 04:00 IST deadline. If time permits later, that comparison would show at most
   the difference between a working house-number key and a redundant one — likely small, since
   `house` is already the dominant blocking key kind by candidate count.

## LLC gate (closed same session, unrelated to ZIP)

`llc` count in `name_clean` OR `name_legal_form`, computed with `collapse_abbreviations()` applied
first (so "p.l.l.c." collapses to "pllc" before the standalone-`llc` check, avoiding
PLLC/LCSW false positives): exact match between the raw recount and the existing post-C3 combined
count on all 6 tables. See `src/amlc/cleaning/llc_gate_evidence.py`.

| Table | raw_llc (collapsed, excl. PLLC) | combined post-C3 |
|---|---:|---:|
| train_source1 | 377,579 | 377,579 |
| train_source2 | 594,080 | 594,080 |
| train_source3 | 633,596 | 633,596 |
| test_source1 | 189,062 | 189,062 |
| test_source2 | 355,882 | 355,882 |
| test_source3 | 376,810 | 376,810 |
