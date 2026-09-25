# AMLC 2026 — Full Project Context (portable summary)

Paste this into a new chat to restore full context without re-deriving anything.

---

## What this project is

Amazon ML Challenge 2026: **Business Entity Resolution.** For each Source-1 (S1) business record, find all matching records in Source-2 (S2) and Source-3 (S3). Metric: **macro F_0.5 per S1 entity** (precision weighted 2×; a singleton correctly predicted empty scores 1.0, any prediction on it scores 0.0). Data covers US + India in training; test adds **France**, which has no labels.

Rules: no external lookups/geocoding/APIs. Final model must be MIT/Apache-2.0, ≤8B params.

Kaggle workspace: `divyanshsingh0967/mlchallenge-2026` (private dataset), profile `kaggle.com/divyanshsingh0967`.

## Where everything lives

| What | Path |
|---|---|
| **Project code (own git repo)** | `C:\Users\suremdra singh\amlc2026\` |
| Full workflow plan (objectives, mistakes register, W0–W12 steps, guardrails, council review) | `C:\Users\suremdra singh\.claude\plans\amlc2026-workflow-plan.md` |
| Stage 0 data-foundation spec (the exact prompt executed) | `C:\Users\suremdra singh\.claude\plans\amlc2026-stage0-data-foundation-prompt.md` |
| **Stage 1 cleaning spec (written, NOT executed; D5–D7 await explicit confirmation)** | `C:\Users\suremdra singh\.claude\plans\amlc2026-stage1-cleaning-prompt.md`. Order is C1 text → C2 script/transliteration → C3 names → C4 addresses → C5 stats; transliteration must run before legal-form extraction. The Indic vocabulary is closed: the same 1,537 tokens appear in train and test. |
| Old plan file (now corrupted on disk — contains only "c", do not use) | `expressive-orbiting-fiddle.md` — superseded, ignore |
| Old Gemini-built pipeline (read-only historical record; **rule: never reopen it**, see below) | `C:\Users\suremdra singh\kaggle-review\` |
| Original competition zip (verified source of truth) | `C:\Users\suremdra singh\Downloads\6ab10eb3b23ba_student_resource.zip` |
| Submission template (unfilled) | `amlc2026\submission\Documentation_template.md` |

## Standing rules (never violate these)

1. **No code or changes without an explicit "go" from the user.** A plan being approved is NOT a go.
2. **Never reopen the old Gemini pipeline files again.** All 12 of its bugs are already catalogued in the workflow plan in plain language. Stage 1 (cleaning) onward must be written purely from: the README spec, the measured data profile, and the workflow plan's guardrails — not from the old code's structure, even to "just check" something.
3. **No claim without a measurement.** Every number reported must come from a logged run. Anything unmeasured is labeled NOT VERIFIED.
4. FIT/VALIDATION/LOCKBOX split (see below) is frozen — content hash is verified on every load, LOCKBOX can be read exactly once ever.

## What the old (Gemini) pipeline got wrong — 12 measured mistakes (M1–M12)

Full detail in the workflow plan §2. Highlights:
- **Never used training labels at all** — thresholds were pure guesses, F_0.5 was never computed.
- **Worst bug:** treated any accented character (é, à, ç) as "a different script" (regex `[^\x00-\x7F]`), which routed those records into a weak fallback that matches on generic shared address words. Measured precision on that path: **0.11 India / 0.42 US.** France is heavily exposed (16–38% of its records have accents). This causes **false merges**, not missed matches (an earlier report of mine said "recall" — that was wrong; corrected in the plan).
- Dotted abbreviations broken (`L.L.C.` → `l l c`), legal forms became top blocking keys, ZIP/PIN codes deleted, country-blind street-word expansion, hardcoded country list, many-to-many output (ignored that ground truth is a strict 1-S2/S3-to-1-S1 partition), candidate caps never measured for recall loss, and "validator PASS" (format-only) was mistaken for a quality signal.
- Also debunked several of Gemini's own claims in its self-report: it said the France bug hurt *recall* (wrong — precision), claimed 5–10 min runtime (its own later run took 37.7 min), and claimed <0.2% loss from its candidate cap (never measured).

## Key facts about the data (all measured directly, not assumed)

- train S1: 2,206,821 rows (US 1,323,633 / India 883,188). S1 is clean/canonical: no empty fields, no null fields, Title Case, no Indic script. It does have a handful of placeholder address **components** (`N/A`, `None`, or empty from `,,`): exactly 8 train S1 rows and 29 test S1 rows (corrected 2026-09-26; the earlier "fully clean" claim was never measured).
- train S2: 5,034,616 / S3: 5,285,603. All noise lives in S2/S3.
- test S1: 1,732,544 (India 809,986 / US 663,106 / **France 259,452**). test S2: 4,887,273 / S3: 5,082,316.
- **Ground truth is a strict partition:** all 7,638,365 links, 0 S2/S3 records linked to more than one S1. 5.58% of S1 are singletons (median 3 matches).
- Test has more S2/S3 per S1 than train in **every** country (not just France): US 4.674→5.756, India 4.680→5.824, France test-only 5.531.
- S2/S3 noise: ALL-CAPS names 15–21%, Indic-script names ~13–23% of India, injected accents 6–24% depending on country, empty/null addresses ~3%, double spaces ~10%. 24% of India true pairs share **zero** name tokens; 8% of US true pairs likewise.
- No cross-country true matches (0/69,157 sampled). No ID string shared between train and test.

## Decisions already made (don't re-ask these)

- **D1 (data provenance):** user's Kaggle upload verified **byte-identical** (SHA-256) to the original competition zip. Confirmed source of truth.
- **D2 (storage):** Parquet files as canonical store + DuckDB views on top for SQL queries (chosen over DuckDB-native tables — Parquet is portable, hashable per-file, and everything downstream is Polars/dataframe-native, not SQL-native).
- **D3 (split):** 70/20/10 FIT/VALIDATION/LOCKBOX, seed = `20260925`, stratified by country × match-count bucket (0/1-2/3-4/5+). Frozen — never change the seed.
- **D4 (repo structure):** `amlc2026\` is its own independent git repo (not nested in the home-folder repo, which has an unrelated exposed private key staged in it — separate issue, not fixed by us).
- **Indic transliteration:** dictionary learned from FIT pairs only. **No transliteration library** (user's explicit choice).
- **Label-free IDF on test S2/S3:** allowed (needed for France, which has no labels). Documented as transductive, not a leak.
- **Model:** LightGBM (MIT license), not the old rule-based heuristics.

## Foundation v2: ✅ COMPLETE 2026-09-26 (supersedes the v1 split and access; v1 kept read-only)

A review of Stage 0 found errors, now fixed in v2 (built from commit f505e5f; all 31 gates passed; 29/29 tests pass):
- **Stress flag:** v1 drew it from all splits (84,699 VALIDATION and 42,222 LOCKBOX flagged). v2 draws it **from FIT only** (US 248,831, India 173,510). The achieved ratios equal the test ratios, and the check is proven able to fail.
- **Label exposure:** v1's `load_split()` exposed `n_total`, `is_singleton` and `bucket` for VALIDATION/LOCKBOX. The v2 split has only `s1_gid, s1_id, country, split, stress_hidden`. Counts moved behind `access.load_match_counts(split)`, which is gated like the labels.
- **Scorer and lockbox modules expose scores only**, never raw labels. `leakscan` (run in tests) flags any production code that reads label files or requests VALIDATION/LOCKBOX data.
- **Manifest:** `MANIFEST_v2.json` computes every number, records every gate, and lists corrected expectations. `access.py` hash-verifies every file it serves against it.
- **Unchanged:** raw, bronze, gt_raw, gt_links, and **FIT/VALIDATION/LOCKBOX membership** (0 differences from v1).
- **Corrected expectation:** S1 does have placeholder address components (exactly 8 train rows and 29 test rows). The assumed "0" was never measured.

## Stage 0: Data Foundation v1 (historical; split/access superseded by v2 above)

Built exactly per the Stage 0 spec. Summary:
- Verified raw files against the **original zip** (not just Kaggle) — all 7 SHA-256 hashes match exactly.
- Converted to lossless Parquet ("bronze" layer); proved losslessness by round-tripping back to TSV and hash-matching the original bytes exactly.
- Built label tables (`gt_links`, `s1_match_counts`) with full integrity checks (0 unresolved links, 0 double-linked records, 0 cross-country links).
- Built the frozen FIT/VALIDATION/LOCKBOX split — every stratum within 0.5pp of 70/20/10, cross-split distributions match to <0.001pp.
- Built the stress-test flag (hides ~18.8% of US FIT / ~19.65% of India FIT S1s so validation's S2/S3-per-S1 ratio matches test's, since test has more distractors).
- Built a profile snapshot; **cross-checked 6 numbers against my earlier manual analysis — all matched within 0.5pp** (this caught nothing wrong, confirming both measurements agree).
- Built `amlc.foundation.access` — the only sanctioned way later code may read data. Enforces: FIT open to anyone; VALIDATION labels only loadable from `amlc.eval.scorer`; LOCKBOX labels only loadable from `amlc.eval.lockbox`, and **only once ever** (logged, refuses a second read). Proved with 8 passing pytest tests, including a tampered-file-rejection test.
- Built `MANIFEST_v1.json` recording every hash, count, and parameter; froze all data files read-only.
- **F11 determinism proof:** rebuilt everything from raw bytes into a separate temp dir, compared content hashes — all 7 tables matched exactly. (Caught and fixed one real bug along the way: my first verification attempt had a broken hash function computing a Python generator's memory address instead of the actual row content — investigated properly rather than assuming the pipeline was non-deterministic; the pipeline itself was fine.)

Project layout now:
```
amlc2026/
  src/amlc/foundation/   ← Stage 0 code (ingest, validate, split, profile, access-control)
  src/amlc/eval/         ← stub modules only; real F_0.5 scorer is Stage/step W0, not built yet
  tests/foundation/      ← 8 passing tests for the access API
  data/                  ← 3.6GB Parquet+raw, gitignored, frozen read-only, NOT for submission
  submission/            ← Documentation_template.md (unfilled placeholder)
  requirements.lock      ← polars 1.44.2, pyarrow 25.0.1, duckdb 1.5.5, pytest 9.1.1 (pinned)
```

## The workflow plan (what comes after Stage 0) — not started yet

Full detail in `amlc2026-workflow-plan.md`. Summary of steps W0–W12:

| Step | What |
|---|---|
| W0 | F_0.5 scorer + baselines (all-empty, old matcher) on VALIDATION |
| W1 | Candidate generation (blocking); measure recall ceiling + candidate count on FULL pool |
| W2 | ~15 pair features (name/address similarity, house number, ZIP, legal form, source S2-vs-S3, co-tenant distinctiveness); adversarial-validation check that features look the same train vs test |
| W3 | Training pairs from FIT's blocked candidates only (no random negatives) |
| W4 | LightGBM, calibrated on an inner split of FIT |
| W5 | Assign each S2/S3 record to at most one S1 (enforces the ground-truth partition) |
| W6 | Threshold chosen on VALIDATION (global, Version A — council rejected the combinatorial Version B as too risky for France); France gets a stricter threshold shift from pseudo-France |
| W7 | Bounded error-analysis loop, one change at a time, must show ≥+0.002 F_0.5 with no slice hurt >0.005 |
| W8 | Distractor stress test, pseudo-France check, cross-country transfer test (train US→test India, reverse) as our best proxy for France generalization |
| W9 | Freeze recipe, open LOCKBOX exactly once (a pre-registered fallback recipe also gets scored in that same single read) |
| W10–12 | Test inference, output validation (official `validate_submission.py`), build the submission zip |

Tech stack decided: Polars (lazy) + Parquet, rapidfuzz, LightGBM, scikit-learn (hashing/calibration), pytest. Full runs happen on Kaggle (30GB RAM/4 cores/12h limit), one notebook per stage, only with explicit go. Banned: GPL/AGPL libs, external lookups.

## Council review already done

Local Claude 4-role council (Devil's Advocate, Simplicity Champion, Security Auditor, Scalability Architect — same underlying model, different roles, NOT cross-vendor) reviewed the workflow plan and found real issues, since adopted into the plan:
- My original VALIDATION design would have overstated precision (fixed: all train S1s must compete in every eval run, not just the split being scored).
- The one-S2/S3-per-S1 partition should be a core modeling step, not "post-processing".
- Added: source (S2 vs S3) as a feature, co-tenant-address feature, lockbox must beat baselines not just match validation, a pre-registered fallback recipe.
- One separate OpenRouter council run (GLM-5.2 only — other 2 free models were rate-limited both times) found additional real gaps: absolute score floors needed (not just relative deltas), riskiest choice was the Version-B combinatorial threshold approach (rejected in favor of simpler Version A + France shift), scale concerns about TfidfVectorizer (switched to HashingVectorizer/Polars LazyFrame).

## What's NOT done yet / explicitly NOT VERIFIED

- Stage 1 (cleaning) — not started, waiting for "go".
- Whether test's larger S2/S3-per-S1 ratio reflects more distractors or genuinely more true matches per S1 (assumed pessimistically = distractors).
- Whether test has the same 5.58% singleton rate as train.
- Whether France behaves like US/India (the pseudo-France + cross-country-transfer checks in W8 are our best proxy, not proof).

## Immediate next step

Say **"go"** to start **Stage 1: cleaning** — the text-normalization pipeline (Stage 1–5 of the cleaning plan: nulls, Unicode/accent-folding, name normalization, address normalization, Indic script handling, word-frequency statistics), written from scratch per the isolation rule above, operating on the Stage 0 bronze Parquet tables.
