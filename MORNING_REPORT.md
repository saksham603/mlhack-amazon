# Morning report — overnight run, 2026-09-26 04:57–07:10 IST

Full step-by-step log: `OVERNIGHT_LOG.md` (append-only, every SELF-CHECK and LESSON entry).
This report summarizes it per the master prompt's T8 spec.

## 1. Task status

| Task | Status | Commit |
|---|---|---|
| T1 Re-check state, council check | Done. State matched §2 exactly. No council/ollama processes found (the earlier regex hits were my own command lines containing the search string, not real matches). | — |
| T2 Fix known bug, add CG-27/28/29 coverage | Done. Fixed the known column-ordering bug **and found a second bug** via the new end-to-end test (L9). Audited all new modules for Unicode-unsafe patterns; only one real hit (already fixed pre-04:40, L2). | (rolled into T5) |
| T3 1% dry run, twice | Done. All slice-runnable gates pass both times; output byte-identical across two runs with the same code (determinism). One invalid comparison noted and explained (L-note 07:05), not a bug. | (rolled into T5) |
| T4 Decision checkpoint | Done. D-C3a applied (empty names left as-is; traced to real degenerate raw names like "The"/"MR"). No rule changed. | — |
| T5 Commit | Done. | `8c397ec` |
| T6 Full silver build | **Done and published.** All 6 tables, all gates pass. | `8c397ec` (code); publish is a data artifact, not a commit |
| T7 W1 blocking dry run | Done (measurement only, as scoped). **Critical finding: recall is well below the workflow plan's targets** — see §4. | `71538e8`, `6894a41` |
| T8 Morning report | This file. | (this commit) |

## 2. Gates — measured vs. expected

### T6 (silver v1) — full data, all PASS

| Gate | Measured | Expected | Source |
|---|---|---|---|
| G-C2-L1 (0 non-FIT ids reach the dictionary builder) | 0 | 0 | access.load_split() |
| G-SIL-1 (row count = C1 output), all 6 tables | exact match, all 6 | C1 output row counts | interim/c1/v2 |
| G-SIL-13 (idempotence), all 6 tables | `{bad_names: 0, bad_addresses: 0}`, all 6 | same | recount-style, own check |
| G-C2-coverage-train | 0.97753 | ≥ 0.949 | Stage 1 prompt §1 |
| G-SIL-C5-pool | train 3,205,354 / test 3,020,000 *(vocabulary size, i.e. distinct (country,token) pairs — mislabeled in the gate description as "rows"; corrected in OVERNIGHT_LOG.md 06:50; not a data bug)* | by construction | — |
| G-SIL-L2 (leakscan) | `[]` | `[]` | leakscan.scan |
| G-P1-peak-ram | 4896.3 MB | ≥ 398.0 MB (largest table) | polars estimated_size |
| G-SIL-5 (determinism, fresh process) | identical | identical | subprocess rerun |

Indic coverage: **train 0.9775, test 0.9639** (both above the §1 baseline of 0.949/0.934). Dictionary: 1,347 mappings (§1's raw-text measurement found 1,355 — close, expected given a different tokenizer stage; not re-diagnosed further, D5b stayed diagnosis-only as decided).

C3 `llc` totals are much larger than §1's raw dotted-`L.L.C.`-pattern counts (e.g. train S1: 377,579 vs. 21,843). This is very likely because §1 only counted the **dotted** variant; plain `LLC` (no periods) never needed the collapse step and is far more common — **not independently verified this morning**, flagged as NOT VERIFIED rather than assumed.

C4 `addr_zip` non-null rate: 6.3–6.6% (train), 4.3% (test). §1 only gave whole-table counts (zip5+pin6 combined) for **train**, not test, so test's rate is reported for information only, not gated (no §1 target exists for it).

### T3 (dry run, slice) — all PASS
Every gate in the T3 section of `OVERNIGHT_LOG.md` (07:05 entry) passed on the final rerun; earlier failures (L8–L12) were all found, fixed, and reverified before committing.

### T7 (W1 blocking dry run) — measurement, no pass/fail gate (by design)
Not a gate in the pass/fail sense — T7's job was to measure, not decide. See §4.

## 3. Real bugs found and fixed overnight (not assumed correct — see OVERNIGHT_LOG.md for full detail)

| # | What | Found by | Fix |
|---|---|---|---|
| L8/L9 | `pipeline.py`: a `with_columns()` column-ordering bug, then a **second** bug (a DataFrame passed directly to `with_columns()`, silently coerced to a single Object column) | The new CG-27 end-to-end test, which is exactly what it's for | Split into two `with_columns()` calls; unpack the DataFrame's columns explicitly |
| L10 | `names.py`: legal-form extraction stopped at the first match per call, leaving a second legal-form token (e.g. "DDS ... PC") in `name_clean` | `silver_run.py`'s per-table idempotence gate, real train S1 data (gid 56200) | Loop extraction to a fixpoint, dedupe, join with spaces |
| L11 | `names.py`: `strip_leading` ran once, before legal-form extraction; extracting a *leading* legal form (e.g. "Inc The-Anchor...") exposed a new leading "the" that was never re-checked | Same idempotence gate, real train S2 data | Loop `strip_leading` + extraction together to a fixpoint |
| L2 (pre-04:40) | `script.py`: `all()` over an empty "has an alphabetic char" generator counted punctuation-only tokens as Indic | Implausibly low coverage (89.9%) | Explicit empty-case handling |
| L1 (pre-04:40) | `names.py`: punctuation stripping used `\w`'s complement, which strips Indic combining marks | Found while writing, before any run | Unicode-category (P/S) translate table |
| L12 | A full-size single-table clean crashed the Polars allocator (~1.5 GB allocation failure), reproducibly, even with 9.82 GB free | Direct measurement, done proactively before the real T6 run | Stream each table in 10 gid-modulo chunks via `pyarrow.parquet.ParquetWriter`, keeping the single-file-per-table layout (§2) unchanged |
| L13 | `dryrun_w1.py` had no RAM budget check at all (unlike `silver_run.py`); the real run peaked at 10.1 GB without one | Reviewing the run's own peak-RAM output after the fact | Added `free_ram_mb()`/`check_ram_budget()` (not rerun tonight; fix is for daytime reuse) |

Pattern noted (in the log, worth repeating here): L10 and L11 are the same root cause — a fixed-order pipeline of "extract one thing" steps that don't re-check earlier steps after a later step changes the string. Worth watching for in any future multi-step text pipeline.

## 4. The most important finding: blocking recall is well below target

Measured on real data (silver v1), 1% of FIT S1 per country vs. the **full** S2/S3 pool:

| Country | Best recall (cap=5000, k=100) | Target (workflow plan W1 exit gate) | Gap |
|---|---|---|---|
| US | 0.8198 | ≥ 0.97 | −0.15 |
| India | 0.4976 | ≥ 0.95 | **−0.45** |

Recall was still rising at the top of the tried range (cap 100→5000, k 10→100) in both countries, especially India — the curve had not flattened, so the true ceiling with more cap/k is unknown, but even generous extrapolation is very unlikely to close India's gap with the current 3 key types alone (rare core-name tokens, house-number+address-token composite, ZIP/PIN).

This lines up with the workflow plan's own documented risk (M8): "24% of India true pairs share no name token at all." The char-3gram fallback pass is explicitly named in the plan for exactly this case — it is **not installed tonight** (scikit-learn is missing from the venv) and was correctly not attempted, per the environment note and CG-23 (no rule change, no new scope, under time pressure or otherwise).

**Full-scale estimate (corrected):** the code originally computed a RAM-based extrapolation that gave an implausible ~493 GB — that formula was wrong (conflated dictionary-build-style fixed costs with a workload that doesn't have one here) and is **not usable**; flagged rather than reported as fact. A better estimate, scaling the *measured pair counts* by the ratio of full FIT S1 to sampled S1 (pair generation is roughly linear in S1 count for a fixed pool and cap):

| Country | cap | Sampled pairs (1% S1) | **Estimated full-S1 pairs** |
|---|---|---|---|
| US | 100 | 538,765 | ~54M |
| US | 5000 | 18,464,413 | **~1.85 billion** |
| India | 100 | 212,018 | ~21M |
| India | 5000 | 5,444,451 | **~544 million** |

This is an ESTIMATE, not a measurement of the full run. It strongly suggests the real W1 run will need S1-batch or key-hash chunking (R6 already anticipates this) even at modest caps — this is useful to know before S8 starts, not a decision made here.

## 5. Runtimes, peak RAM, free RAM

| Run | Runtime | Peak RAM | Free RAM before |
|---|---|---|---|
| T3 dry run (final, ×2) | ~53s each | ~2.5 GB | 9.14–9.82 GB (fluctuated across the session) |
| T6 full run | 4045.3s (~67.4 min) | 4896.3 MB | 9.14 GB (budget: 7.64 GB) |
| T7 W1 dry run | 234.9s | **10,092.7 MB** | not re-checked immediately before starting — see L13 |

T6 took about 2× the pre-run ETA (20–30 min) because the estimate didn't account for `G-SIL-5`'s determinism check reprocessing all 6 tables a **second** time in a fresh subprocess. Noted as a gap in the pre-run estimate, not a measurement bug — the actual single-pass work was in the estimated range.

## 6. Every STOP and why

None of tonight's work hit a full STOP-and-abandon. Two things were **flagged rather than silently decided** (the CG-14 discipline):
- The C3 `llc` gate's literal wording contradicts the same section's "legal-form tokens removed from `name_clean`" rule. Implemented as `name_clean` OR `name_legal_form` (the only self-consistent reading) — **needs your confirmation**, not applied as a silent redefinition anywhere else.
- The G-SIL-C5-pool gate's numbers are vocabulary sizes, not pool sizes (a labeling bug in the gate description, corrected in the log, not silently left wrong).

## 7. Decisions waiting for you

1. **Confirm or correct the `llc` gate reading** (§6 above) — does "name_clean OR name_legal_form" match what you intended, or should the gate be redefined?
2. **India blocking recall (0.50 at best) is far below the 0.95 target.** Options, none applied: extend the cap/k grid further; add the char-3gram pass (needs `pip install scikit-learn`, blocked overnight); accept a lower India target and rely more heavily on W2 features/W6 decision logic to compensate. This is a real design decision, not a bug to fix.
3. **US blocking recall (0.82 at best) is also below the 0.97 target**, though less severely. Same set of options.
4. **The full-scale W1 pair-count estimate (~1.85 billion at cap=5000 for US) means W1 needs chunking design**, not just "run it bigger." Worth planning before S8 starts.
5. Whether to `pip install scikit-learn`/`rapidfuzz`/`lightgbm` now that a natural checkpoint has been reached (all blocked overnight per the master prompt's settings).
6. `PROJECT_CONTEXT.md` was correctly left untouched (read-only per §3) — it needs updating to reflect silver v1 being published and the T7 findings before the next session relies on it.

## 8. Exact commands for 10:00

```bash
cd "C:\Users\suremdra singh\amlc2026"
git log --oneline -6                          # see tonight's 4 commits
git status --porcelain                         # should be empty except this report until you commit it
cat OVERNIGHT_LOG.md                           # full step-by-step trace, all 13 lessons
cat data/MANIFEST_silver_v1.json                # T6 gates, in full
cat data/_dryrun_w1/w1_dryrun_report.json       # T7 recall/pair-count numbers, in full
.venv\Scripts\python.exe -m pytest -q          # 147/147 should pass
```

Once you've reviewed:
- Say **go** to continue with S9 (W2 features) if you're satisfied silver v1 is good to build on, or
- Say what to change first if the `llc` gate reading or the blocking-recall gap need addressing before S9.

## 9. What was NOT done (in scope but not reached, or explicitly out of scope)

- W1 itself (the full run, choosing k or a cap, writing `candidate_pairs.tsv`) — explicitly out of scope tonight (§1 "Not authorised").
- Any feature or model code (W2 and later) — same.
- Any Kaggle or network action — same, and hard-blocked in settings.
- Extending the D5b Indic alignment beyond diagnosis — pre-decided to stay diagnosis-only (§6).
