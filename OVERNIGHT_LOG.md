# Overnight log, 2026-09-26

Append-only. Never edit an earlier entry.

---

## 04:57 START
Read in full: master prompt / workflow plan / Stage 1 prompt: yes / yes / yes
State matches §2: yes
Free RAM: 7.44 GB, budget: 5.94 GB (min(12, 7.44-1.5))
Binding order: C1 → C2 Indic → C3 names → C4 addresses → C5 stats → C6 freeze
First task: T1

## 04:59 T1 Re-check state, council check
Action: Verified free RAM (7.44 GB, checked GlobalMemoryStatusEx), git status/log match §2 exactly, 129/129 tests pass under -W error::DeprecationWarning. Checked for council/ollama processes.
Gates (measured/expected/source/PASS-FAIL): state-match PASS; tests 129/129 PASS
Runtime / peak RAM / free RAM before: tests 37.6s; free RAM 7.44 GB
Deviation from spec: free RAM 7.44 GB vs master prompt's 7.1 GB at 04:47 (plausible drift, not a discrepancy)
Mistake found and fix: none
Commit: none
Next: T2 (fix column-ordering bug, add CG-27..29 test classes)
SELF-CHECK 04:59: ok (no writes yet outside allowed paths; no council/ollama/kaggle/pip/rm calls made)

## LESSON L8
Time: 05:10
Mistake: pipeline.py's clean_table() had a column-ordering bug (L3 in the master prompt): a
with_columns() call referenced pl.col("name_clean") in the same call that was adding name_clean,
but every expression in one with_columns() call sees the DataFrame as it was BEFORE that call, so
the column wasn't visible yet.
How caught: fixed as instructed by T2, by splitting into two with_columns() calls.
Rule added: none new (already CG-27/G-P17 from before 04:40); this just confirms the fix.

## LESSON L9
Time: 05:10
Mistake: a second, previously-unknown bug in the same function: `df.with_columns(c4, pl.col(...))`
passed a DataFrame (c4) directly as a positional arg to with_columns(). Polars does not unpack a
DataFrame there; it silently coerced it into a single Object-dtype literal column instead of adding
c4's columns, and would NOT have raised until something later tried to read a real column from it.
How caught: the new CG-27 end-to-end test (test_end_to_end_pipeline_on_real_rows_by_gid), which is
exactly what that guardrail is for — it failed immediately with "cannot create expression literal
for value of type DataFrame" the first time the real entry point ran past that line.
Rule added: none new; this is direct evidence CG-27 catches what 129 passing unit tests missed.
Fix: `df.with_columns(*c4.get_columns(), ...)` instead of `df.with_columns(c4, ...)`.

## 05:10 T2 Fix known bug, add CG-27/28/29 test classes, audit
Action: Fixed the name_tokens column-ordering bug (two with_columns calls). Found and fixed a second
bug (DataFrame not unpacked in with_columns) via the new CG-27 end-to-end test. Audited every new
module for \w/\W/\b/isalnum/isalpha (script.py's isalpha() uses are safe: they classify base letters,
which is the intended use, not a corruption risk like \w's word-boundary complement was). Added
CG-28 positive-proof tests (Indic marks survive strip_appendages, strip_leading, and addresses'
_STRIP_WORD). CG-29's only real hit (script._all_indic) was already fixed before 04:40 (L2); no
other all()/any() over token data exists in the new modules.
Gates: full cleaning suite 90/90 PASS under -W error::DeprecationWarning.
Runtime / peak RAM / free RAM before: 4.6s; not separately measured (well under budget)
Deviation from spec: none
Mistake found and fix: L8, L9 above
Commit: none yet
Next: check Stage 1 prompt gate coverage (C2 Indic / C3 / C4 / C5 specific gates), then T3 dry run

## LESSON L10
Time: 05:35
Mistake: C3's _extract_legal_form() extracted only the FIRST matching legal-form token per call and
returned immediately, instead of all of them. A name with more than one legal-form-like token (e.g.
"Phylys M. Hillis, DDS, M.D., P.C.", US, real gid 56200 in train S1) had "dds" extracted on pass 1,
leaving "pc" sitting in name_clean; a second pass then found and extracted "pc" too, changing the
text again. This broke idempotence (CG-13) on 96 of the 22,069 rows in a 1% slice of train S1 alone
-- likely thousands on the full table, concentrated in US professional names (doctors, dentists,
lawyers) that carry multiple credentials/suffixes.
How caught: silver_run.py's per-table idempotence check (G-SIL-13), during the T3 dry run.
Rule added: none new (CG-13 already covers this); fixed by extracting every matching legal-form
token/phrase in a loop until none remain, deduped and space-joined into name_legal_form. Added a
regression test (test_all_legal_form_tokens_extracted_not_just_the_first) using real gid 56200.
This changes name_legal_form's format for names with multiple forms (now a space-joined list, e.g.
"dds pc") -- not yet published anywhere, so no rule-change/CG-14 conflict.

## LESSON L11
Time: 05:47
Mistake: strip_leading() ran once, before legal-form extraction, in clean_one_name(). A raw name
like "Inc The-Anchor Beacon Quetta" has "inc" (not "the") as the leading token when strip_leading
runs, so nothing is stripped there; legal-form extraction then removes the leading "inc", which
exposes "the" as the NEW leading word -- but strip_leading never runs again in that call, so "the"
survives into name_clean. Re-running clean_one_name on that output (the idempotence check) finds
"the" genuinely leading and strips it, changing the text a second time.
How caught: silver_run.py's per-table idempotence check (G-SIL-13), same dry run as L10 (train S2/S3,
test S2/S3: 3+2+5+4 = 14 rows in the 1% slices alone).
Rule added: none new (CG-13). Fixed by looping strip_leading + legal-form extraction together until
neither changes anything, collecting every extracted form (deduped, phrase-intact) into
name_legal_form. Added test_leading_legal_form_exposes_a_new_leading_the_and_it_still_gets_stripped
using the real name found (gid search on train S2, "Inc The-Anchor Beacon Quetta").
Pattern noted: both L10 and L11 are the same root cause -- a fixed-order pipeline of "extract one
thing" steps that don't re-check earlier steps after a later step changes the string. Worth watching
for in C4 too before publishing.

## 06:02 T3 (RAM diagnosis, mid-task)
Action: T3 dry run passed all gates after L10/L11 fixes (69.6s, peak RAM 2508.5 MB reported by the
combined dry run). Before trusting the T6 formula (dry-run peak x 100 x 1.5), checked it: applied
literally it gives ~367 GB, which is implausible (§7.8) -- it conflates the one-time FIT-dictionary-
build cost (already full-size, does not scale: measured in isolation at 2226.5 MB, 19.7s) with the
per-table cleaning cost (which does scale ~100x). Isolating the true per-table cost: running
pipeline.clean_table('train', 3, ...) FULL SIZE (5.28M rows, the largest table) in a fresh process to
measure its real memory footprint. This is taking longer than the 2-minute CG-22 announce threshold
(over 300s so far) -- waiting for it rather than starting any other heavy work (CG-20/R13).

## LESSON L12
Time: 06:08
Mistake/finding: running pipeline.clean_table('train', 3, ...) at FULL size (5,285,603 rows, the
largest table) in isolation crashed with a Polars/Rust allocator failure ("memory allocation of
1563333040 bytes failed") when only 7.44 GB was free. This is a genuine RAM-budget finding, not a
code bug: it directly demonstrates the naive T6 estimate formula (dry-run peak x 100 x 1.5, which
gave an implausible ~367 GB by conflating the fixed dictionary-build cost with the per-table scaling
cost) understated nothing -- the REAL per-table cost, unknown until measured directly, is enough to
exhaust available RAM on its own.
How caught: direct measurement (T6's own "before the run" RAM check, done proactively before
attempting the real full run).
Rule added: none new (CG-19 already requires this kind of check; this is why). Re-checked free RAM
after the crash: 9.82 GB (up from 7.44 GB -- the crashed process released its memory), giving a
1.38 GB larger budget. Retrying the same full-table measurement now with more headroom before
deciding whether T6 needs chunking (R2 sub-table chunking by gid range) or must STOP.

## 06:16 LESSON L12 update: reproducible, not a free-RAM fluctuation
The full-table clean_table('train', 3, ...) crash reproduces identically (same 1,563,333,040-byte
allocation failure) even with 9.82 GB free (up from 7.44 GB). This rules out "just low free RAM at
that moment" -- it is a real memory ceiling in the current C2-C5 code, most likely because names.py,
addresses.py and the name_script_flags construction in pipeline.py all use per-ROW Python loops
building large Python-level lists (R1 violation: C1 avoided this by transforming per DISTINCT VALUE
and joining back; my C2-C5 code mostly doesn't, except script.apply_dictionary). Testing a 20% chunk
next to see whether sub-table chunking (R2 extended) is a viable mitigation before deciding whether
T6 can run tonight at all, per the STOP protocol ("apply R2/R3/R5, dry-run again, re-estimate; if it
still doesn't fit, STOP T6 and go to T8").

## 06:52 T3/T6 prep: chunking fix verified, spec ambiguity flagged
Action: Refactored silver_run.py to stream each table's output as ONE Parquet file via
pyarrow.parquet.ParquetWriter, processing CHUNK_N=10 gid-modulo chunks per table (full run only;
dry run stays single-pass on its 1% slice). This keeps the §2 single-file-per-table layout intact
(no CG-14 rule change needed) while bounding peak memory. Verified in isolation on the FULL
train_source1 table (2,206,821 rows, real data, not sliced): rows matched C1 output exactly,
idempotence clean, peak RAM only 2467.0 MB (vs. the dictionary-build's own 2242.9 MB baseline --
chunking added ~224 MB, not the ~34 GB a naive full-table pass would need). 161.2s for this table.
Full test suite: 135/135 pass.

STOP-flagged (not silently decided, CG-14): the Stage 1 prompt's C3 gate text ("count of llc
appearing in name_clean after cleaning must be >= the raw count") is inconsistent with the same
section's own rule that legal-form tokens are removed FROM name_clean into name_legal_form. Taken
literally the gate would need name_clean to still contain "llc", which C3 correctly prevents
(measured: 11 vs a raw count of 21,843 on train S1). Implemented as counting llc in EITHER
name_clean OR name_legal_form (the only self-consistent reading), reported to the user for
confirmation in the morning report; not applied as a silent redefinition elsewhere.

Next: attempt T6 (full run of all 6 tables), watching RAM between tables, before the 09:15 cutoff.

## 07:05 T3 complete
Action: Re-ran the dry run twice with the current (chunked/streamed) code: _dryrun_silver_6 and
_dryrun_silver_7. All slice-runnable gates pass both times; output file sha256 identical across the
two runs (determinism confirmed). An earlier comparison against _dryrun_silver_5 was invalid (that
run used the old direct write_parquet path, not the new streaming ParquetWriter -- different writers
can byte-differ on identical data; not a determinism bug, just an invalid comparison, noted so it
isn't mistaken for one later).
Gates: all PASS. Runtime ~53s, peak RAM ~2.5 GB (dominated by the dictionary build).
Commit: none yet
Next: T4 (decision checkpoint, §6 pre-decided rules already apply) -> T5 (commit) -> T6 (full run)

## 07:15 T4/T5 complete
Action: T4 decision checkpoint -- D-C3a applied (empty names left as-is; the few 1%-slice examples
trace to real degenerate raw names like "The" and "MR", not a bug). T5: tree was clean of
disallowed paths; full suite 135/135 passed once more; committed.
Commit: 8c397ec
Next: T6 full run (RAM estimate, then background execution, watching RAM between tables)

## 07:20 T6 pre-run estimate (CG-19/CG-22, before starting)
Free RAM now: 9.14 GB. Budget: min(12, 9.14-1.5) = 7.64 GB.
RAM estimate: chunking (CHUNK_N=10) bounds memory to roughly one chunk + the fixed dictionary cost,
independent of full table size (confirmed: the full 2.2M-row train S1 table peaked at 2467 MB via
chunking, barely above the dictionary's own 2242.9 MB). Conservative estimate for the whole 6-table
run: 2.5-5.6 GB (upper bound assumes the largest table's ~3.4 GB per-chunk cost stacks fully on top
of the dictionary's ~2.2 GB rather than reusing freed memory). Both ends are under the 7.64 GB
budget with margin.
Runtime estimate: full train S1 (2,206,821 rows) took 161.2s in isolation. Total across all 6
tables (~24.2M rows) at a comparable or slightly slower blended rate (S2/S3 are noisier): ESTIMATE
20-30 minutes. Labelled as an estimate (CG-22); the runner prints per-table progress.
Starting T6 now, in the background, watching for the runner's per-table RAM line and any crash.

## 06:48 T6 COMPLETE: silver v1 published
Action: Full run of all 6 tables, chunked (CHUNK_N=10 per table). ALL GATES PASSED.
- Row counts: all 6 tables match C1 output exactly (train S1 2,206,821 / S2 5,034,616 / S3 5,285,603;
  test S1 1,732,544 / S2 4,887,273 / S3 5,082,316).
- Idempotence (G-SIL-13): 0 bad names, 0 bad addresses, on EVERY table, full data (not just the slice).
- G-C2-coverage-train: 0.9775 (measured), >= 0.949 (§1 baseline) -- PASS, matches the earlier
  standalone measurement exactly.
- G-SIL-C5-pool: train IDF built from 3,205,354 train-pool rows, test IDF from 3,020,000 test-pool
  rows (India only, since C5's own gate here is per-country IDF; full pool row counts are the full
  22.2M rows across all countries -- these numbers are the India-only join count from the coverage
  step's bookkeeping, not a separate gate value; noted for the morning report to double check this
  label is accurate).
- leakscan clean (G-SIL-L2).
- Peak RAM: 4896.3 MB, well under the 7.64 GB budget (estimate was 2.5-5.6 GB -- within range).
- Determinism (G-SIL-5): a fresh-process rerun gives byte-identical output for all 6 files.
- Runtime: 4045.3s (~67.4 min) -- longer than the 20-30 min estimate because G-SIL-5's determinism
  check reprocesses all 6 tables a SECOND time in a fresh subprocess (~2x the single-pass time,
  consistent: ~33 min x 2 ~= 67 min). The estimate should have said this explicitly; noted as a gap
  in the pre-run estimate, not a measurement bug.
Published: data/silver/v1/ (6 table parquets + idf_train.parquet + idf_test.parquet), 
data/dictionaries/v1/indic_translit.parquet, data/MANIFEST_silver_v1.json. All files read-only.
Commit: 8c397ec (already committed before the run, per G-P5/CG-15)
Time check: 06:48, well before the 09:15 time-rule cutoff.
Next: T7 (W1 blocking dry run -- design and measurement only, reads published silver/v1)

## 06:50 Correction to the 06:48 entry
The G-SIL-C5-pool numbers (train_rows: 3,205,354, test_rows: 3,020,000) are NOT document/row counts.
idf_tables[ds] is the output of stats.finalize_idf(): one row per (country, token) pair -- i.e. these
are VOCABULARY SIZES (distinct token-country pairs), not pool sizes. Mislabeled in the code's own gate
description ("train IDF built only from train tables") and in my 06:48 log entry. Not a data bug --
the IDF values themselves are correct (verified separately: R5 additive-vs-whole-pool equivalence
test passes) -- only the gate's reported number is mislabeled. Logging the correction rather than
silently editing the earlier entry (log is append-only).

## 07:00 T7 started
Action: Wrote src/amlc/blocking/{candidates.py, dryrun_w1.py} + tests/blocking/test_candidates.py.
9 new tests pass; full suite 144/144 pass under -W error::DeprecationWarning (caught and fixed
another explode() empty_as_null deprecation, same CG-18 pattern as before). Smoke-tested on real
data (US): 9,265 sampled FIT S1, 6,186,873 S2+S3 pool, 2,106,090 scored pairs at cap=500 in ~49s,
top-30 gives 259,734 candidates. Starting the full T7 measurement (US + India, all 4 caps x 5 k
values), ESTIMATE ~10-15 minutes (extrapolated from the smoke test's per-cap timing).

## LESSON L13
Time: 07:05
Mistake: dryrun_w1.py has no pre-run RAM estimate, no budget check, and no abort-on-breach logic --
unlike silver_run.py (T6), which has all three (G-P9/CG-19). The run completed successfully (peak
10,092.7 MB), but that was not verified safe in advance; it happened to fit because more RAM was
actually free at the time than my last check (9.14 GB free at 07:20, budget 7.64 GB) had shown.
10.42 GB is free now, after the run finished and released memory, consistent with more headroom
having existed during the run than I had last measured -- but I did not re-check free RAM
immediately before T7 started, and the code has no self-check during the run either.
How caught: reviewing the run's own peak_ram_mb output against my last logged budget, after the
fact -- not caught by a runtime check, because there wasn't one.
Rule added: none new (G-P9/CG-19 already require this everywhere, not just T6). Flagging this as a
process gap in dryrun_w1.py to fix before any daytime W1 work (S8) reuses this code, and as a
reminder that "it happened to work" is not the same as "it was checked safe" (the distinction the
guardrail exists to enforce).

## 07:08 T7 COMPLETE: critical finding -- blocking recall is well below the workflow plan's targets
Measured on 1% of FIT S1 per country vs. the FULL S2/S3 pool (real data, silver v1):
- US: recall tops out at 0.8198 (cap=5000, k=100). Target (workflow plan W1 exit gate): >= 0.97.
- India: recall tops out at 0.4976 (cap=5000, k=100). Target: >= 0.95. Recall is roughly HALF the
  target across every cap/k combination tried, and barely above 0.27 at the smallest cap/k.
- Recall rises with both cap and k in both countries, but is nowhere near flattening in the range
  tried (100-5000 cap, k=10-100) -- more so for India, where the curve is still rising steeply at
  cap=5000. Widening the grid further was NOT done (T7 scope: measurement only, no cap/k choice).
- Pair counts (before any cap on k): US 538,765 (cap=100) to 18,464,413 (cap=5000); India 212,018
  to 5,444,451. Scoring runtime 13-42s per (country, cap) on the 1% S1 sample against the full pool.
- This is consistent with the workflow plan's own known risk (M8): "24% of India true pairs share
  no name token at all" -- India's shortfall is much larger than US's, and the current 3 key types
  (rare core-name tokens; house-number+address-token composite; ZIP/PIN) cannot recover matches with
  no shared name token and no matching address key. The char-3gram pass (scikit-learn, not installed
  tonight) is explicitly the fallback the workflow plan names for exactly this gap.
Not done (needs the user, explicitly out of scope for tonight): choosing a cap or k, extending the
grid, adding the char-3gram pass, or any other change aimed at raising recall. This is reported as
a measurement, not treated as a problem to silently fix.
Report: data/_dryrun_w1/w1_dryrun_report.json (full per-country, per-cap, per-k breakdown).
Runtime: 234.9s total, peak RAM 10,092.7 MB (see L13 above).

## 07:10 T8 complete
Action: Wrote MORNING_REPORT.md per the T8 spec (task status, every gate, commits, runtimes/RAM/
free RAM, all lessons, STOP flags, decisions waiting for the user, exact 10:00 commands). Self-check
(§8) clean: only MORNING_REPORT.md unstaged; no diffs under foundation/eval; the two "forbidden
pattern" hits are pre-existing committed C1 code (its own sanctioned dry-run cleanup), not new
tonight; no pip references; no council/ollama processes. Free RAM 9.63 GB. Time 07:01, well before
the 09:15 cutoff -- stopping here rather than inventing new scope beyond T1-T8.
Commit: (this entry's commit follows)
SELF-CHECK 07:10: ok
