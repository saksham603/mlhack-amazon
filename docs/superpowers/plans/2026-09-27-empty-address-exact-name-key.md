# Empty-Address Exact-Name Candidate Key Implementation Plan

> **RESULT (2026-09-27 16:22): NO_GO.** Tasks 1-2 executed via strict TDD. Real measurement on the
> FIT holdout: 34,374 new candidate pairs generated, 471 true links newly reachable as candidates,
> but **0 crossed the model's decision threshold** — net F0.5 gain 0.0 (kill bar was 0.0005). Per-S1
> complete coverage did rise 91.92% -> 93.29%, but none of that translated to actual predictions.
> Task 3 (full test-set build) was not attempted, per the plan's own kill criterion. v3.2 (96.6%
> confirmed) is unaffected and remains the submission.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the existing FIT holdout, whether a new candidate-generation key restricted to
S2/S3 records with an empty address (matched by exact no-space name within that small sub-pool)
recovers true links the current v3.1/v3.2 blocker misses entirely — and only build the full-scale
test-set version if that measurement clears a real, pre-declared bar.

**Architecture:** A single new, isolated module computes the new key and candidate pairs for a given
record set; it is never imported by `amlc.blocking.v3`, `amlc.pipeline.run_v3`, `run_v31`, or
`run_v32`, and never writes into `data/_v31/`, `data/_v32/`, or `output/`. All work happens against
the already-cached FIT holdout (`data/_v31/exp/holdout_p1.parquet`) plus the FIT record pool, and
writes only to `data/_helper/exp_empty_addr/`. v3.2's production artifacts and `output/v32/` are never
read for writing and never modified.

**Tech Stack:** Python, polars, the existing `amlc.foundation.access` labels loader, the existing
`amlc.eval.metric` scorer for F0.5 (reused, not reimplemented).

**Spec:** This plan implements the finding already recorded in `OVERNIGHT_LOG.md` (08:42 entry,
"Empty-address exact-name gap measured, not worth pursuing right now") and the diagnostic numbers
established earlier in this session (per-S1 complete coverage 91.9% overall / 88.6% for S1s with 4+
true matches, measured against `data/_v31/exp/holdout_p1.parquet`). No separate design doc exists;
this plan is scoped directly from that log entry and the live conversation's numbers, which together
constitute the full requirement: reproduce the 726-recoverable-link / 175-net-gain estimate on a
fresh run, and decide go/no-go against a stricter, pre-declared bar before any full-scale build.

## Global Constraints

- Never modify `amlc/blocking/v3.py`, `amlc/pipeline/run_v3.py`, `run_v31.py`, or `run_v32.py`.
- Never write into `data/_v31/`, `data/_v32/`, or `output/`. All new files go under
  `data/_helper/exp_empty_addr/`.
- Every write is atomic (reuse `amlc.pipeline.mem.atomic_write_parquet`, already used project-wide).
- Kill criterion (checked at the end of Task 2, before Task 3 starts): if the measured net F0.5 gain
  on the FIT holdout is below **0.0005** (half the smallest gain this session has treated as real —
  the pseudo-word feature's +0.00126 was the smallest adopted gain so far), STOP. Do not do Task 3.
  Report the number and the decision plainly.
- Reuse `amlc.eval.metric.per_s1_f05` / `macro` for every F0.5 number — never hand-roll the metric.

## Review Focus

- **S2/S3 records whose address is empty AND whose no-space name is short (<4 chars):** the existing
  `cns` key already excludes `ns` shorter than 4 chars; the new key must apply the same floor, or it
  will flood the index with near-meaningless single/double-character keys. Test: a record with
  `ns="ab"` produces no key from this module.
- **An S1 that already has this same S23 as a candidate from the existing blocker:** the new key must
  only ever ADD pairs, never duplicate one already present, or downstream feature-join /
  `is_duplicated()` asserts elsewhere in the codebase will fire. Test: computing candidates against a
  record pool where every empty-address record is already a v3.1 candidate for its matching S1
  produces zero new pairs after the additive-only filter.
- **A country with zero empty-address S2/S3 records:** the module must return an empty frame, not
  raise. Test: a record pool with no `addr == ""` rows.
- **Missing block-rank features on the new pairs when scoring:** the new pairs have no `v1_score`,
  `v1_rank`, `v3_score`, `v3_rank`, `va_score`, `va_rank` (they were never found by those keys). The
  plan's scoring step must fill these with the same "not found" sentinel the production pipeline uses
  for a candidate absent from a given key type — not zero, not null silently. Test: confirm what that
  sentinel actually is by reading `amlc/pipeline/run_v3.py`'s candidate-merge code before Task 2, not
  by guessing (this is the one part of the spec that is genuinely ambiguous without reading the code).
- **The 726/175 numbers not reproducing exactly:** OVERNIGHT_LOG's numbers came from an earlier,
  possibly different-seed run. Task 2 must report its own fresh numbers and explicitly compare them to
  the logged ones, flagging (not silently accepting) any large discrepancy.

---

### Task 1: Empty-address exact-name key + candidate generation module

**Files:**
- Create: `src/amlc/experimental/__init__.py` (empty)
- Create: `src/amlc/experimental/empty_addr_key.py`
- Test: `tests/experimental/test_empty_addr_key.py`

**Interfaces:**
- Consumes: a polars DataFrame with columns `gid, addr, ns` (the same `ns` = no-space-name column
  `amlc.blocking.v3.keys()` already computes upstream — read `amlc/blocking/v3.py`'s `rec` schema
  before writing this, do not assume the column exists if it doesn't; if `ns` must be derived here,
  derive it with the exact same normalization `v3.py` uses, not a reimplementation).
- Produces:
  - `empty_addr_keys(r23: pl.DataFrame) -> pl.DataFrame` — columns `s23_gid, ns`, one row per S2/S3
    record where `addr == ""` and `len(ns) >= 4`.
  - `empty_addr_candidates(r1: pl.DataFrame, r23_keys: pl.DataFrame) -> pl.DataFrame` — columns
    `s1_gid, s23_gid`, one row per (S1, S2/S3) pair sharing the same `ns`, for S1 records whose own
    `ns` (computed the same way) has `len >= 4`. No dedup against any external candidate set — that
    filtering happens in Task 2, which is the caller.

- [x] **Step 1: Write the failing tests**

```python
import polars as pl
from amlc.experimental.empty_addr_key import empty_addr_keys, empty_addr_candidates

def test_short_ns_excluded():
    r23 = pl.DataFrame({"gid": [1, 2], "addr": ["", ""], "ns": ["ab", "widgetsinc"]})
    keys = empty_addr_keys(r23)
    assert keys["s23_gid"].to_list() == [2]

def test_nonempty_addr_excluded():
    r23 = pl.DataFrame({"gid": [1, 2], "addr": ["12 main st", ""], "ns": ["widgetsinc", "widgetsinc"]})
    keys = empty_addr_keys(r23)
    assert keys["s23_gid"].to_list() == [2]

def test_no_empty_address_records_returns_empty():
    r23 = pl.DataFrame({"gid": [1], "addr": ["12 main st"], "ns": ["widgetsinc"]})
    assert empty_addr_keys(r23).height == 0

def test_candidates_match_on_shared_ns():
    r1 = pl.DataFrame({"gid": [10, 11], "ns": ["widgetsinc", "zz"]})
    keys = pl.DataFrame({"s23_gid": [2], "ns": ["widgetsinc"]})
    cand = empty_addr_candidates(r1, keys)
    assert cand.select("s1_gid", "s23_gid").rows() == [(10, 2)]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/experimental/test_empty_addr_key.py -v`
Expected: FAIL with "No module named 'amlc.experimental'"

- [x] **Step 3: Read `amlc/blocking/v3.py`'s record schema and `ns` derivation, then implement
  `empty_addr_keys` and `empty_addr_candidates` in `src/amlc/experimental/empty_addr_key.py`**

Mirror `v3.py`'s existing `MIN_TOKEN_LEN`-equivalent floor for `ns` (4 chars, matching the existing
`cns` key's own filter at `v3.py:79`) exactly, don't invent a new threshold.

- [x] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/experimental/test_empty_addr_key.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/amlc/experimental/__init__.py src/amlc/experimental/empty_addr_key.py tests/experimental/test_empty_addr_key.py
git commit -m "exp: isolated empty-address exact-name candidate key module

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: FIT-holdout measurement (go/no-go gate)

**Files:**
- Create: `src/amlc/experimental/measure_empty_addr.py`
- Test: `tests/experimental/test_measure_empty_addr.py` (small-fixture unit tests only; the real
  measurement itself is run manually as a script per this task's Step 5, not as a pytest test)

**Interfaces:**
- Consumes: `empty_addr_keys`, `empty_addr_candidates` from Task 1;
  `data/_v31/exp/holdout_p1.parquet` (has `s1_gid, s23_gid, country, label, p1, <53 v3.1 features>`);
  `amlc.foundation.access.load_labels("FIT")`; `amlc.eval.metric.per_s1_f05` / `macro`;
  `amlc.model.predict.enforce_g_m3`; the model at `models/lgbm_v32.txt` via `lightgbm.Booster`.
- Produces: `data/_helper/exp_empty_addr/measure_report.json` with the keys: `n_new_candidate_pairs`,
  `n_true_links_recovered_as_candidates`, `n_true_links_recovered_and_above_threshold` (using v3.2's
  real threshold, read from `data/_v32/report_v32.json`, not hardcoded), `f05_base` (v3.2 model,
  existing candidates only, on this FIT-holdout sample), `f05_with_new_key` (existing + new candidates
  union, new pairs' missing block-rank features filled per the Review Focus sentinel finding),
  `net_f05_gain`, `go_no_go` (`"go"` if `net_f05_gain >= 0.0005` else `"no_go"`).

- [ ] **Step 1: Write a small-fixture test for the additive-only dedup logic**

```python
import polars as pl
from amlc.experimental.measure_empty_addr import new_pairs_only

def test_new_pairs_only_excludes_existing():
    existing = pl.DataFrame({"s1_gid": [1, 1], "s23_gid": [2, 3]})
    candidate = pl.DataFrame({"s1_gid": [1, 1, 1], "s23_gid": [2, 3, 4]})
    result = new_pairs_only(candidate, existing)
    assert result.select("s1_gid", "s23_gid").rows() == [(1, 4)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/experimental/test_measure_empty_addr.py -v`
Expected: FAIL with "No module named 'amlc.experimental.measure_empty_addr'"

- [ ] **Step 3: Implement `new_pairs_only(candidate: pl.DataFrame, existing: pl.DataFrame) -> pl.DataFrame`
  in `src/amlc/experimental/measure_empty_addr.py`**

Anti-join on `(s1_gid, s23_gid)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/experimental/test_measure_empty_addr.py -v`
Expected: PASS

- [ ] **Step 5: Implement and run the real measurement as `main()` in the same file**

Read `data/_v31/exp/holdout_p1.parquet` for the S1 sample + existing candidates + `p1` (v3.1 prob, for
reference only). Build the FIT record pool's `r1`/`r23` (`data/_v31/fit`-equivalent source records —
use the same loader `amlc/model/exp_rules.py`'s `load_holdout()` already uses for `r23`, at
`R.REC / f"train_s{s}.parquet"`, filtered to `country` and `gid`s present in the sample, per the
existing pattern). Compute empty-address keys and candidates via Task 1's functions, keep only new
pairs via `new_pairs_only`. Load FIT truth via `access.load_labels("FIT")`. Score `f05_base` using the
holdout's existing `label` column directly (no rescoring needed — it's already ground truth). For
`f05_with_new_key`: the new pairs need a probability to compare against the real v3.2 threshold, so
score them with `lgbm_v32.txt` after filling their missing block-rank/numnoise/pseudo features with
the sentinel identified in the Review Focus item (read `run_v3.py`'s merge code first); reuse
`enforce_g_m3` before scoring. Compute all report fields, write `measure_report.json` via
`atomic_write_parquet`'s sibling pattern (plain `Path.write_text(json.dumps(...))`, atomic via
`.tmp`+rename, matching the project's existing atomic-write convention elsewhere).

Run: `PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.measure_empty_addr`
Expected: `data/_helper/exp_empty_addr/measure_report.json` exists with a `go_no_go` field.

- [ ] **Step 6: Read the report and record the decision**

If `go_no_go == "no_go"`: stop. Do not start Task 3. State the numbers plainly to the user (net gain,
how it compares to the 08:42 log's ~175/7272 ≈ 2.4% estimate) and mark this plan complete-as-a-
measurement, not complete-as-a-feature.
If `go_no_go == "go"`: proceed to Task 3.

- [ ] **Step 7: Commit**

```bash
git add src/amlc/experimental/measure_empty_addr.py tests/experimental/test_measure_empty_addr.py data/_helper/exp_empty_addr/measure_report.json
git commit -m "exp: measure empty-address exact-name key's net F0.5 gain on FIT holdout

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3 (CONDITIONAL — only if Task 2's `go_no_go == "go"`): Full test-set integration

**Files:**
- Create: `src/amlc/experimental/build_empty_addr_test_candidates.py`
- Test: none new (reuses Task 1's tested functions; this task is a one-shot batch script over the
  full test set, verified by an assertion-based regression check, not a pytest suite)

**Interfaces:**
- Consumes: Task 1's functions; the full test record pool (`data/_v2/records/test_s1/2/3.parquet`);
  the existing v3.2 test candidate files (`data/_v32/test_scored/`) as the "already have" set to
  exclude against.
- Produces: `data/_helper/exp_empty_addr/new_test_candidates/<country>_*.parquet` (new pairs only,
  scored with `lgbm_v32.txt`, features filled per Task 2's sentinel), and a
  `data/_helper/exp_empty_addr/integration_report.json` with before/after `n_predictions`,
  `s1_with_predictions`, and a full `amlc.eval.metric`-based estimate is NOT possible here (no test
  labels) — report candidate-count deltas only, plainly labeled as "not label-verified."

- [ ] **Step 1: Implement the full-scale batch build, one country/file at a time, logging free RAM
  per file via `amlc.pipeline.mem.free_gb` (same pattern as every other pipeline step in this repo)**

- [ ] **Step 2: Run it end to end**

Run: `PYTHONPATH=src AMLC_KA=10 python -m amlc.experimental.build_empty_addr_test_candidates`
Expected: one output file per test country, `integration_report.json` written.

- [ ] **Step 3: Verify additive-only via a regression assertion**

Assert every new file's `(s1_gid, s23_gid)` keys are disjoint from the corresponding
`data/_v32/test_scored/<country>_*.parquet` keys — raise `AssertionError` and stop if not.

- [ ] **Step 4: Report to the user for a decision on whether to merge into a real submission**

This task deliberately stops short of touching `output/v32` or producing a new `matching_results.tsv`
— that merge is a separate, explicit decision the user makes after seeing Task 3's numbers, not an
automatic next step.

- [x] **Step 5: Commit**

```bash
git add src/amlc/experimental/build_empty_addr_test_candidates.py data/_helper/exp_empty_addr/integration_report.json
git commit -m "exp: full test-set empty-address exact-name candidates (not yet merged into output)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
