# AMLC 2026 — Agent Context

This file is read automatically by Codex/agent CLIs opened in this repo. It is the
up-to-date source of truth. `PROJECT_CONTEXT.md` is an **earlier, now-stale** snapshot
from the data-cleaning stage (Stage 0/1) — useful only for early-stage historical decisions
(data split seed, provenance verification, cleaning gate discipline); everything about the
actual modeling pipeline and current results supersedes it and lives here and in `README.md`.

## What this project is

Amazon ML Challenge 2026: business entity resolution across India, US and France. For
each query record (S1), predict which records in two candidate pools (S2, S3) refer to
the same real-world business. Metric: **macro F0.5** (β=0.25, precision-weighted),
averaged per-S1 then across all S1. Rules: no external lookups/APIs, final model must be
MIT/Apache-2.0 licensed and ≤8B params.

**Team:** Divyansh Singh, Saksham Bansal, Siddharth Mohan Jha, Ayudh.

## Current state (read this first)

- **Confirmed real leaderboard score: 96.6% (v3.2 submission).** This is the protected
  baseline — never overwrite `models/lgbm_v32.txt` or `output/v32/` without an explicit
  go-ahead, and never re-run `run_v32.py`'s training step casually, since it's the known-
  good artifact everything else is measured against.
- v3.2's own held-out VAL-B: F0.5 0.9763, precision 0.994, recall 0.943, threshold 0.74.
- Four same-day experiments (empty-address candidate key, a context-feature fix, Soundex
  phonetic features, GPU embedding retrieval) all came back **NO_GO / DROP** against
  pre-registered kill bars — see README.md's experiment table for exact numbers. This was
  treated as a real signal, not a failure: the architecture is near a genuine ceiling
  (~96-97%) for this feature/blocking/model design without a fundamentally different
  retrieval mechanism.

## Key decisions worth knowing (so they aren't re-litigated)

1. **No code changes without an explicit go.** A plan being approved is not itself a go.
   This discipline was followed throughout and should continue.
2. **Never reopen old/abandoned code "for reference."** Early in the project there was a
   prior pipeline (by a different tool) with 12 catalogued correctness bugs (worst: treated
   any accented character as "different script," causing false merges especially exposed
   in France — measured precision 0.11 India / 0.42 US on that code path). The rule set
   then and still in force: write fresh from spec/tests, never peek at discarded buggy code
   even "just to check."
3. **No claim without a measurement.** Every reported number must come from a logged run.
4. **FIT / VALIDATION / LOCKBOX split is frozen.** 70/20/10, seed `20260925`, stratified by
   country × match-count bucket. VAL-A is used to tune (e.g. threshold), VAL-B is reported
   on and never tuned against — this split discipline is why validation numbers can be
   trusted and is enforced in `amlc/model/validate.py`.
5. **`scorer.py` is the only module allowed to see raw VALIDATION labels** — keeps the rest
   of the codebase leakage-safe by construction, not by convention.
6. **Experiments are isolated and kill-bar gated.** Every experimental module in
   `src/amlc/experimental/` is built without touching the frozen production pipeline,
   measures real net F0.5 gain against a bar decided *before* seeing the result, and is
   reported honestly even when the result is negative. This discipline is why the four
   same-day experiments above were correctly called NO_GO instead of being rationalized in.
7. **Only `matching_results.tsv` is uploaded to the leaderboard** —
   `candidate_pairs.tsv` is an internal diagnostic, not part of the official format.
8. **Large binaries are not in git.** `data/`, `models/*.txt`, `output/` are gitignored —
   too large / regenerable from the pipeline. They must be transferred separately
   (shared folder / drive) if you need to actually run the pipeline on a new machine,
   not just read the code.

## How it runs

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.lock
pytest                          # full test suite, mirrors src/amlc/ structure

# full confirmed pipeline (requires data/ already populated — see "transferring" above)
PYTHONPATH=src AMLC_KA=10 python -m amlc.pipeline.run_v32
```

See `README.md` for the full architecture breakdown (blocking -> features -> model ->
decision layer -> assembly), module-by-module descriptions, and the experiment results
table. See `submission/package_README.md` for the exact upload steps.

## Working style established on this project

- TDD (red-green-commit) for new features/bugfixes, one test file per source module under
  `tests/`.
- Pre-registered success/kill criteria for any experiment, decided before running it.
- Every real retrain/experiment result cross-checked (independent reproduction, checksum
  on any handed-off file) before being trusted or wired in.
