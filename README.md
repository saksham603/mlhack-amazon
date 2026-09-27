# AMLC 2026 — Entity Resolution

Business-record matching across India, US and France for the Amazon ML Challenge 2026.
Given a query business record (S1) and two candidate pools (S2, S3), predict which S2/S3
records refer to the same real-world business as each S1 record. Scored as macro F0.5
(beta = 0.25, precision-weighted), averaged per S1 then across all S1.

**Team:** Divyansh Singh, Saksham Bansal, Siddharth Mohan Jha, Ayudh

## Current result

| Submission | Real leaderboard F0.5 | Status |
|---|---|---|
| **v3.2** | **96.6%** | Confirmed, protected fallback |
| v3.1 | — | Validation-only estimate (96.46%); real score was 95.2% (France scored 0 — no training labels for that country) |

v3.2's own held-out validation split (VAL-B, 20,000 scored S1 records): F0.5 = 0.9763,
precision 0.994, recall 0.943, threshold 0.74. Per-country: India 0.9745, US 0.9782.

## Architecture

```
raw records (S1/S2/S3, 3 countries)
        |
   cleaning/            -- name & address normalization, tokenization
        |
   blocking/v3.py        -- candidate generation (name-key + address-key blocking,
        |                   IDF-capped, separate top-k per key kind)
        |
   features/              -- w3.py (fuzzy name/address sim, live ctx_* rank/gap features),
        |                   numnoise.py (address number-noise), pseudo-word features
        |
   model/lowmem_train.py  -- LightGBM, single pre-sized float32 matrix, streaming fit
        |
   model/predict.py       -- threshold + enforce_g_m3 (one S1 per S23 group constraint)
        |
   pipeline/assemble_merge.py -- merges independently-scored per-country submissions
        |
   output/matching_results.tsv  -- final leaderboard upload
```

### Pipeline versions (`src/amlc/pipeline/`)

- **`run_v3.py`** — base pipeline, 53 features, threshold grid search, low-memory streaming
  training.
- **`run_v31.py`** — v3 + separate name/address blocking indexes, memory-safe candidate
  generation for the full country scale.
- **`run_v32.py`** — v3.1 + number-noise address features (8) + pseudo-word features (4) =
  65 features total. **This is the confirmed 96.6% submission.**
- **`chain_day2.py` / `chain_v32.py`** — Windows Task Scheduler drivers that chain the
  pipeline stages unattended.

### Key modules

- `amlc/blocking/v3.py` — candidate generation. `NAME_KINDS` (exact-normalized, rare-name-
  sketch, rare-name-key) vs `ADDR_KINDS` (address-normalized, city-normalized-street),
  IDF rarity caps per key, `999` / `0.0` as the "not found" sentinel convention throughout
  the codebase.
- `amlc/features/w3.py` — fuzzy name/address similarity features plus live per-S1
  context features (`ctx_n_rank/gap`, `ctx_a_rank/gap`, `ctx_c_rank/gap`) computed via
  Polars window functions (`.over("s1_gid")`).
- `amlc/model/lowmem_train.py` — trains LightGBM from one pre-sized float32 matrix built
  incrementally per-file, to keep peak memory bounded on a laptop-scale machine.
- `amlc/model/predict.py` — `enforce_g_m3` applies the competition's G-M3 constraint (each
  S23 group may match at most one S1) at prediction time.
- `amlc/eval/metric.py` / `amlc/eval/scorer.py` — macro F0.5 computation; `scorer.py` is the
  only module allowed to see raw VALIDATION labels, to keep the rest of the codebase
  leakage-safe.
- `amlc/model/validate.py` — VAL-A/VAL-B split discipline (tune threshold on A, report
  honestly on B) so no number is ever tuned against the data it's reported on.

### Experimental modules (`src/amlc/experimental/`)

Every experiment here follows the same protocol: build in isolation from the frozen
production pipeline, measure real net F0.5 gain against a pre-registered kill bar
**before** deciding whether to integrate, and never touch the confirmed v3.2 model or
submission while measuring.

| Experiment | Result | Verdict |
|---|---|---|
| Empty-address exact-name candidate key | +471 true links reachable as candidates, but 0 crossed the decision threshold | **NO_GO** |
| `ctx_c_adj_*` context-feature fix (mirrors name context onto missing-address records) | VAL-B delta −0.00019 | **DROP** |
| Soundex phonetic name features (`snd_eq`, `snd_jacc`) | VAL-B delta +0.00015 (below +0.002 keep bar) | **DROP** |
| GPU embedding-based auxiliary retrieval (multilingual MiniLM, Kaggle T4) | 403,928 new candidate pairs, only 196 true links (0.05% precision), net F0.5 −0.0000005 | **NO_GO** |

Conclusion from today's experiments: the architecture is close to a genuine ceiling
(~96–97%) for this feature/blocking/model design. Meaningful further gains would need a
fundamentally different retrieval or matching mechanism, or resolving inherent
data ambiguity (some records are byte-identical in name+address text with different
ground-truth labels — not resolvable by any feature).

## Data layout

```
data/            -- not tracked in git (see .gitignore); raw + processed records, splits,
                     candidates, VAL-A/VAL-B holdouts, per-experiment scratch output
models/          -- trained LightGBM boosters (*.txt); not tracked in git, regenerate via
                     the pipeline entry points below
output/          -- final submission files (matching_results.tsv, candidate_pairs.tsv);
                     not tracked in git
```

## Running it

```bash
# install
python -m venv .venv
.venv/Scripts/activate       # Windows
pip install -r requirements.lock

# run tests
pytest

# run the confirmed v3.2 pipeline (requires data/ already populated)
PYTHONPATH=src AMLC_KA=10 python -m amlc.pipeline.run_v32
```

Key dependencies (pinned in `requirements.lock`): `lightgbm==4.7.0`, `polars==1.44.2`,
`numpy==2.5.3`, `pyarrow==25.0.1`, `duckdb==1.5.5`, `pytest==9.1.1`.

## Testing discipline

Every non-trivial module has a matching test file under `tests/`, mirroring `src/amlc/`'s
structure. New features and bug fixes follow TDD (red-green-commit): write the failing
test, watch it fail for the right reason, write the minimal code to pass it.

## Submission

Only `matching_results.tsv` is uploaded to the leaderboard (`candidate_pairs.tsv` is an
internal diagnostic artifact, not part of the official submission format). See
`submission/package_README.md` for the exact upload steps used for the confirmed v3.2
submission.
