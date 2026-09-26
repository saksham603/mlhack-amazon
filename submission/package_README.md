# Business Entity Resolution — reproduction guide

Pipeline: raw TSVs → lossless Parquet (bronze) → cleaned fields (silver) → cleanup v2 records →
candidate generation (v1 IDF keys ∪ v3 name×address keys) → 51 pairwise features → LightGBM →
threshold + one-S1-per-record assignment → `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

All code is in `src/amlc/`. Every command below is run from this folder with `PYTHONPATH=src`.
Hardware used: 4-core Intel i5-1135G7 laptop, 16 GB RAM, no GPU. Runtimes are for that machine.

## 0. Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
```

The project root is set in `src/amlc/foundation/access.py` (`ROOT`) and `src/amlc/foundation/ingest.py`
(`RAW`, `BRONZE`); change those two lines if you run from another folder. Copy the six competition
TSVs (`train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`) into `data/raw/v1/`.

## 1. Data foundation (bronze, labels, frozen split) — ~10 min

```bash
python -m amlc.foundation.ingest          # TSV -> bronze Parquet, all columns kept as strings
python -m amlc.foundation.f6_labels       # ground truth -> (s1, s23) link table
python -m amlc.foundation.f7_split        # FIT / VALIDATION / LOCKBOX split of train S1 (70/20/10, seed 20260925)
python -m amlc.foundation.f8_profile      # data profile
python -m amlc.foundation.f10_manifest    # MANIFEST_v1 (hashes of every artifact)
python -m amlc.foundation.build_v2        # manifest-verified v2 split + label artifacts
```

## 2. Cleaning (silver) — ~25 min

```bash
python -m amlc.cleaning.c1_run            # Unicode / script normalisation, learned Indic transliteration table
python -m amlc.cleaning.silver_run        # name tokens, legal form, address components, house number, ZIP
```

## 3. v1 candidates (IDF-weighted rare keys, budget 150) — ~4 h in total

```bash
python -m amlc.model.build_datasets India 3000     # FIT sample + VALIDATION candidates
python -m amlc.model.build_datasets US 3000
python -m amlc.model.build_fit_extra India 3000    # 250k extra FIT S1 per country
python -m amlc.model.build_fit_extra US 3000
python -m amlc.model.run_test_country India 3000   # TEST candidates (one country per process)
python -m amlc.model.run_test_country US 3000
python -m amlc.model.run_test_country France 3000
```

## 4. Cleanup v2 records — ~11 min

```bash
python -m amlc.features.norm_v2           # data/_v2/records/*.parquet (+ native-script state map, FIT pairs only)
```

## 5. Candidates v3, features, model, test scoring — ~3 h

```bash
python -m amlc.pipeline.run_v3 val        # VALIDATION candidates (v1 top-30 ∪ v3 top-20) + features
python -m amlc.pipeline.run_v3 fit        # FIT candidates + features + labels
python -m amlc.pipeline.run_v3 train      # LightGBM, threshold tuned on VALIDATION half A, report on half B
python -m amlc.pipeline.run_v3 test       # TEST candidates + features + probabilities
python -m amlc.pipeline.run_v3 assemble final   # output/final/{matching_results,candidate_pairs}.tsv
```

## 6. Validate

```bash
python <student_resource>/utils/validate_submission.py --matching output/final/matching_results.tsv \
    --candidate output/final/candidate_pairs.tsv --test-dir <student_resource>/dataset/test --check-ids
```

## Notes

- Models: LightGBM (MIT licence) trained only on the competition training data. No pretrained or
  external models. Libraries: polars, numpy, lightgbm, rapidfuzz (MIT), anyascii (ISC).
- External data: none. Hand-typed lists only: US state names/codes, Indian state names/codes, legal
  forms, street-type abbreviations, titles. The native-script state map is learned from FIT pairs.
- VALIDATION labels are only read inside `amlc.eval.scorer`; LOCKBOX is read once, at the end.
- `candidate_pairs.tsv` is exactly the candidate set the model scores.
