# V1 submission report (2026-09-26)

## Submission files (official validator PASS with `--check-ids`, 72 s)

| File | Size | Rows |
|---|---|---|
| `output/matching_results.tsv` | 77.4 MB (77,421,830 B) | 1,732,544 S1 (1,482,874 non-empty) |
| `output/candidate_pairs.tsv` | 687.0 MB (686,995,782 B) | 1,732,544 S1 (45 empty), 51,567,118 pairs |

The k=150 versions are kept as `output/*_k150.tsv` (2.69 GB candidates). They cannot pass the official
validator on this 16 GB machine, which holds every candidate id in Python sets.

## Model and score

- LightGBM `models/lgbm_v1.txt`, 9 pairwise features, trained on 50k FIT S1 per country (India+US),
  12.0M candidate rows (299,524 positive). France has no labels and uses the same model.
- Threshold 0.64 was tuned on VAL-A (F0.5 0.8281). G-M3 is enforced globally: each S2/S3 is kept only on its best S1.
- Leakscan found 0 hits before any F0.5 was trusted.

## Candidate budget k (VAL-B, 20,000 S1, threshold 0.64)

| k | VAL-B F0.5 | India | US | cand/S1 |
|---|---|---|---|---|
| 5 | 0.7277 | 0.6115 | 0.8452 | 5.0 |
| 10 | 0.7514 | 0.6461 | 0.8579 | 10.0 |
| 20 | 0.7680 | 0.6719 | 0.8652 | 19.9 |
| **30 (submitted)** | **0.7780** | **0.6894** | **0.8676** | **29.8** |
| 50 | 0.7939 | 0.7193 | 0.8694 | 49.2 |
| 150 | 0.8259 | 0.7767 | 0.8757 | 120.5 |

The rule "smallest k within 0.002 of k=150" would pick k=150. We chose k=30 so the submission passes the
validator; it costs 0.048 F0.5, almost all of it in India.

## Test run (full TEST: 1,732,544 S1)

| Country | S1 | Candidates (k=150) | Runtime | Peak RSS |
|---|---|---|---|---|
| India | 809,986 | 84,319,233 | 57 min (21:49–22:46) | 8.89 GB |
| US | 663,106 | 88,832,105 | ~2h09 wall (two session restarts, resumed) | 9.38 GB |
| France | 259,452 | 33,554,286 | 16 min | 9.41 GB |

India and US passed a pair-level candidate/feature/scored integrity check on every batch (0 bad
batches in the final state). Output assembly took 42 s at k=30, local checks 9 s, and the official validator 72 s.

## Known gaps (ranked by measured loss)

1. **Model error ~12 pts** (FIT holdout: F0.5 0.824 on all true links vs 0.876 on in-candidate links).
   Per-S1 context features gave +1.0 pt. Next steps are a bigger FIT set, context features, and address/IDF features.
2. **Candidate budget / blocking rank.** At k=30 India loses 0.087 F0.5 against k=150. True matches are
   ranked too low (83% of India's blocking misses share a name token but lose on rank/cap). Better
   blocking ranking would let a small k keep most of the k=150 score.
3. **Blocking recall** (~5 pts): India 79.5%, US 93.9% on the FIT sample.
4. **Blocking is not bit-reproducible.** Candidate sets differ slightly between runs, likely from multithreaded float sums flipping ties at the cutoff.
5. `requirements.lock` has not been regenerated since lightgbm was installed.
