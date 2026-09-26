"""Low-memory LightGBM training from per-batch parquet files (added 2026-09-27 05:15 after Python
peaks near 8 GB froze the laptop). Pass 1 reads only the columns the negative subsampling needs, to
size one float32 matrix; pass 2 fills it file by file, training rows from the front and holdout rows
(10% of S1s by hash) from the back, so the holdout is a view, never a copy. The matrix is released as
soon as LightGBM has binned it (LightGBM keeps its own 1-byte-per-value copy)."""
import gc

import lightgbm as lgb
import numpy as np
import polars as pl

from amlc.pipeline.mem import free_gb

SUB_COLS = ["s1_gid", "s23_gid", "label", "v1_rank", "v3_rank", "n_tset", "a_tset"]


def _holdout(d: pl.DataFrame, seed: int) -> np.ndarray:
    return ((d["s1_gid"].hash(seed=seed + 1) % 10) == 0).to_numpy()


def fit_lowmem(n_files: int, load, feats: list, subsample, params: dict, seed: int, model_path, log,
               sub_cols: list = SUB_COLS) -> lgb.Booster:
    """load(i, cols) -> DataFrame of file i with at least cols (None = all needed); subsample(df) -> df with 'w'."""
    counts = []
    for i in range(n_files):
        d = subsample(load(i, sub_cols))
        h = _holdout(d, seed)
        counts.append((int((~h).sum()), int(h.sum())))
    ntr, nho = sum(c[0] for c in counts), sum(c[1] for c in counts)
    log(f"lowmem: {ntr:,} train + {nho:,} holdout rows x {len(feats)} features = {(ntr + nho) * len(feats) * 4 / 2**30:.2f} GB [free {free_gb()} GB]")
    X = np.empty((ntr + nho, len(feats)), dtype=np.float32)
    y = np.empty(ntr + nho, dtype=np.float32)
    w = np.empty(ntr + nho, dtype=np.float32)
    a, b = 0, ntr
    need = list(dict.fromkeys(sub_cols + feats))
    for i in range(n_files):
        d = subsample(load(i, need))
        h = _holdout(d, seed)
        xs = d.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()
        ys, ws = d["label"].to_numpy().astype(np.float32), d["w"].to_numpy()
        if (int((~h).sum()), int(h.sum())) != counts[i]:
            raise AssertionError(f"file {i}: row counts changed between passes. STOP.")
        n1, n2 = counts[i]
        X[a:a + n1], y[a:a + n1], w[a:a + n1] = xs[~h], ys[~h], ws[~h]
        X[b:b + n2], y[b:b + n2], w[b:b + n2] = xs[h], ys[h], ws[h]
        a, b = a + n1, b + n2
        del d, xs
    dtr = lgb.Dataset(X[:ntr], label=y[:ntr], weight=w[:ntr], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X[ntr:], label=y[ntr:], weight=w[ntr:], reference=dtr, feature_name=feats, free_raw_data=True)
    dtr.construct()
    dva.construct()
    del X, y, w
    gc.collect()
    log(f"lowmem: datasets binned, raw matrix released [free {free_gb()} GB]")
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    log(f"lowmem: saved {model_path.name}, {booster.best_iteration} rounds [free {free_gb()} GB]")
    return booster
