"""Experiment (2026-09-27 07:30, light enough to run beside the chain): before spending 3 laptop hours on
the pass-2 chain steps, measure whether a second pass or a decision rule beats the plain v3.1 model.

Data: half of the FIT early-stopping holdout (S1s the v3.1 model never trained on; FIT labels only,
full truth incl. links the candidates missed). S1s are split by hash into three parts:
  P0 trains the proxy second-pass model, P1 tunes every method's parameters, P2 reports.
Methods (all end with G-M3, scored with amlc.eval.metric):
  base     : p1 >= t
  bestset  : per S1 the top-k set with the highest expected F0.5 (F_k = 1.25*sum_top_k(p) / (0.25*S + k),
             S = sum of p over the S1's candidates + c; empty scores prod(1-p)); only p >= floor eligible
  rescue   : base, plus candidates with p1 >= t_low whose name+address resemble a confident sibling
  pass2    : LightGBM on base features + pass-2 features, trained on P0 (much less data than the real
             pass-2, so this under-states it)
  pass2+bestset
Writes data/_v31/exp/rules_proxy.json.  Run: PYTHONPATH=src AMLC_KA=10 python -m amlc.model.exp_rules
"""
import ctypes
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from amlc.eval import metric
from amlc.foundation import access
from amlc.model.pass2 import P2_FEATURES, add_pass2
from amlc.model.predict import enforce_g_m3
from amlc.pipeline import run_v3 as R
from amlc.pipeline.mem import atomic_write_parquet, free_gb

THREADS = 4
EXP = R.OUT / "exp"
FEATS2 = R.FEATURES + P2_FEATURES


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, f"[free RAM {free_gb()} GB]", flush=True)


def load_holdout() -> pl.DataFrame:
    cache = EXP / "holdout_p1.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    parts = []
    for f in sorted(glob.glob(str(R.OUT / "fit" / "*.parquet"))):
        d = pl.read_parquet(f, columns=["s1_gid", "s23_gid", "country", "label", *R.FEATURES])
        parts.append(d.filter((d["s1_gid"].hash(seed=R.SEED + 1) % 20) == 0))  # subset of the %10 holdout
    d = pl.concat(parts)
    log(f"holdout rows {d.height:,}, S1 {d['s1_gid'].n_unique():,}")
    booster = lgb.Booster(model_file=str(R.MODEL_PATH))
    d = d.with_columns(pl.Series("p1", booster.predict(R.to_x(d), num_threads=THREADS)))
    log("p1 predicted")
    r23_gids = d["s23_gid"].unique()
    r23 = pl.concat([pl.scan_parquet(R.REC / f"train_s{s}.parquet").select("gid", "core", "addr")
                     .filter(pl.col("gid").is_in(r23_gids)).collect() for s in (2, 3)])
    chunks = []
    for k in range(8):  # pass-2 features per S1 chunk, to keep the sibling pairs small
        c = d.filter((d["s1_gid"].hash(seed=5) % 8) == k)
        p2 = add_pass2(c.select("s1_gid", "s23_gid"), c["p1"].to_numpy(), r23).drop("p1")
        chunks.append(c.join(p2, on=["s1_gid", "s23_gid"], how="left", validate="1:1"))
    d = pl.concat(chunks)
    log("pass-2 features added")
    atomic_write_parquet(d, cache)
    return d


def f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.Series, prob: str) -> float:
    return metric.macro(metric.per_s1_f05(enforce_g_m3(pred.select("s1_gid", "s23_gid", pl.col(prob).alias("prob")))
                                          .select("s1_gid", "s23_gid"), truth, s1))


def by_country(pred, truth, s1_df, prob) -> dict:
    return {c: round(f05(pred.join(g, on="s1_gid", how="semi"), truth, g["s1_gid"], prob), 5)
            for c, g in ((c, s1_df.filter(pl.col("country") == c)) for c in sorted(s1_df["country"].unique()))}


def bestset(d: pl.DataFrame, prob: str, floor: float, c: float) -> pl.DataFrame:
    s = d.sort(["s1_gid", prob], descending=[False, True]).with_columns(
        (pl.col(prob).sum().over("s1_gid") + c).alias("S"),
        pl.col(prob).cum_sum().over("s1_gid").alias("tp"),
        pl.col(prob).cum_count().over("s1_gid").alias("k"),
        (1 - pl.col(prob)).clip(1e-9, 1).log().sum().over("s1_gid").exp().alias("f_empty"))
    s = s.with_columns((1.25 * pl.col("tp") / (0.25 * pl.col("S") + pl.col("k"))).alias("fk"))
    elig = s.filter(pl.col(prob) >= floor)
    best = elig.group_by("s1_gid").agg(pl.col("fk").max().alias("fbest"), pl.col("k").sort_by("fk").last().alias("kbest"),
                                       pl.col("f_empty").first())
    keep = best.filter(pl.col("fbest") > pl.col("f_empty")).select("s1_gid", "kbest")
    return s.join(keep, on="s1_gid").filter(pl.col("k") <= pl.col("kbest")).select("s1_gid", "s23_gid", prob)


def main():
    if os.name == "nt":  # below-normal priority: the chain's test scoring keeps the CPU first
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    EXP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load_holdout()
    part = (d["s1_gid"].hash(seed=11) % 3).cast(pl.Int8)
    d = d.with_columns(pl.Series("part", part))
    s1_df = d.select("s1_gid", "country", "part").unique("s1_gid")
    truth = access.load_labels("FIT").select("s1_gid", "s23_gid").join(s1_df.select("s1_gid"), on="s1_gid", how="semi")
    P = {k: s1_df.filter(pl.col("part") == k) for k in (0, 1, 2)}
    D = {k: d.filter(pl.col("part") == k) for k in (0, 1, 2)}
    tr = {k: truth.join(P[k].select("s1_gid"), on="s1_gid", how="semi") for k in (0, 1, 2)}
    rep = {"rows": d.height, "s1": s1_df.height, "s1_per_part": {k: P[k].height for k in P}}

    def score(k, pred, prob):
        return f05(pred, tr[k], P[k]["s1_gid"], prob)

    def tune_t(prob):
        curve = {t: score(1, D[1].filter(pl.col(prob) >= t), prob) for t in R.THRESHOLDS}
        return max(curve, key=curve.get)

    # base
    t = tune_t("p1")
    base = D[2].filter(pl.col("p1") >= t)
    rep["base"] = {"t": t, "f05_P2": score(2, base, "p1"), "by_country": by_country(base, tr[2], P[2], "p1")}
    log(f"base t={t} P2 {rep['base']['f05_P2']:.5f} {rep['base']['by_country']}")

    # best-set rule on p1
    grid = [(fl, c) for fl in (0.05, 0.1, 0.2, 0.3, 0.4) for c in (0.0, 0.05, 0.1, 0.2)]
    g = {(fl, c): score(1, bestset(D[1], "p1", fl, c), "p1") for fl, c in grid}
    fl, c = max(g, key=g.get)
    bs = bestset(D[2], "p1", fl, c)
    rep["bestset"] = {"floor": fl, "c": c, "f05_P2": score(2, bs, "p1"), "by_country": by_country(bs, tr[2], P[2], "p1")}
    log(f"bestset floor={fl} c={c} P2 {rep['bestset']['f05_P2']:.5f}")

    # sibling rescue on p1
    best, arg = -1, None
    for tl in (0.1, 0.2, 0.3, 0.4, 0.5):
        for s in (80, 85, 90, 95):
            for tt in (t, round(t + 0.06, 2)):
                sel = D[1].filter((pl.col("p1") >= tt) | ((pl.col("p1") >= tl) & (pl.col("sib_both_max") >= s)))
                v = score(1, sel, "p1")
                if v > best:
                    best, arg = v, (tt, tl, s)
    tt, tl, s = arg
    rs = D[2].filter((pl.col("p1") >= tt) | ((pl.col("p1") >= tl) & (pl.col("sib_both_max") >= s)))
    rep["rescue"] = {"t": tt, "t_low": tl, "sib": s, "f05_P2": score(2, rs, "p1"), "by_country": by_country(rs, tr[2], P[2], "p1")}
    log(f"rescue {arg} P2 {rep['rescue']['f05_P2']:.5f}")

    # proxy pass-2 model trained on P0
    def xs(df):
        return df.select([pl.col(f).cast(pl.Float32) for f in FEATS2]).to_numpy()
    es = (D[0]["s1_gid"].hash(seed=3) % 10) == 0
    params = dict(R.PARAMS, learning_rate=0.05, num_threads=THREADS)
    dtr = lgb.Dataset(xs(D[0].filter(~es)), label=D[0].filter(~es)["label"].to_numpy(), feature_name=FEATS2)
    dva = lgb.Dataset(xs(D[0].filter(es)), label=D[0].filter(es)["label"].to_numpy(), reference=dtr)
    b2 = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    for k in (1, 2):
        D[k] = D[k].with_columns(pl.Series("p2", b2.predict(xs(D[k]), num_iteration=b2.best_iteration, num_threads=THREADS)))
    t2 = tune_t("p2")
    p2 = D[2].filter(pl.col("p2") >= t2)
    rep["pass2"] = {"t": t2, "rounds": b2.best_iteration, "f05_P2": score(2, p2, "p2"), "by_country": by_country(p2, tr[2], P[2], "p2"),
                    "top_gain": sorted(zip(FEATS2, b2.feature_importance("gain").round().tolist()), key=lambda x: -x[1])[:10]}
    log(f"pass2 t={t2} rounds={b2.best_iteration} P2 {rep['pass2']['f05_P2']:.5f}")
    g2 = {(fl, c): score(1, bestset(D[1], "p2", fl, c), "p2") for fl, c in grid}
    fl2, c2 = max(g2, key=g2.get)
    p2b = bestset(D[2], "p2", fl2, c2)
    rep["pass2_bestset"] = {"floor": fl2, "c": c2, "f05_P2": score(2, p2b, "p2"), "by_country": by_country(p2b, tr[2], P[2], "p2")}
    log(f"pass2+bestset P2 {rep['pass2_bestset']['f05_P2']:.5f}")

    # where the remaining loss is (P2, base): missing from candidates vs below threshold vs false positives
    cand = D[2].select("s1_gid", "s23_gid")
    tp_all = tr[2].join(cand, on=["s1_gid", "s23_gid"], how="semi").height
    rep["base_errors_P2"] = {"true_links": tr[2].height, "in_candidates": tp_all,
                             "predicted_true": base.filter(pl.col("label") == 1).height,
                             "false_pos": base.filter(pl.col("label") == 0).height}
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "rules_proxy.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log(json.dumps({k: (v["f05_P2"] if isinstance(v, dict) and "f05_P2" in v else v) for k, v in rep.items()}, default=str))


if __name__ == "__main__":
    main()
