"""Day-2 decision rule (plan Phase D, pulled forward): per-country main threshold plus a
best-candidate rescue. An S1 whose best candidate is below the main threshold still gets that one
candidate when its probability clears a lower bar (94% of train S1s have at least one match; v1 left
22% of India test S1s empty). G-M3 (each S2/S3 on its best S1) is enforced after.

Tuned on VALIDATION half A through amlc.eval.scorer (aggregate metrics only), reported on half B.
Countries without VALIDATION labels (France) use the pooled setting.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.decide
"""
import json
import time

import polars as pl

from amlc.eval import scorer
from amlc.model import validate as V
from amlc.model.predict import enforce_g_m3
from amlc.model.run_v2 import REPORTS

MAIN_GRID = [round(0.02 * i, 2) for i in range(10, 48)]   # 0.20 .. 0.94
LOW_GRID = [round(0.05 * i, 2) for i in range(1, 19)]     # 0.05 .. 0.90


def best_per_s1(scored: pl.DataFrame) -> pl.DataFrame:
    return (scored.sort(["s1_gid", "prob", "s23_gid"], descending=[False, True, False])
            .group_by("s1_gid", maintain_order=True).head(1))


def predict(scored: pl.DataFrame, best: pl.DataFrame, t_main: dict, t_low: dict, default: str = "_pooled") -> pl.DataFrame:
    """t_main / t_low: {country: threshold}; countries not listed use t_main[default] / t_low[default]."""
    def thr(table, col):
        return (pl.col("country").replace_strict(table, default=table[default], return_dtype=pl.Float64)
                if len(table) > 1 else pl.lit(table[default])).alias(col)
    main = scored.with_columns(thr(t_main, "tm")).filter(pl.col("prob") >= pl.col("tm"))
    rescue = (best.with_columns(thr(t_main, "tm"), thr(t_low, "tl"))
              .filter((pl.col("prob") < pl.col("tm")) & (pl.col("prob") >= pl.col("tl"))))
    pred = pl.concat([main.select("s1_gid", "s23_gid", "prob"), rescue.select("s1_gid", "s23_gid", "prob")])
    return enforce_g_m3(pred.unique(["s1_gid", "s23_gid"])).select("s1_gid", "s23_gid")


def tune(scored: pl.DataFrame, a: pl.Series) -> dict:
    sa = scored.filter(pl.col("s1_gid").is_in(a.implode()))
    best = best_per_s1(sa)
    countries = sorted(sa["country"].unique().to_list())
    # 1) main threshold per country, no rescue (t_low = t_main disables it)
    res = {}
    for t in MAIN_GRID:
        r = scorer.score(predict(sa, best, {"_pooled": t}, {"_pooled": t}), s1_subset=a)
        res[t] = r
    t_pool = max(MAIN_GRID, key=lambda t: res[t]["f05"])
    t_c = {c: max(MAIN_GRID, key=lambda t: res[t]["by_country"][c]["f05"]) for c in countries}
    # 2) rescue bar per country, given its main threshold
    low_res = {}
    for tl in LOW_GRID:
        tm = {"_pooled": t_pool, **t_c}
        tlo = {"_pooled": min(tl, t_pool), **{c: min(tl, t_c[c]) for c in countries}}
        low_res[tl] = scorer.score(predict(sa, best, tm, tlo), s1_subset=a)
    l_pool = max(LOW_GRID, key=lambda t: low_res[t]["f05"])
    l_c = {c: max(LOW_GRID, key=lambda t: low_res[t]["by_country"][c]["f05"]) for c in countries}
    return {"t_main": {"_pooled": t_pool, **t_c}, "t_low": {"_pooled": min(l_pool, t_pool), **{c: min(l_c[c], t_c[c]) for c in countries}},
            "val_a_no_rescue_pooled": res[t_pool]["f05"], "val_a_rescue_pooled": low_res[l_pool]["f05"]}


def main() -> int:
    t0 = time.time()
    scored = pl.read_parquet(REPORTS / "val_scored.parquet")
    a, b = V.sample_validation_halves()
    tuned = tune(scored, a)
    sb = scored.filter(pl.col("s1_gid").is_in(b.implode()))
    best_b = best_per_s1(sb)
    out = {**tuned}
    out["val_b_single_threshold"] = scorer.score(
        predict(sb, best_b, {"_pooled": tuned["t_main"]["_pooled"]}, {"_pooled": tuned["t_main"]["_pooled"]}), s1_subset=b)
    out["val_b_per_country"] = scorer.score(predict(sb, best_b, tuned["t_main"], tuned["t_main"]), s1_subset=b)
    out["val_b_per_country_rescue"] = scorer.score(predict(sb, best_b, tuned["t_main"], tuned["t_low"]), s1_subset=b)
    out["runtime_s"] = round(time.time() - t0, 1)
    (REPORTS / "decide_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for k in ("val_b_single_threshold", "val_b_per_country", "val_b_per_country_rescue"):
        r = out[k]
        print(k, round(r["f05"], 4), {c: round(v["f05"], 4) for c, v in r["by_country"].items()},
              "empty", round(r["pred_empty_share"], 4), "P", round(r["pair_precision"], 4), "R", round(r["pair_recall"], 4))
    print("t_main", tuned["t_main"], "t_low", tuned["t_low"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
