"""Re-measure blocking recall with fixes 1-3 on the SAME evaluation as T7 and the diagnosis
(user go, 2026-09-26 15:00). One country per process:  python -m amlc.blocking.eval_w1_v2 India

Same population: candidates.sample_fit_s1 (1% of FIT S1, seed 20260925). Its S1 count and true-link
count must equal the earlier runs (India 6,182 / 21,387; US 9,265 / 31,991) or the run stops. The
drawn S1 list is frozen to disk on first use and checked on every later run.
Same ground truth: FIT links of the sampled S1s. Same metric: share of true links in the final set.
Same rarity cap (5,000) and the same per-S1 budget of 100 candidates for the headline comparison.

Variants (all deterministic, ties broken by s23_gid):
  A  old: one combined score over single tokens + house + zip, top-100          (before)
  B  Fix 2 only: name (single tokens) top-70  U  address top-30
  C  Fix 2 + Fix 1: name (tokens + pairs) top-70  U  address top-30
  D  Fix 2 + Fix 1 + Fix 3: name (tokens + pairs + no-space) top-70  U  address top-30   (after)
  E1 Fix 1 on the old combined ranking, top-100;  E2 Fix 1 + Fix 3 on it, top-100   (other order)
  D+ D at a larger budget (name 100 U address 50), to show how far recall still climbs
"""
import json
import sys
import time

import polars as pl

from amlc.blocking import candidates as C
from amlc.blocking import v2
from amlc.blocking.dryrun_w1 import OUT_DIR, check_ram_budget, free_ram_mb
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access

CAP = 5000
EXPECTED = {"India": (6182, 21387), "US": (9265, 31991)}   # from T7 and the diagnosis
T7_RECALL = {"India": 0.4975919951372329, "US": 0.8197618080085024}  # cap=5000, k=100, T7 report
COLS = ["gid", "name_clean", "name_tokens", "addr_components", "addr_house_number", "addr_zip"]
BYTES_PER_JOIN_ROW = 48


def load_pool(country: str) -> pl.DataFrame:
    parts = []
    for src in (2, 3):
        df = pl.read_parquet(C.SILVER_DIR / f"train_source{src}.parquet", columns=COLS)
        meta = access.load_bronze("train", src, columns=["gid", "country"])
        parts.append(df.join(meta.filter(pl.col("country") == country).select("gid"), on="gid", how="semi"))
    return pl.concat(parts, how="vertical").rename({"gid": "s23_gid"})


def frozen_sample(country: str) -> pl.DataFrame:
    s1 = C.sample_fit_s1(country)
    n_s1, n_links = EXPECTED[country]
    path = OUT_DIR / f"eval_sample_{country}.parquet"
    gids = s1.select("s1_gid").sort("s1_gid")
    if path.exists():
        frozen = pl.read_parquet(path)
        if not frozen.equals(gids):
            raise RuntimeError(f"{country}: drawn sample differs from the frozen one at {path}. STOP.")
    elif s1.height != n_s1:
        raise RuntimeError(f"{country}: sample has {s1.height} S1, expected {n_s1}. STOP.")
    else:
        gids.write_parquet(path)
    return s1.select(COLS[1:] + ["s1_gid"])


def evaluate(final: pl.DataFrame, truth: pl.DataFrame, n_s1: int, base_hits: pl.DataFrame | None) -> dict:
    hits = truth.join(final, on=["s1_gid", "s23_gid"], how="semi")
    per = final.group_by("s1_gid").len()
    out = {"recall": hits.height / truth.height, "hits": hits.height,
           "missed": truth.height - hits.height,
           "candidates": final.height, "candidates_per_s1_mean": final.height / n_s1,
           "candidates_per_s1_p99": float(per["len"].quantile(0.99)) if per.height else 0.0}
    if base_hits is not None:
        out["recovered_vs_A"] = hits.join(base_hits, on=["s1_gid", "s23_gid"], how="anti").height
        out["lost_vs_A"] = base_hits.join(hits, on=["s1_gid", "s23_gid"], how="anti").height
    return out


def timed(label: str, fn, timings: dict):
    t = time.time()
    r = fn()
    timings[label] = round(time.time() - t, 1)
    return r


def run(country: str) -> dict:
    t0 = time.time()
    timings: dict = {}
    check_ram_budget()
    budget_rows = int(max(free_ram_mb() - 1536, 512) * 2**20 / BYTES_PER_JOIN_ROW)

    s1 = frozen_sample(country)
    truth = (access.load_labels("FIT").select("s1_gid", "s23_gid")
             .join(s1.select("s1_gid"), on="s1_gid", how="semi"))
    if (s1.height, truth.height) != EXPECTED[country]:
        raise RuntimeError(f"{country}: population {(s1.height, truth.height)} != {EXPECTED[country]}. STOP.")
    s23 = timed("load_pool", lambda: load_pool(country), timings)
    rep = {"country": country, "n_s1": s1.height, "n_true_links": truth.height, "n_s23_pool": s23.height,
           "t7_reported_recall_A": T7_RECALL[country], "join_row_budget": budget_rows}

    # ---- A: old strategy, recomputed deterministically ----
    scored_a = timed("A_score", lambda: C.candidates_for_cap(s1, s23, CAP, budget_rows * BYTES_PER_JOIN_ROW), timings)
    rep["A_scored_pairs"] = scored_a.height
    final_a = timed("A_topk", lambda: C.candidates_topk(scored_a, 100).select("s1_gid", "s23_gid"), timings)
    del scored_a
    rep["A"] = evaluate(final_a, truth, s1.height, None)
    base_hits = truth.join(final_a, on=["s1_gid", "s23_gid"], how="semi")
    del final_a

    # ---- v2 keys ----
    check_ram_budget()
    kinds = v2.KINDS_NAME + v2.KINDS_ADDR
    k1 = timed("keys_s1", lambda: v2.build_keys(s1.select("s1_gid", *COLS[1:]), "s1_gid", kinds), timings)
    k23 = timed("keys_s23", lambda: v2.build_keys(s23, "s23_gid", kinds), timings)
    rep["s23_keys_by_kind"] = dict(k23.group_by("kind").len().select(pl.col("kind").cast(pl.String), "len").iter_rows())
    rare = timed("idf", lambda: v2.idf_table(k23, s23.height, CAP), timings)
    rep["rare_keys_by_kind"] = dict(rare.group_by("kind").len().select(pl.col("kind").cast(pl.String), "len").iter_rows())

    # nospace: vectorised == reference on every real sampled S1 name and 200k pool names (gate)
    probe = pl.concat([s1.select("name_tokens"), s23.select("name_tokens").head(200_000)])
    vec = probe.select(v2.nospace_expr().alias("k"))["k"].to_list()
    ref = [v2.nospace_value(t) for t in probe["name_tokens"].to_list()]
    rep["nospace_vectorised_mismatches"] = sum(a != b for a, b in zip(vec, ref))
    if rep["nospace_vectorised_mismatches"]:
        raise AssertionError(f"nospace vectorised != reference on {rep['nospace_vectorised_mismatches']} names")

    del s23
    scores = {}
    for label, ks in (("name_tok", ("tok",)), ("name_tokpair", ("tok", "pair")),
                      ("name_all", ("tok", "pair", "nospace")), ("addr", ("house", "zip"))):
        check_ram_budget()
        scores[label] = timed(f"score_{label}", lambda ks=ks: v2.score(k1, k23, rare, ks, budget_rows), timings)
    rep["scored_pairs"] = {k: v.height for k, v in scores.items()}
    del k23

    def combined(name_label: str) -> pl.DataFrame:   # old-style single ranking over the v2 keys
        return (pl.concat([scores[name_label], scores["addr"]], how="vertical")
                .group_by("s1_gid", "s23_gid").agg(pl.col("score").sum()))

    t = time.time()
    variants = {
        "name_only_before": v2.topk(scores["name_tok"], 100),
        "name_only_after": v2.topk(scores["name_all"], 100),
        "address_only": v2.topk(scores["addr"], 100),
        "B_fix2": v2.union_candidates(v2.topk(scores["name_tok"], 70), v2.topk(scores["addr"], 30)),
        "C_fix2_fix1": v2.union_candidates(v2.topk(scores["name_tokpair"], 70), v2.topk(scores["addr"], 30)),
        "D_all": v2.union_candidates(v2.topk(scores["name_all"], 70), v2.topk(scores["addr"], 30)),
        "E1_fix1_old_ranking": v2.topk(combined("name_tokpair"), 100),
        "E2_fix1_fix3_old_ranking": v2.topk(combined("name_all"), 100),
        "D_plus_budget_150": v2.union_candidates(v2.topk(scores["name_all"], 100), v2.topk(scores["addr"], 50)),
    }
    timings["rank_all_variants"] = round(time.time() - t, 1)
    for name, final in variants.items():
        rep[name] = evaluate(final, truth, s1.height, base_hits)

    # ---- remaining misses of D, by cause (same classes as the diagnosis) ----
    d = variants["D_all"]
    missed = truth.join(d, on=["s1_gid", "s23_gid"], how="anti")
    scored_any = pl.concat([scores["name_all"].select("s1_gid", "s23_gid"),
                            scores["addr"].select("s1_gid", "s23_gid")]).unique()
    shared = (missed.join(k1.select("s1_gid", "key"), on="s1_gid")
              .join(v2.build_keys(load_pool(country).join(missed.select("s23_gid").unique(), on="s23_gid",
                                                          how="semi"), "s23_gid", kinds).select("s23_gid", "key"),
                    on=["s23_gid", "key"])
              .select("s1_gid", "s23_gid").unique())
    ranked_out = missed.join(scored_any, on=["s1_gid", "s23_gid"], how="semi").height
    any_key = missed.join(shared, on=["s1_gid", "s23_gid"], how="semi").height
    rep["D_remaining_misses"] = {"total": missed.height, "ranked_out": ranked_out,
                                 "cap_dropped": any_key - ranked_out, "no_key": missed.height - any_key}
    rep["timings_s"] = timings
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["peak_ram_mb"] = peak_ram_mb()
    return rep


def main() -> int:
    country = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== {country} (free RAM {free_ram_mb()} MB)", flush=True)
    rep = run(country)
    (OUT_DIR / f"w1_v2_{country}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
