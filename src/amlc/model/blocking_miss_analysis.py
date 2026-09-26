"""Where does blocking lose true matches? For the same FIT S1 sample the training set used (50k per
country, dataset.sample_fit_s1_gids), finds FIT true links that never became candidates and sorts
them into categories that each point at a different fix. FIT labels only; light on RAM (reads only
the missed pairs' silver rows).

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.blocking_miss_analysis
"""
import json
import random

import polars as pl

from amlc.blocking.candidates import SILVER_DIR
from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.foundation import access
from amlc.model import dataset

TRAIN_CAND = access.DATA / "_v1" / "training" / "candidates"
ANALYSIS_OUT = access.DATA / "_v1" / "analysis"
COUNTRIES = ("India", "US")
COLS = ["gid", "name_clean", "name_tokens", "addr_clean", "addr_house_number"]
N_EXAMPLES = 15


def char3(s: str) -> set[str]:
    s = f"  {s or ''}  "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def category(r: dict) -> str:
    t1, t2 = set(r["tok_1"] or []), set(r["tok_2"] or [])
    shared = t1 & t2
    if shared:
        return "shares_name_token"          # a key existed; lost to rarity cap or rank/budget
    c3 = jaccard(char3(r["name_1"]), char3(r["name_2"]))
    hn = r["hn_1"] and r["hn_1"] == r["hn_2"]
    if c3 >= 0.3:
        return "no_token_but_char3_similar"  # spelling/transliteration/spacing: char-3gram target
    if hn:
        return "no_name_sim_same_house_no"   # address-only match
    return "no_name_no_house_signal"         # hard: neither name nor house number overlaps


def load_rows(path, gids: pl.DataFrame, suffix: str) -> pl.DataFrame:
    df = (pl.scan_parquet(path).select(COLS).join(gids.lazy(), on="gid", how="semi").collect())
    return df.rename({"gid": f"gid{suffix}", "name_clean": f"name{suffix}", "name_tokens": f"tok{suffix}",
                      "addr_clean": f"addr{suffix}", "addr_house_number": f"hn{suffix}"})


def analyze_country(country: str, truth_all: pl.DataFrame) -> dict:
    s1 = dataset.sample_fit_s1_gids(country)
    truth = truth_all.join(s1, on="s1_gid", how="semi")
    cand = pl.concat([pl.read_parquet(f, columns=["s1_gid", "s23_gid"])
                      for f in sorted((TRAIN_CAND / country).glob("candidates_*.parquet"))], how="vertical")
    missed = truth.join(cand, on=["s1_gid", "s23_gid"], how="anti")
    del cand

    s1_rows = load_rows(SILVER_DIR / "train_source1.parquet",
                        missed.select(pl.col("s1_gid").alias("gid")).unique(), "_1")
    s23_rows = pl.concat([load_rows(SILVER_DIR / f"train_source{s}.parquet",
                                    missed.select(pl.col("s23_gid").alias("gid")).unique(), "_2") for s in (2, 3)],
                         how="vertical")
    joined = (missed.join(s1_rows, left_on="s1_gid", right_on="gid_1", how="left")
              .join(s23_rows, left_on="s23_gid", right_on="gid_2", how="left"))
    rows = joined.to_dicts()
    for r in rows:
        r["cat"] = category(r)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cat"]] = counts.get(r["cat"], 0) + 1
    rng = random.Random(20260926)
    examples = {}
    for cat in counts:
        pool = [r for r in rows if r["cat"] == cat]
        examples[cat] = [{"s1_name": r["name_1"], "s23_name": r["name_2"],
                          "s1_addr": r["addr_1"], "s23_addr": r["addr_2"]}
                         for r in rng.sample(pool, min(N_EXAMPLES, len(pool)))]
    return {"country": country, "n_s1": s1.height, "n_true_links": truth.height, "n_missed": missed.height,
            "fit_sample_recall": 1 - missed.height / truth.height,
            "missed_by_category": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "missed_share_of_all_true_links": {k: v / truth.height for k, v in counts.items()},
            "examples": examples}


def main() -> int:
    truth_all = access.load_labels("FIT").select("s1_gid", "s23_gid")
    report = {c: analyze_country(c, truth_all) for c in COUNTRIES}
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_OUT / "blocking_miss_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {c: {k: v for k, v in r.items() if k != "examples"} for c, r in report.items()}
    (OUT_DIR / "blocking_miss_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
