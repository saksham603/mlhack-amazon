"""Measure blocking v3 on the FIT v1 sample (50k S1 per country, full train S2/S3 pool): recall of v1
(budget 150 and cut to k), v3 alone, and unions, with candidates per S1. FIT labels only. Saves the
v3 candidates of the sample (data/_v2/blocking/v3_fit_<country>.parquet) for training features.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.blocking.eval_v3 [k3]
"""
import json
import sys
import time

import polars as pl

from amlc.blocking import v3
from amlc.foundation import access

REC = access.DATA / "_v2" / "records"
OUT = access.DATA / "_v2" / "blocking"
COLS = ["gid", "country", "core", "addr", "region", "ns", "skel"]
BATCH = 5000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def recall(cands: pl.DataFrame, truth: pl.DataFrame) -> float:
    return truth.join(cands.select("s1_gid", "s23_gid").unique(), on=["s1_gid", "s23_gid"], how="semi").height / truth.height


def main() -> int:
    k3 = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT.mkdir(parents=True, exist_ok=True)
    labels = access.load_labels("FIT").select("s1_gid", "s23_gid")
    report = {}
    for c in ("India", "US"):
        t0 = time.time()
        r1 = pl.read_parquet(REC / "train_s1.parquet", columns=COLS).filter(pl.col("country") == c)
        r23 = pl.concat([pl.read_parquet(REC / f"train_s{s}.parquet", columns=COLS) for s in (2, 3)]).filter(pl.col("country") == c)
        k23, rare = v3.build_index(r23)
        del r23
        log(f"{c}: index {k23.height:,} keys, {rare.height:,} rare ({time.time() - t0:.0f}s)")
        v1 = pl.read_parquet(access.DATA / "_v1" / "training" / f"labeled_{c}.parquet", columns=["s1_gid", "s23_gid", "blocking_rank"])
        s1s = v1.select("s1_gid").unique().sort("s1_gid")
        truth = labels.join(s1s, on="s1_gid", how="semi")
        r1s = r1.join(s1s.rename({"s1_gid": "gid"}), on="gid", how="semi")
        parts = []
        for i in range(0, r1s.height, BATCH):
            parts.append(v3.candidates(r1s.slice(i, BATCH), k23, rare, k3))
        cand3 = pl.concat(parts)
        cand3.write_parquet(OUT / f"v3_fit_{c}.parquet")
        n = s1s.height
        rep = {"s1": n, "true_links": truth.height, "v3_seconds": round(time.time() - t0)}
        for k in (30, 50, 150):
            rep[f"v1_k{k}"] = {"recall": round(recall(v1.filter(pl.col("blocking_rank") <= k), truth), 4),
                               "per_s1": round(v1.filter(pl.col("blocking_rank") <= k).height / n, 1)}
        for k in sorted({10, 20, 30, k3}):
            if k <= k3:
                c3 = cand3.filter(pl.col("v3_rank") <= k)
                rep[f"v3_k{k}"] = {"recall": round(recall(c3, truth), 4), "per_s1": round(c3.height / n, 1)}
                for kv1 in (30, 50, 150):
                    u = pl.concat([v1.filter(pl.col("blocking_rank") <= kv1).select("s1_gid", "s23_gid"),
                                   c3.select("s1_gid", "s23_gid")]).unique()
                    rep[f"v1_k{kv1}+v3_k{k}"] = {"recall": round(recall(u, truth), 4), "per_s1": round(u.height / n, 1)}
        report[c] = rep
        log(f"{c}: " + json.dumps(rep))
        del k23, rare, cand3
    (OUT / f"eval_v3_k{k3}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
