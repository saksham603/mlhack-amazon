"""Measure blocking v3.1 address keys (address-token pairs + exact no-space name) on the FIT v1 sample:
recall and candidates per S1 of v1 top-30 + v3 top-20 (current) plus va top-ka. FIT labels only.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.blocking.eval_v31
"""
import json
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


def main() -> int:
    labels = access.load_labels("FIT").select("s1_gid", "s23_gid")
    report = {}
    for c in ("India", "US"):
        t0 = time.time()
        r1 = pl.read_parquet(REC / "train_s1.parquet", columns=COLS).filter(pl.col("country") == c)
        r23 = pl.concat([pl.read_parquet(REC / f"train_s{s}.parquet", columns=COLS) for s in (2, 3)]).filter(pl.col("country") == c)
        k23, rare = v3.build_index(r23)
        del r23
        log(f"{c}: index {k23.height:,} keys ({time.time() - t0:.0f}s)")
        v1 = pl.read_parquet(access.DATA / "_v1" / "training" / f"labeled_{c}.parquet", columns=["s1_gid", "s23_gid", "blocking_rank"])
        s1s = v1.select("s1_gid").unique().sort("s1_gid")
        truth = labels.join(s1s, on="s1_gid", how="semi")
        base = pl.concat([v1.filter(pl.col("blocking_rank") <= 30).select("s1_gid", "s23_gid"),
                          pl.read_parquet(OUT / f"v3_fit_{c}.parquet").filter(pl.col("v3_rank") <= 20).select("s1_gid", "s23_gid")]).unique()
        r1s = r1.join(s1s.rename({"s1_gid": "gid"}), on="gid", how="semi")
        va = pl.concat([v3.candidates_addr(r1s.slice(i, BATCH), k23, rare, 20) for i in range(0, r1s.height, BATCH)])
        va.write_parquet(OUT / f"va_fit_{c}.parquet")
        n = s1s.height

        def rec(cands):
            u = cands.unique()
            return {"recall": round(truth.join(u, on=["s1_gid", "s23_gid"], how="semi").height / truth.height, 4),
                    "per_s1": round(u.height / n, 1)}
        rep = {"base_v1k30_v3k20": rec(base)}
        for ka in (5, 10, 20):
            rep[f"base+va_k{ka}"] = rec(pl.concat([base, va.filter(pl.col("va_rank") <= ka).select("s1_gid", "s23_gid")]))
        rep["seconds"] = round(time.time() - t0)
        report[c] = rep
        log(f"{c}: {json.dumps(rep)}")
        del k23, rare
    (OUT / "eval_v31.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
