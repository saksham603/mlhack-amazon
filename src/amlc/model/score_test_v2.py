"""Day-2 Phase B1: rescore the existing v1 TEST candidates (budget 150) with lgbm_v2 + v3 features,
then assemble output/<name>/matching_results.tsv. The candidate set is unchanged, so
output/candidate_pairs_k150.tsv is the matching candidate file (every prediction is inside it).
Resumable: a batch whose scored file exists is skipped. G-M3 is enforced globally before writing.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.score_test_v2 score
     PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.score_test_v2 assemble <name> [threshold]
"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl

from amlc.features import w3
from amlc.foundation import access
from amlc.model.assemble_output import build_id_maps, links_to_tsv, write_tsv
from amlc.model.predict import predictions_at_threshold
from amlc.model.run_v2 import MODEL_PATH, REPORTS, to_x

TEST_V1 = access.DATA / "_v1" / "test"
SCORED = access.DATA / "_v2" / "test_scored"
OUTPUT = access.ROOT / "output"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def countries() -> list[str]:
    return sorted(p.name for p in (TEST_V1 / "candidates").iterdir() if p.is_dir())


def score() -> None:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    r1, r23 = w3.load_records("test", "s1"), w3.load_records("test", "s23")
    log(f"test records: s1 {r1.height:,}, s23 {r23.height:,}")
    for c in countries():
        out_dir = SCORED / c
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((TEST_V1 / "candidates" / c).glob("candidates_*.parquet"))
        n_rows = 0
        for f in files:
            dst = out_dir / f.name.replace("candidates_", "scored_")
            if dst.exists():
                continue
            cand = pl.read_parquet(f)
            feat = pl.read_parquet(TEST_V1 / "features" / c / f.name.replace("candidates_", "features_"))
            pairs = cand.join(feat, on=["s1_gid", "s23_gid"], how="inner", validate="1:1")
            if pairs.height != cand.height:
                raise AssertionError(f"{c}/{f.name}: {cand.height} candidates vs {pairs.height} joined. STOP.")
            x = w3.pair_features(pairs, r1, r23)
            if x.height != cand.height:
                raise AssertionError(f"{c}/{f.name}: feature rows changed. STOP.")
            (x.select("s1_gid", "s23_gid", "blocking_rank").with_columns(pl.Series("prob", booster.predict(to_x(x))))
             .write_parquet(dst))
            n_rows += cand.height
        log(f"{c}: scored {len(files)} batches ({n_rows:,} new rows)")


def assemble(name: str, threshold: float, k: int = 150) -> dict:
    parts = []
    for c in countries():
        for p in sorted((SCORED / c).glob("scored_*.parquet")):
            parts.append(pl.scan_parquet(p).filter((pl.col("blocking_rank") <= k) & (pl.col("prob") >= threshold))
                         .select("s1_gid", "s23_gid", "prob").collect())
    pred = predictions_at_threshold(pl.concat(parts), threshold)  # G-M3 across all countries
    s1_map, s23_map = build_id_maps()
    all_s1 = s1_map.select("s1_id")
    tsv = links_to_tsv(pred, s1_map, s23_map, all_s1, "matched_id")
    out = OUTPUT / name / "matching_results.tsv"
    write_tsv(tsv, "matched_id", "matched_entity_ids", out)
    rep = {"name": name, "threshold": threshold, "k": k, "n_predictions": pred.height,
           "s1_rows": tsv.height, "s1_empty": int((tsv["matched_id"] == "").sum()), "path": str(out)}
    (OUTPUT / name / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> int:
    if sys.argv[1] == "score":
        score()
    else:
        name = sys.argv[2]
        t = float(sys.argv[3]) if len(sys.argv) > 3 else json.loads(
            (REPORTS / "phase_a_report.json").read_text(encoding="utf-8"))["threshold"]
        print(json.dumps(assemble(name, t), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
