"""Sprint step 6: assemble output/candidate_pairs.tsv and output/matching_results.tsv from the
per-country TEST candidates + scored predictions. G-M3 enforced GLOBALLY (across all 3 countries)
on the final predictions before writing, even though blocking is per-country and cross-country
candidates should not occur structurally.
"""
import json

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.foundation import access
from amlc.model.id_map import load_gid_id_map
from amlc.model.predict import predictions_at_threshold

TEST_OUT = access.DATA / "_v1" / "test"
OUTPUT_DIR = access.ROOT / "output"
COUNTRIES = ("India", "US", "France")


def load_all_scored() -> pl.DataFrame:
    return pl.concat([pl.read_parquet(TEST_OUT / f"scored_{c}.parquet") for c in COUNTRIES], how="vertical")


def load_all_candidates() -> pl.DataFrame:
    parts = []
    for c in COUNTRIES:
        cdir = TEST_OUT / "candidates" / c
        parts.append(pl.concat([pl.read_parquet(f) for f in sorted(cdir.glob("candidates_*.parquet"))], how="vertical"))
    return pl.concat(parts, how="vertical").select("s1_gid", "s23_gid")


def build_id_maps() -> tuple[pl.DataFrame, pl.DataFrame]:
    s1_map = load_gid_id_map("test", 1).rename({"gid": "s1_gid", "entity_id": "s1_id"})
    s23_map = pl.concat([load_gid_id_map("test", 2), load_gid_id_map("test", 3)], how="vertical") \
        .rename({"gid": "s23_gid", "entity_id": "s23_id"})
    return s1_map, s23_map


def links_to_tsv(links: pl.DataFrame, s1_map: pl.DataFrame, s23_map: pl.DataFrame,
                  all_s1_ids: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """links: (s1_gid, s23_gid). Returns one row per ALL_S1_IDS s1_id (missing/empty ones get "")."""
    joined = links.join(s1_map, on="s1_gid", how="inner").join(s23_map, on="s23_gid", how="inner")
    grouped = (joined.group_by("s1_id")
               .agg(pl.col("s23_id").unique().sort().str.join(",").alias(id_col)))
    return (all_s1_ids.join(grouped, on="s1_id", how="left")
            .with_columns(pl.col(id_col).fill_null("")))


def write_tsv(df: pl.DataFrame, id_col: str, header2: str, out_path) -> None:
    out = df.rename({"s1_id": "source1_entity_id", id_col: header2})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_path, separator="\t", quote_style="never")


def assemble(threshold: float) -> dict:
    s1_map, s23_map = build_id_maps()
    all_s1_ids = s1_map.select("s1_id")

    candidates = load_all_candidates()
    cand_tsv = links_to_tsv(candidates, s1_map, s23_map, all_s1_ids, "candidate_id")
    write_tsv(cand_tsv, "candidate_id", "candidate_entity_ids", OUTPUT_DIR / "candidate_pairs.tsv")

    scored = load_all_scored()
    predictions = predictions_at_threshold(scored.select("s1_gid", "s23_gid", "prob"), threshold)
    match_tsv = links_to_tsv(predictions, s1_map, s23_map, all_s1_ids, "matched_id")
    write_tsv(match_tsv, "matched_id", "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")

    return {
        "threshold": threshold, "n_test_s1": all_s1_ids.height,
        "n_candidates": candidates.height, "n_predictions": predictions.height,
        "candidate_rows_written": cand_tsv.height, "match_rows_written": match_tsv.height,
    }


def main() -> int:
    report_path = OUT_DIR / "step4_model_report.json"
    threshold = json.loads(report_path.read_text(encoding="utf-8"))["best_threshold"]
    summary = assemble(threshold)
    (OUT_DIR / "assemble_output.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
