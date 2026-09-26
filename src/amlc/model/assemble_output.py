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


def load_scored(country: str, columns: list[str]) -> pl.DataFrame:
    """Batched layout (scored/<country>/scored_*.parquet) if present, else the older single file."""
    batch_dir = TEST_OUT / "scored" / country
    if batch_dir.is_dir() and any(batch_dir.glob("scored_*.parquet")):
        return pl.concat([pl.read_parquet(f, columns=columns) for f in sorted(batch_dir.glob("scored_*.parquet"))],
                         how="vertical")
    return pl.read_parquet(TEST_OUT / f"scored_{country}.parquet", columns=columns)


def load_predicted(threshold: float) -> pl.DataFrame:
    """Only rows at/above threshold (a few million), filtered per country before concatenating, so
    the full ~230M-row scored set is never in memory at once. G-M3 is applied by the caller."""
    return pl.concat([load_scored(c, ["s1_gid", "s23_gid", "prob"]).filter(pl.col("prob") >= threshold)
                      for c in COUNTRIES], how="vertical")


def group_ids(links: pl.DataFrame, s1_map: pl.DataFrame, s23_map: pl.DataFrame) -> pl.DataFrame:
    """links: (s1_gid, s23_gid) -> (s1_id, ids) with sorted, de-duplicated, comma-joined S2/S3 ids."""
    return (links.join(s1_map, on="s1_gid", how="inner").join(s23_map, on="s23_gid", how="inner")
            .group_by("s1_id").agg(pl.col("s23_id").unique().sort().str.join(",").alias("ids")))


def write_candidate_tsv_streaming(out_path, s1_map, s23_map, all_s1_ids: set[str], internal_dir) -> dict:
    """Streams candidate_pairs.tsv one candidate batch at a time (each S1's candidates live entirely
    inside one batch, since batches partition S1s), then appends an empty row for every test S1
    blocking found nothing for. Also writes the internal parquet (s1_id, candidate_id, rank, score)
    per batch for tomorrow's budget sweep (sprint sec5 / D-S3)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    n_pairs = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for c in COUNTRIES:
            idir = internal_dir / c
            idir.mkdir(parents=True, exist_ok=True)
            for p in sorted((TEST_OUT / "candidates" / c).glob("candidates_*.parquet")):
                batch = pl.read_parquet(p)
                n_pairs += batch.height
                (batch.join(s1_map, on="s1_gid", how="inner").join(s23_map, on="s23_gid", how="inner")
                 .select("s1_id", pl.col("s23_id").alias("candidate_id"),
                         pl.col("blocking_rank").alias("rank"), pl.col("blocking_score").alias("score"))
                 .write_parquet(idir / f"{p.stem}.parquet"))
                rows = group_ids(batch.select("s1_gid", "s23_gid"), s1_map, s23_map)
                dup = written.intersection(rows["s1_id"].to_list())
                if dup:
                    raise RuntimeError(f"S1 ids appear in more than one candidate batch, e.g. {list(dup)[:3]}. STOP.")
                written.update(rows["s1_id"].to_list())
                f.write("".join(f"{a}\t{b}\n" for a, b in rows.iter_rows()))
        empty = all_s1_ids - written
        f.write("".join(f"{a}\t\n" for a in sorted(empty)))
    return {"n_candidate_pairs": n_pairs, "s1_with_candidates": len(written), "s1_without_candidates": len(empty)}


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

    cand_rep = write_candidate_tsv_streaming(OUTPUT_DIR / "candidate_pairs.tsv", s1_map, s23_map,
                                             set(all_s1_ids["s1_id"].to_list()),
                                             access.DATA / "_v1" / "test_candidates")

    predictions = predictions_at_threshold(load_predicted(threshold), threshold)
    match_tsv = links_to_tsv(predictions, s1_map, s23_map, all_s1_ids, "matched_id")
    write_tsv(match_tsv, "matched_id", "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")

    return {
        "threshold": threshold, "n_test_s1": all_s1_ids.height, **cand_rep,
        "n_predictions": predictions.height, "match_rows_written": match_tsv.height,
        "s1_with_predictions": predictions["s1_gid"].n_unique(),
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
