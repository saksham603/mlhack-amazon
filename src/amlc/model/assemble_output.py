"""Sprint step 6: assemble output/candidate_pairs.tsv and output/matching_results.tsv from the
per-country scored TEST parquets, restricted to candidates with blocking_rank <= k. Building both
files from the same scored rows guarantees predictions are a subset of candidates. G-M3 enforced
GLOBALLY (across all 3 countries) on the final predictions before writing.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.model.assemble_output 30
"""
import json
import sys

import polars as pl

from amlc.blocking.dryrun_w1 import OUT_DIR
from amlc.foundation import access
from amlc.model.id_map import load_gid_id_map
from amlc.model.predict import predictions_at_threshold

TEST_OUT = access.DATA / "_v1" / "test"
OUTPUT_DIR = access.ROOT / "output"
COUNTRIES = ("India", "US", "France")


def scored_files(country: str) -> list:
    """Batched layout (scored/<country>/scored_*.parquet) if present, else the older single file.
    Each S1's candidates live entirely inside one file either way."""
    batch_dir = TEST_OUT / "scored" / country
    if batch_dir.is_dir() and any(batch_dir.glob("scored_*.parquet")):
        return sorted(batch_dir.glob("scored_*.parquet"))
    return [TEST_OUT / f"scored_{country}.parquet"]


def load_predicted(threshold: float, k: int) -> pl.DataFrame:
    """Rows with blocking_rank <= k and prob >= threshold, filtered per file before concatenating so
    the full scored set is never in memory at once. G-M3 is applied by the caller."""
    return pl.concat([pl.scan_parquet(p).filter((pl.col("blocking_rank") <= k) & (pl.col("prob") >= threshold))
                      .select("s1_gid", "s23_gid", "prob").collect()
                      for c in COUNTRIES for p in scored_files(c)], how="vertical")


def group_ids(links: pl.DataFrame, s1_map: pl.DataFrame, s23_map: pl.DataFrame) -> pl.DataFrame:
    """links: (s1_gid, s23_gid) -> (s1_id, ids) with sorted, de-duplicated, comma-joined S2/S3 ids."""
    return (links.join(s1_map, on="s1_gid", how="inner").join(s23_map, on="s23_gid", how="inner")
            .group_by("s1_id").agg(pl.col("s23_id").unique().sort().str.join(",").alias("ids")))


def write_candidate_tsv_streaming(out_path, s1_map, s23_map, all_s1_ids: set[str], k: int) -> dict:
    """Streams candidate_pairs.tsv one scored file at a time (rows with blocking_rank <= k), then
    appends an empty row for every test S1 with no candidates."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    n_pairs = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for c in COUNTRIES:
            for p in scored_files(c):
                batch = (pl.scan_parquet(p).filter(pl.col("blocking_rank") <= k)
                         .select("s1_gid", "s23_gid").collect())
                n_pairs += batch.height
                rows = group_ids(batch, s1_map, s23_map)
                dup = written.intersection(rows["s1_id"].to_list())
                if dup:
                    raise RuntimeError(f"S1 ids appear in more than one scored file, e.g. {list(dup)[:3]}. STOP.")
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


def assemble(threshold: float, k: int) -> dict:
    s1_map, s23_map = build_id_maps()
    all_s1_ids = s1_map.select("s1_id")

    cand_rep = write_candidate_tsv_streaming(OUTPUT_DIR / "candidate_pairs.tsv", s1_map, s23_map,
                                             set(all_s1_ids["s1_id"].to_list()), k)

    predictions = predictions_at_threshold(load_predicted(threshold, k), threshold)
    match_tsv = links_to_tsv(predictions, s1_map, s23_map, all_s1_ids, "matched_id")
    write_tsv(match_tsv, "matched_id", "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")

    return {
        "threshold": threshold, "k": k, "n_test_s1": all_s1_ids.height, **cand_rep,
        "n_predictions": predictions.height, "match_rows_written": match_tsv.height,
        "s1_with_predictions": predictions["s1_gid"].n_unique(),
    }


def main() -> int:
    k = int(sys.argv[1])
    threshold = json.loads((OUT_DIR / "step4_model_report.json").read_text(encoding="utf-8"))["best_threshold"]
    summary = assemble(threshold, k)
    (OUT_DIR / f"assemble_output_k{k}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
