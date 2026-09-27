"""Merge a per-country submission from independently-scored country files (2026-09-27 08:45): built
so France can come from a different model/threshold (the Ryzen's region-remapped rescoring) than
India/US (the laptop's v31 or v32), without touching either side's own output folders.

Each source is (parquet with s1_gid, s23_gid, prob for exactly one country's candidate pairs, threshold).
G-M3 is enforced globally (harmless even though country S2/S3 gid pools are disjoint) and every test S1
gets a row, matched or empty. candidate_pairs.tsv = the union of every source's own candidate rows,
i.e. still exactly the model input for each country -- no candidates are added or dropped by merging.

Run: PYTHONPATH=src python -m amlc.pipeline.assemble_merge <name> <country=path:threshold> [country=path:threshold ...]
Example: PYTHONPATH=src python -m amlc.pipeline.assemble_merge v32_france \
    India=data/_v32/test_scored/India_*.parquet:0.72 US=data/_v32/test_scored/US_*.parquet:0.72 \
    France=data/_shared/ryzen_out/france_v32/france_scored.parquet:0.60
"""
import glob
import json
import sys

import polars as pl

from amlc.model.assemble_output import build_id_maps, group_ids, links_to_tsv, write_tsv
from amlc.model.predict import predictions_at_threshold
from amlc.pipeline.run_v3 import OUTPUT
from amlc.foundation import access


def parse_source(country: str, spec: str) -> tuple[pl.DataFrame, float, str | None]:
    """spec: 'path:threshold' or 'path:threshold:reference_glob'. `reference_glob`, when given, is an
    original (untouched) candidate file set whose (s1_gid, s23_gid) key SET `path` must match exactly
    (same rows, only `prob` may differ) -- the guard for an externally-delivered rescoring (e.g. the
    Ryzen's France remap), which must never add, drop or leave out any of the country's candidate rows."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{country}={spec}: expected path:threshold[:reference_glob]")
    path, thr_s, *ref = parts
    threshold = float(thr_s)
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"{country}: no files match {path!r}")
    d = pl.concat([pl.read_parquet(f, columns=["s1_gid", "s23_gid", "prob"]) for f in files])
    if d.select("s1_gid", "s23_gid").is_duplicated().any():
        raise AssertionError(f"{country}: duplicate (s1_gid, s23_gid) rows in {path!r}. STOP.")
    return d, threshold, (ref[0] if ref else None)


def check_reference(country: str, cand: pl.DataFrame, reference_glob: str) -> None:
    ref_files = sorted(glob.glob(reference_glob))
    if not ref_files:
        raise FileNotFoundError(f"{country}: reference glob matched nothing: {reference_glob!r}")
    ref = pl.concat([pl.read_parquet(f, columns=["s1_gid", "s23_gid"]) for f in ref_files])
    a, b = cand.select("s1_gid", "s23_gid"), ref.select("s1_gid", "s23_gid")
    if a.height != b.height or not a.sort("s1_gid", "s23_gid").equals(b.sort("s1_gid", "s23_gid")):
        only_a, only_b = a.join(b, on=["s1_gid", "s23_gid"], how="anti").height, b.join(a, on=["s1_gid", "s23_gid"], how="anti").height
        raise AssertionError(f"{country}: candidate keys don't match the reference {reference_glob!r} exactly "
                             f"({a.height} vs {b.height} rows; {only_a} only in delivered, {only_b} only in reference). STOP.")


def assemble(name: str, sources: dict) -> dict:
    """sources: {country: (candidates_df[s1_gid,s23_gid,prob], threshold, reference_glob_or_None)}.
    Verifies each candidates_df's s1_gid set actually belongs to that country (against test_s1 records),
    and, when a reference_glob is given, that its (s1_gid, s23_gid) keys match that reference exactly."""
    s1c = pl.read_parquet(access.DATA / "_v2" / "records" / "test_s1.parquet", columns=["gid", "country"])
    all_cand, all_pred, per_country = [], [], {}
    for country, (cand, threshold, reference) in sources.items():
        if reference:
            check_reference(country, cand, reference)
        bad = cand.join(s1c.filter(pl.col("country") != country).rename({"gid": "s1_gid"}), on="s1_gid", how="semi")
        if bad.height:
            raise AssertionError(f"{country}: {bad.height} candidate rows whose s1_gid is not {country}. STOP.")
        own = s1c.filter(pl.col("country") == country)
        if not cand.join(own.rename({"gid": "s1_gid"}), on="s1_gid", how="semi").equals(cand):
            raise AssertionError(f"{country}: some candidate s1_gid rows are missing from that country. STOP.")
        pred = predictions_at_threshold(cand, threshold)
        all_cand.append(cand.select("s1_gid", "s23_gid"))
        all_pred.append(pred)
        per_country[country] = {"threshold": threshold, "candidate_pairs": cand.height,
                                "s1_with_candidates": cand["s1_gid"].n_unique(),
                                "predictions": pred.height, "s1_with_predictions": pred["s1_gid"].n_unique()}
    covered = set(sources)
    missing = set(s1c["country"].unique().to_list()) - covered
    if missing:
        raise AssertionError(f"no source given for countries {missing}; every test country needs one. STOP.")
    cand_all = pl.concat(all_cand)
    pred_all = pl.concat(all_pred)  # each source already thresholded + G-M3'd; country S2/S3 pools are disjoint
    if pred_all["s23_gid"].is_duplicated().any():
        raise AssertionError("a s23_gid was predicted by more than one country's source -- pools should be disjoint. STOP.")
    s1_map, s23_map = build_id_maps()
    all_s1 = s1_map.select("s1_id")
    dst = OUTPUT / name
    write_tsv(links_to_tsv(pred_all, s1_map, s23_map, all_s1, "matched_id"), "matched_id", "matched_entity_ids",
              dst / "matching_results.tsv")
    rows = group_ids(cand_all, s1_map, s23_map)
    written = set(rows["s1_id"].to_list())
    empty = set(all_s1["s1_id"].to_list()) - written
    with open(dst / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        fh.write("".join(f"{x}\t{y}\n" for x, y in rows.iter_rows()))
        fh.write("".join(f"{x}\t\n" for x in sorted(empty)))
    rep = {"name": name, "sources": {c: {**v, "path_pattern": None} for c, v in per_country.items()},
           "n_predictions": pred_all.height, "candidate_pairs": cand_all.height,
           "s1_with_predictions": pred_all["s1_gid"].n_unique(), "n_test_s1": all_s1.height}
    (dst / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    name = sys.argv[1]
    srcs = {}
    for arg in sys.argv[2:]:
        country, _, spec = arg.partition("=")
        srcs[country] = parse_source(country, spec)
    print(json.dumps(assemble(name, srcs), indent=1))
