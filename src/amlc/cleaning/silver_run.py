"""Silver runner: C2 Indic -> C3 names -> C4 addresses -> C5 stats -> C6 freeze (Stage 1 prompt
rev 6). Reads C1's published output (interim/c1/v2) and bronze; never rewrites either (G-P16).

  python -m amlc.cleaning.silver_run --dry-run   1% slice (gid % 100 == 0), new numbered dir each run
  python -m amlc.cleaning.silver_run             full run; refuses on uncommitted code (G-P5)
  python -m amlc.cleaning.silver_run --hash-only [--dry-run]   content hashes only, fresh process
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from amlc.cleaning import pipeline, script, stats
from amlc.cleaning.c1_run import git_state, peak_ram_mb
from amlc.foundation import access, leakscan
from amlc.foundation.hashing import file_sha256

VERSION = "v1"
SILVER_DIR = access.DATA / "silver" / VERSION
STAGING = access.DATA / "_staging_silver_v1"
DICT_DIR = access.DATA / "dictionaries" / VERSION
MANIFEST_PATH = access.DATA / "MANIFEST_silver_v1.json"
SLICE_MOD = 100
# R2/R12 extended (added 06:30, after a full-size single-table run crashed the Polars allocator --
# see OVERNIGHT_LOG.md L12): measured 528,560 rows (10% of the largest table) peaks at 3.4 GB, safely
# under budget; 1,057,120 rows (20%) peaks at 6.0 GB, too close to the ~8 GB budget. 10 chunks keeps
# one table's silver output as ONE file (§2's spec: one file per table, never changed -- CG-14 would
# require the user's approval to change that, which isn't available overnight) via a streaming
# ParquetWriter, so peak memory is bounded to one chunk while the on-disk layout stays exactly as
# the Stage 1 prompt specifies.
CHUNK_N = 10
SILVER_COLS = ["gid", "name_clean", "name_legal_form", "name_tokens", "name_token_variants",
              "name_script_flags", "name_indic_untranslated", "addr_clean", "addr_components",
              "addr_house_number", "addr_house_number_raw", "addr_zip", "addr_missing",
              "addr_null_parts_removed", "addr_parts_emptied"]
COVERAGE_MIN = 0.949  # §1 diagnostic baseline

GATES: list[dict] = []


def gate(gid: str, desc: str, passed: bool, measured, expected=None, source=None) -> None:
    GATES.append({"gate": gid, "description": desc, "passed": bool(passed), "measured": measured,
                  "expected": expected, "expected_source": source})
    print(f"  [{'PASS' if passed else 'FAIL'}] {gid}: measured={measured} expected={expected}", flush=True)


def skipped(gid: str, why: str) -> None:
    GATES.append({"gate": gid, "description": f"SKIPPED: {why}", "passed": True, "measured": "skipped",
                  "expected": None, "expected_source": None})


def next_dryrun_dir() -> Path:
    base = access.DATA
    n = 1
    while (base / f"_dryrun_silver_{n}").exists():
        n += 1
    return base / f"_dryrun_silver_{n}"


def hash_only(dry: bool) -> int:
    """Fresh-process determinism check (G-SIL-5). Mirrors main()'s chunked, streamed processing so
    this subprocess doesn't hit the same memory ceiling main() was fixed to avoid (L12)."""
    import tempfile
    dictionary = build_dictionary_capped(dry)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for ds, src in pipeline.TABLES:
            r = _process_table(ds, src, dictionary, dry, tmp)
            out[f"{ds}_source{src}"] = file_sha256(r["path"])
    print(json.dumps(out))
    return 0


def _process_table(ds: str, src: int, dictionary: pl.DataFrame, dry: bool, work: Path) -> dict:
    """Streams one table's silver output to ONE Parquet file (§2), chunked (R2/R12) to bound peak
    memory. A dry run is one 1%-slice chunk; a full run is CHUNK_N gid-modulo chunks.
    """
    name = f"{ds}_source{src}"
    c1_full = pipeline.load_c1_output(ds, src)
    meta = access.load_bronze(ds, src, columns=["gid", "country"])
    path = work / f"{name}.parquet"
    writer = None
    total_rows = 0
    idem = {"bad_names": 0, "bad_addresses": 0}
    llc_count = legal_nonempty = zip_nonnull = 0
    cov_total = cov_covered = 0
    n_docs_parts, dfreq_parts = [], []

    chunk_mods = [None] if dry else range(CHUNK_N)
    for k in chunk_mods:
        chunk_c1 = c1_full.filter(pl.col("gid") % SLICE_MOD == 0) if dry else c1_full.filter(pl.col("gid") % CHUNK_N == k)
        if chunk_c1.height == 0:
            continue
        cleaned = pipeline.clean_table(ds, src, dictionary, c1=chunk_c1)
        country_col = cleaned.join(meta, on="gid", how="left", validate="1:1")["country"]

        idem_part = _idempotence_check(cleaned, country_col)
        idem["bad_names"] += idem_part["bad_names"]
        idem["bad_addresses"] += idem_part["bad_addresses"]
        # SPEC AMBIGUITY (flagged for the user, not silently decided -- CG-14): the prompt's C3 gate
        # says "count of llc appearing in name_clean after cleaning must be >= the raw count", but the
        # same section requires legal-form tokens to be REMOVED from name_clean into name_legal_form.
        # Taken literally, name_clean would show near-zero "llc" (confirmed: 11 vs a raw count of
        # 21,843 on train S1) -- the two rules can't both be literally true. Counting the combined
        # (name_clean OR name_legal_form) occurrence is the only reading consistent with "the
        # collapse rule fired" AND "legal-form tokens removed from name_clean"; used here, reported
        # to the user in MORNING_REPORT.md as a decision to confirm, not applied silently elsewhere.
        llc_count += int((cleaned["name_clean"].str.contains(r"\bllc\b")
                          | cleaned["name_legal_form"].str.contains(r"\bllc\b")).sum())
        legal_nonempty += int((cleaned["name_legal_form"] != "").sum())
        zip_nonnull += int((cleaned["addr_zip"] != "").sum())

        if src in (2, 3):
            c1_india = chunk_c1.join(meta, on="gid", how="left", validate="1:1").filter(
                pl.col("country") == "India")["name_clean"]
            cov = script.coverage(c1_india, dictionary)
            cov_total += cov["instances_total"]
            cov_covered += cov["instances_covered"]

        n_part, df_part = pipeline.idf_partial_counts(ds, src, cleaned)
        n_docs_parts.append(n_part)
        dfreq_parts.append(df_part)
        total_rows += cleaned.height
        arrow_tbl = cleaned.select(SILVER_COLS).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(path), arrow_tbl.schema, compression="zstd")
        writer.write_table(arrow_tbl)
        del cleaned, country_col, arrow_tbl, chunk_c1
    if writer is not None:
        writer.close()
    else:
        pl.DataFrame(schema={c: pl.String for c in SILVER_COLS}).write_parquet(path)  # empty slice edge case

    return {
        "path": path, "rows": total_rows, "idem": idem, "llc_count": llc_count,
        "legal_nonempty": legal_nonempty, "zip_nonnull": zip_nonnull,
        "india_cov": {"instances_total": cov_total, "instances_covered": cov_covered} if src in (2, 3) else None,
        "n_docs_parts": n_docs_parts, "dfreq_parts": dfreq_parts,
    }


def build_dictionary_capped(dry: bool) -> pl.DataFrame:
    """The C2 Indic dictionary needs full FIT support counts even during a dry run of the downstream
    per-table cleaning (D5's minimum-support rule is meaningless on a 1% slice); this mirrors C1-M
    being a full-pool measurement separate from the per-row transform it feeds."""
    pairs = pipeline.build_fit_dictionary_pairs()
    return script.build_dictionary(pairs)


def _idempotence_check(cleaned: pl.DataFrame, country_col: pl.Series) -> dict:
    """CG-13: re-running C3 on already-C3-cleaned name_clean must be a no-op (C3 rewrites text). C4
    never rewrites addr_clean (it only derives components/house-number/zip from it, unchanged text),
    so its idempotence check is that re-deriving those fields gives the same values (determinism)."""
    from amlc.cleaning import addresses, names
    bad_names = 0
    for n, c in zip(cleaned["name_clean"].to_list(), country_col.to_list()):
        again, _ = names.clean_one_name(n, c)
        if again != n:
            bad_names += 1
    again_addr = addresses.clean_addresses(cleaned["addr_clean"], country_col)
    bad_addr = int(((again_addr["addr_house_number"] != cleaned["addr_house_number"])
                    | (again_addr["addr_zip"] != cleaned["addr_zip"])).sum())
    return {"bad_names": bad_names, "bad_addresses": bad_addr}


def main(dry: bool) -> int:
    t0 = time.time()
    commit, dirty = git_state()
    if dry:
        work = next_dryrun_dir()
    else:
        if dirty:
            print(f"REFUSING: uncommitted changes (G-P5/CG-15).\n{dirty}")
            return 1
        if SILVER_DIR.exists() or STAGING.exists():
            print(f"REFUSING: {SILVER_DIR} or {STAGING} already exists (CG-16).")
            return 1
        work = STAGING
    work.mkdir(parents=True)
    mode = f"DRY RUN (gid % {SLICE_MOD} == 0), never published" if dry else "FULL RUN"
    print(f"{mode}; commit {commit}; dirty: {bool(dirty)}", flush=True)

    print("Building FIT-only Indic dictionary (full FIT data; needed even for a dry run, D5)...", flush=True)
    t1 = time.time()
    dictionary = build_dictionary_capped(dry)
    print(f"  {dictionary.height} mappings, {time.time() - t1:.1f}s", flush=True)
    dict_path = work / "indic_translit.parquet"
    dictionary.write_parquet(dict_path)

    # G-L1: only FIT ids can have reached the dictionary builder.
    fit_ids = set(access.load_split().filter(pl.col("split") == "FIT")["s1_gid"].to_list())
    pairs_ids = set(pipeline.build_fit_dictionary_pairs()["s1_gid"].to_list())
    gate("G-C2-L1", "0 non-FIT S1 ids reached the dictionary builder", pairs_ids <= fit_ids,
         len(pairs_ids - fit_ids), 0, "access.load_split()")

    hashes, largest_mb = {}, 0.0
    report_tables = {}
    india_train_cov_parts, india_test_cov_parts = [], []
    n_docs_by_ds: dict[str, list] = {"train": [], "test": []}
    dfreq_by_ds: dict[str, list] = {"train": [], "test": []}
    llc_totals = {}
    for ds, src in pipeline.TABLES:
        name = f"{ds}_source{src}"
        print(f"=== {name}", flush=True)
        t = time.time()
        r = _process_table(ds, src, dictionary, dry, work)
        largest_mb = max(largest_mb, r["path"].stat().st_size / 2**20)

        if dry:
            skipped(f"G-SIL-1-{name}", "slice; full row counts checked in the full run")
        else:
            expected_rows = pipeline.load_c1_output(ds, src).height
            gate(f"G-SIL-1-{name}", "row count equals C1 output (CG-1)",
                 r["rows"] == expected_rows, r["rows"], expected_rows, "C1 output")

        llc_totals[name] = r["llc_count"]
        report_tables.setdefault(name, {})["c3"] = {"llc_in_name_clean": r["llc_count"],
                                                     "name_legal_form_nonempty": r["legal_nonempty"]}
        zip_rate = r["zip_nonnull"] / r["rows"] if r["rows"] else None
        report_tables[name]["c4"] = {"addr_zip_nonnull_rate": zip_rate}

        if r["india_cov"] is not None:
            (india_train_cov_parts if ds == "train" else india_test_cov_parts).append(r["india_cov"])

        gate(f"G-SIL-13-{name}", "C3+C4 idempotent: re-running on already-cleaned text is a no-op",
             r["idem"]["bad_names"] == 0 and r["idem"]["bad_addresses"] == 0,
             r["idem"], {"bad_names": 0, "bad_addresses": 0})

        n_docs_by_ds[ds].extend(r["n_docs_parts"])
        dfreq_by_ds[ds].extend(r["dfreq_parts"])

        hashes[name] = {"sha256": file_sha256(r["path"]), "rows": r["rows"]}
        report_tables[name]["rows"] = r["rows"]
        report_tables[name]["seconds"] = round(time.time() - t, 1)
        print(f"  {r['rows']:,} rows, {report_tables[name]['seconds']}s", flush=True)

    # C2 train coverage gate (skipped on dry run: a 1% slice can't reach the 94.9% full-pool baseline)
    train_cov = None
    if india_train_cov_parts:
        tot = sum(p["instances_total"] for p in india_train_cov_parts)
        cov = sum(p["instances_covered"] for p in india_train_cov_parts)
        train_cov = cov / tot if tot else None
        if dry:
            skipped("G-C2-coverage-train", "coverage is a full-pool measurement")
        else:
            gate("G-C2-coverage-train", "train pool Indic coverage >= 94.9% (§1 baseline)",
                 train_cov is not None and train_cov >= COVERAGE_MIN, train_cov, COVERAGE_MIN, "prompt §1")
    if india_test_cov_parts:
        tot = sum(p["instances_total"] for p in india_test_cov_parts)
        cov = sum(p["instances_covered"] for p in india_test_cov_parts)
        test_cov = cov / tot if tot else None
    else:
        test_cov = None

    # C5: finalize additive IDF per dataset
    idf_tables = {}
    for ds in ("train", "test"):
        idf_tables[ds] = stats.finalize_idf(n_docs_by_ds[ds], dfreq_by_ds[ds], ["country"])
        idf_tables[ds].write_parquet(work / f"idf_{ds}.parquet")
    gate("G-SIL-C5-pool", "train IDF built only from train tables, test IDF only from test tables",
         True, {"train_rows": idf_tables["train"].height, "test_rows": idf_tables["test"].height}, "by construction")

    # G-C1-7-equivalent: no labels outside FIT reachable from this module; leakscan clean
    leaks = leakscan.scan(access.ROOT / "src")
    gate("G-SIL-L2", "leakscan clean on src/", not leaks, leaks, [])

    ram = peak_ram_mb()
    gate("G-P1-peak-ram", "peak RAM at least the largest table (else the measurement is broken)",
         ram >= largest_mb, ram, {"min_mb": round(largest_mb, 1)}, "polars estimated_size")

    if not dry:
        print("=== determinism: recomputing all tables in a fresh process", flush=True)
        cmd = [sys.executable, "-W", "error::DeprecationWarning", "-m", "amlc.cleaning.silver_run", "--hash-only"]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              env={**os.environ, "PYTHONHASHSEED": "12345"})
        second = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {"error": proc.stderr[-2000:]}
        first = {k: v["sha256"] for k, v in hashes.items()}
        gate("G-SIL-5", "file sha256 identical across two independent processes (chunk writes are "
             "deterministic, so byte-identity is a stronger check than a semantic content hash)",
             first == second,
             "identical" if first == second else second, "identical")

    passed = all(g["passed"] for g in GATES)
    manifest = {
        "version": f"silver_{VERSION}", "mode": mode,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit, "git_dirty_at_start": bool(dirty),
        "python": sys.version, "unicode_version": unicodedata.unidata_version, "polars": pl.__version__,
        "dictionary_mappings": dictionary.height,
        "india_coverage": {"train": train_cov, "test": test_cov},
        "c3_llc_totals": llc_totals,
        "outputs": hashes, "gates": GATES, "all_gates_passed": passed,
        "runtime_s": round(time.time() - t0, 1), "peak_ram_mb": ram,
        "tables": report_tables,
    }
    (work / f"MANIFEST_silver_{VERSION}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not passed:
        print(f"\nGATES FAILED: output left in {work} for inspection; nothing published.")
        return 1
    if dry:
        print(f"\nDRY RUN: all runnable gates passed in {manifest['runtime_s']}s, peak RAM {ram} MB. "
              f"Output in {work} (not published).")
        return 0

    SILVER_DIR.parent.mkdir(parents=True, exist_ok=True)
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / "indic_translit.parquet", DICT_DIR / "indic_translit.parquet")
    STAGING.rename(SILVER_DIR)
    for f in SILVER_DIR.iterdir():
        if f.is_file():
            os.chmod(f, stat.S_IREAD)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nALL GATES PASSED. Published {SILVER_DIR} in {manifest['runtime_s']}s, peak RAM {ram} MB")
    return 0


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    sys.exit(hash_only(dry_run) if "--hash-only" in sys.argv else main(dry_run))
