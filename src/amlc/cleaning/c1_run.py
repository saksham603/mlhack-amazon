"""C1-only runner (Stage 1 prompt rev 4, step-by-step mode).

Reads bronze through access.load_bronze(), writes data/interim/c1/<VERSION>/ (NOT silver), a manifest
and C1_REPORT.md, then stops. Outputs are staged and published only if every gate passes.
  python -m amlc.cleaning.c1_run --dry-run   1% slice (gid % 100 == 0), never published (G-C2)
  python -m amlc.cleaning.c1_run             full run; refuses to start on uncommitted code (G-P5)
  python -m amlc.cleaning.c1_run --hash-only [--dry-run]  content hashes in a fresh process (G-C1-5)
"""
import ast
import ctypes
import ctypes.wintypes
import json
import os
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import polars as pl

from amlc.cleaning import recount, text
from amlc.foundation import access, leakscan
from amlc.foundation.hashing import content_hash, file_sha256

VERSION = "v2"  # v1 was built from uncommitted code; superseded and kept untouched (CG-15, CG-16)
C1_DIR = access.DATA / "interim" / "c1"
OUT = C1_DIR / VERSION
STAGING = C1_DIR / f"_staging_{VERSION}"
DRYRUN = C1_DIR / f"_dryrun_{VERSION}"  # never published; rewritten by every dry run
MEASURE_JSON = C1_DIR / "measure_v3" / "c1m_measurements.json"
PROFILE = access.DATA / "profile" / "v2" / "profile.parquet"
MANIFEST_V1 = access.DATA / "MANIFEST_v1.json"
TABLES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]
OUT_COLS = ["gid", "name_clean", "addr_clean", "addr_missing", "addr_null_parts_removed", "addr_parts_emptied"]
SLICE_MOD = 100  # dry run keeps gid % 100 == 0 (1%)
INDIC = (0x0900, 0x0D7F)
# §1's definition: a maximal run of Indic characters, NOT a whitespace token (rev 4, user-approved).
# Only this definition reproduces 1,537 (identical train/test) on raw text; whitespace tokens give 1,518
# because ZWNJ (Cf, outside this range) sits inside words and splits one word into two runs.
INDIC_RUN = re.compile("[ऀ-ൿ]+")
EXPECTED_INDIC_VOCAB = 1537
ALLOWED_CHARS = set(string.ascii_lowercase + string.digits + string.punctuation + " ")
CLEANING_SRC = Path(__file__).resolve().parent
LABEL_LOADERS = {"load_labels", "load_match_counts"}
REPAIR_CANDIDATES = "[\x00-\x08\x0e-\x1f\x7f\u0080-ÿ]"

GATES: list[dict] = []


def gate(gid: str, desc: str, passed: bool, measured, expected=None, source=None) -> None:
    GATES.append({"gate": gid, "description": desc, "passed": bool(passed), "measured": measured,
                  "expected": expected, "expected_source": source})
    print(f"  [{'PASS' if passed else 'FAIL'}] {gid}: measured={measured} expected={expected}", flush=True)


def skipped(gid: str, why: str) -> None:
    GATES.append({"gate": gid, "description": f"SKIPPED: {why}", "passed": True, "measured": "skipped",
                  "expected": None, "expected_source": None})


def code_identifiers(path: Path, pred, strings: bool = True) -> list[str]:
    """Names, attributes, arguments, imports and (if strings) short string constants such as column
    names in a module's code, excluding docstrings, that satisfy pred."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    found = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            s = n.id
        elif isinstance(n, ast.Attribute):
            s = n.attr
        elif isinstance(n, ast.arg):
            s = n.arg
        elif isinstance(n, ast.alias):
            s = n.name
        elif strings and isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings and len(n.value) <= 40:
            s = n.value
        else:
            continue
        if pred(s):
            found.append(s)
    return sorted(set(found))


def is_indic(ch: str) -> bool:
    return INDIC[0] <= ord(ch) <= INDIC[1]


def peak_ram_mb() -> float:
    """Peak working set of this process. Raises if the OS call fails (CG-11: never a silent default)."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    # Without argtypes/restype, ctypes passes GetCurrentProcess()'s pseudo-handle as a 32-bit int and the
    # call fails; the unchecked failure reported a false "0.0 MB" in the first C1 run (2026-09-26).
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    get_process_memory_info.restype = ctypes.wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not get_process_memory_info(get_current_process(), ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError()
    return round(pmc.PeakWorkingSetSize / 2**20, 1)


def git_state() -> tuple[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=access.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"], cwd=access.ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return commit, dirty


def load(ds: str, src: int, dry: bool) -> pl.DataFrame:
    df = access.load_bronze(ds, src, columns=["gid", "country", "business_name", "business_address"])
    return df.filter(pl.col("gid") % SLICE_MOD == 0) if dry else df


def transform(df: pl.DataFrame) -> pl.DataFrame:
    addr = text.clean_addresses(df["business_address"])
    return pl.DataFrame({"gid": df["gid"], "name_clean": text.clean_names(df["business_name"])}).hstack(addr).select(OUT_COLS)


def hash_only(dry: bool) -> int:
    print(json.dumps({f"{ds}_source{src}": content_hash(transform(load(ds, src, dry)), OUT_COLS, "gid")
                      for ds, src in TABLES}))
    return 0


def expected_rows(prof: pl.DataFrame, ds: str, src: int, col: str) -> dict[str, int]:
    p = prof.filter((pl.col("dataset") == ds) & (pl.col("src") == src))
    return {c: round(pct * n / 100) for c, pct, n in p.select("country", col, "n").iter_rows()}


def main(dry: bool) -> int:
    t0 = time.time()
    commit, dirty = git_state()
    if dry:
        work = DRYRUN
        if work.exists():
            shutil.rmtree(work)  # dry-run output is never published (CG-16 exception, like staging)
    else:
        if dirty:
            print(f"REFUSING: uncommitted changes in src/ or tests/ (G-P5 / CG-15). Commit first.\n{dirty}")
            return 1
        for p in (OUT, STAGING):
            if p.exists():
                print(f"REFUSING: {p} exists. Published output is never overwritten or deleted (CG-16).")
                return 1
        work = STAGING
    work.mkdir(parents=True)
    mode = f"DRY RUN on 1% slice (gid % {SLICE_MOD} == 0), never published" if dry else "FULL RUN"
    print(f"{mode}; commit {commit}; tree dirty: {bool(dirty)}", flush=True)

    m1 = json.loads(MANIFEST_V1.read_text(encoding="utf-8"))
    m2 = access._manifest()
    prof_sha = file_sha256(PROFILE)
    gate("C0-profile-hash", "profile v2 sha256 equals MANIFEST_v2", prof_sha == m2["new_artifacts"]["profile"]["profile.parquet"],
         prof_sha, m2["new_artifacts"]["profile"]["profile.parquet"], "MANIFEST_v2.new_artifacts.profile")
    prof = pl.read_parquet(PROFILE)
    c1m = {t["table"]: t for t in json.loads(MEASURE_JSON.read_text(encoding="utf-8"))["tables"]}
    s1_exact = {g["gate"]: g["measured"]["measured_rows"] for g in m2["gates"] if g["gate"].endswith("S1-null-component")}

    hashes, report = {}, {"tables": {}}
    vocab_raw = {"train": set(), "test": set()}
    vocab_clean_ws = {"train": set(), "test": set()}
    largest_table_mb = 0.0
    for ds, src in TABLES:
        name = f"{ds}_source{src}"
        print(f"=== {name}", flush=True)
        t = time.time()
        df = load(ds, src, dry)
        largest_table_mb = max(largest_table_mb, df.estimated_size("mb"))
        if dry:
            skipped(f"G-C1-1-{name}-rows-in", "slice; full row counts checked in the full run")
        else:
            gate(f"G-C1-1-{name}-rows-in", "bronze rows equal MANIFEST_v1 row_counts", df.height == m1["row_counts"][name],
                 df.height, m1["row_counts"][name], "MANIFEST_v1.row_counts")
        out = transform(df)
        gate(f"G-C1-1-{name}-rows-out", "output rows and gid order equal input",
             out.height == df.height and out["gid"].equals(df["gid"]), out.height, df.height, "input")
        rep = {"rows": df.height}

        # ---- G-C1-2 / G-C1-3 / G-C1-9: independent recount vs profile v2, C1-M and text.py ----
        rc = recount.recount_addresses(df["business_address"].to_list(), df["country"].to_list())
        if dry:
            for g in ("2a", "2b", "3a"):
                skipped(f"G-C1-{g}-{name}", "expectation is a full-data count")
        else:
            exp_s0 = expected_rows(prof, ds, src, "addr_null_component_pct")
            got_s0 = {c: v["stage0"] for c, v in rc["by_country"].items()}
            gate(f"G-C1-2a-{name}", "rows with a Stage 0 null part: recount == profile v2, per country",
                 got_s0 == exp_s0, got_s0, exp_s0, "profile/v2 addr_null_component_pct x n")
            if src == 1:
                key = f"C4-{ds}S1-null-component"
                gate(f"G-C1-2a-{name}-exact", "S1 rows with a Stage 0 null part", rc["total"]["stage0"] == s1_exact[key],
                     rc["total"]["stage0"], s1_exact[key], f"MANIFEST_v2 gate {key}")
            exp_v = c1m[name]["a_case_variant_null_rows"]
            gate(f"G-C1-2b-{name}", "rows with a case-variant null part: recount == C1-M v2",
                 rc["total"]["variant"] == exp_v, rc["total"]["variant"], exp_v, "measure_v2")
            exp_empty = expected_rows(prof, ds, src, "empty_addr_pct")
            got_empty = {c: v["raw_empty"] for c, v in rc["by_country"].items()}
            gate(f"G-C1-3a-{name}", "raw empty addresses: recount == profile v2, per country",
                 got_empty == exp_empty, got_empty, exp_empty, "profile/v2 empty_addr_pct x n")
        text_rows = int((out["addr_null_parts_removed"] > 0).sum())
        gate(f"G-C1-2c-{name}", "rows with >=1 part removed: text.py == recount (union)",
             text_rows == rc["total"]["c1"], text_rows, rc["total"]["c1"], "recount.py")
        text_parts = int(out["addr_null_parts_removed"].sum())
        gate(f"G-C1-2d-{name}", "total parts removed: text.py == recount",
             text_parts == rc["total"]["parts_removed"], text_parts, rc["total"]["parts_removed"], "recount.py")
        text_missing = int(out["addr_missing"].sum())
        gate(f"G-C1-3b-{name}", "addr_missing rows: text.py == recount, and >= raw empty",
             text_missing == rc["total"]["all_missing"] and text_missing >= rc["total"]["raw_empty"],
             text_missing, {"recount": rc["total"]["all_missing"], "min": rc["total"]["raw_empty"]}, "recount.py")
        emptied = {"parts": int(out["addr_parts_emptied"].sum()), "rows": int((out["addr_parts_emptied"] > 0).sum())}
        exp_emptied = {"parts": rc["total"]["parts_emptied"], "rows": rc["total"]["rows_with_emptied"]}
        gate(f"G-C1-9-{name}", "D-C1f parts emptied by cleaning: text.py == recount", emptied == exp_emptied,
             emptied, exp_emptied, "recount.py")
        rep["missing"] = {"recount_total": rc["total"], "recount_by_country": rc["by_country"]}

        # ---- G-C1-4 idempotence (every distinct output value) ----
        names_u = out["name_clean"].unique()
        bad_n = [x for x in names_u.to_list() if text.clean_text(x) != x]
        addr_u = out["addr_clean"].unique()
        again = text.clean_addresses(addr_u)
        bad_a = (again["addr_clean"] != addr_u) | ((addr_u != "") & (
            (again["addr_null_parts_removed"] > 0) | (again["addr_parts_emptied"] > 0)))
        gate(f"G-C1-4-{name}", "C1(C1(x)) == C1(x) on every distinct name and address",
             not bad_n and not bad_a.any(), {"bad_names": len(bad_n), "bad_addresses": int(bad_a.sum()),
                                             "examples": bad_n[:5] + addr_u.filter(bad_a).head(5).to_list()}, 0)

        # ---- G-C1-8 Indic characters neither added nor removed ----
        both = pl.DataFrame({"raw": pl.concat([df["business_name"], df["business_address"]]),
                             "clean": pl.concat([out["name_clean"], out["addr_clean"]])}).unique()
        both = both.filter(pl.col("raw").str.contains("[ऀ-ൿ]") | pl.col("clean").str.contains("[ऀ-ൿ]"))
        bad8 = []
        for r, c in both.iter_rows():
            ref = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", r))
            if Counter(ch for ch in ref if is_indic(ch)) != Counter(ch for ch in c if is_indic(ch)):
                bad8.append((r, c))
        gate(f"G-C1-8-{name}", "Indic character multiset unchanged (vs NFC(NFKC(raw)))", not bad8,
             {"bad": len(bad8), "examples": bad8[:3]}, 0)

        # ---- repairs (D-C1b, D-C1e) and G-C1-10 ----
        g = Counter()
        for col, series in (("name", df["business_name"]), ("addr", text.remove_missing_parts(df["business_address"])["addr_joined"])):
            tmp = pl.DataFrame({"v": series}).filter(pl.col("v").str.contains(REPAIR_CANDIDATES)).group_by("v").len()
            for v, n in tmp.iter_rows():
                for k, cnt in text.repair_counts(v).items():
                    g[f"{col}: {k}"] += cnt * n
        rep["repairs"] = dict(g.most_common())
        if dry:
            skipped(f"G-C1-10-{name}", "expectation is a full-data count")
        else:
            got_ctrl = {c: sum(v for k, v in g.items() if k.startswith(f"{c}: ascii_control")) for c in ("name", "addr")}
            exp_ctrl = {c: sum(c1m[name][f"g_{c}_ascii_control_non_ws"].values()) for c in ("name", "addr")}
            gate(f"G-C1-10-{name}", "D-C1e ASCII controls replaced == C1-M v2 count", got_ctrl == exp_ctrl,
                 got_ctrl, exp_ctrl, "measure_v2")

        # ---- report material: Q3, Q4, Q1 ----
        full = df.select("country", "business_name").hstack(out.select("name_clean", "addr_clean"))
        unexpected, unexpected_ex = Counter(), {}
        for col in ("name_clean", "addr_clean"):
            d = full.filter(pl.col(col).str.contains(r"[^ -~]")).group_by(col, "country").len()
            for v, ctry, n in d.iter_rows():
                for ch in set(v):
                    if ch not in ALLOWED_CHARS and not is_indic(ch):
                        k = f"U+{ord(ch):04X} {unicodedata.name(ch, '?')} | {col} | {ctry}"
                        unexpected[k] += n * v.count(ch)
                        unexpected_ex.setdefault(k, v)
        rep["unexpected_top50"] = [(k, c, unexpected_ex[k]) for k, c in unexpected.most_common(50)]

        tc, changed_rows = Counter(), 0
        for raw, clean, ctry, n in full.group_by("business_name", "name_clean", "country").len().iter_rows():
            b, a = len(raw.split()), len(clean.split())
            tc[(ctry, "before", min(b, 13))] += n
            tc[(ctry, "after", min(a, 13))] += n
            changed_rows += n if a != b else 0
        rep["name_token_counts"] = {f"{c}|{w}|{k if k < 13 else '13+'}": v for (c, w, k), v in sorted(tc.items())}
        rep["name_token_count_changed_rows"] = changed_rows

        if src in (2, 3):
            ind = full.filter(pl.col("country") == "India").select("business_name", "name_clean").unique()
            for raw, clean in ind.iter_rows():
                vocab_raw[ds].update(INDIC_RUN.findall(raw))
                vocab_clean_ws[ds].update(tok for tok in clean.split() if any(map(is_indic, tok)))

        path = work / f"{name}.parquet"
        out.write_parquet(path, compression="zstd")
        hashes[name] = {"sha256": file_sha256(path), "content_hash": content_hash(out, OUT_COLS, "gid"), "rows": out.height}
        rep["seconds"] = round(time.time() - t, 1)
        report["tables"][name] = rep
        del df, out, full, both

    # ---- Q1 Indic vocabulary ----
    q1_raw = {k: len(vocab_raw[k]) for k in ("train", "test")}
    if dry:
        skipped("G-Q1-raw-vocab", "vocabulary is a full-data count")
    else:
        gate("G-Q1-raw-vocab", "raw Indic vocab (maximal-run def.) reproduces §1 (1,537, identical train/test)",
             q1_raw["train"] == q1_raw["test"] == EXPECTED_INDIC_VOCAB and vocab_raw["train"] == vocab_raw["test"],
             q1_raw, EXPECTED_INDIC_VOCAB, "prompt §1")
    all_raw = vocab_raw["train"] | vocab_raw["test"]
    tok_map = {tok: text.clean_text(tok) for tok in all_raw}
    by_clean: dict[str, list[str]] = {}
    for r, c in tok_map.items():
        by_clean.setdefault(c, []).append(r)
    report["q1"] = {
        "raw_maximal_run_counts": q1_raw,
        "clean_whitespace_token_counts": {k: len(vocab_clean_ws[k]) for k in ("train", "test")},
        "clean_whitespace_train_equals_test": vocab_clean_ws["train"] == vocab_clean_ws["test"],
        "run_tokens_not_whole_words": sorted(all_raw - (vocab_clean_ws["train"] | vocab_clean_ws["test"])),
        "changed": sorted([(r, c) for r, c in tok_map.items() if r != c]),
        "merged": sorted([(c, sorted(rs)) for c, rs in by_clean.items() if len(rs) > 1]),
    }

    # ---- G-C1-5 determinism in a fresh process ----
    print("=== G-C1-5: recomputing all tables in a fresh process", flush=True)
    cmd = [sys.executable, "-W", "error::DeprecationWarning", "-m", "amlc.cleaning.c1_run", "--hash-only"] + (["--dry-run"] if dry else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONHASHSEED": "12345"})
    second = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {"error": proc.stderr[-2000:]}
    first = {k: v["content_hash"] for k, v in hashes.items()}
    gate("G-C1-5", "content hashes identical across two independent processes", first == second,
         "identical" if first == second else second, "identical")

    # ---- G-C1-6 / G-C1-7 static checks ----
    country_refs = code_identifiers(CLEANING_SRC / "text.py", lambda s: "country" in s.lower())
    gate("G-C1-6", "text.py code never references a country (identifiers and column-name strings)",
         not country_refs, country_refs, [])
    label_refs = {f.name: code_identifiers(f, lambda s: s in LABEL_LOADERS, strings=False) for f in sorted(CLEANING_SRC.glob("*.py"))}
    label_refs = {k: v for k, v in label_refs.items() if v}
    leaks = leakscan.scan(access.ROOT / "src")
    gate("G-C1-7", "cleaning code never loads labels; leakscan clean", not label_refs and not leaks,
         {"label_refs": label_refs, "leakscan": leaks}, "none")

    # ---- G-P1 plausibility of the resource measurement (CG-11) ----
    ram = peak_ram_mb()
    gate("G-P1-peak-ram", "peak RAM is at least the largest table loaded (else the measurement is broken)",
         ram >= largest_table_mb, ram, {"min_mb": round(largest_table_mb, 1)}, "polars estimated_size")

    passed = all(g["passed"] for g in GATES)
    manifest = {
        "version": f"c1_{VERSION}", "mode": mode, "status": "interim, NOT silver",
        "supersedes": "c1_v1: built from uncommitted code (CG-15); kept untouched (CG-16)",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit, "git_dirty_at_start": bool(dirty),
        "python": sys.version, "unicode_version": unicodedata.unidata_version, "polars": pl.__version__,
        "rules": {"missing_words_lowercase": text.MISSING_WORDS, "ascii_control": text.ASCII_CONTROL.pattern,
                  "garbled_lost": text.GARBLED_LOST.pattern, "garbled_utf8": text.GARBLED_UTF8.pattern,
                  "c1_control": text.C1_CONTROL.pattern, "ligatures": text.LIGATURES, "latin_ranges": text.LATIN_RANGES,
                  "accent_marks": text.ACCENT_MARKS, "space_before_comma": text.SPACE_BEFORE_COMMA.pattern,
                  "order": "1 missing parts (raw) > 2 ascii controls, garbled repair, Cf removal > 3 NFKC > "
                           "4 casefold+NFKC > 5 latin fold+ligatures+NFC > 6 whitespace, no space before comma > "
                           "7 drop parts emptied by cleaning"},
        "decisions": {"D-C1a": "case-insensitive missing words", "D-C1b": "repair garbled text to intended char",
                      "D-C1c": "fold oe/ae ligatures", "D-C1d": "names exactly 'NA' left unchanged",
                      "D-C1e": "ASCII controls become a space", "D-C1f": "drop address parts emptied by cleaning",
                      "rev4": "Q1 maximal-run definition and space-before-comma rule, approved retroactively"},
        "inputs": {"MANIFEST_v2_sha256": file_sha256(access.MANIFEST), "profile_v2_sha256": prof_sha,
                   "c1m_measurements_v2_sha256": file_sha256(MEASURE_JSON)},
        "outputs": hashes, "gates": GATES, "all_gates_passed": passed,
        "runtime_s": round(time.time() - t0, 1), "peak_ram_mb": ram,
    }
    (work / f"MANIFEST_c1_{VERSION}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (work / "C1_REPORT.md").write_text(render_report(manifest, report), encoding="utf-8")
    if not passed:
        print(f"\nGATES FAILED: outputs left in {work} for inspection; nothing published.")
        return 1
    if dry:
        print(f"\nDRY RUN: all runnable gates passed in {manifest['runtime_s']}s, peak RAM {ram} MB. Output in {work} (not published).")
        return 0
    STAGING.rename(OUT)
    for f in OUT.iterdir():
        os.chmod(f, stat.S_IREAD)
    print(f"\nALL GATES PASSED. Published {OUT} in {manifest['runtime_s']}s, peak RAM {ram} MB")
    return 0


def render_report(m: dict, r: dict) -> str:
    L = [f"# C1 report {m['version']} (interim, NOT silver)", "", f"**{m['mode']}**", "",
         f"Built {m['built_at_utc']} from commit {m['git_commit']} (dirty at start: {m['git_dirty_at_start']}), "
         f"runtime {m['runtime_s']}s, peak RAM {m['peak_ram_mb']} MB, Unicode {m['unicode_version']}, "
         f"all gates passed: **{m['all_gates_passed']}**", "",
         "## Gates", "", "| gate | passed | measured | expected | source |", "|---|---|---|---|---|"]
    for g in m["gates"]:
        L.append(f"| {g['gate']} | {'PASS' if g['passed'] else 'FAIL'} | {json.dumps(g['measured'], ensure_ascii=False, default=str)[:300]} "
                 f"| {json.dumps(g['expected'], ensure_ascii=False, default=str)[:200]} | {g['expected_source'] or ''} |")
    q1 = r.get("q1", {})
    L += ["", "## Q1 Indic vocabulary", "",
          f"Raw, maximal-run definition (matches §1): {q1.get('raw_maximal_run_counts')} (gate target 1,537)",
          f"Post-C1, whitespace-token definition (what C2 aligns on): {q1.get('clean_whitespace_token_counts')}, "
          f"train==test: {q1.get('clean_whitespace_train_equals_test')}",
          f"Run-tokens that are not whole words after C1 (ZWNJ halves etc.): {len(q1.get('run_tokens_not_whole_words', []))}",
          f"Raw run-tokens changed by C1: {len(q1.get('changed', []))}; clean tokens that merged >1 raw token: "
          f"{len(q1.get('merged', []))}", "",
          "### Run-tokens that are not whole words after C1", ""]
    L += [f"- `{x}`" for x in q1.get("run_tokens_not_whole_words", [])]
    L += ["", "### Merged (clean token <- raw tokens)", ""]
    L += [f"- `{c}` <- {', '.join(f'`{x!r}`' for x in rs)}" for c, rs in q1.get("merged", [])]
    L += ["", "### Changed (raw -> clean)", ""] + [f"- `{a!r}` -> `{b!r}`" for a, b in q1.get("changed", [])]
    for name, t in r["tables"].items():
        L += ["", f"## {name} ({t['rows']:,} rows, {t['seconds']}s)", "",
              "### Q2 missing and emptied address parts (recount.py)", "", f"Total: {t['missing']['recount_total']}", "",
              f"By country: {t['missing']['recount_by_country']}", "",
              "### Repairs (D-C1b garbled text, D-C1e ASCII controls)", ""] + [f"- {k}: {v:,}" for k, v in t["repairs"].items()]
        L += ["", "### Q3 unexpected characters after C1 (top 50, rows weighted)", ""] + \
             [f"- {k}: {c:,} e.g. `{ex[:120]}`" for k, c, ex in t["unexpected_top50"]]
        L += ["", f"### Q4 name token counts (rows whose count changed: {t['name_token_count_changed_rows']:,})", ""] + \
             [f"- {k}: {v:,}" for k, v in t["name_token_counts"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    sys.exit(hash_only(dry_run) if "--hash-only" in sys.argv else main(dry_run))
