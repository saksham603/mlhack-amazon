"""Unattended day-2 chain (2026-09-27 05:30): runs every remaining step in order, independent of the
Claude session (launched by Windows Task Scheduler). Each step: skipped if done (marker file or its
product exists), partial files quarantined (moved, never deleted) before it runs, run as a subprocess
with its own log, marker written on exit 0; the chain stops at the first failure and records it in
data/_v31/chain/status.json. One heavy job at a time. Env: PYTHONPATH=src, AMLC_KA=10.

Run: python -m amlc.pipeline.chain_day2 [--dry-run]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(r"C:\Users\suremdra singh\amlc2026")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
V31 = ROOT / "data" / "_v31"
CHAIN = V31 / "chain"
LOGS = ROOT / "data" / "_v2" / "logs"
VALIDATOR = Path(r"C:\Users\suremdra singh\Desktop\dataset_student_resource\student_resource\utils\validate_submission.py")
TEST_DIR = VALIDATOR.parent.parent / "dataset" / "test"


def now():
    return time.strftime("%H:%M:%S")


def status(**kw):
    CHAIN.mkdir(parents=True, exist_ok=True)
    p = CHAIN / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {"steps": {}}
    for k, v in kw.items():
        if k == "step":
            s["steps"][v[0]] = v[1]
        else:
            s[k] = v
    s["updated"] = now()
    p.write_text(json.dumps(s, indent=2))


def quarantine(dirs) -> int:
    """Move leftover .tmp files and unreadable parquet files out of the way (never delete)."""
    q = CHAIN / "quarantine" / time.strftime("%H%M%S")
    moved = 0
    for d in dirs:
        d = V31 / d
        if not d.exists():
            continue
        for f in list(d.glob("*.tmp")) + list(d.glob("*.parquet")):
            bad = f.suffix == ".tmp"
            if not bad:
                try:
                    pl.read_parquet_schema(f)
                except Exception:
                    bad = True
            if bad:
                q.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(q / f.name))
                moved += 1
    return moved


def exists(rel: str) -> bool:
    return (V31 / rel).exists()


STEPS = [  # name, argv, dirs to check for partial files, extra done-check
    ("v31_val", ["-m", "amlc.pipeline.run_v31", "val"], ["val"], None),
    ("v31_fit", ["-m", "amlc.pipeline.run_v31", "fit"], ["fit"], None),
    ("v31_train", ["-m", "amlc.pipeline.run_v3", "train"], [], lambda: exists("report_v3.json")),
    ("v31_test", ["-m", "amlc.pipeline.run_v31", "test"], ["test_c3", "test_feats_scored"], None),
    ("assemble_base", ["-m", "amlc.pipeline.run_v3", "assemble", "v31_base"], [], None),
    ("validate_base", ["VALIDATE", "v31_base"], [], None),
    ("p2_folds", ["-m", "amlc.model.run_pass2", "folds"], [], None),
    ("p2_feats", ["-m", "amlc.model.run_pass2", "feats"], ["fit_p2", "val_p2", "test_feats_scored_p2"], None),
    ("p2_train", ["-m", "amlc.model.run_pass2", "train"], [], lambda: exists("report_pass2.json")),
    ("p2_test", ["-m", "amlc.model.run_pass2", "test"], ["test_feats_scored_p2"], None),
    ("assemble_p2", ["-m", "amlc.model.run_pass2", "assemble", "v31_pass2"], [], None),
    ("validate_p2", ["VALIDATE", "v31_pass2"], [], None),
]


def run(name: str, argv: list) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), AMLC_KA="10", PYTHONIOENCODING="utf-8")
    log = LOGS / f"chain_{name}.log"
    if argv[0] == "VALIDATE":  # official validator on the matching file + our streaming candidate check
        out = ROOT / "output" / argv[1]
        cmd = [PY, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
               "--candidate", str(out / "__skip__.tsv"), "--test-dir", str(TEST_DIR), "--check-ids"]
        cmd2 = [PY, "-m", "amlc.pipeline.check_candidates", str(out)]
        with open(log, "w", encoding="utf-8") as fh:
            r1 = subprocess.run(cmd, cwd=VALIDATOR.parent.parent, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            r2 = subprocess.run(cmd2, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        txt = log.read_text(encoding="utf-8", errors="replace")
        return 0 if (r1 == 0 and r2 == 0 and "PASS" in txt and "CANDIDATES OK" in txt) else 1
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {now()} start {name}\n")
        fh.flush()
        return subprocess.run([PY, "-u", *argv], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> int:
    dry = "--dry-run" in sys.argv
    CHAIN.mkdir(parents=True, exist_ok=True)
    if not dry:
        status(state="running", pid=os.getpid(), started=now())
    for name, argv, dirs, done_check in STEPS:
        marker = CHAIN / f"{name}.done"
        if marker.exists() or (done_check is not None and done_check()):
            print(now(), f"{name}: done, skipping", flush=True)
            continue
        if dry:
            print(now(), f"{name}: WOULD RUN {' '.join(argv)}", flush=True)
            continue
        moved = quarantine(dirs)
        print(now(), f"{name}: start (quarantined {moved} partial files)", flush=True)
        status(step=(name, {"state": "running", "start": now()}))
        t0 = time.time()
        rc = run(name, argv)
        mins = round((time.time() - t0) / 60, 1)
        if rc != 0:
            print(now(), f"{name}: FAILED rc={rc} after {mins} min -- chain stopped, see data/_v2/logs/chain_{name}.log", flush=True)
            status(state="failed", step=(name, {"state": "failed", "rc": rc, "minutes": mins}))
            return rc
        marker.write_text(now())
        print(now(), f"{name}: ok ({mins} min)", flush=True)
        status(step=(name, {"state": "ok", "minutes": mins}))
    if not dry:
        status(state="finished")
    print(now(), "chain finished" if not dry else "dry run finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
