"""Unattended v3.2 chain (2026-09-27 08:15), launched by Task Scheduler like chain_day2. It waits until no
other amlc python process runs (the day-2 chain driver included, so it starts only after that chain has
exited), refuses to start unless the v3.1 test scoring finished (its done marker), then runs the v3.2
steps in order. Done markers, status.json and quarantine live in data/_v32/chain/.

Run: python -m amlc.pipeline.chain_v32 [--dry-run]
"""
import json
import os
import shutil
import subprocess
import sys
import time

import polars as pl

from amlc.pipeline import chain_day2 as C

V32 = C.ROOT / "data" / "_v32"
CHAIN = V32 / "chain"
PRECONDITION = C.CHAIN / "v31_test.done"  # the v3.2 steps read every v3.1 test file

STEPS = [  # name, argv, dirs (under data/_v32) to check for partial files, extra done-check
    ("x_val", ["-m", "amlc.pipeline.run_v32", "x", "val"], ["x_val"], None),
    ("x_fit", ["-m", "amlc.pipeline.run_v32", "x", "fit"], ["x_fit"], None),
    ("x_test", ["-m", "amlc.pipeline.run_v32", "x", "test"], ["x_test"], None),
    ("v32_train", ["-m", "amlc.pipeline.run_v32", "train"], [], lambda: (V32 / "report_v32.json").exists()),
    ("v32_test", ["-m", "amlc.pipeline.run_v32", "test"], ["test_scored"], None),
    ("assemble_v32", ["-m", "amlc.pipeline.run_v32", "assemble", "v32"], [], None),
    ("validate_v32", ["VALIDATE", "v32"], [], None),
]


def status(**kw):
    CHAIN.mkdir(parents=True, exist_ok=True)
    p = CHAIN / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {"steps": {}}
    for k, v in kw.items():
        if k == "step":
            s["steps"][v[0]] = v[1]
        else:
            s[k] = v
    s["updated"] = C.now()
    p.write_text(json.dumps(s, indent=2))


def quarantine(dirs) -> int:
    """Move leftover .tmp files and unreadable parquet files out of the way (never delete)."""
    q = CHAIN / "quarantine" / time.strftime("%H%M%S")
    moved = 0
    for d in dirs:
        d = V32 / d
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


def other_pipeline_pids() -> list:
    """PIDs of python processes running any amlc module (the day-2 chain driver included) except this driver."""
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return [-1]  # unknown -> treat as busy
    if not out:
        return []
    rows = json.loads(out)
    rows = rows if isinstance(rows, list) else [rows]
    return [r["ProcessId"] for r in rows
            if "amlc." in (r.get("CommandLine") or "") and "chain_v32" not in (r.get("CommandLine") or "")]


def wait_for_idle() -> None:
    clean = 0
    while clean < 2:
        pids = other_pipeline_pids()
        clean = clean + 1 if not pids else 0
        if pids:
            print(C.now(), f"waiting: other pipeline processes running {pids}", flush=True)
            status(state="waiting", waiting_for=pids)
        time.sleep(60 if pids else 30)


def main() -> int:
    dry = "--dry-run" in sys.argv
    CHAIN.mkdir(parents=True, exist_ok=True)
    if not dry:
        wait_for_idle()
    if not PRECONDITION.exists():
        print(C.now(), f"BLOCKED: {PRECONDITION} missing -- the v3.1 test scoring did not finish; nothing run", flush=True)
        if not dry:
            status(state="blocked", reason="v31_test.done missing")
            return 2
    if not dry:
        status(state="running", pid=os.getpid(), started=C.now())
    for name, argv, dirs, done_check in STEPS:
        marker = CHAIN / f"{name}.done"
        if marker.exists() or (done_check is not None and done_check()):
            print(C.now(), f"{name}: done, skipping", flush=True)
            continue
        if dry:
            print(C.now(), f"{name}: WOULD RUN {' '.join(argv)}", flush=True)
            continue
        moved = quarantine(dirs)
        print(C.now(), f"{name}: start (quarantined {moved} partial files)", flush=True)
        status(step=(name, {"state": "running", "start": C.now()}))
        t0 = time.time()
        rc = C.run(name, argv)
        mins = round((time.time() - t0) / 60, 1)
        if rc != 0:
            print(C.now(), f"{name}: FAILED rc={rc} after {mins} min -- chain stopped, see data/_v2/logs/chain_{name}.log", flush=True)
            status(state="failed", step=(name, {"state": "failed", "rc": rc, "minutes": mins}))
            return rc
        marker.write_text(C.now())
        print(C.now(), f"{name}: ok ({mins} min)", flush=True)
        status(step=(name, {"state": "ok", "minutes": mins}))
    if not dry:
        status(state="finished")
    print(C.now(), "chain finished" if not dry else "dry run finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
