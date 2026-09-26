"""T7: W1 blocking dry run (design and measurement only; overnight master prompt). Samples 1% of
FIT S1 per country against the FULL S2/S3 pool of that country. Never runs on full S1 data, never
chooses k or a cap, never writes candidate_pairs.tsv (T7 scope, not W1 itself).
"""
import ctypes
import json
import time
from pathlib import Path

import polars as pl

from amlc.blocking import candidates as C
from amlc.cleaning.c1_run import peak_ram_mb
from amlc.foundation import access

OUT_DIR = access.DATA / "_dryrun_w1"
COUNTRIES = ("US", "India")
RAM_BUDGET_BYTES = 6 * 2**30  # generous for the 1% sample; not the full-run budget
MIN_FREE_RAM_MB = 1024  # STOP floor (master prompt §7.4)


class _MemStat(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
               ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
               ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
               ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
               ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_ram_mb() -> float:
    """CG-11: checked, never a silent default. Added after L13 -- this module had no RAM check at
    all before, unlike silver_run.py, even though a real run of it peaked at 10.1 GB."""
    fn = ctypes.windll.kernel32.GlobalMemoryStatusEx
    fn.argtypes = [ctypes.POINTER(_MemStat)]
    fn.restype = ctypes.c_int
    m = _MemStat()
    m.dwLength = ctypes.sizeof(_MemStat)
    if not fn(ctypes.byref(m)):
        raise ctypes.WinError()
    if m.ullAvailPhys <= 0 or m.ullAvailPhys > m.ullTotalPhys:
        raise RuntimeError(f"implausible free RAM reading: {m.ullAvailPhys} of {m.ullTotalPhys}")
    return round(m.ullAvailPhys / 2**20, 1)


def check_ram_budget() -> None:
    """CG-19/G-P9, applied here after L13 found this module lacked it. Raises rather than continuing
    into a step that could exhaust memory; the caller logs and moves on (§7.4 STOP condition)."""
    free = free_ram_mb()
    if free < MIN_FREE_RAM_MB:
        raise MemoryError(f"free RAM {free} MB is below the {MIN_FREE_RAM_MB} MB floor; STOP (§7.4)")


def _true_links_for_sample(s1_gids: list[int]) -> pl.DataFrame:
    links = access.load_labels("FIT").select("s1_gid", "s23_gid")
    sample = pl.DataFrame({"s1_gid": pl.Series(s1_gids, dtype=pl.UInt32)})
    return links.join(sample, on="s1_gid", how="semi")


def run_country(country: str) -> dict:
    t_country0 = time.time()
    s1 = C.sample_fit_s1(country)  # already named s1_gid, via the split join
    s23 = C.load_s23_pool(country).rename({"gid": "s23_gid"})
    truth = _true_links_for_sample(s1["s1_gid"].to_list())

    report = {"country": country, "n_s1_sampled": s1.height, "n_s23_pool": s23.height,
              "n_true_links": truth.height, "caps": {}}

    for cap in C.CAPS:
        check_ram_budget()
        t0 = time.time()
        ram_before = peak_ram_mb()
        try:
            scored = C.candidates_for_cap(s1, s23, cap, RAM_BUDGET_BYTES)
        except MemoryError as e:
            report["caps"][cap] = {"error": str(e)}
            continue
        cap_report = {"scored_pairs": scored.height, "seconds_scoring": round(time.time() - t0, 1),
                      "peak_ram_mb": peak_ram_mb(), "ks": {}}
        for k in C.KS:
            tk0 = time.time()
            topk = C.candidates_topk(scored, k)
            per_s1 = topk.group_by("s1_gid").len()
            mean_cand = float(per_s1["len"].mean()) if per_s1.height else 0.0
            p99_cand = float(per_s1["len"].quantile(0.99)) if per_s1.height else 0.0
            rec = C.recall_at_k(topk, truth)
            cap_report["ks"][k] = {
                "recall": rec["recall"], "n_true_links": rec["n_true_links"],
                "mean_candidates_per_s1": mean_cand, "p99_candidates_per_s1": p99_cand,
                "seconds_topk": round(time.time() - tk0, 1),
            }
            del topk
        report["caps"][cap] = cap_report
        del scored

    report["seconds_total"] = round(time.time() - t_country0, 1)
    report["peak_ram_mb"] = peak_ram_mb()
    return report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for country in COUNTRIES:
        check_ram_budget()
        print(f"=== {country} (free RAM {free_ram_mb()} MB)", flush=True)
        results[country] = run_country(country)
        print(f"  {json.dumps({k: v for k, v in results[country].items() if k != 'caps'})}", flush=True)
        for cap, cr in results[country]["caps"].items():
            if "error" in cr:
                print(f"  cap={cap}: {cr['error']}", flush=True)
                continue
            print(f"  cap={cap}: {cr['scored_pairs']:,} scored pairs, {cr['seconds_scoring']}s", flush=True)
            for k, kr in cr["ks"].items():
                print(f"    k={k}: recall={kr['recall']}, mean/p99 candidates/S1="
                      f"{kr['mean_candidates_per_s1']:.1f}/{kr['p99_candidates_per_s1']:.1f}", flush=True)

    ram = peak_ram_mb()
    runtime = round(time.time() - t0, 1)
    # Extrapolation to the FULL train pool (S1 is sampled at 1%; the S2/S3 pool is already full-size,
    # so RAM should scale roughly with S1 row count, not x100 the whole process -- reported as an
    # estimate, not used to choose anything).
    extrapolation = {
        "note": "ESTIMATE only, not used to choose k or a cap. S2/S3 pool was already full-size in "
                "this dry run; only the S1 side is 1%, so RAM/time for the real W1 run is estimated "
                "as this run's peak x ~(100 / n_countries) rather than a flat x100.",
        "dryrun_peak_ram_mb": ram, "dryrun_runtime_s": runtime,
        "rough_full_run_ram_mb_estimate": round(ram * 100 / len(COUNTRIES), 1),
        "rough_full_run_runtime_s_estimate": round(runtime * 100 / len(COUNTRIES), 1),
    }

    manifest = {"built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "countries": results, "peak_ram_mb": ram, "runtime_s": runtime,
               "extrapolation_estimate": extrapolation,
               "caps_tried": list(C.CAPS), "ks_tried": list(C.KS), "seed": C.SEED,
               "sample_fraction": C.SAMPLE_FRACTION}
    (OUT_DIR / "w1_dryrun_report.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nDone. {runtime}s, peak RAM {ram} MB. Report: {OUT_DIR / 'w1_dryrun_report.json'}")
    print("No k or cap chosen; no candidate_pairs.tsv written (T7 scope). Decision needs the user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
