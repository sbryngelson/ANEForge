#!/usr/bin/env python3
"""Roofline suite: one command that fingerprints the machine, runs the full
roofline battery, and emits a single merged report keyed to that machine.

Design (agreed): this SITS ABOVE the existing bench scripts rather than forking
them. The performance rooflines already live in dedicated scripts, each writing
its own canonical results/*.json that the paper cites - so the suite subprocesses
those (leaving their JSONs untouched and citable) and collects them by reference.
The one plane that has no script yet - the numeric CORRECTNESS cliffs from issue
#115 and docs/cross-chip.md - runs in-process via bench/numeric_cliffs.py.

The merged report carries the machine fingerprint (bench/_machine.py), so many
contributors on different Apple Silicon machines can each commit a
non-colliding roofline-<chip>-<model>-<hwhash>-<runid>.json, and a later
aggregator can pool them by hardware_hash without mixing an M2 Air with an M2
Mac mini.

Run:
  PYTHONPATH=. python3 bench/roofline_suite.py            # fingerprint + numeric cliffs (fast, no sudo)
  PYTHONPATH=. python3 bench/roofline_suite.py --perf     # also run the perf scripts (slow; sudo for watts)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling bench modules
import _machine  # noqa: E402
import numeric_cliffs  # noqa: E402

RESULTS = REPO / "bench" / "results"
ROOFLINES_DIR = RESULTS / "rooflines"   # one immutable JSON per submission; PR'd in

# The existing perf-roofline scripts, in dependency order (roofline_analysis reads
# the saturation + bandwidth JSONs, so it must run last). Each entry is the script
# and the canonical result file it writes, which the suite collects by reference.
PERF_SCRIPTS = [
    ("device_saturation_sweep.py", "device_saturation_sweep_results.json"),
    ("device_bandwidth_roofline.py", "device_bandwidth_roofline_results.json"),
    ("device_serving_sweep.py", "device_serving_sweep_results.json"),
    ("decode_measurement.py", "decode_measurement_results.json"),
    ("device_compare_wattcomplete.py", "device_compare_wattcomplete_results.json"),
    ("roofline_analysis.py", "roofline_analysis_results.json"),
]


def _run_perf_script(script: str, result_file: str, extra_args: list[str]) -> dict:
    """Subprocess one existing bench script; collect its canonical JSON by reference.

    Non-destructive: the per-script results/*.json are the repo's committed
    reference numbers that the paper cites, so we snapshot the prior file, read
    the fresh output into the merged report, then RESTORE the prior bytes. The
    contributor's data lives in the fingerprinted merged report, not by dirtying
    a tracked reference file.
    """
    path = REPO / "bench" / script
    out_path = RESULTS / result_file
    prior = out_path.read_bytes() if out_path.exists() else None
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path), *extra_args],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    entry = {"script": script, "result_file": result_file,
             "returncode": proc.returncode, "seconds": round(dt, 1)}
    try:
        if proc.returncode != 0:
            entry["error"] = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        else:
            try:
                entry["summary"] = json.loads(out_path.read_text())
            except Exception as e:
                entry["error"] = f"could not read {result_file}: {e}"
    finally:
        # restore the reference file to exactly its prior state (committed or absent)
        if prior is not None:
            out_path.write_bytes(prior)
        elif out_path.exists():
            out_path.unlink()
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", action="store_true",
                    help="also run the existing perf-roofline scripts (slow; sudo needed for watts)")
    ap.add_argument("--quick", action="store_true",
                    help="pass --quick through to perf scripts that support it")
    ap.add_argument("--out", default=None, help="explicit output path (default: fingerprinted name)")
    ap.add_argument("--contributor", default=None,
                    help="your GitHub handle to be credited (overrides auto-detection)")
    args = ap.parse_args()

    fp = _machine.fingerprint()
    print("machine:", fp["hardware"]["chip"], fp["hardware"]["model_identifier"],
          "  hwhash", fp["hardware_hash"], "  sudo", fp["environment"]["have_sudo"])

    # Credit: explicit flag wins; otherwise auto-detect from a GitHub noreply email.
    contributor = args.contributor or _machine.github_handle()
    contributor = contributor.lstrip("@") if contributor else None
    if contributor:
        src = "flag" if args.contributor else "auto-detected from git noreply email"
        print(f"contributor: @{contributor} ({src})")
    else:
        print("contributor: none (git email is not a GitHub noreply; pass --contributor to be credited)")

    report = {
        "suite": "roofline",
        "schema_version": 2,
        "contributor": contributor,
        "machine": fp,
        "numeric_cliffs": None,
        "perf_rooflines": None,
    }

    print("\n[numeric cliffs] correctness rooflines (issue #115, docs/cross-chip.md) ...")
    report["numeric_cliffs"] = numeric_cliffs.run()
    mm = report["numeric_cliffs"]["matmul_saturation"]["by_K"][0].get("cliff")
    sl = report["numeric_cliffs"]["slice_saturation"].get("cliff")
    rex = report["numeric_cliffs"]["reduce_exactness"].get("last_all_exact")
    print(f"  matmul inf-cliff ~{mm}  |  slice cliff {sl} (None=exact/A16+)  |  reduce exact <= {rex}")

    if args.perf:
        if not fp["environment"]["have_sudo"]:
            print("\n[perf] WARNING: no passwordless sudo -> powermetrics watts will be missing.")
        pw = fp["environment"]["power"]
        if pw.get("is_laptop") and pw.get("source") == "battery":
            print(f"\n[perf] WARNING: on battery ({pw.get('battery_pct')}%) -> clocks throttle; "
                  "plug in for representative perf rooflines. (Numeric cliffs are unaffected.)")
        extra = ["--quick"] if args.quick else []
        collected = []
        for script, result_file in PERF_SCRIPTS:
            print(f"\n[perf] {script} ...", flush=True)
            entry = _run_perf_script(script, result_file, extra)
            state = "ok" if entry["returncode"] == 0 else f"FAILED rc={entry['returncode']}"
            print(f"  {state} in {entry.get('seconds')}s")
            collected.append(entry)
        report["perf_rooflines"] = collected
    else:
        print("\n[perf] skipped (pass --perf to run the perf-roofline scripts)")

    out = Path(args.out) if args.out else ROOFLINES_DIR / _machine.result_filename(fp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote merged report -> {out}")
    print("to publish: commit this file and run `python3 bench/aggregate_rooflines.py`, then open a PR")


if __name__ == "__main__":
    main()
