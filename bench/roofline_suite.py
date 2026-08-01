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
  PYTHONPATH=. python3 bench/roofline_suite.py             # fingerprint + numeric cliffs (fast, no sudo)
  PYTHONPATH=. python3 bench/roofline_suite.py --perf      # + fast headline perf (a few min; prompts once
                                                           #   for sudo to read watts -- see --no-sudo)
  PYTHONPATH=. python3 bench/roofline_suite.py --perf --no-sudo  # skip the watts prompt (perf/W blank)
  PYTHONPATH=. python3 bench/roofline_suite.py --perf-full # + full paper-grade battery (~30 min)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
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

# Perf scripts run in dependency order (roofline_analysis reads the saturation +
# bandwidth JSONs, so it runs last). Each entry: (script, result_file, fast_args).
#
# FAST is the contributor default (`--perf`): only the scripts the headline table
# needs, with --quick sweeps and short power windows -> a few minutes. FULL is the
# complete paper-grade battery (`--perf-full`): every script at full sampling.
PERF_FAST = [
    ("device_saturation_sweep.py", "device_saturation_sweep_results.json", ["--quick"]),
    ("device_bandwidth_roofline.py", "device_bandwidth_roofline_results.json", ["--quick", "--window", "2"]),
    ("decode_measurement.py", "decode_measurement_results.json", ["--quick"]),
    ("roofline_analysis.py", "roofline_analysis_results.json", []),
]
PERF_FULL = [
    ("device_saturation_sweep.py", "device_saturation_sweep_results.json", []),
    ("device_bandwidth_roofline.py", "device_bandwidth_roofline_results.json", []),
    ("device_serving_sweep.py", "device_serving_sweep_results.json", []),
    ("decode_measurement.py", "decode_measurement_results.json", []),
    ("device_compare_wattcomplete.py", "device_compare_wattcomplete_results.json", []),
    ("roofline_analysis.py", "roofline_analysis_results.json", []),
]


def _run_perf_script(script: str, result_file: str, extra_args: list[str]) -> dict:
    """Subprocess one existing bench script and collect its canonical JSON by reference.

    Does NOT restore the result file here: some scripts read earlier scripts'
    outputs (roofline_analysis synthesizes from the saturation + bandwidth JSONs),
    so restoring per-script would feed the synthesis stale committed data instead
    of this run's fresh numbers. The whole batch is snapshotted and restored around
    the loop in main() instead, keeping the dependency chain intact.
    """
    path = REPO / "bench" / script
    out_path = RESULTS / result_file
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path), *extra_args],
        cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    entry = {"script": script, "result_file": result_file,
             "returncode": proc.returncode, "seconds": round(dt, 1)}
    if proc.returncode != 0:
        entry["error"] = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
    else:
        try:
            entry["summary"] = json.loads(out_path.read_text())
        except Exception as e:
            entry["error"] = f"could not read {result_file}: {e}"
    return entry


def _snapshot(result_files: list[str]) -> dict[str, bytes | None]:
    """Prior bytes of each reference JSON (None == did not exist)."""
    return {rf: (RESULTS / rf).read_bytes() if (RESULTS / rf).exists() else None
            for rf in result_files}


def _restore(snap: dict[str, bytes | None]) -> None:
    """Put each reference JSON back exactly as it was before the batch ran."""
    for rf, prior in snap.items():
        p = RESULTS / rf
        if prior is not None:
            p.write_bytes(prior)
        elif p.exists():
            p.unlink()


def _authorize_sudo() -> bool:
    """Interactively cache sudo credentials (prompts once) so the perf scripts'
    `sudo -n powermetrics` calls succeed without passwordless sudo. Needs a TTY."""
    try:
        return subprocess.run(["sudo", "-v"]).returncode == 0   # inherits stdio -> prompts
    except Exception:
        return False


def _sudo_keepalive_start() -> threading.Event:
    """Refresh the sudo timestamp every 60s so it does not expire mid-run (the long
    saturation sweep can outlast the default 5-min sudo timeout between power reads)."""
    stop = threading.Event()

    def loop():
        while not stop.wait(60):
            subprocess.run(["sudo", "-n", "true"], capture_output=True)

    threading.Thread(target=loop, daemon=True).start()
    return stop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", action="store_true",
                    help="fast headline perf run (a few min; --quick sweeps, short power windows)")
    ap.add_argument("--perf-full", dest="perf_full", action="store_true",
                    help="full paper-grade perf battery (all scripts, full sampling; ~30 min)")
    ap.add_argument("--no-sudo", dest="no_sudo", action="store_true",
                    help="never prompt for sudo; skip powermetrics watts (perf/W left blank)")
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

    if args.perf or args.perf_full:
        scripts = PERF_FULL if args.perf_full else PERF_FAST
        label = "full paper-grade battery (~30 min)" if args.perf_full else "fast headline run (a few min)"
        print(f"\n[perf] {label}")
        # Power/watt reads use `sudo powermetrics`. Passwordless sudo is used
        # automatically; otherwise we prompt once (default) unless --no-sudo. The
        # prompt is skipped when there is no terminal, so non-interactive/CI runs
        # never hang -- they just proceed without watts.
        have_sudo = fp["environment"]["have_sudo"]
        keepalive = None
        if not have_sudo and not args.no_sudo and sys.stdin.isatty():
            print("[perf] Per-rail power needs sudo. The perf scripts run this exact command:")
            print("[perf]     sudo powermetrics --samplers ane_power,cpu_power,gpu_power")
            print("[perf] (read-only power sampling; nothing is modified). Authorizing now -- you")
            print("[perf] may be prompted for your login password. Re-run with --no-sudo to skip it.")
            if _authorize_sudo():
                have_sudo = True
                keepalive = _sudo_keepalive_start()
                report["machine"]["environment"]["sudo_authorized"] = True
                print("[perf] sudo authorized -> watts will be measured.")
            else:
                print("[perf] sudo not granted -> continuing WITHOUT watts (perf/W left blank).")
        if not have_sudo:
            why = ("--no-sudo set" if args.no_sudo
                   else "not a terminal" if not sys.stdin.isatty() else "not granted")
            print(f"[perf] note: no sudo ({why}) -> perf/W and per-rail watts skipped; "
                  "every other number is unaffected and the run still submits.")
        pw = fp["environment"]["power"]
        if pw.get("is_laptop") and pw.get("source") == "battery":
            mode = (pw.get("energy_mode") or {}).get("mode")
            if mode == "high_power":
                print(f"[perf] note: on battery ({pw.get('battery_pct')}%) but in High Power mode "
                      "-> close to AC over short bench windows; the state is recorded in the report.")
            else:
                print(f"[perf] WARNING: on battery ({pw.get('battery_pct')}%), energy mode "
                      f"'{mode}' -> clocks may throttle; High Power mode or AC gives cleaner perf "
                      "rooflines. (Numeric cliffs are unaffected.) The state is recorded either way.")
        # Snapshot every reference JSON once, run the whole batch (so roofline_analysis
        # reads THIS run's fresh saturation/bandwidth), then restore them all at the end.
        snap = _snapshot([rf for _, rf, _ in scripts])
        collected = []
        try:
            for script, result_file, fast_args in scripts:
                print(f"\n[perf] {script} ...", flush=True)
                entry = _run_perf_script(script, result_file, fast_args)
                state = "ok" if entry["returncode"] == 0 else f"FAILED rc={entry['returncode']}"
                print(f"  {state} in {entry.get('seconds')}s")
                collected.append(entry)
        finally:
            _restore(snap)
            if keepalive is not None:
                keepalive.set()
        report["perf_rooflines"] = collected
    else:
        print("\n[perf] skipped (pass --perf for the fast headline run, --perf-full for everything)")

    out = Path(args.out) if args.out else ROOFLINES_DIR / _machine.result_filename(fp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote merged report -> {out}")
    print("to publish: commit this file and run `python3 bench/aggregate_rooflines.py`, then open a PR")


if __name__ == "__main__":
    main()
