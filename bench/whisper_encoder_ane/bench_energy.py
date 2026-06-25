#!/usr/bin/env python3
"""Whisper-tiny encoder energy (ANE vs Metal GPU), idle-subtracted via powermetrics. Run: PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_energy.py"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import encoder as E  # noqa: E402

RAILS = ("CPU", "GPU", "ANE")


def sample(workload, seconds):
    """Loop `workload` ~`seconds` while powermetrics samples; returns (inf/s, {rail: mean mW})."""
    n = int(seconds * 1000 / 500) + 2
    proc = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
         "-i", "500", "-n", str(n)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        workload()
        iters += 1
    elapsed = time.perf_counter() - t0
    out, _ = proc.communicate(timeout=60)
    means = {}
    for r in RAILS:
        vals = [float(v) for v in re.findall(rf"{r} Power:\s*([\d.]+)\s*mW", out)]
        means[r] = sum(vals) / len(vals) if vals else 0.0
    return (iters / elapsed if elapsed else 0.0), means


def pkg(means):
    return sum(means[r] for r in RAILS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=8.0, help="seconds per workload sample")
    ap.add_argument("--real", action="store_true",
                    help="trained whisper-tiny weights (downloads; the representative number)")
    args = ap.parse_args()

    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        sys.exit("powermetrics needs sudo: run `sudo -v` first (passwordless sudo).")

    enc, sd = E.real_encoder() if args.real else E.make_encoder()
    print(f"weights: {'trained whisper-tiny' if args.real else 'random init'}")
    mel = E.mel_input()
    net = E.build_cf(sd)                              # channels-first (fast) encoder
    mel4 = mel[:, :, None, :].astype("float16")
    pos = sd["embed_positions.weight"].T.reshape(1, E.D, 1, E.CTX).astype("float16")
    ane_call = lambda: net(mel4, pos)

    enc_mps = enc.to("mps").half()
    x_mps = torch.from_numpy(mel).to("mps").half()
    def mps_call():
        with torch.no_grad():
            enc_mps(x_mps)
        torch.mps.synchronize()

    print("sampling idle baseline...")
    _, idle = sample(lambda: time.sleep(0.01), 5)
    print(f"  idle  CPU {idle['CPU']:.0f}  GPU {idle['GPU']:.0f}  ANE {idle['ANE']:.0f} mW  (pkg {pkg(idle):.0f})")

    rows = []
    for name, call in (("ANE", ane_call), ("MPS (Metal GPU)", mps_call)):
        print(f"sampling {name}...")
        thru, m = sample(call, args.window)
        active_w = max(0.0, pkg(m) - pkg(idle)) / 1000.0
        mj = active_w / thru * 1000.0 if thru else 0.0
        rows.append((name, thru, m, mj))

    print("\n=== whisper-tiny encoder, idle-subtracted whole-package energy ===")
    print(f"{'engine':16s} {'enc/s':>7} {'ms/enc':>7} {'CPU mW':>7} {'GPU mW':>7} {'ANE mW':>7} {'mJ/enc':>8}")
    for name, thru, m, mj in rows:
        ms = 1000 / thru if thru else 0
        print(f"{name:16s} {thru:7.1f} {ms:7.2f} {m['CPU']:7.0f} {m['GPU']:7.0f} {m['ANE']:7.0f} {mj:8.1f}")
    if len(rows) == 2 and rows[0][3] and rows[1][3]:
        print(f"\nANE uses {rows[1][3] / rows[0][3]:.1f}x less energy per encode than MPS.")


if __name__ == "__main__":
    main()
