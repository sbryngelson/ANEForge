#!/usr/bin/env python3
"""GEMV weight-stream bandwidth sweep (corroborates the ~112 GB/s).

Peer-review blocker: the paper's "~112 GB/s ANE weight-stream bandwidth" (the
*second* bandwidth, distinct from the choked ~24 GB/s standalone-activation path)
is inferred from a SINGLE GEMV point (M=1, K=N=4096, 466% of the drawn ~24 GB/s
roof) with no climb-to-plateau. The same evidence standard the paper applies to
the 24 GB/s elementwise number (flat-with-size plateau) must be applied here.

This script sweeps the decode GEMV (M=1) on the ANE across K=N in
{512,1024,2048,4096,6144,8192} (plus a few rectangular K!=N), and for each computes
the ACHIEVED EFFECTIVE WEIGHT-STREAM BANDWIDTH:

    eff_BW = (weight_bytes + input_bytes + output_bytes) / min_latency

with weight_bytes = K*N*2 (fp16), input = K*2, output = N*2. The weight matrix
dominates at M=1, so this is the weight-DMA bandwidth the decode GEMV actually
drives. We report GB/s per K and ask: does it plateau ~112 GB/s (supporting the
two-bandwidth claim) or vary (undermining it)?

The same GEMV sweep is run on GPU (MLX fp16) and CPU (numpy/Accelerate fp32) for
context. Power (idle-subtracted total-package, median + CV) is measured at the
4096 and 8192 points via the rigorous harness imported from
device_compare_wattcomplete.

Run from repo root (energy needs passwordless sudo for powermetrics):

    PYTHONPATH=. python3 bench/gemv_bandwidth_sweep.py

Writes bench/results/gemv_bandwidth_sweep_results.json. --quick caps sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import device_compare as dc            # noqa: E402  min_latency, relerr, runners
import device_compare_wattcomplete as wc  # noqa: E402  measure_energy, sample_idle, IDLE

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
min_latency = dc.min_latency
relerr = dc.relerr

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx


def eff_bytes(K, N):
    """Total bytes streamed for one M=1 GEMV in fp16: weight + input + output."""
    return K * N * 2 + K * 2 + N * 2


def bw_gbps(K, N, lat_s):
    return eff_bytes(K, N) / lat_s / 1e9


def gflops(K, N, lat_s):
    return (2.0 * K * N) / lat_s / 1e9


def sweep_gemv(K, N, *, reps, warmup, measure_pw):
    """One GEMV point (M=1): x[1,K] @ W -> [1,N]. Returns a dict of per-device results."""
    tag = f"K{K}_N{N}"
    label = f"GEMV M=1, K={K}, N={N}"
    print(f"\n=== {label} === (weight {K*N*2/1e6:.1f} MB fp16)", flush=True)
    rng = np.random.default_rng(0)
    # well-scaled so fp16 is meaningful; aneforge linear wants W as [out,in]=[N,K]
    x32 = (rng.standard_normal((1, K)).astype(np.float32) / np.sqrt(K))
    W32 = (rng.standard_normal((N, K)).astype(np.float32) / np.sqrt(K))  # [out,in]
    ref = x32.astype(np.float64) @ W32.astype(np.float64).T              # [1,N]
    res = {"K": K, "N": N, "weight_MB": K * N * 2 / 1e6,
           "eff_bytes": eff_bytes(K, N), "devices": {}}

    # ANE (aneforge fused single-program, fp16) ----------------------------- #
    if HAVE_ANE:
        try:
            net = af.compile(af.input((1, K)).linear(W32.astype(np.float16)))
            xf = x32.astype(np.float16)
            out_h = {}

            def run():
                out_h["o"] = net(xf)
            lat = min_latency(run, reps=reps, warmup=warmup)
            out = np.asarray(out_h["o"])
            d = {"dtype": "fp16", "lat_ms": lat * 1e3,
                 "eff_GBps": bw_gbps(K, N, lat), "GFLOPs": gflops(K, N, lat),
                 "relerr": relerr(out, ref)}
            if measure_pw and HAVE_SUDO:
                e = wc.measure_energy(lambda: net(xf), tag=f"gemv_{tag}_ane")
                if e:
                    apw = e.get("active_pkg_W")
                    d["active_pkg_W"] = apw
                    d["pkg_cv_pct"] = e.get("active_pkg_cv_pct")
                    d["ane_active_mW"] = e.get("ane_active_mW")
                    d["energy_iter_ms"] = e.get("iter_ms")
                    d["flags"] = e.get("flags")
                    if apw and apw > 0:
                        d["eff_GBps_per_W"] = d["eff_GBps"] / apw
            res["devices"]["ANE"] = d
            print(f"  ANE  {lat*1e3:8.3f} ms  {d['eff_GBps']:7.1f} GB/s  "
                  f"{d['GFLOPs']:7.1f} GFLOP/s  relerr {d['relerr']:.2e}")
        except Exception as e:
            res["devices"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  ANE  FAILED: {type(e).__name__}: {e}")

    # GPU (MLX fp16) -------------------------------------------------------- #
    if HAVE_MLX:
        try:
            xg = mx.array(x32.astype(np.float16))
            Wg = mx.array(W32.T.astype(np.float16))     # [K,N]
            out_h = {}

            def run():
                o = xg @ Wg
                mx.eval(o)
                out_h["o"] = o
            lat = min_latency(run, reps=reps, warmup=warmup)
            out = np.array(out_h["o"], copy=False)
            d = {"dtype": "fp16", "lat_ms": lat * 1e3,
                 "eff_GBps": bw_gbps(K, N, lat), "GFLOPs": gflops(K, N, lat),
                 "relerr": relerr(out, ref)}
            if measure_pw and HAVE_SUDO:
                e = wc.measure_energy(lambda: mx.eval(xg @ Wg), tag=f"gemv_{tag}_gpu")
                if e:
                    apw = e.get("active_pkg_W")
                    d["active_pkg_W"] = apw
                    d["pkg_cv_pct"] = e.get("active_pkg_cv_pct")
                    d["energy_iter_ms"] = e.get("iter_ms")
                    d["flags"] = e.get("flags")
                    if apw and apw > 0:
                        d["eff_GBps_per_W"] = d["eff_GBps"] / apw
            res["devices"]["GPU"] = d
            print(f"  GPU  {lat*1e3:8.3f} ms  {d['eff_GBps']:7.1f} GB/s  "
                  f"{d['GFLOPs']:7.1f} GFLOP/s  relerr {d['relerr']:.2e}")
        except Exception as e:
            res["devices"]["GPU"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  GPU  FAILED: {type(e).__name__}: {e}")

    # CPU (numpy / Accelerate fp32) ----------------------------------------- #
    try:
        Wt = W32.T.copy()    # [K,N]
        out_h = {}

        def run():
            out_h["o"] = x32 @ Wt
        lat = min_latency(run, reps=max(10, reps // 2), warmup=warmup)
        out = out_h["o"]
        # CPU fp32 moves 4 B/elem for the weight; report its own effective BW honestly
        cpu_bytes = K * N * 4 + K * 4 + N * 4
        d = {"dtype": "fp32", "lat_ms": lat * 1e3,
             "eff_GBps": cpu_bytes / lat / 1e9, "GFLOPs": gflops(K, N, lat),
             "relerr": relerr(out, ref)}
        res["devices"]["CPU"] = d
        print(f"  CPU  {lat*1e3:8.3f} ms  {d['eff_GBps']:7.1f} GB/s (fp32 4B/elem)  "
              f"{d['GFLOPs']:7.1f} GFLOP/s")
    except Exception as e:
        res["devices"]["CPU"] = {"error": f"{type(e).__name__}: {e}"}

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=12)
    args = ap.parse_args()

    print("=" * 90)
    print(" gemv_bandwidth_sweep - ANE weight-stream bandwidth climb-to-plateau")
    print("=" * 90)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}  "
          f"sudo={'yes' if HAVE_SUDO else 'NO (no power)'}")

    if HAVE_SUDO:
        print("\n sampling idle baseline...", flush=True)
        wc.sample_idle(3.0)
        print(f" idle pkg {wc.IDLE_PKG:.0f} mW  (ANE {wc.IDLE.get('ane',0):.0f} / "
              f"GPU {wc.IDLE.get('gpu',0):.0f} / CPU {wc.IDLE.get('cpu',0):.0f})")

    square = [512, 1024, 2048, 4096] if args.quick else [512, 1024, 2048, 4096, 6144, 8192]
    # power measured at EVERY square point so the weight-stream tier carries a
    # GB/s/W column (the only sweep that previously lacked one - closes the
    # "watt-complete" asymmetry: the 24 GB/s activation path has GB/s/W in
    # Table 4, the weight path did not). Each point adds one power window.
    pw_at = set(square)

    sweep = []
    for KN in square:
        sweep.append(sweep_gemv(KN, KN, reps=args.reps, warmup=args.warmup,
                                measure_pw=(KN in pw_at)))

    # a couple of rectangular K!=N points (same M=1 decode shape)
    rect = [] if args.quick else [(8192, 2048), (2048, 8192), (4096, 11008)]
    rect_res = []
    for (K, N) in rect:
        rect_res.append(sweep_gemv(K, N, reps=args.reps, warmup=args.warmup, measure_pw=False))

    # -------- plateau analysis on the ANE square sweep -------------------- #
    print("\n" + "=" * 90)
    print(" ANE WEIGHT-STREAM BANDWIDTH (square M=1 GEMV)")
    print("=" * 90)
    print(f" {'K=N':>6} {'weightMB':>9} {'ANE GB/s':>10} {'GPU GB/s':>10} {'CPU GB/s':>10} "
          f"{'ANE GFLOP/s':>12}")
    ane_bw = []
    for r in sweep:
        a = r["devices"].get("ANE", {}); g = r["devices"].get("GPU", {}); c = r["devices"].get("CPU", {})
        abw = a.get("eff_GBps");
        if abw is not None:
            ane_bw.append((r["K"], abw))
        print(f" {r['K']:>6} {r['weight_MB']:>9.1f} "
              f"{(a.get('eff_GBps') or float('nan')):>10.1f} "
              f"{(g.get('eff_GBps') or float('nan')):>10.1f} "
              f"{(c.get('eff_GBps') or float('nan')):>10.1f} "
              f"{(a.get('GFLOPs') or float('nan')):>12.1f}")

    verdict = {}
    if ane_bw:
        bws = np.array([b for _, b in ane_bw])
        # plateau = the large-K tail (>=2048), where small-size dispatch overhead is amortized
        tail = np.array([b for k, b in ane_bw if k >= 2048])
        verdict = {
            "ane_bw_all_GBps": {str(k): round(b, 1) for k, b in ane_bw},
            "ane_bw_max_GBps": round(float(bws.max()), 1),
            "ane_bw_tail_median_GBps": round(float(np.median(tail)), 1) if len(tail) else None,
            "ane_bw_tail_cv_pct": round(float(tail.std() / tail.mean() * 100), 1) if len(tail) else None,
            "paper_claim_GBps": 112,
        }
        tm = verdict["ane_bw_tail_median_GBps"]
        print(f"\n ANE max eff BW = {verdict['ane_bw_max_GBps']} GB/s; "
              f"tail (K>=2048) median = {tm} GB/s, CV {verdict['ane_bw_tail_cv_pct']}%")
        if tm is not None:
            within = abs(tm - 112) / 112
            verdict["tail_vs_112_pct"] = round(within * 100, 1)
            if within <= 0.25:
                print(f" => tail bandwidth is within {within*100:.0f}% of the paper's 112 GB/s - "
                      f"CORROBORATES the second (weight-stream) bandwidth.")
            else:
                print(f" => tail bandwidth differs from 112 GB/s by {within*100:.0f}% - "
                      f"two-bandwidth claim may need softening.")

    out = Path(__file__).resolve().parent / "results" / "gemv_bandwidth_sweep_results.json"
    out.write_text(json.dumps({
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
        "idle_pkg_mW": wc.IDLE_PKG, "idle_mW": wc.IDLE,
        "square_sweep": sweep, "rect_sweep": rect_res,
        "verdict": verdict,
        "method": "eff_BW = (K*N + K + N)*2 bytes (fp16 weight+io) / min_latency; "
                  "CPU uses 4B/elem fp32. M=1 decode GEMV. min over reps after warmup.",
    }, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
