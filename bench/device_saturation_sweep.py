#!/usr/bin/env python3
"""Saturation (roofline) sweep — PEAK throughput + PEAK perf/watt per compute engine.

The watt-complete device comparison (``device_compare_wattcomplete.py``) raced each
device on *representative* workload shapes. But representative != saturating: a 512x512
GEMM runs the GPU at ~4% of its fp16 peak, so "the GPU is X GFLOP/s" measured there
UNDER-reports the silicon. The fair "how fast / how efficient is each device at its
BEST" answer requires sizing the workload until the device's cores/units are FULLY
UTILIZED. That is what this script does.

We scale two primitives that actually saturate compute:

  1. GEMM   — square NxN @ NxN, FLOPs = 2 N^3. Sweep N until each device plateaus.
  2. conv   — 3x3 same-pad conv, channels C with batch sized to fill, FLOPs =
              2 * B * Cout * Cin * 9 * H * W. The ANE's home turf — find its conv peak.

For each (primitive, size, device in {CPU, GPU, ANE}):
  * THROUGHPUT  = GFLOP/s from the MIN latency over reps (device forced to sync:
                  mx.eval / the compiled aneforge net / numpy inline).
  * RELERR      vs an fp64 (GEMM) / fp32 (conv) reference, reported next to every
                throughput number — the ANE's fp16 error grows at large N and that is
                part of the story (a 30 TFLOP/s GPU number at 4e-4 != an ANE number
                at 3e-2).
  * POWER       at the SATURATING sizes (the plateau region, where the loop is
                naturally multi-second) — idle-subtracted ACTIVE package power
                (median + CV) via the REUSED harness in device_compare_wattcomplete
                (measure_energy / sample_idle). perf/watt = GFLOP/s / active_W.

DTYPE ASYMMETRY IS REAL AND LABELED. CPU is fp32 (numpy/Accelerate-AMX cannot do a fast
fp16 GEMM — it upcasts), GPU and ANE are fp16. So the CPU peak is an fp32 number; it is
NOT the same product as the fp16 peaks and is labeled as such everywhere.

Run from repo root (energy needs passwordless sudo for powermetrics):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 bench/device_saturation_sweep.py

Writes bench/results/device_saturation_sweep_results.json alongside the printed
tables. --quick trims the largest (multi-second) sizes and shortens the power window
for a smoke test. --no-power skips the energy phase (throughput curves only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# Reuse the watt-complete harness for power (idle-subtracted ACTIVE package, median+CV,
# sampler-exit-driven loop) and device_compare for the latency/relerr helpers.
import device_compare as dc            # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
relerr = dc.relerr
min_latency = dc.min_latency

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx

# sizes (overridable by --quick which trims the multi-second tail)
GEMM_NS = [512, 1024, 2048, 4096, 6144, 8192]
GEMM_NS_QUICK = [512, 1024, 2048, 4096]
# conv: fixed-ish spatial 64x64, 3x3, batch chosen so total work tracks GEMM scale.
# we scale channels C (=Cin=Cout) and pick batch B to keep the tensor fillable.
CONV_SPATIAL = 64
CONV_CONFIGS = [  # (C, B)
    (64, 16), (128, 16), (256, 8), (512, 4),
]
CONV_CONFIGS_QUICK = [(64, 8), (128, 8), (256, 4), (512, 2)]

REPS = 8           # reps for the min-latency probe (large GEMMs are seconds each)
WARMUP = 3
POWER_WINDOW = 8.0  # seconds for the sustained-power loop (>=16 samples @ 500ms -> clears
                    # the watt-complete harness's <12-sample "short window" flag)

RESULTS: dict = {"gemm": [], "conv": []}


def _gflops(flops, lat_s):
    return flops / lat_s / 1e9


def _min_lat(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _power_at(run_once, tag):
    """Idle-subtracted ACTIVE package power (median W + CV%) over a sustained loop,
    via the reused watt-complete harness. Returns the harness dict or None."""
    if not HAVE_SUDO:
        return None
    return wc.measure_energy(run_once, tag=tag, window=POWER_WINDOW)


# GEMM sweep
def sweep_gemm(ns, do_power):
    print("\n" + "=" * 92)
    print(" GEMM SATURATION SWEEP — square NxN @ NxN, FLOPs = 2 N^3")
    print("=" * 92)
    for N in ns:
        flops = 2.0 * N * N * N
        rng = np.random.default_rng(0)
        x32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))
        W32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))  # [out,in]
        # fp64 reference (subsample rows for huge N to bound memory/time, relerr is a
        # norm so a representative row-block is a faithful estimate)
        rblk = min(N, 256)
        ref_blk = x32[:rblk].astype(np.float64) @ W32.astype(np.float64).T
        row = {"N": N, "gflop": flops / 1e9, "devices": {}}
        print(f"\n N={N}  ({flops/1e9:.1f} GFLOP)")

        # --- CPU (fp32, numpy/Accelerate-AMX) ---
        # IMPORTANT: feed CONTIGUOUS operands. A transposed view (W32.T) makes
        # Accelerate fall off its fast fp32 GEMM path (~4x slower, ~fp64-rate) — the
        # honest AMX peak needs both operands C-contiguous.
        Wt = np.ascontiguousarray(W32.T)
        lat = _min_lat(lambda: x32 @ Wt)
        out = x32 @ Wt
        err = relerr(out[:rblk], ref_blk)
        row["devices"]["CPU"] = {"dtype": "fp32", "lat_ms": lat * 1e3,
                                 "gflops": _gflops(flops, lat), "relerr": err}
        print(f"   CPU  fp32  {lat*1e3:9.3f} ms  {_gflops(flops,lat):9.1f} GFLOP/s  relerr {err:.2e}")

        # --- GPU (MLX fp16) ---
        if HAVE_MLX:
            xg = mx.array(x32.astype(np.float16))
            Wg = mx.array(W32.T.astype(np.float16))
            def grun(xg=xg, Wg=Wg):
                o = xg @ Wg
                mx.eval(o)
                return o
            lat = _min_lat(grun)
            out = np.array(grun(), copy=False)
            err = relerr(out[:rblk], ref_blk)
            row["devices"]["GPU"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                     "gflops": _gflops(flops, lat), "relerr": err}
            print(f"   GPU  fp16  {lat*1e3:9.3f} ms  {_gflops(flops,lat):9.1f} GFLOP/s  relerr {err:.2e}")

        # --- ANE (aneforge fp16) ---
        if HAVE_ANE:
            try:
                net = af.compile(af.input((N, N)).linear(W32.astype(np.float16)))
                xf = x32.astype(np.float16)
                net(xf)
                lat = _min_lat(lambda: net(xf))
                out = net(xf)
                err = relerr(out[:rblk], ref_blk)
                row["devices"]["ANE"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                         "gflops": _gflops(flops, lat), "relerr": err}
                print(f"   ANE  fp16  {lat*1e3:9.3f} ms  {_gflops(flops,lat):9.1f} GFLOP/s  relerr {err:.2e}")
            except Exception as e:
                row["devices"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"   ANE  fp16  CAP — {type(e).__name__}: {e}")
        RESULTS["gemm"].append(row)
    if do_power:
        _power_phase_gemm(ns)


def _power_phase_gemm(ns):
    """Measure active package power for EVERY swept GEMM size (so the power-vs-size
    curve shows the climb toward the plateau), per device."""
    print("\n" + "-" * 92)
    print(" GEMM power-vs-size (idle-subtracted active package W) — utilization evidence")
    print("-" * 92)
    by_n = {r["N"]: r for r in RESULTS["gemm"]}
    for N in ns:
        row = by_n[N]
        flops = 2.0 * N * N * N
        rng = np.random.default_rng(0)
        x32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))
        W32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))
        print(f"\n N={N}")
        # CPU
        Wt = W32.T.copy()
        _attach_power(row["devices"]["CPU"], lambda: x32 @ Wt, f"sat_gemm{N}_cpu", flops, "CPU")
        if HAVE_MLX:
            xg = mx.array(x32.astype(np.float16)); Wg = mx.array(W32.T.astype(np.float16))
            _attach_power(row["devices"]["GPU"], lambda: mx.eval(xg @ Wg), f"sat_gemm{N}_gpu", flops, "GPU")
        if HAVE_ANE and "error" not in row["devices"].get("ANE", {"error": 1}):
            net = af.compile(af.input((N, N)).linear(W32.astype(np.float16)))
            xf = x32.astype(np.float16); net(xf)
            _attach_power(row["devices"]["ANE"], lambda: net(xf), f"sat_gemm{N}_ane", flops, "ANE")


# conv sweep
def sweep_conv(configs, do_power):
    print("\n" + "=" * 92)
    print(f" CONV SATURATION SWEEP — 3x3 same-pad, {CONV_SPATIAL}x{CONV_SPATIAL}, "
          f"FLOPs = 2*B*Cout*Cin*9*H*W")
    print("=" * 92)
    H = W = CONV_SPATIAL
    k = 3
    for C, B in configs:
        flops = 2.0 * B * C * C * k * k * H * W
        rng = np.random.default_rng(1)
        x32 = rng.standard_normal((B, C, H, W)).astype(np.float32)
        w32 = (rng.standard_normal((C, C, k, k)).astype(np.float32)
               * np.sqrt(2.0 / (C * k * k)))
        # fp32 numpy reference on a single batch element (relerr is a norm)
        ref0 = dc._np_conv2d(x32[:1].astype(np.float32), w32, 1).astype(np.float64)
        row = {"C": C, "B": B, "HxW": f"{H}x{W}", "gflop": flops / 1e9, "devices": {}}
        print(f"\n C={C} B={B}  ({flops/1e9:.1f} GFLOP)")

        # --- CPU (fp32) ---
        lat = _min_lat(lambda: dc._np_conv2d(x32[:1].astype(np.float32), w32, 1), reps=max(3, REPS // 2))
        # CPU only times one batch element to keep it bounded; scale to full-batch flops
        cpu_flops = flops / B
        out = dc._np_conv2d(x32[:1].astype(np.float32), w32, 1)
        err = relerr(out, ref0)
        row["devices"]["CPU"] = {"dtype": "fp32", "lat_ms": lat * 1e3,
                                 "gflops": _gflops(cpu_flops, lat), "relerr": err,
                                 "note": "B=1 timed, per-element GFLOP/s"}
        print(f"   CPU  fp32  {lat*1e3:9.3f} ms/img  {_gflops(cpu_flops,lat):9.1f} GFLOP/s  relerr {err:.2e}  (B=1)")

        # --- GPU (MLX fp16, NHWC) ---
        if HAVE_MLX:
            xg = mx.array(np.transpose(x32, (0, 2, 3, 1)).astype(np.float16))
            wg = mx.array(np.transpose(w32, (0, 2, 3, 1)).astype(np.float16))
            def grun(xg=xg, wg=wg):
                o = mx.conv2d(xg, wg, stride=1, padding=1)
                mx.eval(o)
                return o
            lat = _min_lat(grun)
            out = np.transpose(np.array(grun(), copy=False), (0, 3, 1, 2))
            err = relerr(out[:1], ref0)
            row["devices"]["GPU"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                     "gflops": _gflops(flops, lat), "relerr": err}
            print(f"   GPU  fp16  {lat*1e3:9.3f} ms  {_gflops(flops,lat):9.1f} GFLOP/s  relerr {err:.2e}")

        # --- ANE (aneforge fp16, NCHW) ---
        if HAVE_ANE:
            try:
                net = af.compile(af.conv(af.input((B, C, H, W)), w32.astype(np.float16),
                                         stride=1, pad=1))
                xf = x32.astype(np.float16)
                net(xf)
                lat = _min_lat(lambda: net(xf))
                out = net(xf)
                err = relerr(np.asarray(out)[:1], ref0)
                row["devices"]["ANE"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                         "gflops": _gflops(flops, lat), "relerr": err}
                print(f"   ANE  fp16  {lat*1e3:9.3f} ms  {_gflops(flops,lat):9.1f} GFLOP/s  relerr {err:.2e}")
            except Exception as e:
                row["devices"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"   ANE  fp16  CAP — {type(e).__name__}: {e}")
        RESULTS["conv"].append(row)
    if do_power:
        _power_phase_conv(configs)


def _power_phase_conv(configs):
    print("\n" + "-" * 92)
    print(" CONV power-vs-size (idle-subtracted active package W) — utilization evidence")
    print("-" * 92)
    H = W = CONV_SPATIAL
    k = 3
    by_c = {(r["C"], r["B"]): r for r in RESULTS["conv"]}
    for C, B in configs:
        row = by_c[(C, B)]
        flops = 2.0 * B * C * C * k * k * H * W
        rng = np.random.default_rng(1)
        x32 = rng.standard_normal((B, C, H, W)).astype(np.float32)
        w32 = (rng.standard_normal((C, C, k, k)).astype(np.float32)
               * np.sqrt(2.0 / (C * k * k)))
        print(f"\n C={C} B={B}")
        # CPU power: drive full batch to make it a sustained load
        _attach_power(row["devices"]["CPU"],
                      lambda: dc._np_conv2d(x32.astype(np.float32), w32, 1),
                      f"sat_conv{C}_cpu", flops, "CPU")
        if HAVE_MLX:
            xg = mx.array(np.transpose(x32, (0, 2, 3, 1)).astype(np.float16))
            wg = mx.array(np.transpose(w32, (0, 2, 3, 1)).astype(np.float16))
            _attach_power(row["devices"]["GPU"],
                          lambda: mx.eval(mx.conv2d(xg, wg, stride=1, padding=1)),
                          f"sat_conv{C}_gpu", flops, "GPU")
        if HAVE_ANE and "error" not in row["devices"].get("ANE", {"error": 1}):
            net = af.compile(af.conv(af.input((B, C, H, W)), w32.astype(np.float16),
                                     stride=1, pad=1))
            xf = x32.astype(np.float16); net(xf)
            _attach_power(row["devices"]["ANE"], lambda: net(xf),
                          f"sat_conv{C}_ane", flops, "ANE")


# shared power attach
def _attach_power(dev_row, run_once, tag, flops, device):
    e = _power_at(run_once, tag)
    if e is None:
        return
    apw = e.get("active_pkg_W", float("nan"))
    sane = apw == apw and apw > 0
    rec = {"active_pkg_W": apw,
           "active_pkg_cv_pct": e.get("active_pkg_cv_pct", float("nan")),
           "iter_ms": e.get("iter_ms"),
           "n_pm_samples": e.get("n_pm_samples"),
           "ane_active_mW": e.get("ane_active_mW", float("nan")),
           "gpu_active_mW": e.get("gpu_active_mW", float("nan")),
           "cpu_active_mW": e.get("cpu_active_mW", float("nan")),
           "flags": list(e.get("flags", []))}
    if sane and e.get("iter_ms"):
        thr = flops / (e["iter_ms"] / 1e3)  # FLOP/s at the sustained-loop rate
        rec["gflops_sustained"] = thr / 1e9
        rec["perf_per_W"] = (thr / 1e9) / apw  # GFLOP/s/W
    # plausibility flag: ANE work that reads ~0 on the ANE rail is a sampling miss
    if device == "ANE" and e.get("ane_active_mW", 0.0) < 5.0 and not rec["flags"]:
        rec["flags"].append("ANE rail ~0 mW during ANE work — likely a 100ms sampling miss")
    dev_row["power"] = rec
    pw = rec.get("perf_per_W")
    print(f"   [pwr {device:<3}] active pkg {apw:6.2f} W (CV {rec['active_pkg_cv_pct']:.0f}%, "
          f"{rec['n_pm_samples']} smp)  {(f'{pw:.2f} GFLOP/s/W' if pw else '-')}"
          f"  | ANE {rec['ane_active_mW']:.0f}/GPU {rec['gpu_active_mW']:.0f}/CPU {rec['cpu_active_mW']:.0f} mW")
    for f in rec["flags"]:
        print(f"      FLAG: {f}")


# peak extraction + reporting
def _peaks(prim):
    """Per-device peak GFLOP/s (+ size) and peak GFLOP/s/W (+ size) for a primitive."""
    rows = RESULTS[prim]
    keyname = "N" if prim == "gemm" else "C"
    out = {}
    for dev in ("CPU", "GPU", "ANE"):
        best_thr = (-1.0, None, None)   # (gflops, size, relerr)
        best_ppw = (-1.0, None)         # (perf_per_W, size)
        for r in rows:
            d = r["devices"].get(dev, {})
            if "gflops" in d and d["gflops"] > best_thr[0]:
                best_thr = (d["gflops"], r[keyname], d.get("relerr"))
            p = d.get("power", {})
            if p.get("perf_per_W") and p["perf_per_W"] > best_ppw[0]:
                best_ppw = (p["perf_per_W"], r[keyname])
        if best_thr[1] is not None:
            out[dev] = {"peak_gflops": best_thr[0], "peak_gflops_size": best_thr[1],
                        "peak_gflops_relerr": best_thr[2],
                        "peak_perf_per_W": best_ppw[0] if best_ppw[1] is not None else None,
                        "peak_perf_per_W_size": best_ppw[1]}
    return out


def print_report():
    print("\n" + "=" * 92)
    print(" SATURATION / ROOFLINE — PEAK TABLE")
    print(" dtype labeled: CPU=fp32 (Accelerate/AMX, upcasts — NOT same product as fp16), "
          "GPU/ANE=fp16")
    print("=" * 92)
    for prim in ("gemm", "conv"):
        pk = _peaks(prim)
        szlbl = "N" if prim == "gemm" else "C"
        print(f"\n {prim.upper()} peaks:")
        print(f"   {'device':<6} {'dtype':<6} {'peak GFLOP/s':>13} {f'@{szlbl}':>7} "
              f"{'relerr':>10} {'peak GFLOP/s/W':>15} {f'@{szlbl}':>7}")
        for dev in ("CPU", "GPU", "ANE"):
            if dev not in pk:
                continue
            p = pk[dev]
            dtype = "fp32" if dev == "CPU" else "fp16"
            ppw = f"{p['peak_perf_per_W']:.2f}" if p['peak_perf_per_W'] else "  -"
            ppws = p['peak_perf_per_W_size'] if p['peak_perf_per_W_size'] is not None else "-"
            print(f"   {dev:<6} {dtype:<6} {p['peak_gflops']:13.1f} {str(p['peak_gflops_size']):>7} "
                  f"{p['peak_gflops_relerr']:10.2e} {ppw:>15} {str(ppws):>7}")
    RESULTS["peaks"] = {"gemm": _peaks("gemm"), "conv": _peaks("conv")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="trim multi-second sizes + short power window")
    ap.add_argument("--no-power", action="store_true", help="throughput curves only, skip energy")
    args = ap.parse_args()
    global POWER_WINDOW
    gemm_ns = GEMM_NS_QUICK if args.quick else GEMM_NS
    conv_cfg = CONV_CONFIGS_QUICK if args.quick else CONV_CONFIGS
    if args.quick:
        POWER_WINDOW = 2.5
    do_power = HAVE_SUDO and not args.no_power

    print("=" * 92)
    print(" device_saturation_sweep — PEAK throughput + PEAK perf/watt (roofline)")
    print("=" * 92)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}  "
          f"powermetrics(sudo)={'yes' if HAVE_SUDO else 'NO — power skipped'}")
    print(f" GEMM N = {gemm_ns}")
    print(f" conv (C,B) = {conv_cfg} @ {CONV_SPATIAL}x{CONV_SPATIAL} 3x3")
    print(f" power window = {POWER_WINDOW}s  reps = {REPS}")

    if do_power:
        print("\n sampling idle baseline (no workload)...", flush=True)
        wc.sample_idle(3.0)
        print(f" idle: ANE {wc.IDLE.get('ane',0):.0f} / GPU {wc.IDLE.get('gpu',0):.0f} / "
              f"CPU {wc.IDLE.get('cpu',0):.0f} mW (pkg {wc.IDLE_PKG:.0f} mW)")

    sweep_gemm(gemm_ns, do_power)
    sweep_conv(conv_cfg, do_power)
    print_report()

    RESULTS["meta"] = {
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
        "gemm_ns": gemm_ns, "conv_configs": conv_cfg, "conv_spatial": CONV_SPATIAL,
        "power_window_s": POWER_WINDOW, "reps": REPS,
        "idle_mW": dict(wc.IDLE) if do_power else {}, "idle_pkg_mW": wc.IDLE_PKG if do_power else 0.0,
    }
    out = Path(__file__).resolve().parent / "results" / "device_saturation_sweep_results.json"
    out.write_text(json.dumps(RESULTS, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
