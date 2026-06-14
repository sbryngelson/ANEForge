#!/usr/bin/env python3
"""Watt-complete cross-device comparison - ANE (aneforge fp16) vs GPU (MLX) vs CPU.

Companion to ``device_compare.py``. That harness covers raw latency + precision
across every workload class but measures ENERGY for only two sustained loops. This
script makes energy/watt *complete across every meaningful workload class*, with the
power methodology a reviewer will attack first done right:

  * IDLE SUBTRACTION. We sample a no-workload idle baseline (per rail) once at start
    and report ACTIVE = (loaded - idle). The marginal cost of the workload is what
    matters; reporting raw loaded power double-counts the ~0.6 W the CPU burns at rest.

  * HEADLINE = TOTAL-PACKAGE ACTIVE POWER (sum of ane+gpu+cpu rails, idle-subtracted).
    This is the honest number: an ANE workload still burns CPU on dispatch and an MLX
    workload burns CPU too, so attributing only the "device" rail would flatter both.
    We ALSO report the per-rail breakdown so the attribution is visible
    (e.g. "ANE 1.2 W + CPU 0.8 W dispatch").

  * CONFIDENCE. We report the MEDIAN total-package power AND the spread (CV% over the
    per-sample package totals, plus min/median/p90). A workload whose power CV is large
    is FLAGGED low-confidence rather than reported as a clean number. Implausible reads
    (e.g. ANE 0 mW *during* an ANE workload - a 100 ms sampling miss) are flagged.

  * STEADY STATE. Each sustained loop runs >= the requested window (default 4 s) after
    a warmup, long enough for stable 100 ms-interval sampling. perf/watt is throughput
    / active_W; for the real models we also report energy-per-inference in mJ.

  * ACCURACY ALONGSIDE. Every workload still reports relerr vs an fp32/fp64 numpy
    reference, presented next to the speed/watt numbers - a speed win at worse accuracy
    must be visible (GPU is more accurate than the ANE at large K, the ANE matches or
    beats it where the math is well-conditioned).

SCOPE = workload classes where the ANE-vs-GPU choice is REAL:
  1. GEMM at K = 256 (floor) / 1024 (bandwidth) / 4096 (compute)
  2. conv: a single 3x3 + a ResNet-ish 3x3 stack
  3. attention: the ViT self-attention block (vit_demo S=197 shape) + a long-seq S=512
     shape; both via the in-graph af.mha (decomposed-SDPA fused route) - the FAIR
     native-attention path. (NOT the subprocess-bridge SDPA - see EXCLUSIONS.)
  4. norm family: layer_norm, rms_norm, group_norm at representative sizes
  5. scientific kernels: matmul-DFT, a 2D 5-point stencil step, a fixed-iter Jacobi solve
  6. real models: ResNet-18, MiniLM encoder, full ViT-B/16 forward

EXCLUSIONS (stated honestly): the netplist *bridge* ops (sdpa-bridge / argmax / fps /
cost_volume / radius_search / sort via the subprocess dispatch path) are NOT raced
here. They run 25 ms - 2.5 s in the current subprocess path due to a DISPATCH artifact, not
silicon speed; racing them against MLX would misrepresent the hardware. They are an
ANE-EXCLUSIVE CAPABILITY, dispatch-bound in the current path - a separate story, not a
fair speed race. The native in-graph af.sdpa / af.mha attention route (no subprocess)
IS fair and is included as the attention class.

Run from repo root (energy needs passwordless sudo for powermetrics):

    PYTHONPATH=. python3 bench/device_compare_wattcomplete.py

Writes bench/results/device_compare_wattcomplete_results.json alongside the printed
tables. --quick runs a reduced rep/window for a smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

# Reuse the sibling harness's helpers + workload math so we don't duplicate it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_compare as dc  # noqa: E402

HAVE_ANE, HAVE_MLX, HAVE_TV, HAVE_HF = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_TV, dc.HAVE_HF
HAVE_SUDO = dc.HAVE_SUDO
relerr = dc.relerr
min_latency = dc.min_latency
min_latency_with_out = dc.min_latency_with_out

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx

_RAIL = {"ane": re.compile(r"ANE Power:\s*([\d.]+)\s*mW"),
         "cpu": re.compile(r"CPU Power:\s*([\d.]+)\s*mW"),
         "gpu": re.compile(r"GPU Power:\s*([\d.]+)\s*mW")}
# powermetrics prints the OS-computed package total once per sample - use it as the
# authoritative per-sample package number (avoids the GPU-rail double-print, which
# appears twice per block and would misalign a hand-summed per-sample total).
_PKG = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")

WINDOW = 6.0          # sustained-loop seconds (overridden by --quick)
# powermetrics 'Power' is the AVERAGE over the sample interval. A sub-ms op sampled at
# 100 ms catches the package mid-duty-cycle (huge CV - the exact artifact a reviewer
# would flag). We integrate over a coarser 500 ms interval so each sample averages
# many iterations -> the duty cycle is smoothed into a stable mean, and the residual
# CV is real run-to-run variation, not a phase artifact.
PM_INTERVAL_MS = 500

# accumulate every workload's rows + per-device energy here
RESULTS: dict[str, dict] = {}
IDLE: dict[str, float] = {}   # per-rail idle mW, sampled once at start
IDLE_PKG = 0.0                # total-package idle mW


# rigorous power harness
def _parse_pm_per_sample(txt: str) -> tuple[dict[str, list[float]], list[float]]:
    """Per-rail lists of per-sample mW + the per-sample OS-combined package total.
    The GPU rail is printed twice per sample block by powermetrics, so we keep the
    aligned subset (every other match) to match the ANE/CPU one-per-sample cadence."""
    per = {}
    for rail, rx in _RAIL.items():
        vals = [float(m) for m in rx.findall(txt)]
        if rail == "gpu" and len(vals) >= 2:
            # GPU prints twice/sample; the first per block is the rail reading we want
            ane_n = len(_RAIL["ane"].findall(txt))
            if ane_n and len(vals) == 2 * ane_n:
                vals = vals[0::2]
        per[rail] = vals
    pkg = [float(m) for m in _PKG.findall(txt)]
    return per, pkg


def sample_idle(seconds: float) -> None:
    """Sample a no-workload idle baseline (per rail) ONCE. Stores median mW/rail."""
    global IDLE_PKG
    if not HAVE_SUDO:
        return
    samples = max(20, int(seconds / (PM_INTERVAL_MS / 1000.0)) + 5)
    log = Path("/tmp/pm_wattc_idle.log")
    pm = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "ane_power,cpu_power,gpu_power",
         "--sample-rate", str(PM_INTERVAL_MS), "--sample-count", str(samples)],
        stdout=open(log, "w"), stderr=subprocess.DEVNULL)
    pm.wait()
    per, pkg = _parse_pm_per_sample(log.read_text())
    for rail, v in per.items():
        IDLE[rail] = float(np.median(v)) if v else 0.0
    IDLE_PKG = float(np.median(pkg)) if pkg else sum(IDLE.values())


def measure_energy(run_once, *, tag: str, window: float = WINDOW) -> dict | None:
    """Sustained-loop powermetrics around run_once(), with the rigor fixes:
       idle-subtracted ACTIVE per rail + total-package, median + CV + min/p90 over the
       per-sample package totals, perf-counter iter time, and low-confidence flags."""
    if not HAVE_SUDO:
        return None
    for _ in range(5):              # warmup before the sampling window
        run_once()
    # Size the sampler to the window and keep the workload loop running until the
    # sampler EXITS (poll), so every integrated sample is captured under load - the
    # earlier bug was the loop stopping at `window` while powermetrics kept sampling
    # idle for its remaining count, which manufactured the high CV.
    samples = max(8, int(window / (PM_INTERVAL_MS / 1000.0)))
    log = Path(f"/tmp/pm_wattc_{tag}.log")
    pm = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "ane_power,cpu_power,gpu_power",
         "--sample-rate", str(PM_INTERVAL_MS), "--sample-count", str(samples)],
        stdout=open(log, "w"), stderr=subprocess.DEVNULL)
    time.sleep(0.35)                # let the sampler spin up before we count iters
    t0 = time.perf_counter()
    n = 0
    while pm.poll() is None:        # drive work for the ENTIRE sampling window
        run_once()
        n += 1
    dt = time.perf_counter() - t0
    pm.wait()
    per, pkg = _parse_pm_per_sample(log.read_text())
    ns = len(pkg)
    flags: list[str] = []

    out: dict = {"iter_ms": dt / n * 1e3, "iters": n, "n_pm_samples": ns}
    # per-rail active (idle-subtracted), using the per-rail median over samples
    rail_active = {}
    for rail in _RAIL:
        v = per.get(rail, [])
        loaded = float(np.median(v)) if v else float("nan")
        active = max(0.0, loaded - IDLE.get(rail, 0.0)) if not np.isnan(loaded) else float("nan")
        out[f"{rail}_loaded_mW"] = loaded
        out[f"{rail}_active_mW"] = active
        rail_active[rail] = active

    # per-sample OS-combined package total (powermetrics' own "Combined Power" line,
    # which is the average over each sample interval - not an instantaneous spot read).
    # CV is computed on the RAW loaded samples (no per-sample clamp - clamping at the
    # idle floor manufactures variance); idle is subtracted ONCE, from the median.
    if pkg:
        arr = np.array(pkg)
        med = float(np.median(arr))
        mean = float(arr.mean())
        cv = float(arr.std() / mean * 100.0) if mean > 0 else float("nan")
        out["loaded_pkg_W"] = med / 1000.0
        out["active_pkg_W"] = max(0.0, med - IDLE_PKG) / 1000.0
        out["active_pkg_mean_W"] = max(0.0, mean - IDLE_PKG) / 1000.0
        out["active_pkg_min_W"] = max(0.0, float(arr.min()) - IDLE_PKG) / 1000.0
        out["active_pkg_p90_W"] = max(0.0, float(np.percentile(arr, 90)) - IDLE_PKG) / 1000.0
        out["active_pkg_cv_pct"] = cv          # CV of the loaded package draw
        if cv > 35.0:
            flags.append(f"loaded-package CV {cv:.0f}% (>35%) - sub-ms op vs sampler, "
                         f"low confidence")
    else:
        out["active_pkg_W"] = float("nan")
        out["active_pkg_cv_pct"] = float("nan")
        flags.append("no powermetrics samples parsed")

    if ns and ns < 12:
        flags.append(f"only {ns} pm samples - short window, treat as indicative")
    out["flags"] = flags
    return out


def _attach_energy(wl: str, device: str, run_once, *, tag: str,
                   flops: float | None = None, items: float | None = None,
                   per_inf: bool = False, window: float = WINDOW):
    """Run the energy harness for one device of a workload and attach perf/watt."""
    e = measure_energy(run_once, tag=tag, window=window)
    if e is None:
        return
    apw = e.get("active_pkg_W", float("nan"))
    sane = apw == apw and apw > 0
    # plausibility: an ANE workload reading ~0 on the ANE rail is a sampling miss
    if device == "ANE" and e.get("ane_active_mW", 0.0) < 5.0 and not e["flags"]:
        e["flags"].append("ANE rail ~0 mW during ANE workload - likely a 100ms sampling miss")
    if flops is not None and sane:
        e["perf_per_W"] = (flops / (e["iter_ms"] / 1e3)) / 1e9 / apw  # GFLOP/s per W
        e["perf_unit"] = "GFLOP/s/W"
    elif items is not None and sane:
        e["perf_per_W"] = (items / (e["iter_ms"] / 1e3)) / apw
        e["perf_unit"] = "items/s/W"
    if per_inf and sane:
        e["mJ_per_inf"] = apw * e["iter_ms"]   # W * ms = mJ
    RESULTS[wl].setdefault("energy", {})[device] = e


# result accumulation (mirrors device_compare's add_row but local)
def add_row(wl, device, dtype, lat_s, throughput, rerr):
    RESULTS.setdefault(wl, {"rows": []})
    RESULTS[wl]["rows"].append({
        "device": device, "dtype": dtype,
        "latency_ms": (lat_s * 1e3) if lat_s is not None else None,
        "throughput": throughput, "relerr": rerr})


def note(wl, text):
    RESULTS.setdefault(wl, {"rows": []})["note"] = text


def gfs(flops, lat_s):
    return f"{flops / lat_s / 1e9:.1f} GFLOP/s"


# WORKLOADS (latency + precision + energy for ANE & MLX-fp16)
def wl_gemm(M, K, N, tag):
    wl = f"GEMM {tag} (M={M},K={K},N={N})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(0)
    x32 = rng.standard_normal((M, K)).astype(np.float32) / np.sqrt(K)
    W32 = rng.standard_normal((N, K)).astype(np.float32) / np.sqrt(K)
    ref = x32.astype(np.float64) @ W32.astype(np.float64).T
    flops = 2.0 * M * K * N
    note(wl, f"{flops/1e9:.3f} GFLOP. ref = fp64 numpy.")

    if HAVE_ANE:
        net = af.compile(af.input((M, K)).linear(W32.astype(np.float16)))
        xf = x32.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: net(xf), tag=f"gemm_{tag}_ane", flops=flops)
    if HAVE_MLX:
        xg32, Wg32 = mx.array(x32), mx.array(W32.T)
        lat, out = dc.mlx_run(lambda: xg32 @ Wg32)
        add_row(wl, "GPU", "fp32", lat, gfs(flops, lat), relerr(out, ref))
        xg16, Wg16 = mx.array(x32.astype(np.float16)), mx.array(W32.T.astype(np.float16))
        lat, out = dc.mlx_run(lambda: xg16 @ Wg16)
        add_row(wl, "GPU", "fp16", lat, gfs(flops, lat), relerr(out, ref))
        _attach_energy(wl, "GPU", lambda: mx.eval(xg16 @ Wg16), tag=f"gemm_{tag}_gpu", flops=flops)
    lat, out = dc.cpu_run(lambda: x32 @ W32.T)
    add_row(wl, "CPU", "fp32", lat, gfs(flops, lat), relerr(out, ref))


def wl_conv(Cin, Cout, H, W, k, depth, tag):
    wl = f"conv {tag} (C={Cout},{H}x{W},k={k},depth={depth})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(1)
    pad = k // 2
    x32 = rng.standard_normal((1, Cin, H, W)).astype(np.float32)
    Ws = [rng.standard_normal((Cout, (Cin if d == 0 else Cout), k, k)).astype(np.float32)
          * np.sqrt(2.0 / ((Cin if d == 0 else Cout) * k * k)) for d in range(depth)]
    flops = sum(2.0 * (Cin if d == 0 else Cout) * Cout * k * k * H * W for d in range(depth))
    refr = dc._np_conv_stack(x32, Ws, pad, relu=True)
    note(wl, f"{flops/1e9:.3f} GFLOP, same-pad, He init. ref = fp32 numpy.")

    if HAVE_ANE:
        h = af.input((1, Cin, H, W))
        for w in Ws:
            h = af.conv(h, w.astype(np.float16), stride=1, pad=pad).relu()
        net = af.compile(h)
        xf = x32.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), relerr(out, refr))
        _attach_energy(wl, "ANE", lambda: net(xf), tag=f"conv_{tag}_ane", flops=flops)
    if HAVE_MLX:
        def mk(dt):
            xg = mx.array(np.transpose(x32, (0, 2, 3, 1)).astype(dt))
            Wg = [mx.array(np.transpose(w, (0, 2, 3, 1)).astype(dt)) for w in Ws]
            return xg, Wg
        for dt, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg, Wg = mk(dt)
            def run(xg=xg, Wg=Wg):
                hh = xg
                for w in Wg:
                    hh = mx.maximum(mx.conv2d(hh, w, stride=1, padding=pad), 0)
                return hh
            lat, out = dc.mlx_run(run)
            outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
            add_row(wl, "GPU", name, lat, gfs(flops, lat), relerr(outn, refr))
            if dt == np.float16:
                _attach_energy(wl, "GPU", lambda run=run: mx.eval(run()), tag=f"conv_{tag}_gpu", flops=flops)


def wl_attention(SEQ, DIM, HEADS, tag):
    wl = f"attention {tag} (S={SEQ},D={DIM},H={HEADS})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(5)
    dh = DIM // HEADS
    sc = 1.0 / np.sqrt(DIM)
    def mkw(o, i): return rng.standard_normal((o, i)).astype(np.float32) * sc
    def mkb(o): return rng.standard_normal(o).astype(np.float32) * 0.01
    Wq, Wk, Wv, Wo = (mkw(DIM, DIM) for _ in range(4))
    bq, bk, bv, bo = (mkb(DIM) for _ in range(4))
    x = rng.standard_normal((SEQ, DIM)).astype(np.float32)

    def ref_attn(xx, dt):
        xx = xx.astype(dt)
        def lin(a, W, b): return a @ W.astype(dt).T + b.astype(dt)
        q = lin(xx, Wq, bq).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        kk = lin(xx, Wk, bk).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        v = lin(xx, Wv, bv).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        s = (q @ kk.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a = a / a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
        return lin(o, Wo, bo)
    ref = ref_attn(x, np.float64)
    flops = 4.0 * 2 * SEQ * DIM * DIM + 2.0 * 2 * HEADS * SEQ * SEQ * dh
    note(wl, f"{flops/1e9:.3f} GFLOP. in-graph af.mha (decomposed-SDPA fused route). "
             f"ref = fp64. softmax fp16-stable (wide accum).")

    if HAVE_ANE:
        y = af.mha(af.input((SEQ, DIM)), Wq.astype(np.float16), bq, Wk.astype(np.float16), bk,
                   Wv.astype(np.float16), bv, Wo.astype(np.float16), bo, HEADS)
        net = af.compile(y)
        xf = x.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: net(xf), tag=f"attn_{tag}_ane", flops=flops)
    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(x.astype(d))
            Wqg, Wkg, Wvg, Wog = (mx.array(w.T.astype(d)) for w in (Wq, Wk, Wv, Wo))
            bqg, bkg, bvg, bog = (mx.array(b.astype(d)) for b in (bq, bk, bv, bo))
            def run(xg=xg, Wqg=Wqg, Wkg=Wkg, Wvg=Wvg, Wog=Wog, bqg=bqg, bkg=bkg, bvg=bvg, bog=bog):
                q = (xg @ Wqg + bqg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                kk = (xg @ Wkg + bkg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                v = (xg @ Wvg + bvg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                s = (q @ kk.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
                a = mx.softmax(s, axis=-1)
                o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
                return o @ Wog + bog
            lat, out = dc.mlx_run(run)
            add_row(wl, "GPU", name, lat, gfs(flops, lat), relerr(np.array(out, copy=False), ref))
            if d == np.float16:
                _attach_energy(wl, "GPU", lambda run=run: mx.eval(run()), tag=f"attn_{tag}_gpu", flops=flops)
    lat, _ = dc.cpu_run(lambda: ref_attn(x, np.float32), reps=10)
    add_row(wl, "CPU", "fp32", lat, gfs(flops, lat), relerr(ref_attn(x, np.float32), ref))


def wl_norm(kind, shape, tag, num_groups=32):
    """layer_norm / rms_norm over last dim of [S,D]; group_norm over [1,C,H,W]."""
    wl = f"{kind} {tag} {shape}"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(7)
    x32 = rng.standard_normal(shape).astype(np.float32)
    items = float(np.prod(shape))   # elements normalized/step
    if kind in ("layer_norm", "rms_norm"):
        D = shape[-1]
        g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0)
        b = rng.standard_normal(D).astype(np.float32) * 0.1
        mu = x32.mean(-1, keepdims=True) if kind == "layer_norm" else 0.0
        var = ((x32 - mu) ** 2).mean(-1, keepdims=True)
        ref = ((x32 - mu) / np.sqrt(var + 1e-5)) * g + (b if kind == "layer_norm" else 0.0)
        if kind == "rms_norm":
            rms = np.sqrt((x32 ** 2).mean(-1, keepdims=True) + 1e-5)
            ref = (x32 / rms) * g
    else:  # group_norm [1,C,H,W]
        C = shape[1]
        g = (rng.standard_normal(C).astype(np.float32) * 0.1 + 1.0)
        b = rng.standard_normal(C).astype(np.float32) * 0.1
        xr = x32.reshape(1, num_groups, C // num_groups, shape[2], shape[3])
        mu = xr.mean((2, 3, 4), keepdims=True); var = xr.var((2, 3, 4), keepdims=True)
        ref = (((xr - mu) / np.sqrt(var + 1e-5)).reshape(shape) * g[None, :, None, None]
               + b[None, :, None, None])
    note(wl, f"{items/1e6:.2f}M elems normalized. ref = fp32 numpy.")

    if HAVE_ANE:
        xin = af.input(shape)
        if kind == "layer_norm":
            y = xin.layer_norm(g.astype(np.float16), b.astype(np.float16))
        elif kind == "rms_norm":
            y = xin.rms_norm(g.astype(np.float16))
        else:
            y = xin.group_norm(g.astype(np.float16), b.astype(np.float16), num_groups)
        net = af.compile(y)
        xf = x32.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        add_row(wl, "ANE", "fp16", lat, f"{items/lat/1e9:.2f} Gelem/s", relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: net(xf), tag=f"{kind}_{tag}_ane", items=items)
    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(x32.astype(d)); gg = mx.array(g.astype(d)); bb = mx.array(b.astype(d))
            if kind == "layer_norm":
                def run(xg=xg, gg=gg, bb=bb):
                    mu = mx.mean(xg, axis=-1, keepdims=True)
                    var = mx.mean((xg - mu) ** 2, axis=-1, keepdims=True)
                    return (xg - mu) * mx.rsqrt(var + 1e-5) * gg + bb
            elif kind == "rms_norm":
                def run(xg=xg, gg=gg):
                    ms = mx.mean(xg ** 2, axis=-1, keepdims=True)
                    return xg * mx.rsqrt(ms + 1e-5) * gg
            else:
                C = shape[1]
                def run(xg=xg, gg=gg, bb=bb, C=C):
                    xr = xg.reshape(1, num_groups, C // num_groups, shape[2], shape[3])
                    mu = mx.mean(xr, axis=(2, 3, 4), keepdims=True)
                    var = mx.mean((xr - mu) ** 2, axis=(2, 3, 4), keepdims=True)
                    xn = ((xr - mu) * mx.rsqrt(var + 1e-5)).reshape(shape)
                    return xn * gg.reshape(1, C, 1, 1) + bb.reshape(1, C, 1, 1)
            lat, out = dc.mlx_run(run)
            add_row(wl, "GPU", name, lat, f"{items/lat/1e9:.2f} Gelem/s",
                    relerr(np.array(out, copy=False), ref))
            if d == np.float16:
                _attach_energy(wl, "GPU", lambda run=run: mx.eval(run()), tag=f"{kind}_{tag}_gpu", items=items)


def wl_dft(Nn):
    wl = f"DFT-as-matmul (N={Nn})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(Nn).astype(np.float64)
    n = np.arange(Nn)
    ang = -2 * np.pi * np.outer(n, n) / Nn
    Fr, Fi = np.cos(ang), np.sin(ang)
    ref_r, ref_i = Fr @ x, Fi @ x
    flops = 2.0 * (2 * Nn * Nn)
    note(wl, f"{flops/1e9:.3f} GFLOP. split-real DFT matmul; ref = fp64. Oscillatory => "
             f"fp16 cancellation stress.")
    x1 = x.reshape(1, Nn)
    if HAVE_ANE:
        netr = af.compile(af.input((1, Nn)).linear(Fr.astype(np.float16)))
        neti = af.compile(af.input((1, Nn)).linear(Fi.astype(np.float16)))
        xf = x1.astype(np.float16)
        lat = min_latency(lambda: (netr(xf), neti(xf)))
        yr, yi = netr(xf).ravel(), neti(xf).ravel()
        err = (relerr(yr, ref_r) + relerr(yi, ref_i)) / 2
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), err)
        _attach_energy(wl, "ANE", lambda: (netr(xf), neti(xf)), tag=f"dft{Nn}_ane", flops=flops)
    if HAVE_MLX:
        for dt, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(x1.astype(dt))
            Frg, Fig = mx.array(Fr.T.astype(dt)), mx.array(Fi.T.astype(dt))
            lat, _ = dc.mlx_run(lambda xg=xg, Frg=Frg, Fig=Fig: (xg @ Frg) + (xg @ Fig))
            yr = np.array(xg @ Frg, copy=False).ravel(); yi = np.array(xg @ Fig, copy=False).ravel()
            err = (relerr(yr, ref_r) + relerr(yi, ref_i)) / 2
            add_row(wl, "GPU", name, lat, gfs(flops, lat), err)
            if dt == np.float16:
                _attach_energy(wl, "GPU", lambda xg=xg, Frg=Frg, Fig=Fig: mx.eval((xg @ Frg) + (xg @ Fig)),
                               tag=f"dft{Nn}_gpu", flops=flops)


def wl_stencil(H, W, steps):
    wl = f"stencil 5pt ({H}x{W}, steps={steps})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(3)
    u0 = rng.standard_normal((1, 1, H, W)).astype(np.float32)
    dt = 0.1
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    K = np.zeros((1, 1, 3, 3), dtype=np.float32)
    K[0, 0] = dt * lap; K[0, 0, 1, 1] += 1.0
    flops = steps * 2.0 * 9 * H * W
    note(wl, f"{flops/1e9:.4f} GFLOP. heat-eqn step as identity+dt*Laplacian 3x3 conv; "
             f"ref = fp64. diffusive => fp16-friendly.")
    def np_step(u):
        u = u.astype(np.float64)
        for _ in range(steps):
            u = dc._np_conv2d(u.astype(np.float32), K, 1).astype(np.float64)
        return u
    ref = np_step(u0)
    if HAVE_ANE:
        h = af.input((1, 1, H, W))
        for _ in range(steps):
            h = af.conv(h, K.astype(np.float16), stride=1, pad=1)
        net = af.compile(h)
        uf = u0.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(uf))
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: net(uf), tag="stencil_ane", flops=flops)
    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(np.transpose(u0, (0, 2, 3, 1)).astype(d))
            Kg = mx.array(np.transpose(K, (0, 2, 3, 1)).astype(d))
            def run(xg=xg, Kg=Kg):
                h = xg
                for _ in range(steps):
                    h = mx.conv2d(h, Kg, stride=1, padding=1)
                return h
            lat, out = dc.mlx_run(run)
            outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
            add_row(wl, "GPU", name, lat, gfs(flops, lat), relerr(outn, ref))
            if d == np.float16:
                _attach_energy(wl, "GPU", lambda run=run: mx.eval(run()), tag="stencil_gpu", flops=flops)
    lat, _ = dc.cpu_run(lambda: np_step(u0), reps=5)
    add_row(wl, "CPU", "fp32", lat, gfs(flops, lat), relerr(np_step(u0), ref))


def wl_jacobi(n, iters):
    """Fixed-iteration Jacobi solve of a diagonally-dominant SPD system, as ONE
    fused graph: x <- Dinv * (b - R x), R = A - diag(A). The fair iterative-solver
    comparison (steady-state silicon, not a per-call recompile)."""
    wl = f"Jacobi solve (n={n}, iters={iters})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(8)
    Mx = rng.standard_normal((n, n)).astype(np.float32)
    A = (Mx @ Mx.T) / n + n * np.eye(n, dtype=np.float32)   # diag-dominant SPD
    b = rng.standard_normal(n).astype(np.float32)
    d = np.diag(A).copy()
    R = A.copy(); np.fill_diagonal(R, 0.0)
    Dinv = (1.0 / d).reshape(1, n).astype(np.float32)
    flops = iters * (2.0 * n * n + 3.0 * n)
    # fixed-iteration reference (this exact recurrence in fp64) and the true solve
    xj = np.zeros(n, np.float64)
    for _ in range(iters):
        xj = (b.astype(np.float64) - R.astype(np.float64) @ xj) / d.astype(np.float64)
    ref = xj
    note(wl, f"{flops/1e9:.3f} GFLOP. {iters} Jacobi iters as one fused graph; "
             f"ref = same-iter fp64 recurrence (true solve relerr ~1e-3 at this iters).")
    if HAVE_ANE:
        xin = af.input((1, n)); bin_ = af.input((1, n)); Dv = af.input((1, n))
        x = xin
        for _ in range(iters):
            x = (bin_ - x.linear(R.astype(np.float16))) * Dv
        net = af.compile(x)
        z = np.zeros((1, n), np.float16); bf = b.reshape(1, n).astype(np.float16); Df = Dinv.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(z, bf, Df))
        add_row(wl, "ANE", "fp16", lat, gfs(flops, lat), relerr(out.ravel(), ref))
        _attach_energy(wl, "ANE", lambda: net(z, bf, Df), tag="jacobi_ane", flops=flops)
    if HAVE_MLX:
        for dt, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            Rg = mx.array(R.astype(dt)); bg = mx.array(b.reshape(1, n).astype(dt))
            Dg = mx.array(Dinv.astype(dt)); z0 = mx.zeros((1, n), dtype=Rg.dtype)
            def run(Rg=Rg, bg=bg, Dg=Dg, z0=z0):
                x = z0
                for _ in range(iters):
                    x = (bg - x @ Rg.T) * Dg
                return x
            lat, out = dc.mlx_run(run)
            add_row(wl, "GPU", name, lat, gfs(flops, lat), relerr(np.array(out, copy=False).ravel(), ref))
            if dt == np.float16:
                _attach_energy(wl, "GPU", lambda run=run: mx.eval(run()), tag="jacobi_gpu", flops=flops)
    lat, _ = dc.cpu_run(lambda: _np_jacobi(R, b, d, iters), reps=5)
    add_row(wl, "CPU", "fp32", lat, gfs(flops, lat), relerr(_np_jacobi(R, b, d, iters), ref))


def _np_jacobi(R, b, d, iters):
    x = np.zeros_like(b, dtype=np.float32)
    for _ in range(iters):
        x = ((b - R @ x) / d).astype(np.float32)
    return x


# real models
def wl_resnet18():
    wl = "ResNet-18 forward (1x3x224x224)"
    print(f"\n=== {wl} ===", flush=True)
    if not HAVE_TV:
        note(wl, "skipped - torchvision unavailable.")
        return
    rng = np.random.default_rng(6)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    import torch, torchvision as tv
    m = tv.models.resnet18(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0].astype(np.float64)
    note(wl, "ref = torch-CPU fp32. ANE = aneforge fused conv graph (BN folded).")
    lat_cpu, _ = dc.cpu_run(lambda: dc._torch_fwd(m, img, "cpu"), reps=5)
    add_row(wl, "CPU(torch)", "fp32", lat_cpu, None, 0.0)
    if torch.backends.mps.is_available():
        mm = m.to("mps")
        lat, out = min_latency_with_out(lambda: dc._torch_fwd(mm, img, "mps"), reps=15, warmup=5)
        add_row(wl, "GPU(MPS)", "fp32", lat, None, relerr(out, ref))
        _attach_energy(wl, "GPU(MPS)", lambda: dc._torch_fwd(mm, img, "mps"),
                       tag="resnet_mps", per_inf=True)
    if HAVE_ANE:
        clf = af.load_resnet18()
        lat, out = min_latency_with_out(lambda: clf(img), reps=15, warmup=5)
        add_row(wl, "ANE", "fp16", lat, None, relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: clf(img), tag="resnet_ane", per_inf=True)


def wl_minilm():
    wl = "MiniLM encoder (1 sentence)"
    print(f"\n=== {wl} ===", flush=True)
    if not HAVE_HF:
        note(wl, "skipped - transformers unavailable.")
        return
    NAME = "sentence-transformers/all-MiniLM-L6-v2"
    text = "The Apple Neural Engine is a specialized accelerator for matrix math."
    note(wl, "ref = HF torch-CPU fp32 (mean-pooled, L2-normed). ANE = af.load.")
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(NAME)
        hf = AutoModel.from_pretrained(NAME).eval()
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            hs = hf(**ids).last_hidden_state[0].numpy()
        ref = hs.mean(0); ref = (ref / np.linalg.norm(ref)).astype(np.float64)
        lat_cpu, _ = dc.cpu_run(lambda: dc._hf_embed(hf, ids), reps=5)
        add_row(wl, "CPU(torch)", "fp32", lat_cpu, None, 0.0)
    except Exception as e:
        print(f"  CPU ref: {type(e).__name__}: {e}"); return
    if HAVE_ANE:
        enc = af.load(NAME); enc(text)
        lat, _ = min_latency_with_out(lambda: enc(text), reps=15, warmup=3)
        out = enc(text)[0].astype(np.float64)
        add_row(wl, "ANE", "fp16", lat, None, relerr(out, ref))
        _attach_energy(wl, "ANE", lambda: enc(text), tag="minilm_ane", per_inf=True)


def wl_vit_b16():
    """Full ViT-B/16 forward (vit_demo build), real torchvision pretrained weights."""
    wl = "ViT-B/16 forward (1x3x224x224, 197 tokens)"
    print(f"\n=== {wl} ===", flush=True)
    if not HAVE_TV:
        note(wl, "skipped - torchvision unavailable.")
        return
    sys.path.insert(0, str(REPO / "examples"))
    import vit_demo as vd
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, vd.IMG, vd.IMG)).astype(np.float32)
    m, sd = vd.load_vit_weights()
    cls_const = sd["class_token"].reshape(1, vd.DIM).astype(np.float32)
    pos_const = sd["encoder.pos_embedding"].reshape(vd.SEQ, vd.DIM).astype(np.float32)
    # compile full 12-layer if possible, else fall back (mirror vit_demo)
    n_layers, net = vd.N_LAYERS, None
    for k in (vd.N_LAYERS, 8, 6, 4, 2, 1):
        try:
            net = af.compile(vd.build_vit(sd, k)); n_layers = k; break
        except Exception as e:
            print(f"  [compile] {k} layers failed ({type(e).__name__}) -> fewer"); net = None
    if net is None:
        note(wl, "ANE compile failed at all layer counts."); return
    ref = vd.torch_ref(m, img, n_layers).astype(np.float64)
    note(wl, f"ANE = {n_layers}-layer fused ViT-B/16, real IMAGENET1K weights; "
             f"ref = torch-CPU fp32 (same {n_layers} layers).")
    import torch
    lat_cpu, _ = dc.cpu_run(lambda: vd.torch_ref(m, img, n_layers), reps=4)
    add_row(wl, "CPU(torch)", "fp32", lat_cpu, None, 0.0)
    if torch.backends.mps.is_available():
        mm = m.to("mps")
        def mps_fwd():
            with torch.no_grad():
                x = mm._process_input(torch.from_numpy(img).to("mps"))
                cls = mm.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1) + mm.encoder.pos_embedding
                x = mm.encoder.dropout(x)
                for i in range(n_layers):
                    x = mm.encoder.layers[i](x)
                x = mm.encoder.ln(x)[:, 0]
                o = mm.heads(x); torch.mps.synchronize()
                return o.to("cpu").numpy()[0]
        lat, out = min_latency_with_out(mps_fwd, reps=12, warmup=4)
        add_row(wl, "GPU(MPS)", "fp32", lat, None, relerr(out, ref))
        _attach_energy(wl, "GPU(MPS)", mps_fwd, tag="vit_mps", per_inf=True)
    lat, out = min_latency_with_out(lambda: net(img, cls_const, pos_const), reps=12, warmup=4)
    add_row(wl, "ANE", "fp16", lat, None, relerr(out[0], ref))
    _attach_energy(wl, "ANE", lambda: net(img, cls_const, pos_const), tag="vit_ane", per_inf=True)


# reporting
def fmt_lat(ms):
    if ms is None:
        return "    -   "
    return f"{ms:8.3f} ms" if ms < 1000 else f"{ms/1000:7.3f} s "


def fmt_err(e):
    if e is None:
        return "   -    "
    if e == 0.0:
        return " ref    "
    return f"{e:.2e}"


def print_tables():
    print("\n" + "=" * 100)
    print(" WATT-COMPLETE PER-WORKLOAD RESULTS")
    print(" latency = MIN over reps (end-to-end incl. host/dispatch); power = idle-subtracted ACTIVE")
    print("=" * 100)
    if HAVE_SUDO:
        print(f" idle baseline (median mW): ANE {IDLE.get('ane',0):.0f} | GPU {IDLE.get('gpu',0):.0f} "
              f"| CPU {IDLE.get('cpu',0):.0f} | package {IDLE_PKG:.0f}")
    for wl, data in RESULTS.items():
        print(f"\n{wl}")
        if data.get("note"):
            print(f"  {data['note']}")
        rows = data.get("rows", [])
        if not rows:
            print("  (no rows)"); continue
        print(f"  {'device':<11} {'dtype':<6} {'latency':>12} {'throughput':>16} {'relerr':>12}")
        print("  " + "-" * 62)
        for r in rows:
            print(f"  {r['device']:<11} {r['dtype']:<6} {fmt_lat(r['latency_ms']):>12} "
                  f"{(r['throughput'] or '-'):>16} {fmt_err(r['relerr']):>12}")
        for dev, e in data.get("energy", {}).items():
            apw = e.get("active_pkg_W", float("nan"))
            cv = e.get("active_pkg_cv_pct", float("nan"))
            pw = e.get("perf_per_W"); unit = e.get("perf_unit", "")
            mj = e.get("mJ_per_inf")
            extra = (f", {pw:.2f} {unit}" if pw is not None else "") + \
                    (f", {mj:.1f} mJ/inf" if mj is not None else "")
            print(f"  [energy {dev:<9}] {e['iter_ms']:.3f} ms/iter | active pkg {apw:.2f} W "
                  f"(CV {cv:.0f}%, {e.get('n_pm_samples','?')} smp) | "
                  f"ANE {e.get('ane_active_mW',float('nan')):.0f} / GPU {e.get('gpu_active_mW',float('nan')):.0f} "
                  f"/ CPU {e.get('cpu_active_mW',float('nan')):.0f} mW active{extra}")
            for f in e.get("flags", []):
                print(f"      FLAG: {f}")


def main() -> int:
    global WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced window/reps smoke test")
    ap.add_argument("--window", type=float, default=None)
    args = ap.parse_args()
    if args.quick:
        WINDOW = 2.0
    if args.window:
        WINDOW = args.window

    print("=" * 100)
    print(" device_compare_wattcomplete - ANE vs MLX-GPU vs CPU, energy/watt complete")
    print("=" * 100)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}  "
          f"torchvision={'yes' if HAVE_TV else 'no'}  transformers={'yes' if HAVE_HF else 'no'}")
    print(f" powermetrics(sudo)={'yes' if HAVE_SUDO else 'NO - energy skipped'}  window={WINDOW}s")

    if HAVE_SUDO:
        print("\n sampling idle baseline (no workload)...", flush=True)
        sample_idle(3.0)
        print(f" idle: ANE {IDLE.get('ane',0):.0f} / GPU {IDLE.get('gpu',0):.0f} / "
              f"CPU {IDLE.get('cpu',0):.0f} mW (pkg {IDLE_PKG:.0f} mW)")

    # 1. GEMM
    wl_gemm(64, 256, 256, "floor")
    wl_gemm(128, 1024, 1024, "bandwidth")
    wl_gemm(256, 4096, 4096, "compute")
    # 2. conv
    wl_conv(16, 64, 32, 32, 3, 1, "single")
    wl_conv(64, 256, 32, 32, 3, 16, "resnet-ish")
    # 3. attention (short + long seq)
    wl_attention(197, 768, 12, "vit")
    wl_attention(512, 768, 12, "longseq")
    # 4. norm family
    wl_norm("layer_norm", (197, 768), "vit")
    wl_norm("rms_norm", (197, 768), "vit")
    wl_norm("group_norm", (1, 256, 64, 64), "fmap", num_groups=32)
    # 5. scientific kernels
    wl_dft(1024)
    wl_stencil(256, 256, 32)
    wl_jacobi(512, 25)
    # 6. real models
    wl_resnet18()
    wl_minilm()
    wl_vit_b16()

    print_tables()

    out = Path(__file__).resolve().parent / "results" / "device_compare_wattcomplete_results.json"
    out.write_text(json.dumps({
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "torchvision": HAVE_TV,
                     "transformers": HAVE_HF, "sudo": HAVE_SUDO},
        "window_s": WINDOW, "pm_interval_ms": PM_INTERVAL_MS,
        "idle_mW": IDLE, "idle_pkg_mW": IDLE_PKG,
        "results": RESULTS,
    }, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
