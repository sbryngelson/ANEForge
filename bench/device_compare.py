#!/usr/bin/env python3
"""Shared device-comparison harness - ANE (aneforge fp16) vs GPU (MLX) vs CPU (numpy).

This module holds the timing, precision, energy, and device-runner helpers plus the
workload builders that the other ``bench/`` scripts import; it is not run on its own.
The single-stream device map is produced by ``device_compare_wattcomplete.py``.

Each helper runs the SAME math on three devices and reports, per workload:

    device | dtype | min-latency | (throughput) | relerr-vs-fp32

across the workloads the project actually runs:

    * GEMM at three regimes (K=256 floor, K=1024 bandwidth, K=4096 compute)
    * conv (a small one + a ResNet-ish 3x3 stack)
    * scientific kernels: DFT-as-matmul, a 2D 5-point stencil step, an N-body
      pairwise-force step
    * real models: ResNet-18 (af.load_resnet18), MiniLM encoder (af.load), a ViT
      self-attention block (af.mha, the vit_demo shape)

Devices
    ANE  = aneforge fp16 (af.compile + call). Compute is fp16-only on the silicon;
           the accumulator is >=fp32 (see ANE_MANUAL), so the precision cost shows
           up only in cancellation-heavy reductions.
    GPU  = MLX (Apple GPU / Metal), run at BOTH fp16 and fp32, so the GPU's own
           fp16 precision cost is explicit and separable from the ANE's.
    CPU  = numpy on Accelerate BLAS, fp32 (the precision reference) + fp16 where
           it is a sensible comparison.

Timing: warmup, then MIN over reps (the clean signal - min rejects scheduler
noise). Each device's number includes its own host/dispatch overhead, which is
called out: the ANE/MLX numbers are end-to-end Python-call latency
(compile-once, run-many), NOT pure silicon time. At small shapes that overhead
dominates and the comparison is a dispatch-cost comparison, not a FLOP one.

Energy: powermetrics is sampled ONLY around the sustained compute-bound loops
(the GEMM-K4096 and conv-stack workloads), where a multi-second loop gives a
trustworthy rail average. Short per-call workloads do NOT get an energy number -
a sub-millisecond op sampled at 100 Hz is an artifact, and we don't report it.
Energy needs passwordless sudo; without it, latency/precision still run.

Precision: every workload has an fp64/fp32 numpy reference. We report relative
L2 error (||x - ref|| / ||ref||) for ANE-fp16, GPU-fp16 and GPU-fp32 against it,
so "what does fp16 cost here" is answerable per workload, per device.
"""
from __future__ import annotations

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

# ---- optional backends; the script degrades gracefully ---------------------- #
HAVE_ANE = HAVE_MLX = HAVE_TORCH = HAVE_TV = HAVE_HF = False
ANE_ERR = MLX_ERR = ""
try:
    import aneforge as af
    HAVE_ANE = True
except Exception as e:  # pragma: no cover
    ANE_ERR = f"{type(e).__name__}: {e}"
try:
    import mlx.core as mx
    HAVE_MLX = True
except Exception as e:  # pragma: no cover
    MLX_ERR = f"{type(e).__name__}: {e}"
try:
    import torchvision  # noqa: F401
    HAVE_TV = True
except Exception:
    pass
try:
    import transformers  # noqa: F401
    HAVE_HF = True
except Exception:
    pass

_RAIL = {"ane": re.compile(r"ANE Power:\s*([\d.]+)\s*mW"),
         "cpu": re.compile(r"CPU Power:\s*([\d.]+)\s*mW"),
         "gpu": re.compile(r"GPU Power:\s*([\d.]+)\s*mW")}
HAVE_SUDO = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


# timing + precision helpers

def min_latency(fn, reps=30, warmup=8) -> float:
    """MIN end-to-end wall time (seconds) over `reps`, after `warmup`. fn must
    block until the device work is complete (mx.eval / sync / .numpy())."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def relerr(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    rn = np.linalg.norm(ref)
    return float(np.linalg.norm(got - ref) / rn) if rn > 0 else float(np.linalg.norm(got - ref))


def measure_energy(run_once, *, seconds: float, tag: str) -> dict | None:
    """Sustained-loop powermetrics average around run_once(). Returns active-W
    and per-rail mW, or None if energy can't be trusted (no sudo)."""
    if not HAVE_SUDO:
        return None
    for _ in range(3):
        run_once()
    samples = max(15, int(seconds / 0.1) + 5)
    log = Path(f"/tmp/pm_devcmp_{tag}.log")
    pm = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "ane_power,cpu_power,gpu_power",
         "--sample-rate", "100", "--sample-count", str(samples)],
        stdout=open(log, "w"), stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        run_once()
        n += 1
    dt = time.perf_counter() - t0
    pm.wait()
    txt = log.read_text()
    out = {"iter_ms": dt / n * 1e3, "iters": n}
    active = 0.0
    for rail, rx in _RAIL.items():
        v = [float(m) for m in rx.findall(txt)]
        mw = (sum(v) / len(v)) if v else float("nan")
        out[rail + "_mW"] = mw
        if not np.isnan(mw):
            active += mw
    out["active_W"] = active / 1000.0
    return out


# result accumulation

RESULTS: dict[str, dict] = {}   # workload -> {"rows": [...], "note": str, ...}


def add_row(workload: str, device: str, dtype: str, lat_s: float | None,
            throughput: str | None, rerr: float | None):
    RESULTS.setdefault(workload, {"rows": []})
    RESULTS[workload]["rows"].append({
        "device": device, "dtype": dtype,
        "latency_ms": (lat_s * 1e3) if lat_s is not None else None,
        "throughput": throughput, "relerr": rerr,
    })


def note(workload: str, text: str):
    RESULTS.setdefault(workload, {"rows": []})
    RESULTS[workload]["note"] = text


def gflops(flops: float, lat_s: float) -> str:
    return f"{flops / lat_s / 1e9:.1f} GFLOP/s"


# device runners (each returns (latency_s, output_array) or None if unavailable)

def cpu_run(build_out, reps=30):
    """build_out() -> np.ndarray. Times the whole call (numpy/Accelerate)."""
    out_holder = {}
    def fn():
        out_holder["o"] = build_out()
    lat = min_latency(fn, reps=reps)
    return lat, out_holder["o"]


def mlx_run(build_mx, reps=30):
    """build_mx() -> mlx array (lazy). We force eval each rep."""
    def fn():
        o = build_mx()
        mx.eval(o)
    lat = min_latency(fn, reps=reps)
    o = build_mx(); mx.eval(o)
    return lat, np.array(o, copy=False)


# WORKLOADS

def w_gemm(M, K, N, tag):
    """GEMM x[M,K] @ W[K,N]. aneforge linear wants W as [out,in]=[N,K]."""
    wl = f"GEMM {tag} (M={M},K={K},N={N})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(0)
    x32 = rng.standard_normal((M, K)).astype(np.float32) / np.sqrt(K)
    W32 = rng.standard_normal((N, K)).astype(np.float32) / np.sqrt(K)  # [out,in]
    ref = x32.astype(np.float64) @ W32.astype(np.float64).T
    flops = 2 * M * K * N
    note(wl, f"{flops/1e9:.3f} GFLOP. ref = fp64 numpy.")

    if HAVE_ANE:
        try:
            xin = af.input((M, K))
            net = af.compile(xin.linear(W32.astype(np.float16)))
            xf16 = x32.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(xf16))
            add_row(wl, "ANE", "fp16", lat, gflops(flops, lat), relerr(out, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        xg32, Wg32 = mx.array(x32), mx.array(W32.T)
        lat, out = mlx_run(lambda: xg32 @ Wg32)
        add_row(wl, "GPU", "fp32", lat, gflops(flops, lat), relerr(out, ref))
        xg16, Wg16 = mx.array(x32.astype(np.float16)), mx.array(W32.T.astype(np.float16))
        lat, out = mlx_run(lambda: xg16 @ Wg16)
        add_row(wl, "GPU", "fp16", lat, gflops(flops, lat), relerr(out, ref))

    # CPU fp32 (Accelerate BLAS - the reference-speed device).
    lat, out = cpu_run(lambda: x32 @ W32.T)
    add_row(wl, "CPU", "fp32", lat, gflops(flops, lat), relerr(out, ref))
    # CPU fp16: numpy has NO BLAS fp16 GEMM kernel (it upcasts element-by-element),
    # so this is pathologically slow and NOT a fair fp16 device - only run it at the
    # FLOOR size to document that "CPU fp16" is not a real option, then skip the big
    # ones (they take seconds for an uninteresting, GPU-fp16-identical relerr).
    if M * K * N <= 64 * 256 * 256:
        xh, Wh = x32.astype(np.float16), W32.T.astype(np.float16)
        lat, out = cpu_run(lambda: (xh @ Wh), reps=5)  # numpy upcasts internally
        add_row(wl, "CPU", "fp16*", lat, gflops(flops, lat), relerr(out, ref))


def min_latency_with_out(fn, reps=30, warmup=8):
    holder = {}
    def wrap():
        holder["o"] = fn()
    lat = min_latency(wrap, reps=reps, warmup=warmup)
    return lat, holder["o"]


def w_conv(Cin, Cout, H, W, k, depth, tag, energy=False):
    """conv stack: `depth` chained Cout-channel kxk same-pad convs (ResNet-ish)."""
    wl = f"conv {tag} (C={Cout},{H}x{W},k={k},depth={depth})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(1)
    pad = k // 2
    x32 = (rng.standard_normal((1, Cin, H, W)).astype(np.float32))
    # He-style variance-preserving init (sqrt(2/fan_in)) so a deep ReLU stack keeps
    # activations O(1) - a real ResNet does this via BatchNorm. Without it the
    # activations decay geometrically below fp16's smallest normal and the fp16
    # error becomes an underflow artifact rather than a real precision number.
    Ws = [rng.standard_normal((Cout, (Cin if d == 0 else Cout), k, k)).astype(np.float32)
          * np.sqrt(2.0 / ((Cin if d == 0 else Cout) * k * k)) for d in range(depth)]
    flops = sum(2 * (Cin if d == 0 else Cout) * Cout * k * k * H * W for d in range(depth))
    note(wl, f"{flops/1e9:.3f} GFLOP, same-pad, He init (O(1) activations). ref = fp32 numpy.")

    # fp32 numpy reference (naive, im2col-free via stride tricks would be heavy; use scipy-free loop)
    _ref = _np_conv_stack(x32, Ws, pad)

    if HAVE_ANE:
        try:
            xin = af.input((1, Cin, H, W))
            h = xin
            for w in Ws:
                h = af.conv(h, w.astype(np.float16), stride=1, pad=pad).relu()
            net = af.compile(h)
            xf = x32.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(xf))
            refr = _np_conv_stack(x32, Ws, pad, relu=True)
            add_row(wl, "ANE", "fp16", lat, gflops(flops, lat), relerr(out, refr))
            if energy:
                e = measure_energy(lambda: net(xf), seconds=5.0, tag="conv_ane")
                if e:
                    RESULTS[wl]["energy_ane"] = e
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        # MLX NHWC; weight [Cout,kH,kW,Cin]
        def mk(dt):
            xg = mx.array(np.transpose(x32, (0, 2, 3, 1)).astype(dt))
            Wg = [mx.array(np.transpose(w, (0, 2, 3, 1)).astype(dt)) for w in Ws]
            return xg, Wg
        for dt, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg, Wg = mk(dt)
            def run():
                hh = xg
                for w in Wg:
                    hh = mx.maximum(mx.conv2d(hh, w, stride=1, padding=pad), 0)
                return hh
            lat, out = mlx_run(run)
            outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
            refr = _np_conv_stack(x32, Ws, pad, relu=True)
            add_row(wl, "GPU", name, lat, gflops(flops, lat), relerr(outn, refr))
            if energy and dt == np.float16:
                e = measure_energy(lambda: mx.eval(run()), seconds=5.0, tag="conv_gpu")
                if e:
                    RESULTS[wl]["energy_gpu"] = e

    # CPU fp32 (numpy naive) - only for the SMALL conv; the big stack is too slow on CPU
    if depth * Cout * H * W <= 256 * 32 * 32 * 4:
        refr = _np_conv_stack(x32, Ws, pad, relu=True)
        lat, _ = cpu_run(lambda: _np_conv_stack(x32, Ws, pad, relu=True), reps=5)
        add_row(wl, "CPU", "fp32", lat, gflops(flops, lat), 0.0)
    else:
        note(wl, RESULTS[wl]["note"] + " CPU skipped (naive conv too slow at this size).")


def _np_conv_stack(x, Ws, pad, relu=False):
    """Naive fp32 NCHW conv stack reference (stride 1, same pad)."""
    h = x.astype(np.float32)
    for w in Ws:
        h = _np_conv2d(h, w, pad)
        if relu:
            h = np.maximum(h, 0.0)
    return h


def _np_conv2d(x, w, pad):
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # im2col
    cols = np.empty((N, Cin, kH, kW, H, W), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            cols[:, :, i, j] = xp[:, :, i:i + H, j:j + W]
    cols = cols.reshape(N, Cin * kH * kW, H * W)
    wf = w.reshape(Cout, Cin * kH * kW)
    out = np.einsum("oc,ncp->nop", wf, cols).reshape(N, Cout, H, W)
    return out


def w_dft(Nn):
    """DFT as matmul: y = F @ x, F[j,k] = exp(-2pi i jk/N). Real-valued split."""
    wl = f"DFT-as-matmul (N={Nn})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(Nn).astype(np.float64)
    n = np.arange(Nn)
    ang = -2 * np.pi * np.outer(n, n) / Nn
    Fr, Fi = np.cos(ang), np.sin(ang)            # real/imag of the DFT matrix
    ref_r = Fr @ x
    ref_i = Fi @ x
    flops = 2 * (2 * Nn * Nn)                     # two real matvecs
    note(wl, f"{flops/1e9:.3f} GFLOP. Split-real DFT matrix; ref = fp64. "
             f"Oscillatory matrix => cancellation, the fp16 stress case.")

    # represent as [1,N] @ [N,N] for each of Fr,Fi
    x1 = x.reshape(1, Nn)
    if HAVE_ANE:
        try:
            xin = af.input((1, Nn))
            netr = af.compile(xin.linear(Fr.astype(np.float16)))
            neti = af.compile(xin.linear(Fi.astype(np.float16)))
            xf = x1.astype(np.float16)
            lat = min_latency(lambda: (netr(xf), neti(xf)))
            yr, yi = netr(xf).ravel(), neti(xf).ravel()
            err = (relerr(yr, ref_r) + relerr(yi, ref_i)) / 2
            add_row(wl, "ANE", "fp16", lat, gflops(flops, lat), err)
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        for dt, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(x1.astype(dt))
            Frg, Fig = mx.array(Fr.T.astype(dt)), mx.array(Fi.T.astype(dt))
            lat, _ = mlx_run(lambda: (xg @ Frg) + (xg @ Fig))
            yr = np.array(xg @ Frg, copy=False).ravel()
            yi = np.array(xg @ Fig, copy=False).ravel()
            err = (relerr(yr, ref_r) + relerr(yi, ref_i)) / 2
            add_row(wl, "GPU", name, lat, gflops(flops, lat), err)

    lat, _ = cpu_run(lambda: (Fr.astype(np.float32) @ x.astype(np.float32),
                              Fi.astype(np.float32) @ x.astype(np.float32)))
    yr = (Fr.astype(np.float32) @ x.astype(np.float32))
    yi = (Fi.astype(np.float32) @ x.astype(np.float32))
    err = (relerr(yr, ref_r) + relerr(yi, ref_i)) / 2
    add_row(wl, "CPU", "fp32", lat, gflops(flops, lat), err)


def w_stencil(H, W, steps):
    """2D 5-point Laplacian stencil step, expressed as a 3x3 conv (the natural
    ANE form). u_{t+1} = u_t + dt * laplacian(u_t), `steps` chained."""
    wl = f"stencil 5pt ({H}x{W}, steps={steps})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(3)
    u0 = rng.standard_normal((1, 1, H, W)).astype(np.float32)
    dt = 0.1
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    K = np.zeros((1, 1, 3, 3), dtype=np.float32)
    K[0, 0] = dt * lap
    K[0, 0, 1, 1] += 1.0                  # identity + dt*lap fused into one 3x3
    flops = steps * 2 * 1 * 1 * 9 * H * W
    note(wl, f"{flops/1e9:.4f} GFLOP. Heat-eqn step as identity+dt*Laplacian 3x3 conv; "
             f"ref = fp64 numpy. Smooth/diffusive => fp16-friendly.")

    def np_step(u):
        u = u.astype(np.float64)
        for _ in range(steps):
            u = _np_conv2d(u.astype(np.float32), K, 1).astype(np.float64)
        return u
    ref = np_step(u0)

    if HAVE_ANE:
        try:
            xin = af.input((1, 1, H, W))
            h = xin
            for _ in range(steps):
                h = af.conv(h, K.astype(np.float16), stride=1, pad=1)
            net = af.compile(h)
            uf = u0.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(uf))
            add_row(wl, "ANE", "fp16", lat, gflops(flops, lat), relerr(out, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(np.transpose(u0, (0, 2, 3, 1)).astype(d))
            Kg = mx.array(np.transpose(K, (0, 2, 3, 1)).astype(d))
            def run():
                h = xg
                for _ in range(steps):
                    h = mx.conv2d(h, Kg, stride=1, padding=1)
                return h
            lat, out = mlx_run(run)
            outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
            add_row(wl, "GPU", name, lat, gflops(flops, lat), relerr(outn, ref))

    lat, _ = cpu_run(lambda: np_step(u0), reps=5)
    add_row(wl, "CPU", "fp32", lat, gflops(flops, lat), relerr(np_step(u0), ref))


def w_nbody(Np):
    """N-body pairwise step: for N particles in 3D, compute pairwise displacement
    norms (the O(N^2) cost core). Expressed as Gram-matrix math:
    D2[i,j] = |p_i|^2 + |p_j|^2 - 2 p_i.p_j  -> the 2 p p^T term is the GEMM."""
    wl = f"N-body pairwise (N={Np})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(4)
    P = rng.standard_normal((Np, 3)).astype(np.float64)
    G = P @ P.T                                   # [N,N] Gram = the GEMM core
    sq = np.sum(P * P, axis=1)
    ref = sq[:, None] + sq[None, :] - 2 * G       # squared distances
    flops = 2 * Np * Np * 3
    note(wl, f"{flops/1e9:.4f} GFLOP (Gram core). Pairwise sq-distance via P@P^T; "
             f"ref = fp64. Small inner dim (3) => mostly dispatch/bandwidth-bound.")

    P32 = P.astype(np.float32)
    if HAVE_ANE:
        try:
            # P @ P^T : linear wants W=[out,in]=[N,3] => W=P (so x@P.T = P@P.T)
            xin = af.input((Np, 3))
            net = af.compile(xin.linear(P32.astype(np.float16)))
            pf = P32.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(pf))
            sqf = np.sum(P32 * P32, axis=1)
            d2 = sqf[:, None] + sqf[None, :] - 2 * out.astype(np.float64)
            add_row(wl, "ANE", "fp16", lat, gflops(flops, lat), relerr(d2, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            Pg = mx.array(P32.astype(d))
            lat, out = mlx_run(lambda: Pg @ Pg.T)
            G2 = np.array(out, copy=False).astype(np.float64)
            sqf = np.sum(P32 * P32, axis=1)
            d2 = sqf[:, None] + sqf[None, :] - 2 * G2
            add_row(wl, "GPU", name, lat, gflops(flops, lat), relerr(d2, ref))

    lat, _ = cpu_run(lambda: P32 @ P32.T)
    G2 = (P32 @ P32.T).astype(np.float64)
    sqf = np.sum(P32 * P32, axis=1)
    d2 = sqf[:, None] + sqf[None, :] - 2 * G2
    add_row(wl, "CPU", "fp32", lat, gflops(flops, lat), relerr(d2, ref))


def w_vit_attention(SEQ=197, DIM=768, HEADS=12):
    """One ViT self-attention block (qkv proj -> MHA -> out proj), vit_demo shape.
    Random weights; ANE uses af.mha (decomposed softmax SDPA, fused e5rt)."""
    wl = f"ViT attention block (S={SEQ},D={DIM},H={HEADS})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(5)
    dh = DIM // HEADS
    sc = 1.0 / np.sqrt(DIM)
    def mkw(o, i): return (rng.standard_normal((o, i)).astype(np.float32) * sc)
    def mkb(o): return (rng.standard_normal(o).astype(np.float32) * 0.01)
    Wq, Wk, Wv, Wo = mkw(DIM, DIM), mkw(DIM, DIM), mkw(DIM, DIM), mkw(DIM, DIM)
    bq, bk, bv, bo = mkb(DIM), mkb(DIM), mkb(DIM), mkb(DIM)
    x = rng.standard_normal((SEQ, DIM)).astype(np.float32)

    # fp64 reference
    def ref_attn(xx, dt):
        xx = xx.astype(dt)
        def lin(a, W, b): return a @ W.astype(dt).T + b.astype(dt)
        q = lin(xx, Wq, bq).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        k = lin(xx, Wk, bk).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        v = lin(xx, Wv, bv).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        s = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s); a = a / a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
        return lin(o, Wo, bo)
    ref = ref_attn(x, np.float64)
    flops = 4 * 2 * SEQ * DIM * DIM + 2 * 2 * HEADS * SEQ * SEQ * dh
    note(wl, f"{flops/1e9:.3f} GFLOP. 4 proj GEMMs + attention scores/context. "
             f"ref = fp64. softmax is fp16-stable (wide accum).")

    if HAVE_ANE:
        try:
            xin = af.input((SEQ, DIM))
            y = af.mha(xin, Wq.astype(np.float16), bq, Wk.astype(np.float16), bk,
                       Wv.astype(np.float16), bv, Wo.astype(np.float16), bo, HEADS)
            net = af.compile(y)
            xf = x.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(xf))
            add_row(wl, "ANE", "fp16", lat, None, relerr(out, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")

    if HAVE_MLX:
        for d, name in ((np.float32, "fp32"), (np.float16, "fp16")):
            xg = mx.array(x.astype(d))
            Wqg, Wkg, Wvg, Wog = (mx.array(w.T.astype(d)) for w in (Wq, Wk, Wv, Wo))
            bqg, bkg, bvg, bog = (mx.array(b.astype(d)) for b in (bq, bk, bv, bo))
            def run():
                q = (xg @ Wqg + bqg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                k = (xg @ Wkg + bkg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                v = (xg @ Wvg + bvg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
                s = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
                a = mx.softmax(s, axis=-1)
                o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
                return o @ Wog + bog
            lat, out = mlx_run(run)
            add_row(wl, "GPU", name, lat, None, relerr(np.array(out, copy=False), ref))

    lat, _ = cpu_run(lambda: ref_attn(x, np.float32), reps=10)
    add_row(wl, "CPU", "fp32", lat, None, relerr(ref_attn(x, np.float32), ref))


def w_resnet18():
    """ResNet-18 ImageNet forward (af.load_resnet18). ANE vs torch-CPU/MPS."""
    wl = "ResNet-18 forward (1x3x224x224)"
    print(f"\n=== {wl} ===", flush=True)
    if not HAVE_TV:
        note(wl, "skipped - torchvision unavailable.")
        return
    rng = np.random.default_rng(6)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

    # torch CPU fp32 reference
    import torch
    tv = __import__("torchvision")
    m = tv.models.resnet18(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0].astype(np.float64)
    note(wl, "ref = torch-CPU fp32. ANE = aneforge fused conv graph (BN folded).")

    lat_cpu, _ = cpu_run(lambda: _torch_fwd(m, img, "cpu"), reps=5)
    add_row(wl, "CPU(torch)", "fp32", lat_cpu, None, 0.0)

    if torch.backends.mps.is_available():
        mm = m.to("mps")
        lat, out = min_latency_with_out(lambda: _torch_fwd(mm, img, "mps"), reps=15, warmup=5)
        add_row(wl, "GPU(MPS)", "fp32", lat, None, relerr(out, ref))

    if HAVE_ANE:
        try:
            clf = af.load_resnet18()
            lat, out = min_latency_with_out(lambda: clf(img), reps=15, warmup=5)
            add_row(wl, "ANE", "fp16", lat, None, relerr(out, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")


def _torch_fwd(model, img, dev):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(img).to(dev)
        o = model(t)
        if dev == "mps":
            torch.mps.synchronize()
        return o.detach().to("cpu").numpy()[0]


def w_minilm():
    """MiniLM sentence encoder (af.load). ANE fused encoder vs torch-CPU/MPS."""
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
        ref = hs.mean(0)
        ref = (ref / np.linalg.norm(ref)).astype(np.float64)
        lat_cpu, _ = cpu_run(lambda: _hf_embed(hf, ids), reps=5)
        add_row(wl, "CPU(torch)", "fp32", lat_cpu, None, 0.0)
    except Exception as e:
        print(f"  CPU ref: {type(e).__name__}: {e}")
        return

    if HAVE_ANE:
        try:
            enc = af.load(NAME)
            enc(text)  # warm/compile
            lat, _ = min_latency_with_out(lambda: enc(text), reps=15, warmup=3)
            out = enc(text)[0].astype(np.float64)
            add_row(wl, "ANE", "fp16", lat, None, relerr(out, ref))
        except Exception as e:
            print(f"  ANE: {type(e).__name__}: {e}")


def _hf_embed(hf, ids):
    import torch
    with torch.no_grad():
        v = hf(**ids).last_hidden_state[0].mean(0).numpy()
    return v / np.linalg.norm(v)

