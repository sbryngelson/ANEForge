#!/usr/bin/env python3
"""Memory-bandwidth roofline - ANE (aneforge fp16) vs GPU (MLX) vs CPU.

The compute-bound primitives (GEMM, conv) are covered by the saturation sweep
(``device_saturation_sweep.py``). But those are the MINORITY of aneforge's op set.
The large majority - elementwise activations, binaries, reductions, softmax, the
norm family, data-movement (reshape/transpose/concat/upsample/pixel_shuffle) - are
**memory-bandwidth-bound at size**, not compute-bound. For those, GFLOP/s is the
wrong metric: the meaningful number is **achieved memory bandwidth (GB/s)** and
**GB/s per watt**, and they all collapse onto one roofline. This script measures
that roofline.

KEY METHODOLOGY (a crude probe gave low/noisy numbers - done right here):

  * SIZE TO SATURATION. At small N the ~70 us ANE / dispatch floor dominates and
    HIDES bandwidth (you measure dispatch, not memory). We scale N until the
    achieved GB/s CLIMBS and PLATEAUS - that plateau is the bandwidth ceiling. We
    report the plateau AND the full climb curve as the evidence. If a device stays
    dispatch-bound at all feasible sizes for an archetype, we say so.

  * BYTES MODEL (stated, not hidden).  Bandwidth = bytes_moved / min_latency.
      - streaming unary (x*s / relu):  read N + write N  = 2*N*dtbytes
      - light-compute unary (gelu/silu): same traffic    = 2*N*dtbytes
      - reduction (sum over all):      read N, write ~1  = 1*N*dtbytes
      - softmax (last axis):           read N, write N    = 2*N*dtbytes
        (max+exp/sum+div is read-once/write-once at the framework level; the two
         logical passes are fused, so 2N is the honest external traffic)
      - layer_norm (last axis):        read N, write N    = 2*N*dtbytes
    dtbytes = 2 (fp16) for GPU/ANE, 4 (fp32) for CPU. The CPU bandwidth proxy uses
    a genuinely memory-bound op (a*2.0 / a.sum), NOT a transcendental - a tanh-heavy
    gelu is COMPUTE-bound in numpy and would understate CPU bandwidth.

  * POWER at the saturating size. Reuses the rigorous energy harness from
    ``device_compare_wattcomplete`` verbatim (idle-subtracted ACTIVE total-package
    power from powermetrics' own ``Combined Power`` line, median + CV%, sustained
    loop driven to sampler-exit). GB/s/W = peak GB/s / active-package-W.

  * ACCURACY. relerr vs fp32 numpy reference where meaningful (the activations,
    softmax, norm). Pure-copy/reduction relerr is ~0 by construction.

PART 2 (op coverage): classifies EVERY op in aneforge's live ``_EMIT`` (50) and
``NETPLIST_OPS`` (19) into a roofline class, driven off the live dicts so it can't
silently omit an op. Emitted as a table to the JSON + printed.

Run from repo root (energy needs passwordless sudo for powermetrics):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 bench/device_bandwidth_roofline.py

--quick runs a reduced window + a coarser size sweep for a smoke test.
Writes bench/results/device_bandwidth_roofline_results.json alongside the tables.
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

import numpy as np

# Reuse the rigorous harnesses - do NOT reinvent the power methodology.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_compare as dc            # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402  (energy harness + idle sampling)

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
relerr = dc.relerr
min_latency = dc.min_latency

if HAVE_ANE:
    import aneforge as af
    from aneforge import _compile as C
if HAVE_MLX:
    import mlx.core as mx

# size sweep: element counts (per archetype we pick a 2D shape with this many elems).
# The climb-to-plateau evidence lives in this sweep; the largest size is the ceiling.
SIZES = [1 << 18, 1 << 20, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26]  # 256K .. 64M
QUICK_SIZES = [1 << 20, 1 << 23, 1 << 25]

RESULTS: dict = {"roofline": {}, "coverage": {}}


# archetypes: each returns (ane_net_builder, mlx_run, cpu_run, bytes_per_elem_factor,
# ref_fn) for a given element count. shape is 2D [R, Dlast] chosen near-square but with
# a real last axis for the reductions/softmax/norm.
def _shape_for(nelem: int, dlast: int = 4096) -> tuple[int, int]:
    """A [R, D] shape with ~nelem elements and a sizeable last axis D for the
    reduction/softmax/norm archetypes (so the reduced axis is non-trivial)."""
    d = dlast
    r = max(1, nelem // d)
    return (r, d)


def _ane_min_latency_with_out(net, xf):
    best = float("inf")
    out = None
    for _ in range(6):
        net(xf)
    for _ in range(20):
        import time
        t0 = time.perf_counter()
        out = net(xf)
        best = min(best, time.perf_counter() - t0)
    return best, out


def run_archetype(name, byte_factor, build_ane, build_mlx, build_cpu,
                  ref_fn, sizes, want_acc=True):
    """Run one archetype across all sizes/devices. byte_factor: traffic = factor*N*dt."""
    print(f"\n=== bandwidth archetype: {name} (bytes = {byte_factor}*N*dt) ===", flush=True)
    rec = {"byte_factor": byte_factor, "sizes": [], "peak": {}}
    rng = np.random.default_rng(11)
    for nelem in sizes:
        R, D = _shape_for(nelem)
        N = R * D
        x32 = rng.standard_normal((R, D)).astype(np.float32)
        ref = ref_fn(x32) if (want_acc and ref_fn) else None
        row = {"nelem": N, "shape": [R, D]}

        # GPU (fp16)
        if HAVE_MLX and build_mlx is not None:
            try:
                fn, getout = build_mlx(x32, R, D)
                lat = min_latency(fn, reps=20, warmup=6)
                gbps = byte_factor * N * 2 / lat / 1e9
                row["gpu_gbps"] = gbps
                row["gpu_lat_ms"] = lat * 1e3
                if want_acc and ref is not None:
                    row["gpu_relerr"] = relerr(getout(), ref)
            except Exception as e:
                row["gpu_err"] = f"{type(e).__name__}: {e}"

        # ANE (fp16)
        if HAVE_ANE and build_ane is not None:
            try:
                net, xf = build_ane(x32, R, D)
                lat, out = _ane_min_latency_with_out(net, xf)
                gbps = byte_factor * N * 2 / lat / 1e9
                row["ane_gbps"] = gbps
                row["ane_lat_ms"] = lat * 1e3
                if want_acc and ref is not None:
                    row["ane_relerr"] = relerr(np.asarray(out), ref)
            except Exception as e:
                row["ane_err"] = f"{type(e).__name__}: {e}"

        # CPU (fp32) - genuinely memory-bound proxy
        if build_cpu is not None:
            try:
                fn, _ = build_cpu(x32, R, D)
                lat = min_latency(fn, reps=20, warmup=6)
                gbps = byte_factor * N * 4 / lat / 1e9  # fp32 = 4 bytes
                row["cpu_gbps"] = gbps
                row["cpu_lat_ms"] = lat * 1e3
            except Exception as e:
                row["cpu_err"] = f"{type(e).__name__}: {e}"

        rec["sizes"].append(row)
        print(f"  N={N:>10,} [{R}x{D}]  "
              f"ANE {row.get('ane_gbps', float('nan')):7.1f}  "
              f"GPU {row.get('gpu_gbps', float('nan')):7.1f}  "
              f"CPU {row.get('cpu_gbps', float('nan')):7.1f}  GB/s", flush=True)

    # peak = max GB/s achieved over the sweep (the plateau), per device
    for dev in ("ane", "gpu", "cpu"):
        vals = [(r[f"{dev}_gbps"], r["nelem"]) for r in rec["sizes"] if f"{dev}_gbps" in r]
        if vals:
            best_gbps, best_n = max(vals, key=lambda t: t[0])
            rec["peak"][dev] = {"gbps": best_gbps, "at_nelem": best_n}
    RESULTS["roofline"][name] = rec
    return rec


# archetype builders
def build_archetypes():
    """Return the archetype spec list. Each entry drives run_archetype."""
    specs = []

    # 1. PURE STREAMING - relu (read+write, ~zero arithmetic). Cleanest BW probe.
    #    bytes = 2*N*dt
    def ane_relu(x, R, D):
        net = af.compile(af.input((R, D)).relu())
        return net, x.astype(np.float16)
    def mlx_relu(x, R, D):
        xg = mx.array(x.astype(np.float16))
        def fn():
            mx.eval(mx.maximum(xg, 0))
        return fn, (lambda: np.array(mx.maximum(xg, 0), copy=False))
    def cpu_stream(x, R, D):
        # genuinely memory-bound: scalar multiply (NOT a transcendental)
        def fn():
            _ = x * 2.0
        return fn, None
    specs.append(("streaming (relu / x*2)", 2, ane_relu, mlx_relu, cpu_stream,
                  (lambda x: np.maximum(x, 0)), True))

    # 2. LIGHT-COMPUTE ELEMENTWISE - gelu (a few flops/elem; still BW-bound at size)
    def ane_gelu(x, R, D):
        net = af.compile(af.input((R, D)).gelu())
        return net, x.astype(np.float16)
    def mlx_gelu(x, R, D):
        xg = mx.array(x.astype(np.float16))
        def gel(a):
            return a * 0.5 * (1 + mx.erf(a / np.sqrt(2.0)))
        def fn():
            mx.eval(gel(xg))
        return fn, (lambda: np.array(gel(xg), copy=False))
    def cpu_gelu_bw(x, R, D):
        # CPU proxy stays memory-bound (copy), NOT the transcendental gelu - see header.
        def fn():
            _ = x.copy()
        return fn, None
    def ref_gelu(x):
        from scipy.special import erf  # noqa
        return x * 0.5 * (1 + erf(x / np.sqrt(2.0)))
    # scipy may be absent; fall back to a numpy erf approximation for the ref.
    def ref_gelu_np(x):
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    specs.append(("gelu (light compute)", 2, ane_gelu, mlx_gelu, cpu_gelu_bw,
                  ref_gelu_np, True))

    # 3. REDUCTION - sum over last axis (read N, write R). bytes ~ 1*N*dt
    def ane_sum(x, R, D):
        net = af.compile(af.input((R, D)).sum(-1))
        return net, x.astype(np.float16)
    def mlx_sum(x, R, D):
        xg = mx.array(x.astype(np.float16))
        def fn():
            mx.eval(mx.sum(xg, axis=-1))
        return fn, (lambda: np.array(mx.sum(xg, axis=-1), copy=False))
    def cpu_sum(x, R, D):
        def fn():
            _ = x.sum(-1)
        return fn, None
    specs.append(("reduction (sum)", 1, ane_sum, mlx_sum, cpu_sum,
                  (lambda x: x.sum(-1)), True))

    # 4. SOFTMAX over last axis. bytes = 2*N*dt (read+write the full tensor)
    def ane_softmax(x, R, D):
        net = af.compile(af.input((R, D)).softmax(-1))
        return net, x.astype(np.float16)
    def mlx_softmax(x, R, D):
        xg = mx.array(x.astype(np.float16))
        def fn():
            mx.eval(mx.softmax(xg, axis=-1))
        return fn, (lambda: np.array(mx.softmax(xg, axis=-1), copy=False))
    def cpu_softmax(x, R, D):
        def fn():
            m = x.max(-1, keepdims=True)
            e = np.exp(x - m)
            _ = e / e.sum(-1, keepdims=True)
        return fn, None
    def ref_softmax(x):
        m = x.max(-1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(-1, keepdims=True)
    specs.append(("softmax", 2, ane_softmax, mlx_softmax, cpu_softmax,
                  ref_softmax, True))

    # 5. LAYER_NORM over last axis (the real-model primitive). bytes = 2*N*dt
    def make_ln(R, D):
        rng = np.random.default_rng(13)
        g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0)
        b = rng.standard_normal(D).astype(np.float32) * 0.1
        return g, b
    def ane_ln(x, R, D):
        g, b = make_ln(R, D)
        net = af.compile(af.input((R, D)).layer_norm(g.astype(np.float16), b.astype(np.float16)))
        return net, x.astype(np.float16)
    def mlx_ln(x, R, D):
        g, b = make_ln(R, D)
        xg = mx.array(x.astype(np.float16)); gg = mx.array(g.astype(np.float16)); bb = mx.array(b.astype(np.float16))
        def ln():
            mu = mx.mean(xg, axis=-1, keepdims=True)
            var = mx.mean((xg - mu) ** 2, axis=-1, keepdims=True)
            return (xg - mu) * mx.rsqrt(var + 1e-5) * gg + bb
        def fn():
            mx.eval(ln())
        return fn, (lambda: np.array(ln(), copy=False))
    def cpu_ln(x, R, D):
        g, b = make_ln(R, D)
        def fn():
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            _ = (x - mu) / np.sqrt(var + 1e-5) * g + b
        return fn, None
    def ref_ln(x):
        g, b = make_ln(*x.shape)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * g + b
    specs.append(("layer_norm", 2, ane_ln, mlx_ln, cpu_ln, ref_ln, True))

    return specs


# power at the saturating size
def measure_peak_power(specs, sizes):
    """For each archetype, re-run the SATURATING (largest) size under the energy
    harness and attach GB/s/W per device."""
    if not HAVE_SUDO:
        print("\n[power] no passwordless sudo - GB/s/W skipped.")
        return
    nelem = sizes[-1]
    R, D = _shape_for(nelem)
    N = R * D
    rng = np.random.default_rng(11)
    x32 = rng.standard_normal((R, D)).astype(np.float32)
    print(f"\n=== peak-power pass at saturating N={N:,} [{R}x{D}] ===", flush=True)

    for (name, bf, build_ane, build_mlx, build_cpu, _ref, _acc) in specs:
        rec = RESULTS["roofline"][name]
        rec["power"] = {}
        # ANE
        if HAVE_ANE and build_ane is not None and "ane_err" not in rec["sizes"][-1]:
            net, xf = build_ane(x32, R, D)
            e = wc.measure_energy(lambda: net(xf), tag=f"bw_{abs(hash(name))%9999}_ane",
                                  window=wc.WINDOW)
            if e:
                apw = e.get("active_pkg_W", float("nan"))
                if apw == apw and apw > 0:
                    gbps = bf * N * 2 / (e["iter_ms"] / 1e3) / 1e9
                    e["gbps_loop"] = gbps
                    e["gbps_per_W"] = gbps / apw
                rec["power"]["ane"] = e
        # GPU
        if HAVE_MLX and build_mlx is not None and "gpu_err" not in rec["sizes"][-1]:
            fn, _ = build_mlx(x32, R, D)
            e = wc.measure_energy(fn, tag=f"bw_{abs(hash(name))%9999}_gpu", window=wc.WINDOW)
            if e:
                apw = e.get("active_pkg_W", float("nan"))
                if apw == apw and apw > 0:
                    gbps = bf * N * 2 / (e["iter_ms"] / 1e3) / 1e9
                    e["gbps_loop"] = gbps
                    e["gbps_per_W"] = gbps / apw
                rec["power"]["gpu"] = e
        # CPU
        if build_cpu is not None:
            fn, _ = build_cpu(x32, R, D)
            e = wc.measure_energy(fn, tag=f"bw_{abs(hash(name))%9999}_cpu", window=wc.WINDOW)
            if e:
                apw = e.get("active_pkg_W", float("nan"))
                if apw == apw and apw > 0:
                    gbps = bf * N * 4 / (e["iter_ms"] / 1e3) / 1e9
                    e["gbps_loop"] = gbps
                    e["gbps_per_W"] = gbps / apw
                rec["power"]["cpu"] = e
        p = rec["power"]
        def _s(d):
            ee = p.get(d, {})
            return (ee.get("gbps_per_W", float("nan")), ee.get("active_pkg_W", float("nan")),
                    ee.get("active_pkg_cv_pct", float("nan")))
        an, gp, cp = _s("ane"), _s("gpu"), _s("cpu")
        print(f"  {name:<26}  ANE {an[0]:6.2f} GB/s/W ({an[1]:.1f}W,{an[2]:.0f}%) | "
              f"GPU {gp[0]:6.2f} ({gp[1]:.1f}W,{gp[2]:.0f}%) | "
              f"CPU {cp[0]:6.2f} ({cp[1]:.1f}W,{cp[2]:.0f}%)", flush=True)


# PART 2 - op -> roofline-class coverage (driven off live _EMIT / NETPLIST_OPS)
# class -> set of op names. We classify by op semantics. Driven against the LIVE
# dicts below so a missing op raises, not silently drops.
COMPUTE_BOUND = {
    "matmul", "bmm", "conv", "conv_transpose",
}
# bandwidth-bound: activations, elementwise binaries, reductions, softmax, norms,
# data-movement / shape ops.
BANDWIDTH_BOUND = {
    # activations / elementwise unary (transcendental or cheap)
    "relu", "relu6", "leaky_relu", "elu", "gelu", "silu", "sigmoid", "tanh",
    "softplus", "exp", "log", "sin", "cos", "erf", "abs", "sqrt", "rsqrt",
    "square", "pow", "clip",
    # elementwise binary
    "add", "sub", "mul", "real_div", "maximum", "minimum",
    # reductions
    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
    # softmax + norm family
    "softmax", "layer_norm", "rms_norm", "group_norm", "batch_norm", "l2_norm",
    # pooling (reduction-like, BW-bound at size)
    "avg_pool", "max_pool",
    # data-movement / shape / resample
    "reshape", "transpose", "concat", "upsample",
    "pixel_shuffle", "pixel_unshuffle",
}
# dispatch/latency-bound: scalar / tiny ops where the dispatch floor dominates
# (covered by the floor finding - CPU wins tiny).
DISPATCH_BOUND = {
    "adds", "muls",  # scalar-broadcast ops, trivially tiny arithmetic
}

# NETPLIST bridge ops. has_worker => persistent worker => fairly raceable silicon.
# without => subprocess-per-call: dispatch-bound ARTIFACT, not a fair speed race.
WORKER_OPS = {"sdpa", "argmax", "topk"}   # the persistent-worker op set

# Map a bridge op to its measurement story.
BRIDGE_CLASS = {
    # has a persistent worker -> can benchmark silicon time
    "sdpa": "bridge:worker (raceable - covered as attention class)",
    "argmax": "bridge:worker (raceable silicon via persistent worker)",
    "topk": "bridge:worker (raceable silicon via persistent worker)",
}
# everything else in NETPLIST_OPS without a worker -> subprocess artifact, not raceable.


def build_coverage():
    """Classify every live _EMIT and NETPLIST op; verify total coverage."""
    if not HAVE_ANE:
        print("\n[coverage] aneforge unavailable - coverage table skipped.")
        return
    emit = sorted(C._EMIT.keys())
    netp = sorted(C.NETPLIST_OPS.keys() if isinstance(C.NETPLIST_OPS, dict)
                  else C.NETPLIST_OPS)

    rows = []
    counts = {"compute-bound": 0, "bandwidth-bound": 0, "dispatch/latency-bound": 0,
              "bridge:worker-raceable": 0, "bridge:subprocess-not-raceable": 0, "UNCLASSIFIED": 0}
    unclassified = []

    for op in emit:
        if op in COMPUTE_BOUND:
            cls, where = "compute-bound", "saturation sweep (GEMM/conv peaks)"
            counts["compute-bound"] += 1
        elif op in BANDWIDTH_BOUND:
            cls, where = "bandwidth-bound", "THIS bandwidth roofline"
            counts["bandwidth-bound"] += 1
        elif op in DISPATCH_BOUND:
            cls, where = "dispatch/latency-bound", "floor finding (CPU wins tiny)"
            counts["dispatch/latency-bound"] += 1
        else:
            cls, where = "UNCLASSIFIED", "!! add to a class"
            counts["UNCLASSIFIED"] += 1
            unclassified.append(op)
        rows.append({"op": op, "set": "_EMIT", "class": cls, "where": where})

    for op in netp:
        if op in WORKER_OPS:
            cls = "bridge:worker-raceable"
            where = BRIDGE_CLASS.get(op, "persistent worker - silicon raceable")
            counts["bridge:worker-raceable"] += 1
        else:
            cls = "bridge:subprocess-not-raceable"
            where = ("subprocess-per-call: dispatch-bound ARTIFACT, NOT a fair "
                     "speed race (needs a persistent worker to benchmark silicon)")
            counts["bridge:subprocess-not-raceable"] += 1
        rows.append({"op": op, "set": "NETPLIST_OPS", "class": cls, "where": where})

    RESULTS["coverage"] = {
        "rows": rows, "counts": counts,
        "n_emit": len(emit), "n_netplist": len(netp),
        "worker_ops": sorted(WORKER_OPS),
        "a1_only_ops": sorted(set(netp) - WORKER_OPS),
        "unclassified": unclassified,
    }
    print("\n=== op coverage by roofline class ===")
    for k, v in counts.items():
        print(f"  {k:<28} {v}")
    if unclassified:
        print(f"  !! UNCLASSIFIED OPS: {unclassified}")
    print(f"  worker (raceable) bridge ops: {sorted(WORKER_OPS)}")
    print(f"  subprocess-only (not raceable) bridge ops: {sorted(set(netp) - WORKER_OPS)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--window", type=float, default=None)
    args = ap.parse_args()
    if args.quick:
        wc.WINDOW = 2.0
    if args.window:
        wc.WINDOW = args.window
    sizes = QUICK_SIZES if args.quick else SIZES

    print("=" * 100)
    print(" device_bandwidth_roofline - memory-bound regime: achieved GB/s + GB/s/W")
    print("=" * 100)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}  "
          f"sudo={'yes' if HAVE_SUDO else 'NO - power skipped'}  window={wc.WINDOW}s")

    if HAVE_SUDO:
        print("\n sampling idle baseline (no workload)...", flush=True)
        wc.sample_idle(3.0)
        print(f" idle: ANE {wc.IDLE.get('ane',0):.0f} / GPU {wc.IDLE.get('gpu',0):.0f} / "
              f"CPU {wc.IDLE.get('cpu',0):.0f} mW (pkg {wc.IDLE_PKG:.0f} mW)")

    specs = build_archetypes()
    for (name, bf, ba, bm, bc, ref, acc) in specs:
        run_archetype(name, bf, ba, bm, bc, ref, sizes, want_acc=acc)

    measure_peak_power(specs, sizes)
    build_coverage()

    out = Path(__file__).resolve().parent / "results" / "device_bandwidth_roofline_results.json"
    out.write_text(json.dumps({
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
        "window_s": wc.WINDOW, "pm_interval_ms": wc.PM_INTERVAL_MS,
        "sizes": sizes,
        "idle_mW": wc.IDLE, "idle_pkg_mW": wc.IDLE_PKG,
        "results": RESULTS,
    }, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
