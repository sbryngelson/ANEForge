"""Cross-path compressed-matmul speed: ANEForge (ANE fp16/int4-LUT) vs MLX (GPU fp16/4-bit). Run: PYTHONPATH=<repo> python3 bench/cross_path_compress_bench.py"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

import aneforge as af
import mlx.core as mx

WARMUP, ITERS, REPS = 25, 120, 5
GROUP = 64  # MLX 4-bit group size (N must be divisible)
OUT_JSON = Path(__file__).resolve().parent / "results" / (Path(__file__).stem + ".json")


def _cos(a, b):
    a, b = np.asarray(a).ravel().astype(np.float64), np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _med_ms(fn):
    for _ in range(WARMUP):
        fn()
    meds = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        meds.append((time.perf_counter() - t0) / ITERS * 1e3)
    return statistics.median(meds)


def ane(W, x, compress, **kw):
    net = af.compile(af.input(x.shape) @ W, compress=compress, **kw)
    out = net(x)
    ms = _med_ms(lambda: net(x))
    net.release()
    return ms, out


def gpu_fp16(W, x):
    Wm, xm = mx.array(W).astype(mx.float16), mx.array(x).astype(mx.float16)
    def run():
        y = xm @ Wm
        mx.eval(y)
        return y
    out = np.array(run().astype(mx.float32))
    return _med_ms(run), out


def gpu_int4(W, x):
    # quantize W (K,N) along its last dim N (group-affine, 4-bit); x @ W via quantized_matmul
    Wm, xm = mx.array(W).astype(mx.float16), mx.array(x).astype(mx.float16)
    wq, sc, bi = mx.quantize(Wm, bits=4, group_size=GROUP)
    def run():
        y = mx.quantized_matmul(xm, wq, scales=sc, biases=bi, transpose=False,
                                bits=4, group_size=GROUP)
        mx.eval(y)
        return y
    out = np.array(run().astype(mx.float32))
    return _med_ms(run), out


def run_case(rng, K, N, B, label, results):
    W = rng.standard_normal((K, N)).astype(np.float32)
    x = rng.standard_normal((B, K)).astype(np.float32)
    ref = (x @ W).astype(np.float32)

    a16_ms, a16 = ane(W, x, None)
    a4_ms, a4 = ane(W, x, "int4", compress_atol=0.5)
    g16_ms, g16 = gpu_fp16(W, x)
    g4_ms, g4 = gpu_int4(W, x)
    row = {
        "label": label, "K": K, "N": N, "B": B,
        "ane_fp16_ms": a16_ms, "ane_int4_ms": a4_ms,
        "gpu_fp16_ms": g16_ms, "gpu_int4_ms": g4_ms,
        "cos": {"ane_fp16": _cos(a16, ref), "ane_int4": _cos(a4, ref),
                "gpu_fp16": _cos(g16, ref), "gpu_int4": _cos(g4, ref)},
        "ane_int4_vs_gpu_int4": g4_ms / a4_ms,   # >1 => ANE int4 faster than GPU int4
        "ane_int4_vs_gpu_fp16": g16_ms / a4_ms,
    }
    results.append(row)
    print(f"{label:18s} ANE[fp16 {a16_ms:.3f} int4 {a4_ms:.3f}]  "
          f"GPU[fp16 {g16_ms:.3f} int4 {g4_ms:.3f}]  "
          f"ANE-int4/GPU-int4 {row['ane_int4_vs_gpu_int4']:.2f}x  "
          f"cos(a4 {row['cos']['ane_int4']:.3f} g4 {row['cos']['gpu_int4']:.3f})")


def main():
    rng = np.random.default_rng(0)
    results = []
    print(f"# cross-path matmul latency, median ms, {REPS}x{ITERS} evals, warmup {WARMUP}")
    print("## single-GEMV (B=1) weight-size sweep")
    for K, N in [(4096, 4096), (8192, 4096), (4096, 11008)]:
        run_case(rng, K, N, 1, f"{K}x{N} B=1", results)
    print("## batch sweep at 4096x4096 (MLX 4-bit degrades at batch; ANE int4 holds)")
    for B in [8, 16, 32, 64, 128]:
        run_case(rng, 4096, 4096, B, f"4096x4096 B={B}", results)
    OUT_JSON.write_text(json.dumps({"warmup": WARMUP, "iters": ITERS, "reps": REPS,
                                    "group_size": GROUP, "results": results}, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
