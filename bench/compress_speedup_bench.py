"""Eval-latency win of compressed weight streaming (int4-LUT, sparse) vs fp16 on the ANE. Run: python3 bench/compress_speedup_bench.py"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

import aneforge as af

WARMUP, ITERS, REPS = 30, 200, 5
OUT_JSON = Path(__file__).resolve().parent / "results" / (Path(__file__).stem + ".json")


def _weights_bin_bytes(W, compress, x, **kw):
    import tempfile
    d = tempfile.mkdtemp(prefix="cbench_")
    af.compile(af.input(x.shape) @ W, compress=compress, build_dir=d, **kw).release()
    return Path(d, "weights.bin").stat().st_size


def _block_median(net, x):
    for _ in range(WARMUP):
        net(x)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        net(x)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def measure(W, x, compress, **kw):
    net = af.compile(af.input(x.shape) @ W, compress=compress, **kw)
    meds = [_block_median(net, x) for _ in range(REPS)]
    net.release()
    return {
        "median_ms": statistics.median(meds),
        "sd_ms": statistics.stdev(meds) if len(meds) > 1 else 0.0,
        "reps": meds,
        "weights_bin_bytes": _weights_bin_bytes(W, compress, x, **kw),
    }


def run_case(rng, IN, OUT, B, label, results):
    Wf = rng.standard_normal((IN, OUT)).astype(np.float32)
    Ws = Wf.copy()
    Ws[np.abs(Ws) < 1.0] = 0.0  # ~68% zeros
    sparsity = float((Ws == 0).mean())
    x = rng.standard_normal((B, IN)).astype(np.float32)

    fp16 = measure(Wf, x, None)
    int8 = measure(Wf, x, "int8")
    int4 = measure(Wf, x, "int4", compress_atol=0.5)
    sparse = measure(Ws, x, "sparse")
    row = {
        "label": label, "in": IN, "out": OUT, "batch": B,
        "fp16": fp16, "int8": int8, "int4": int4, "sparse": sparse,
        "sparsity": sparsity,
        "int8_speedup": fp16["median_ms"] / int8["median_ms"],
        "int4_speedup": fp16["median_ms"] / int4["median_ms"],
        "sparse_speedup": fp16["median_ms"] / sparse["median_ms"],
        "int4_vs_int8": int8["median_ms"] / int4["median_ms"],  # >1 => int4 faster than int8
        "int8_bytes_frac": int8["weights_bin_bytes"] / fp16["weights_bin_bytes"],
        "int4_bytes_frac": int4["weights_bin_bytes"] / fp16["weights_bin_bytes"],
        "sparse_bytes_frac": sparse["weights_bin_bytes"] / fp16["weights_bin_bytes"],
    }
    results.append(row)
    print(f"{label:24s} fp16 {fp16['median_ms']:.3f}  "
          f"int8 {int8['median_ms']:.3f} ({row['int8_speedup']:.2f}x)  "
          f"int4 {int4['median_ms']:.3f} ({row['int4_speedup']:.2f}x, vs-int8 {row['int4_vs_int8']:.2f}x)  "
          f"sparse {sparse['median_ms']:.3f} ({row['sparse_speedup']:.2f}x)")


def main():
    rng = np.random.default_rng(0)
    results = []
    print(f"# eval latency, median ms (B=batch), {REPS}x{ITERS} evals, warmup {WARMUP}")
    print("## weight-size sweep, batch 1 (bandwidth-bound regime)")
    for IN, OUT in [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 4096)]:
        run_case(rng, IN, OUT, 1, f"{IN}x{OUT} B=1", results)
    print("## dispatch-bound control (tiny weight)")
    run_case(rng, 256, 256, 1, "256x256 B=1", results)
    print("## batch sweep at 4096x4096 (does the win persist?)")
    for B in [8, 32, 64]:
        run_case(rng, 4096, 4096, B, f"4096x4096 B={B}", results)
    OUT_JSON.write_text(json.dumps({"warmup": WARMUP, "iters": ITERS, "reps": REPS,
                                    "results": results}, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
