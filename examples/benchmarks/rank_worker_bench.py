"""A1 vs A2 latency benchmark for af.argmax and af.topk (native rank ops).

A1 = correctness-first subprocess-per-call netplist dispatch: every argmax/topk
     call spawns a one-shot ObjC probe (ane_invoke_probe_rank) that
     compiles+loads+maps the netplist, evaluates, exits. For topk that's ONE
     subprocess *per row* (C rows -> C spawns/call).

A2 = persistent Path-A worker (aneforge/_netplist_worker.py +
     ane_persistent_worker probe): compiles+loads+maps ONCE, then services many
     evals over a pipe. topk reuses ONE loaded 1-channel program, eval'd C times.

We measure per-call wall latency under each, single-call (cold) and over a
32-call steady-state loop, and assert A2 produces BIT-IDENTICAL outputs.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 examples/benchmarks/rank_worker_bench.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import aneforge as af

N = 32
ARGMAX_SHAPE = (8, 64)   # [C, W]
TOPK_SHAPE = (8, 64)
TOPK_K = 5


def time_path(label, build_model, inputs):
    """Compile a fresh model (worker is cached against the model lifetime),
    time a cold first call then N steady-state calls."""
    net = build_model()
    t0 = time.perf_counter()
    out0 = net(*inputs)
    cold_ms = (time.perf_counter() - t0) * 1e3
    per_call_ms = []
    for _ in range(N):
        t0 = time.perf_counter()
        net(*inputs)
        per_call_ms.append((time.perf_counter() - t0) * 1e3)
    per_call_ms = np.array(per_call_ms)
    loop_total_ms = per_call_ms.sum()
    workers = len(net._workers)
    net.release()
    print(f"  [{label}] workers={workers} cold {cold_ms:8.1f} ms | "
          f"steady p50 {np.percentile(per_call_ms, 50):8.3f} ms | "
          f"min {per_call_ms.min():8.3f} ms | "
          f"{N}-call loop {loop_total_ms:8.1f} ms")
    return np.percentile(per_call_ms, 50), per_call_ms.min(), loop_total_ms, out0


def bench(name, build_model, inputs):
    print(f"\n{name}:")
    os.environ["ANEFORGE_NETPLIST_WORKER"] = "0"
    a1_p50, a1_min, a1_loop, a1_out = time_path("A1 subprocess/call", build_model, inputs)
    os.environ["ANEFORGE_NETPLIST_WORKER"] = "1"
    a2_p50, a2_min, a2_loop, a2_out = time_path("A2 persistent worker", build_model, inputs)

    bit_identical = np.array_equal(a1_out.astype(np.float16), a2_out.astype(np.float16))
    print(f"  per-call speedup (p50): {a1_p50 / a2_p50:6.1f}x  "
          f"({a1_p50:.3f} -> {a2_p50:.3f} ms)")
    print(f"  per-call speedup (min): {a1_min / a2_min:6.1f}x  "
          f"({a1_min:.3f} -> {a2_min:.3f} ms)")
    print(f"  {N}-call loop speedup:  {a1_loop / a2_loop:6.1f}x  "
          f"({a1_loop:.1f} -> {a2_loop:.1f} ms)")
    print(f"  A1 vs A2 bit-identical: {bit_identical}")
    ok = a2_p50 < a1_p50 and bit_identical
    print(f"  {'OK' if ok else 'FAIL'}: A2 {'beats' if a2_p50 < a1_p50 else 'does NOT beat'} A1, "
          f"outputs {'match' if bit_identical else 'DIFFER'}")
    return ok


def main():
    rng = np.random.default_rng(0)
    C, W = ARGMAX_SHAPE
    xa = rng.standard_normal((C, W)).astype(np.float16)
    ok_argmax = bench(
        f"af.argmax axis=1, shape [{C}, {W}], {N} steady calls",
        lambda: af.compile(af.input(ARGMAX_SHAPE).argmax(axis=1)),
        (xa,))

    C, W = TOPK_SHAPE
    xt = rng.standard_normal((C, W)).astype(np.float16)
    ok_topk = bench(
        f"af.topk k={TOPK_K}, shape [{C}, {W}] ({C} per-row tiles), {N} steady calls",
        lambda: af.compile(af.topk(af.input(TOPK_SHAPE), TOPK_K, largest=True)),
        (xt,))

    print()
    return 0 if (ok_argmax and ok_topk) else 1


if __name__ == "__main__":
    sys.exit(main())
