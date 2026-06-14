"""A1 vs A2 latency benchmark for af.sdpa at a fixed shape.

A1 = correctness-first subprocess-per-call netplist dispatch (the current path):
     every af.sdpa call spawns a one-shot ObjC probe that compiles+loads+maps the
     netplist, evaluates once, exits. Repeated calls pay that tax every time.

A2 = persistent Path-A worker (aneforge/_netplist_worker.py + the
     ane_persistent_worker probe): compiles+loads+maps ONCE, then services many
     evals over a pipe. Repeated calls pay only the eval + host<->surface memcpy.

We measure per-call wall latency under each, single-call and over a 32-call loop,
to show A2's amortization. A2 is selected by default; A1 is forced with
ANEFORGE_NETPLIST_WORKER=0.

    PYTHONPATH=. python3 examples/benchmarks/sdpa_worker_bench.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import aneforge as af

H, S, D = 8, 64, 32
N = 32


def make_inputs(rng):
    return tuple(rng.standard_normal((1, H, S, D)).astype(np.float16) for _ in range(3))


def time_path(label):
    """Compile a fresh SDPA model, then time N calls. The compiled-model
    lifetime is what the worker is cached against, so we build it here."""
    rng = np.random.default_rng(0)
    Q, K, V = make_inputs(rng)
    net = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)), af.input((1, H, S, D))))

    # First call (cold): under A2 this includes the one-time worker
    # compile+load+map; under A1 it's just the first subprocess spawn.
    t0 = time.perf_counter()
    out0 = net(Q, K, V)
    cold_ms = (time.perf_counter() - t0) * 1e3

    # Steady-state: time each subsequent call.
    per_call_ms = []
    for _ in range(N):
        t0 = time.perf_counter()
        net(Q, K, V)
        per_call_ms.append((time.perf_counter() - t0) * 1e3)
    per_call_ms = np.array(per_call_ms)

    loop_total_ms = per_call_ms.sum()
    net.release()
    print(f"  [{label}] cold first-call {cold_ms:8.1f} ms | "
          f"steady p50 {np.percentile(per_call_ms, 50):8.3f} ms | "
          f"min {per_call_ms.min():8.3f} ms | "
          f"{N}-call loop {loop_total_ms:8.1f} ms")
    return np.percentile(per_call_ms, 50), per_call_ms.min(), loop_total_ms, out0


def main():
    print(f"af.sdpa shape [1, {H}, {S}, {D}], {N} steady-state calls\n")

    os.environ["ANEFORGE_NETPLIST_WORKER"] = "0"
    a1_p50, a1_min, a1_loop, a1_out = time_path("A1 subprocess/call")

    os.environ["ANEFORGE_NETPLIST_WORKER"] = "1"
    a2_p50, a2_min, a2_loop, a2_out = time_path("A2 persistent worker")

    # numeric agreement between the two paths
    relerr = float(np.abs(a1_out.astype(np.float32) - a2_out.astype(np.float32)).max()
                   / (np.abs(a1_out).max() + 1e-6))

    print()
    print(f"  steady-state per-call speedup (p50): {a1_p50 / a2_p50:6.1f}x  "
          f"({a1_p50:.3f} ms -> {a2_p50:.3f} ms)")
    print(f"  steady-state per-call speedup (min): {a1_min / a2_min:6.1f}x  "
          f"({a1_min:.3f} ms -> {a2_min:.3f} ms)")
    print(f"  {N}-call loop speedup:               {a1_loop / a2_loop:6.1f}x  "
          f"({a1_loop:.1f} ms -> {a2_loop:.1f} ms)")
    print(f"  A1 vs A2 output relerr:              {relerr:.5f}")
    ok = a2_p50 < a1_p50 and relerr < 0.02
    print(f"\n  {'OK' if ok else 'FAIL'}: A2 {'beats' if a2_p50 < a1_p50 else 'does NOT beat'} A1 "
          f"per-call, numerics match")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
