"""Per-call latency benchmark for aneforge's A2 persistent Path-A workers.

Measures steady-state per-call latency for topk (the per-row-tiled op), plus
sdpa and argmax as controls. topk's per-call cost is set by per-row tiling: the
native TopK keys all channels by one lane, so each of the C rows is its own
1-channel program and gets its own ANE evaluateWithQoS dispatch.

The A2 worker protocol is batched (a single request can carry N input sets and
run N evals in ONE pipe round-trip; see _netplist_worker.eval_batch). For topk,
however, the pipe round-trip is NOT the bottleneck: a single trip+eval is
~0.08 ms and the C sequential ANE dispatches dominate. Batching all C rows into
one round-trip removes only the (near-free) trips and actually regresses the
median (the worker's back-to-back eval loop loses the pipe pacing). So topk
keeps the per-call eval() dispatch. To see the batched path's effect directly:

    --batch   route topk through eval_batch (one round-trip) instead of per-call

    PYTHONPATH=. python3 examples/benchmarks/topk_worker_bench.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import aneforge as af


def bench(fn, *args, warmup=5, iters=32):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    dt = time.perf_counter() - t0
    return dt / iters * 1e3  # ms/call


def main():
    use_batch = "--batch" in sys.argv
    rng = np.random.default_rng(0)

    # topk over [C, W]: C row-tiles => C ANE dispatches.
    C, W, k = 16, 64, 2
    xt = rng.standard_normal((C, W)).astype(np.float16)
    net = af.compile(af.topk(af.input((C, W)), k))
    net(xt)  # build/load worker
    ref = np.sort(xt.astype(np.float32), 1)[:, ::-1][:, :k]

    if use_batch:
        # Drive the worker through eval_batch (all C rows in ONE round-trip) to
        # measure the batched-eval path directly.
        from aneforge import _netplist_worker as nw
        worker, _ = nw.build_worker("topk", (C, W), {"k": k, "largest": True})
        xa = np.ascontiguousarray(xt.reshape(C, W))

        def call():
            outs = worker.eval_batch([[xa[c]] for c in range(C)])
            return np.stack([o[0].reshape(k) for o in outs], axis=0)
        mode = "eval_batch (1 round-trip)"
    else:
        def call():
            return net(xt)
        mode = "per-call eval (C round-trips)"

    out = np.asarray(call(), np.float32)
    bit_ok = np.array_equal(out, ref) or float(np.abs(out - ref).max()) < 1e-3
    ms_single = bench(call, iters=1, warmup=2)
    ms_loop = bench(call, iters=32)
    print(f"topk   C={C} W={W} k={k}  rows={C}  bit_ok={bit_ok}  [{mode}]")
    print(f"  single call : {ms_single*1:.3f} ms")
    print(f"  32-call loop : {ms_loop:.3f} ms/call")

    # argmax control (single eval/call)
    Ca, Wa = 4, 8
    xa = rng.standard_normal((Ca, Wa)).astype(np.float16)
    neta = af.compile(af.input((Ca, Wa)).argmax(axis=-1))
    neta(xa)
    msa = bench(lambda: neta(xa), iters=32)
    print(f"argmax C={Ca} W={Wa}  32-call loop : {msa:.3f} ms/call")

    # sdpa control (single eval/call, big dense tensors)
    H, S, D = 4, 32, 16
    Q, K, V = (rng.standard_normal((1, H, S, D)).astype(np.float16) for _ in range(3))
    nets = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)), af.input((1, H, S, D))))
    nets(Q, K, V)
    mss = bench(lambda: nets(Q, K, V), iters=32)
    print(f"sdpa   H={H} S={S} D={D}  32-call loop : {mss:.3f} ms/call")


if __name__ == "__main__":
    main()
