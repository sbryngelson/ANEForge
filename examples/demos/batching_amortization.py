"""DEMO: batching amortizes the fixed dispatch floor (per-sample cost collapses ~127x).

Reverse-engineering finding: per-CALL latency is ~flat in the batch size up to a point (the
~0.2 ms round-trip dominates), so per-SAMPLE cost falls roughly linearly toward the true
compute rate. Measured on M1: 196 us/sample at N=1 -> ~1.5 us/sample at N=512.

Run:  python3 examples/demos/batching_amortization.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import warnings
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")          # the floor warning is demonstrated in 01
    rng = np.random.default_rng(0)
    print(f"{'batch N':>8} | {'us/call':>8} | {'us/sample':>10}")
    print("-" * 32)
    for N in (1, 8, 32, 128, 512):
        W = (rng.standard_normal((16, 8, 3, 3)) * 0.1).astype(np.float32)
        x = af.input((N, 8, 16, 16))
        net = af.compile(af.conv(x, W, pad=1).relu().mean((2, 3)))
        img = (rng.standard_normal((N, 8, 16, 16)) * 0.5).astype(np.float32)
        for _ in range(20):
            net(img)
        K = 300
        t = time.perf_counter()
        for _ in range(K):
            net(img)
        per_call = (time.perf_counter() - t) / K * 1e6
        print(f"{N:>8} | {per_call:>8.0f} | {per_call / N:>10.2f}")
        net.release()
    print("\nPer-call latency barely moves; per-sample drops ~linearly. Batch to amortize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
