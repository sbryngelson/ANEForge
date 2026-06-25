"""The ANE fp16 compute ceiling: ~4.8 TFLOP/s on M1. Run: python3 examples/demos/roofline_compute.py"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'us/call':>8} | {'GFLOP':>7} | {'TFLOP/s':>8}")
    print("-" * 50)
    for (M, K, N) in [(256, 2048, 2048), (512, 4096, 4096), (2048, 4096, 4096)]:
        A = af.input((M, K))
        Wt = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)   # linear: y = x @ Wt^T
        net = af.compile(A.linear(Wt))
        img = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        for _ in range(8):
            net(img)
        R = 40
        t = time.perf_counter()
        for _ in range(R):
            net(img)
        dt = (time.perf_counter() - t) / R
        flop = 2.0 * M * K * N
        print(f"{M:>6} {K:>6} {N:>6} | {dt*1e6:>8.0f} | {flop/1e9:>7.1f} | {flop/dt/1e12:>8.2f}")
        net.release()
    print("\nPeak ~4.8 TFLOP/s fp16 (4096-dim). The cost-model anchor (1.8/3.25) is the")
    print("dispatch-bound effective rate, deliberately lower; this is the saturating ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
