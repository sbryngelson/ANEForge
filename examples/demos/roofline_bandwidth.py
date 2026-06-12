"""DEMO: the ANE streaming-bandwidth ceiling - measured ~51 GB/s on M1.

Reverse-engineering finding: an M=1 matmul reads each weight exactly once (no reuse), so it
is bandwidth-bound. Measured ~51 GB/s at large sizes on M1 (~74% of the ~68 GB/s unified
memory), which matches the compiler's OWN internal GetEngineBwGbPerS = 50 GB/s constant. This
is the streaming regime weights fall into once they exceed on-chip (KMEM/L2) capacity.

Run:  python3 examples/demos/roofline_bandwidth.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    print(f"{'K':>6} {'N':>6} | {'us/call':>8} | {'weight MB':>9} | {'GB/s':>6}")
    print("-" * 44)
    for (K, N) in [(2048, 2048), (4096, 4096), (8192, 8192)]:
        A = af.input((1, K))
        Wt = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
        net = af.compile(A.linear(Wt))
        img = (rng.standard_normal((1, K)) * 0.1).astype(np.float32)
        for _ in range(8):
            net(img)
        R = 40
        t = time.perf_counter()
        for _ in range(R):
            net(img)
        dt = (time.perf_counter() - t) / R
        wbytes = K * N * 2          # fp16 weights, read once
        print(f"{K:>6} {N:>6} | {dt*1e6:>8.0f} | {wbytes/1e6:>9.1f} | {wbytes/dt/1e9:>6.1f}")
        net.release()
    print("\nSaturates ~51 GB/s (= the compiler's internal GetEngineBwGbPerS=50). Weights that")
    print("exceed on-chip capacity stream from DRAM at this rate -> the compute->BW crossover.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
