"""Graph depth amortizes the dispatch floor: one submit, many ops. Run: python3 examples/demos/chaining_depth.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import warnings
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    print(f"{'depth':>6} | {'#ops fused':>10} | {'us/call':>8} | {'us/conv':>8}")
    print("-" * 42)
    for K in (1, 2, 8, 16, 32):
        x = af.input((1, 16, 16, 16))
        h = x
        for _ in range(K):
            Wt = (rng.standard_normal((16, 16, 3, 3)) * 0.05).astype(np.float32)
            h = af.conv(h, Wt, pad=1).relu()
        net = af.compile(h.mean((2, 3)))
        img = (rng.standard_normal((1, 16, 16, 16)) * 0.5).astype(np.float32)
        for _ in range(15):
            net(img)
        M = 200
        t = time.perf_counter()
        for _ in range(M):
            net(img)
        per_call = (time.perf_counter() - t) / M * 1e6
        print(f"{K:>6} | {net.n_ops:>10} | {per_call:>8.0f} | {per_call / K:>8.2f}")
        net.release()
    print("\nEvery depth compiles to ONE program (the op count is what's fused into it);")
    print("per-call latency stays ~flat, so per-op cost collapses.")
    print("A 32-layer model costs about the same per inference as a 1-layer model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
