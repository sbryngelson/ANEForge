"""Same matmul on ANE vs CPU vs GPU. Run: python3 examples/demos/ane_vs_gpu_cpu.py"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def _bench(fn, K):
    for _ in range(5):
        fn()
    t = time.perf_counter()
    for _ in range(K):
        fn()
    return (time.perf_counter() - t) / K * 1e6


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    M, Kd, N, K = 512, 2048, 2048, 50
    A = (rng.standard_normal((M, Kd)) * 0.05).astype(np.float32)
    B = (rng.standard_normal((Kd, N)) * 0.05).astype(np.float32)
    gflop = 2.0 * M * Kd * N / 1e9

    # ANE
    x = af.input((M, Kd)); net = af.compile(x @ B)
    ane_us = _bench(lambda: net(A), K); net.release()
    # CPU (numpy fp32)
    cpu_us = _bench(lambda: A @ B, K)
    print(f"{'backend':>10} | {'us/call':>9} | {'GFLOP/s':>8}")
    print("-" * 33)
    print(f"{'ANE':>10} | {ane_us:>9.0f} | {gflop/ane_us*1e6:>8.0f}")
    print(f"{'CPU numpy':>10} | {cpu_us:>9.0f} | {gflop/cpu_us*1e6:>8.0f}")
    # GPU (torch MPS), optional
    try:
        import torch
        if torch.backends.mps.is_available():
            ta = torch.tensor(A, device="mps"); tb = torch.tensor(B, device="mps")
            def g():
                (ta @ tb); torch.mps.synchronize()
            gpu_us = _bench(g, K)
            print(f"{'GPU mps':>10} | {gpu_us:>9.0f} | {gflop/gpu_us*1e6:>8.0f}")
    except Exception:
        print("(torch/MPS not available - install the [bench] extra for the GPU row)")
    print("\nPer-call the ANE pays its ~0.2 ms dispatch floor; its edge is efficiency (~1.5 W,")
    print("see 11) and offloading the CPU/GPU - feed it batched/chained work to amortize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
