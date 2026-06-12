"""DEMO: ANE power efficiency - high throughput at very low watts.

Exercises:
  - sustaining a compute-bound matmul and measuring achieved fp16 throughput
  - estimating energy/op from the RE-measured ANE rail draw (~1.48 W sustained on M1)
  - the efficiency story: the ANE is a low-power accelerator, not a latency racer

Note: the exact rail draw needs `sudo powermetrics --samplers cpu_power` (prints "ANE Power");
this demo reports throughput and an energy estimate from the measured 1.48 W so it runs without
sudo. (RE measured: 0 W idle, ~0.75 W dispatch-bound N=1, ~1.48 W sustained N=512.)

Run:  python3 examples/demos/power_efficiency.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af

ANE_RAIL_W = 1.48          # measured sustained ANE-rail draw on M1 (powermetrics)


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    M, K, N = 1024, 4096, 4096
    A = af.input((M, K))
    Wt = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    net = af.compile(A.linear(Wt))
    img = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    for _ in range(8):
        net(img)
    R = 40
    t = time.perf_counter()
    for _ in range(R):
        net(img)
    dt = (time.perf_counter() - t) / R
    tflops = 2.0 * M * K * N / dt / 1e12
    pj_per_flop = ANE_RAIL_W * dt / (2.0 * M * K * N) * 1e12

    print(f"sustained matmul {M}x{K}x{N}: {tflops:.2f} TFLOP/s")
    print(f"at the measured ~{ANE_RAIL_W} W ANE rail: ~{pj_per_flop:.2f} pJ/FLOP")
    print(f"~{tflops*1e12/ANE_RAIL_W/1e9:.1f} GFLOP/s per watt")
    net.release()
    print("\nThe ANE trades latency (the fixed ~0.2 ms dispatch floor) for efficiency: when fed")
    print("(batched/chained), it sustains TFLOP/s at ~1.5 W. Measure the rail directly with")
    print("`sudo powermetrics --samplers cpu_power` during a sustained loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
