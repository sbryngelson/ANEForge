"""DEMO: the cost model + autotuner pick a fast lowering without you measuring by hand.

Exercises:
  - af.estimate(out): a no-device microsecond prediction (the optimizer's pruner)
  - af.tune(out): compile several equivalent lowerings, measure on-device, keep the fastest
  - the cost model orders variants so the tuner skips ones predicted far worse

Run:  python3 examples/demos/optimization_autotune.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((64, 32, 3, 3)) * 0.05).astype(np.float32)
    x = af.input((1, 32, 24, 24))
    out = af.conv(x, W, pad=1).relu()

    print(f"af.estimate(out) = {af.estimate(out):.0f} us   (predicted, no device)")
    img = (rng.standard_normal((1, 32, 24, 24)) * 0.5).astype(np.float32)

    try:
        t = time.perf_counter()
        net = af.tune(out, budget=6, reps=15)        # measures variants, returns the fastest
        tune_s = time.perf_counter() - t
        for _ in range(20):
            net(img)
        K = 200
        t = time.perf_counter()
        for _ in range(K):
            net(img)
        us = (time.perf_counter() - t) / K * 1e6
        print(f"af.tune(...) -> fastest variant: {us:.0f} us/call  (autotuned in {tune_s:.1f}s)")
        net.release()
    except Exception as e:
        print(f"(af.tune unavailable here: {str(e).splitlines()[0][:60]})")
        net = af.compile(out)
        net.release()

    print("\nThe extracted analytic cost model predicts latency per chip with no device, and")
    print("af.tune uses it to prune the search so on-device measurement only checks the")
    print("promising lowerings - measurement-free ordering, measured final selection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
