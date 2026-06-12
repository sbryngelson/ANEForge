"""DEMO: dispatch is single-in-flight per die - threads don't amortize one die.

Reverse-engineering finding (kernel decode): the ANEScheduler dispatches one firmware
command in-flight per die, so two threads submitting to the SAME die serialize (~1x). The
multi-die load balancer (M1 Max/Ultra) does dynamic least-busy steering, but only within a
program's eligible-die mask - so the real multi-die lever is making programs all-die-eligible.
This demo shows the single-die serialization (sequential vs 2-thread ~= same wall time).

Run:  python3 examples/demos/single_in_flight.py
"""
import sys, time, threading, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def _mk(seed):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((16, 8, 3, 3)) * 0.1).astype(np.float32)
    x = af.input((1, 8, 16, 16))
    net = af.compile(af.conv(x, W, pad=1).relu().mean((2, 3)))
    img = (rng.standard_normal((1, 8, 16, 16)) * 0.5).astype(np.float32)
    for _ in range(20):
        net(img)
    return net, img


def main() -> int:
    warnings.simplefilter("ignore")
    a, b, K = _mk(1), _mk(2), 600
    t = time.perf_counter()                       # sequential 2K calls
    for _ in range(K): a[0](a[1])
    for _ in range(K): b[0](b[1])
    seq = time.perf_counter() - t

    def run(net, img):
        for _ in range(K):
            net(img)
    t = time.perf_counter()                       # 2 threads, K each (ctypes drops the GIL)
    t1 = threading.Thread(target=run, args=a); t2 = threading.Thread(target=run, args=b)
    t1.start(); t2.start(); t1.join(); t2.join()
    con = time.perf_counter() - t

    print(f"sequential 2x{K}: {seq*1e3:6.0f} ms")
    print(f"concurrent 2x{K}: {con*1e3:6.0f} ms")
    print(f"speedup: {seq/con:.2f}x  -> {'SERIALIZED (single die)' if seq/con < 1.3 else 'parallel'}")
    print("\nOne in-flight per die. To use a 2nd die on M1 Max/Ultra, programs must be")
    print("created eligible on all dies so the balancer's least-busy steering can spread them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
