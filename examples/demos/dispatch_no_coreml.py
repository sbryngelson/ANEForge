"""Dispatch straight to the ANE, without CoreML. Run: python3 examples/demos/dispatch_no_coreml.py"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((16, 8, 3, 3)) * 0.1).astype(np.float32)
    x = af.input((1, 8, 16, 16))

    t = time.perf_counter()
    net = af.compile(af.conv(x, W, pad=1).relu())
    compile_ms = (time.perf_counter() - t) * 1e3

    img = (rng.standard_normal((1, 8, 16, 16)) * 0.5).astype(np.float32)
    net(img)                                          # warm
    K = 200
    t = time.perf_counter()
    for _ in range(K):                                # compile-once, eval-many
        net(img)
    us = (time.perf_counter() - t) / K * 1e6

    print(f"compile (MIL -> ANECompiler -> e5rt program): {compile_ms:.0f} ms (once)")
    print(f"dispatch to ANE silicon: {us:.0f} us/call ({K} reuses of the same program)")
    print("\nNo CoreML / .mlmodel anywhere: aneforge emits MIL, the on-device ANECompiler")
    print("lowers it, and e5rt executes on the ANE from unentitled user space (see 03 for the")
    print("MIL, 08 for the entitlement boundary).")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
