"""Zero-copy I/O views skip the per-call host<->device memcpy. Run: python3 examples/demos/zero_copy_io.py"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 64
    W = (rng.standard_normal((64, 64, 3, 3)) * 0.05).astype(np.float32)
    x = af.input((N, 64, 16, 16))
    net = af.compile(af.conv(x, W, pad=1).relu())
    img16 = (rng.standard_normal((N, 64, 16, 16)) * 0.5).astype(np.float16)

    # correctness: zero-copy path matches the copy path exactly (same buffer)
    ref = np.asarray(net(img16.astype(np.float32))).astype(np.float64)
    v = net.input_view()
    np.copyto(v, img16)
    net.execute()
    got = np.asarray(net.output_view()).astype(np.float64)
    print(f"zero-copy vs eval max-abs diff: {np.abs(ref - got).max():.6f} (exact == same buffer)")

    # timing: __call__ (host copies) vs zero-copy (write view + execute + read view)
    for _ in range(30):
        net(img16.astype(np.float32))
    K = 400
    t = time.perf_counter()
    for _ in range(K):
        net(img16.astype(np.float32))
    copy_us = (time.perf_counter() - t) / K * 1e6
    v = net.input_view()
    t = time.perf_counter()
    for _ in range(K):
        np.copyto(v, img16); net.execute(); _ = net.output_view()
    zc_us = (time.perf_counter() - t) / K * 1e6
    print(f"copy path : {copy_us:6.0f} us/call")
    print(f"zero-copy : {zc_us:6.0f} us/call  ({(1 - zc_us / copy_us) * 100:.0f}% faster)")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
