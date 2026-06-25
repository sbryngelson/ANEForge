"""DEMO: int8/compression is PER-CHIP gated - M1 folds it, M4+ streams it natively.

Reverse-engineering finding: weight compression is gated by the hardware Compression Engine,
which only arrives at M4. On M1 (h13) only int4-LUT streams natively; int8 and sparse FOLD to
dequant-at-the-engine - so int8 is a weight-SIZE / bandwidth lever (half the weight bytes,
helps bandwidth-bound layers), NOT a guaranteed compute speedup, and can even be slower on a
compute-bound layer because of the dequant overhead. This demo measures it directly so you
see the nuance rather than assuming int8 is always faster.

Run:  python3 examples/demos/weights_compression.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def _time_conv(Cin, Cout, HW, int8):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((Cout, Cin, 3, 3)) * 0.02).astype(np.float32)
    x = af.input((1, Cin, HW, HW))
    net = af.compile(af.conv(x, W, pad=1), **({"int8": True} if int8 else {}))
    img = (rng.standard_normal((1, Cin, HW, HW)) * 0.1).astype(np.float32)
    for _ in range(10):
        net(img)
    R = 80
    t = time.perf_counter()
    for _ in range(R):
        net(img)
    dt = (time.perf_counter() - t) / R * 1e6
    net.release()
    return dt


def main() -> int:
    warnings.filterwarnings("ignore")
    print(f"{'layer':>22} | {'fp16 us':>8} | {'int8 us':>8} | {'ratio':>6}")
    print("-" * 54)
    # a bandwidth-leaning layer (many channels) and a compute-heavy one
    for name, (Cin, Cout, HW) in [("BW-leaning 256ch 16x16", (256, 256, 16)),
                                  ("compute-heavy 512ch 32x32", (512, 512, 32))]:
        f = _time_conv(Cin, Cout, HW, int8=False)
        q = _time_conv(Cin, Cout, HW, int8=True)
        print(f"{name:>22} | {f:>8.0f} | {q:>8.0f} | {f/q:>6.2f}x")
    print("\nOn M1, int8 FOLDS to dequant (only int4-LUT streams) - so the ratio is ~1 or <1,")
    print("not a free speedup. int8's real M1 win is halving weight BYTES (bandwidth/footprint).")
    print("Native compressed-weight compute streaming arrives with the Compression Engine on M4+.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
