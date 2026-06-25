"""The ANE dispatch floor: every call pays a fixed firmware round-trip. Run: python3 examples/demos/execution_model_floor.py"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
    x = af.input((1, 8, 16, 16))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net = af.compile(af.conv(x, W, pad=1).relu().mean((2, 3)))
        floor_warned = any(issubclass(x.category, af.DispatchFloorWarning) for x in w)

    img = (rng.standard_normal((1, 8, 16, 16)) * 0.5).astype(np.float32)
    for _ in range(30):           # warm
        net(img)
    K = 500
    t = time.perf_counter()
    for _ in range(K):
        net(img)
    us = (time.perf_counter() - t) / K * 1e6

    print(f"tiny conv->relu->gap : {us:6.0f} us/call")
    print(f"DispatchFloorWarning fired at compile: {floor_warned}")
    print("\nThe ~0.2 ms is the fixed firmware round-trip, not the compute (which is ~us).")
    print("It does not shrink with smaller work - amortize it (see batching_amortization / chaining_depth).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
