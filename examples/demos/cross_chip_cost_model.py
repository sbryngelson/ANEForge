"""DEMO: the measurement-free cost model + cross-chip peak projection.

Reverse-engineering finding: the ANE compiler's analytic roofline was extracted and walked
for every chip. aneforge exposes af.estimate(out) (predicted microseconds, no device) and
af.project_peak(arch) (fp16 peak per chip). The anchors are dispatch-bound-regime effective
values (validated on M1 convs within +/-17%); the saturating ceilings are higher (see
05/06). This shows the predictor and the cross-generation scaling.

Run:  python3 examples/demos/cross_chip_cost_model.py
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
    W = (rng.standard_normal((32, 16, 3, 3)) * 0.05).astype(np.float32)
    x = af.input((1, 16, 32, 32))
    out = af.conv(x, W, pad=1).relu()

    est = float(af.estimate(out))
    net = af.compile(out)
    img = (rng.standard_normal((1, 16, 32, 32)) * 0.5).astype(np.float32)
    for _ in range(20):
        net(img)
    K = 300
    t = time.perf_counter()
    for _ in range(K):
        net(img)
    meas = (time.perf_counter() - t) / K * 1e6
    net.release()
    print(f"af.estimate(out) : {est:7.0f} us (predicted, no device)")
    print(f"measured         : {meas:7.0f} us/call")

    print("\ncross-chip fp16 peak projection (af.project_peak):")
    for arch in ("h13", "h14", "h17s"):
        try:
            p = af.project_peak(arch)
            print(f"  {arch:>5}: {p['tflops']:5.1f} TFLOP/s  ({p['rel_m1']:.1f}x M1, {p['cores']} cores)")
        except Exception as e:
            print(f"  {arch}: {e}")
    print("\n(h13=M1, h14=M2 Pro, h17s=M5. Anchors are effective/dispatch-bound; saturating")
    print(" ceilings ~4.8 TFLOP/s / ~51 GB/s on M1 are higher - see 05/06.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
