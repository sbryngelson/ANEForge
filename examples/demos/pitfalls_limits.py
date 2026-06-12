"""DEMO: the pitfalls and the hard limits - what to expect and design around.

Exercises:
  - the two compile-time signals: PrecisionWarning (fp16 cancellation) and
    DispatchFloorWarning (tiny dispatch-bound graph)
  - the hard limits the RE established, stated plainly so they're not surprises

Run:  python3 examples/demos/pitfalls_limits.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    rng = np.random.default_rng(0)

    # PITFALL 1: a dispatch-floor-bound graph (tiny work) -> DispatchFloorWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
        net = af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1).relu().mean((2, 3)))
        floor = any(issubclass(x.category, af.DispatchFloorWarning) for x in w)
    net.release()
    print(f"tiny graph -> DispatchFloorWarning: {floor}")

    # PITFALL 2: a signed reduce_sum over a long axis (narrow fp16 accumulator) -> PrecisionWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
        risky = af.compile(af.input((1, 4096)).sum((1,)))
        prec2 = any(issubclass(x.category, af.PrecisionWarning) for x in w)
    risky.release()
    print(f"signed reduce_sum over 4096 -> PrecisionWarning: {prec2}")

    print("\nHard limits the RE established (design around these, don't fight them):")
    print("  - ~0.2 ms fixed firmware round-trip per submit; a tiny call can't be faster")
    print("    (amortize: batch / chain / resident state - batching_amortization /")
    print("    chaining_depth / resident_state).")
    print("  - one request in-flight PER DIE; threads don't amortize one die (single_in_flight);")
    print("    use all dies via all-die eligibility on multi-die Macs.")
    print("  - fp16 compute + fp16 accumulator; cancellation needs care (numerics_fp16).")
    print("  - the compiled program is signature-gated: you optimize the MIL, not the .hwx.")
    print("  - large weights go bandwidth-bound at the compute->stream crossover")
    print("    (roofline_compute / roofline_bandwidth).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
