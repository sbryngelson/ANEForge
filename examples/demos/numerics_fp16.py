"""ANE numerics: fp16 compute, the cancellation pitfall + mitigation. Run: python3 examples/demos/numerics_fp16.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)

    # 1) ordinary fp16 op tracks fp32 closely
    x = af.input((1, 512))
    net = af.compile(x.gelu())
    a = (rng.standard_normal((1, 512)) * 2).astype(np.float32)
    got = np.asarray(net(a)).astype(np.float64)
    ref = 0.5 * a * (1 + np.vectorize(__import__("math").erf)(a / np.sqrt(2)))
    print(f"gelu fp16 vs fp32: relerr = {_common.relerr(got, ref):.2e}  (fp16-close, not exact)")
    net.release()

    # 2) the cancellation hazard: signed reduce_sum over a long axis -> PrecisionWarning
    big = af.input((1, 4096))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
        risky = af.compile(big.sum((1,)))             # narrow fp16 sum over 4096 -> precision risk
        flagged = any(issubclass(x.category, af.PrecisionWarning) for x in w)
    print(f"signed reduce_sum over 4096 compiled; PrecisionWarning raised: {flagged}")
    risky.release()

    print("\nfp16 is the compute type (cosine ~1, ~1e-3 relerr). aneforge structurally flags")
    print("cancellation-risk graphs (PrecisionWarning) and offers paired-fp16 / reduce->matmul")
    print("rewrites (af.paired, af.tune_precision) to recover accuracy where it matters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
