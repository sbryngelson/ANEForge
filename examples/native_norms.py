"""Native ANE normalization ops - l2_norm / minmax_norm / lrn / scaled_elementwise.

l2_norm fuses as plain e5rt MIL (reduce_l2_norm + real_div - no graph cut). The
rest are native hardware layers Apple's public MIL/CoreML pipeline never emits
(MinMaxNormalization, LocalResponseNormalization, ScaledElementWise); each runs
as a netplist-bridge sub-program (a graph cut, like af.sdpa). The bridges in
aneforge/_bridges/ must be reachable on disk.

    python3 examples/native_norms.py
"""
import sys

from _common import report   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def main():
    rng = np.random.default_rng(0)
    ok = []

    # --- l2_norm over the last axis (fused e5rt MIL - no graph cut) -----------
    print("fused e5rt-MIL:")
    x = rng.standard_normal((4, 16)).astype(np.float16)
    net = af.compile(af.input((4, 16)).l2_norm(axis=-1))
    ref = x.astype(np.float32) / np.sqrt((x.astype(np.float32) ** 2).sum(-1, keepdims=True) + 1e-12)
    ok.append(report("l2_norm", net(x), ref, route="fused-MIL"))

    print("native netplist-bridge ops (graph cut, like af.sdpa):")

    # --- minmax_norm: per-row (Width) min-max normalization ------------------
    xm = rng.standard_normal((1, 2, 2, 4)).astype(np.float16)
    net = af.compile(af.minmax_norm(af.input((1, 2, 2, 4)), dimension="Width", eps=1e-4))
    xmf = xm.astype(np.float32)
    mn = xmf.min(axis=3, keepdims=True); mx = xmf.max(axis=3, keepdims=True)
    ok.append(report("minmax_norm", net(xm), (xmf - mn) / (mx - mn + 1e-4)))

    # --- lrn: cross-channel local response normalization ---------------------
    C, H, W = 5, 4, 4
    xl = np.arange(1, C * H * W + 1, dtype=np.float16).reshape(1, C, H, W)
    net = af.compile(af.lrn(af.input((1, C, H, W)), alpha=1.0, beta=0.75, k=1.0))
    xlf = xl.reshape(C, H, W).astype(np.float32)
    sq = (xlf ** 2).sum(axis=0, keepdims=True)
    ref = (xlf / (1.0 + 1.0 * sq) ** 0.75).reshape(1, C, H, W)
    ok.append(report("lrn", net(xl), ref, abserr=0.05))  # near-zero outputs: abs-error metric

    # --- scaled_elementwise: scale * (x OP z) --------------------------------
    xe = rng.standard_normal(8).astype(np.float16)
    ze = rng.standard_normal(8).astype(np.float16)
    net = af.compile(af.scaled_elementwise(af.input((8,)), af.input((8,)), op="Add", scale=2.0))
    ref = 2.0 * (xe.astype(np.float32) + ze.astype(np.float32))
    ok.append(report("scaled_elementwise", net(xe, ze), ref))

    print(f"\n{sum(ok)}/{len(ok)} native normalization ops correct on the ANE")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
