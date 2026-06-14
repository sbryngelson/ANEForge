"""Native ANE pixel-rearrange + layout ops - every lossless data-movement layer.

pixel_shuffle / pixel_unshuffle fuse as plain e5rt MIL (no graph cut). The rest
are native hardware layers Apple's public MIL/CoreML pipeline never emits
(SpaceToChannel, ChannelToSpace, SpaceToBatch, BatchToSpace, Flatten, InputView,
DynamicSlice); each runs as a netplist-bridge sub-program (a graph cut, like
af.sdpa). Inputs are integer-valued fp16 so any mismatch is a true permutation
error, not rounding. The bridges in aneforge/_bridges/ must be reachable on disk.

    python3 examples/native_pixel_ops.py
"""
import sys

from _common import report   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


# numpy references matching each op's exact convention
def ref_space_to_depth_tf(x, r):
    N, C, H, W = x.shape
    out = np.zeros((N, C * r * r, H // r, W // r), np.float32)
    for c in range(C):
        for fy in range(r):
            for fx in range(r):
                out[:, (fy * r + fx) * C + c] = x[:, c, fy::r, fx::r]
    return out


def ref_depth_to_space_tf(x, r):
    N, C2, H, W = x.shape
    C = C2 // (r * r)
    out = np.zeros((N, C, H * r, W * r), np.float32)
    for c in range(C):
        for fy in range(r):
            for fx in range(r):
                out[:, c, fy::r, fx::r] = x[:, (fy * r + fx) * C + c]
    return out


def ref_space_to_batch(x, bh, bw):
    N, C, H, W = x.shape
    out = np.zeros((N * bh * bw, C, H // bh, W // bw), np.float32)
    for n in range(N):
        for i in range(bh):
            for j in range(bw):
                out[(n * bh + i) * bw + j] = x[n, :, i::bh, j::bw]
    return out


def ref_batch_to_space(x, bh, bw):
    B, C, H, W = x.shape
    N = B // (bh * bw)
    out = np.zeros((N, C, H * bh, W * bw), np.float32)
    for n in range(N):
        for i in range(bh):
            for j in range(bw):
                out[n, :, i::bh, j::bw] = x[(n * bh + i) * bw + j]
    return out


def main():
    rng = np.random.default_rng(0)
    ok = []

    # integer-valued fp16 so any mismatch is a true permutation error (not rounding)
    def iota(shape):
        return np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)

    print("fused e5rt-MIL ops:")

    # pixel_shuffle r=2 (torch convention, channel-major)
    xs = iota((1, 8, 3, 5))
    net = af.compile(af.pixel_shuffle(af.input((1, 8, 3, 5)), 2))
    ref = xs.reshape(1, 2, 2, 2, 3, 5).transpose(0, 1, 4, 2, 5, 3).reshape(1, 2, 6, 10)
    ok.append(report("pixel_shuffle", net(xs), ref, route="fused-MIL", exact=True))

    # pixel_unshuffle r=2 (inverse of the above)
    xu = iota((1, 3, 6, 8))
    net = af.compile(af.pixel_unshuffle(af.input((1, 3, 6, 8)), 2))
    ref = xu.reshape(1, 3, 3, 2, 4, 2).transpose(0, 1, 3, 5, 2, 4).reshape(1, 12, 3, 4)
    ok.append(report("pixel_unshuffle", net(xu), ref, route="fused-MIL", exact=True))

    print("native netplist-bridge ops (graph cut, like af.sdpa):")

    # space_to_channel: TF space_to_depth (block-major), C>1
    xsc = iota((1, 3, 6, 8))
    net = af.compile(af.space_to_channel(af.input((1, 3, 6, 8)), 2))
    ok.append(report("space_to_channel", net(xsc), ref_space_to_depth_tf(xsc, 2), exact=True))

    # channel_to_space: TF depth_to_space (block-major), C>1
    xcs = iota((1, 8, 3, 5))  # C=2 after r=2
    net = af.compile(af.channel_to_space(af.input((1, 8, 3, 5)), 2))
    ok.append(report("channel_to_space", net(xcs), ref_depth_to_space_tf(xcs, 2), exact=True))

    # space_to_batch: spatial blocks -> batch (N grows)
    xsb = iota((1, 2, 4, 6))
    net = af.compile(af.space_to_batch(af.input((1, 2, 4, 6)), 2, 2))
    ok.append(report("space_to_batch", net(xsb), ref_space_to_batch(xsb, 2, 2), exact=True))

    # batch_to_space: inverse of space_to_batch (N divisible by bh*bw)
    xbs = iota((4, 2, 2, 3))
    net = af.compile(af.batch_to_space(af.input((4, 2, 2, 3)), 2, 2))
    ok.append(report("batch_to_space", net(xbs), ref_batch_to_space(xbs, 2, 2), exact=True))

    # flatten: NCHW [C,H,W] -> 1-D
    xf = rng.standard_normal((2, 2, 3)).astype(np.float16)
    net = af.compile(af.flatten(af.input((2, 2, 3))))
    ok.append(report("flatten", net(xf), xf.reshape(-1), exact=True))

    # input_view: contiguous Width view x[offset:offset+size]
    xv = np.arange(8, dtype=np.float16)
    net = af.compile(af.input_view(af.input((8,)), offset=2, size=3))
    ok.append(report("input_view", net(xv), xv[2:5], exact=True))

    # dynamic_slice: runtime-parametric slice (W=4, size=2 variant)
    xd = np.array([10, 20, 30, 40], dtype=np.float16)
    net = af.compile(af.dynamic_slice(af.input((4,)), start=1, size=2))
    ok.append(report("dynamic_slice", net(xd), xd[1:3], exact=True))

    print(f"\n{sum(ok)}/{len(ok)} native pixel/layout ops correct on the ANE")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
