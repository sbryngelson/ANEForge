"""aneforge super-resolution demo - ESPCN sub-pixel upscaler on the Apple Neural Engine.

A small ESPCN-style super-resolution network - three convs with ReLU feature
extractors followed by a sub-pixel convolution that upscales by ``r`` via
``af.pixel_shuffle`` (depth-to-space). The whole network, including the
PixelShuffle, fuses into ONE e5rt program: PixelShuffle runs as fused e5rt-MIL,
so there is no graph cut. Output is validated against a numpy reference of the
same forward pass (relative L2 error).

    python3 examples/superres_espcn.py
"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def _conv_np(x, w, b, pad):
    """fp16 NCHW conv reference (zero-pad, stride 1), matching af.conv numerics."""
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Hout, Wout = H + 2 * pad - kH + 1, W + 2 * pad - kW + 1
    out = np.zeros((N, Cout, Hout, Wout), np.float32)
    for i in range(kH):
        for j in range(kW):
            patch = xp[:, :, i:i + Hout, j:j + Wout]          # [N,Cin,Hout,Wout]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, i, j])
    return out + b.reshape(1, Cout, 1, 1)


def _pixel_shuffle_np(x, r):
    """numpy PixelShuffle: [N, C*r*r, H, W] -> [N, C, H*r, W*r]."""
    N, C2, H, W = x.shape
    C = C2 // (r * r)
    x = x.reshape(N, C, r, r, H, W).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(N, C, H * r, W * r)


def main():
    rng = np.random.default_rng(0)
    r, C, Cout = 3, 32, 1                 # upscale x3, 32 feature maps, 1-channel out
    H, W = 24, 24

    # ESPCN weights (small deterministic random; bias zero for clarity).
    w1 = (rng.standard_normal((C, 1, 5, 5)) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((C, C, 3, 3)) * 0.05).astype(np.float32)
    w3 = (rng.standard_normal((Cout * r * r, C, 3, 3)) * 0.05).astype(np.float32)
    b1 = np.zeros(C, np.float32); b2 = np.zeros(C, np.float32)
    b3 = np.zeros(Cout * r * r, np.float32)

    # Build the fused ANE graph: conv->relu->conv->relu->conv->pixel_shuffle.
    x = af.input((1, 1, H, W))
    h = af.conv(x, w1, pad=2, bias=b1).relu()
    h = af.conv(h, w2, pad=1, bias=b2).relu()
    h = af.conv(h, w3, pad=1, bias=b3)
    y = af.pixel_shuffle(h, r)
    net = af.compile(y)
    print(f"ESPCN x{r} SR net: {net.n_ops} ops fused into 1 ANE program "
          f"(3 conv + 2 relu + pixel_shuffle, no graph cut)")

    img = (rng.standard_normal((1, 1, H, W)) * 0.5).astype(np.float32)
    ane = net(img)

    # numpy reference of the identical forward pass.
    ref = _conv_np(img, w1, b1, 2)
    ref = np.maximum(ref, 0.0)
    ref = _conv_np(ref, w2, b2, 1)
    ref = np.maximum(ref, 0.0)
    ref = _conv_np(ref, w3, b3, 1)
    ref = _pixel_shuffle_np(ref, r)

    relerr = float(np.linalg.norm(ane - ref) / (np.linalg.norm(ref) + 1e-12))
    ok = relerr < 2e-2                    # fp16 ANE vs fp32 numpy
    print(f"input  {img.shape[2:]}  ->  output {tuple(ane.shape[2:])}  (x{r} upscale)")
    print(f"relerr(ANE, numpy) = {relerr:.4e}  ->  {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
