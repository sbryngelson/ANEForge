"""aneforge frontend worked examples: a vision CNN and a transformer encoder block, each fused to one ANE program. Run: python3 examples/quickstart.py"""
from __future__ import annotations
import sys

from _common import relerr   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def cnn_example(rng):
    N, Cin, H, W = 1, 8, 16, 16
    W1 = rng.standard_normal((8, Cin, 3, 3)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((16, 8, 1, 1)).astype(np.float32) * 0.1
    Wfc = rng.standard_normal((16, 4)).astype(np.float32) * 0.1
    x = rng.standard_normal((N, Cin, H, W)).astype(np.float32) * 0.5

    def conv(a, w, pad):
        Co, Ci, kh, kw = w.shape; N, _, Hh, Ww = a.shape
        ap = np.pad(a, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Ho, Wo = Hh + 2 * pad - kh + 1, Ww + 2 * pad - kw + 1
        o = np.zeros((N, Co, Ho, Wo), np.float32)
        for i in range(kh):
            for j in range(kw):
                o += np.einsum("nchw,oc->nohw", ap[:, :, i:i + Ho, j:j + Wo], w[:, :, i, j])
        return o
    h = np.maximum(conv(x, W1, 1), 0); h = np.maximum(conv(h, W2, 0), 0)
    ref = h.mean((2, 3)) @ Wfc

    inp = af.input((N, Cin, H, W))
    g = af.conv(inp, W1, pad=1).relu()
    g = af.conv(g, W2, pad=0).relu()
    y = g.mean((2, 3)).reshape(N, 16) @ Wfc
    return ("CNN  (conv->relu->conv->relu->GAP->fc)", y, x, ref)


def encoder_example(rng):
    S, D, F = 16, 64, 256
    g1 = rng.standard_normal(D).astype(np.float32) * 0.1 + 1
    g2 = rng.standard_normal(D).astype(np.float32) * 0.1 + 1
    Wq = rng.standard_normal((D, D)).astype(np.float32) * 0.1
    Wk = rng.standard_normal((D, D)).astype(np.float32) * 0.1
    Wv = rng.standard_normal((D, D)).astype(np.float32) * 0.1
    Wo = rng.standard_normal((D, D)).astype(np.float32) * 0.1
    W1 = rng.standard_normal((D, F)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((F, D)).astype(np.float32) * 0.1
    x = rng.standard_normal((S, D)).astype(np.float32) * 0.5

    import math
    _erf = np.vectorize(math.erf)
    def rms(a, g, eps=1e-5): return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps) * g
    def sm(a): e = np.exp(a - a.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    def gelu(a): return 0.5 * a * (1.0 + _erf(a / np.sqrt(2.0)))
    nx = rms(x, g1)
    Q, K, V = nx @ Wq, nx @ Wk, nx @ Wv
    A = sm((Q @ K.T) * (1.0 / np.sqrt(D)))
    h = x + (A @ V) @ Wo
    nh = rms(h, g2)
    ref = h + gelu(nh @ W1) @ W2

    inp = af.input((S, D))
    nxn = inp.rms_norm(g1)
    Qt, Kt, Vt = nxn @ Wq, nxn @ Wk, nxn @ Wv
    At = ((Qt @ Kt.transpose([1, 0])) * (1.0 / np.sqrt(D))).softmax(-1)
    hh = inp + (At @ Vt) @ Wo
    nhh = hh.rms_norm(g2)
    y = hh + ((nhh @ W1).gelu() @ W2)
    return ("Encoder block (RMSNorm+attn+FFN, single-head)", y, x, ref)


def main():
    rng = np.random.default_rng(0)
    for name, y, x, ref in (cnn_example(rng), encoder_example(rng)):
        print(f"\n=== {name} ===")
        for tag, i8 in [("fp16", False), ("int8", True)]:
            net = af.compile(y, int8=i8)
            out = net(x)
            print(f"  {tag}: {net.n_ops} ops fused into 1 program | out{out.shape} | relerr {relerr(out, ref):.4f}"
                  f"  {'OK' if relerr(out, ref) < 0.05 else 'MISMATCH'}")
            net.release()


if __name__ == "__main__":
    sys.exit(main())
