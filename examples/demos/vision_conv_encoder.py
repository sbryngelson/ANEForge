"""DEMO: a small vision conv encoder fused into one ANE program.

Exercises:
  - a conv->relu->maxpool x2 -> global-avg-pool -> fc encoder, the ANE's home turf
  - the whole stack fused into ONE program (one dispatch for the entire forward pass)
  - correctness vs a numpy fp32 reference (cosine ~1)

Run:  python3 examples/demos/vision_conv_encoder.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def _conv(a, w, pad):                       # numpy reference conv (NCHW, stride 1)
    Co, Ci, kh, kw = w.shape; N, _, H, W = a.shape
    ap = np.pad(a, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = H + 2*pad - kh + 1, W + 2*pad - kw + 1
    o = np.zeros((N, Co, Ho, Wo), np.float64)
    for i in range(kh):
        for j in range(kw):
            o += np.einsum("nchw,oc->nohw", ap[:, :, i:i+Ho, j:j+Wo].astype(np.float64),
                           w[:, :, i, j].astype(np.float64))
    return o


def _pool2(a):                              # 2x2 max pool
    N, C, H, W = a.shape
    return a.reshape(N, C, H//2, 2, W//2, 2).max((3, 5))


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((16, 3, 3, 3)) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((32, 16, 3, 3)) * 0.1).astype(np.float32)
    Wf = (rng.standard_normal((32, 10)) * 0.1).astype(np.float32)
    img = (rng.standard_normal((1, 3, 32, 32)) * 0.5).astype(np.float32)

    x = af.input((1, 3, 32, 32))
    h = af.conv(x, W1, pad=1).relu().max_pool(2)
    h = af.conv(h, W2, pad=1).relu().max_pool(2)
    logits = h.mean((2, 3)).reshape(1, 32) @ Wf
    net = af.compile(logits)
    got = np.asarray(net(img)).astype(np.float64)

    r = np.maximum(_conv(img, W1, 1), 0); r = _pool2(r)
    r = np.maximum(_conv(r, W2, 1), 0); r = _pool2(r)
    ref = r.mean((2, 3)).reshape(1, 32) @ Wf.astype(np.float64)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got)*np.linalg.norm(ref)+1e-30))

    print(f"conv->relu->pool x2 -> GAP -> fc : one program, {net.n_ops} ops fused")
    print(f"cosine vs numpy fp32 reference  : {cos:.4f}")
    print("\nVision encoders are the ANE's sweet spot: a deep conv stack is one fused dispatch.")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
