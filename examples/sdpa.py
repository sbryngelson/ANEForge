"""Native fused attention on the ANE via af.sdpa() (the hardware SDPA layer Apple's MIL compiler never emits). Run: python3 examples/sdpa.py"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def ref_sdpa(Q, K, V, scale):
    H, S, D = Q.shape[1], Q.shape[2], Q.shape[3]
    out = np.zeros((1, H, S, D), np.float32)
    for h in range(H):
        s = (Q[0, h].astype(np.float32) @ K[0, h].astype(np.float32).T) * scale
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        out[0, h] = (e / e.sum(1, keepdims=True)) @ V[0, h].astype(np.float32)
    return out


def main():
    rng = np.random.default_rng(0)
    H, S, D = 4, 32, 16
    scale = 1.0 / D ** 0.5
    Q, K, V = (rng.standard_normal((1, H, S, D)).astype(np.float16) for _ in range(3))
    ok = []

    # (1) standalone native SDPA; opt=0 pins the native layer (default would rewrite short-S to fused decomposition)
    net = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)), af.input((1, H, S, D))), opt=0)
    out = net(Q, K, V)
    r = ref_sdpa(Q, K, V, scale)
    e = float(np.abs(out - r).max() / (np.abs(r).max() + 1e-6))
    print(f"native SDPA on ANE      ({net.n_sdpa} fused-attn sub-program)   relerr {e:.4f}")
    ok.append(e < 0.02)

    # (2) graph-cut hybrid: e5rt ops fused around the native SDPA layer
    q, k, v = af.input((1, H, S, D)), af.input((1, H, S, D)), af.input((1, H, S, D))
    net2 = af.compile((af.sdpa(q * 2.0, k, v) + v).relu(), opt=0)
    out2 = net2(Q, K, V)
    r2 = np.maximum(ref_sdpa((Q.astype(np.float32) * 2).astype(np.float16), K, V, scale) + V.astype(np.float32), 0)
    e2 = float(np.abs(out2 - r2).max() / (np.abs(r2).max() + 1e-6))
    print(f"SDPA + e5rt graph cuts  ({net2.n_ops} e5rt ops + {net2.n_sdpa} SDPA)        relerr {e2:.4f}")
    ok.append(e2 < 0.02)

    print(f"\n{sum(ok)}/{len(ok)} OK - attention running on the ANE's native fused-attention hardware")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
