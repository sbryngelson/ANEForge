"""DEMO: native attention (SDPA) on the ANE's dedicated attention hardware layer.

Exercises:
  - af.sdpa(q, k, v, is_causal=...) - the native ANECSDPALayerDesc hardware layer, a layer
    the public MIL toolchain does not emit, reached from unentitled user space
  - causal scaled-dot-product attention checked against a numpy reference (cos ~1)
  - the KV-cache decode shape (seq_q=1 query over cached K/V) that autoregressive decode uses

Run:  python3 examples/demos/llm_attention_kvcache.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def _ref_attn(Q, K, V, causal):
    H, S, D = Q.shape[1], Q.shape[2], Q.shape[3]
    out = np.zeros((1, H, S, D), np.float64)
    for h in range(H):
        sc = (Q[0, h].astype(np.float64) @ K[0, h].astype(np.float64).T) / np.sqrt(D)
        if causal:
            sc += np.triu(np.full((S, S), -1e9), 1)
        sc -= sc.max(1, keepdims=True)
        p = np.exp(sc); p /= p.sum(1, keepdims=True)
        out[0, h] = p @ V[0, h].astype(np.float64)
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    H, S, D = 4, 32, 16
    Q, K, V = (rng.standard_normal((1, H, S, D)).astype(np.float16) for _ in range(3))

    q, k, v = (af.input((1, H, S, D)) for _ in range(3))
    net = af.compile(af.sdpa(q, k, v, is_causal=True), opt=0)
    got = np.asarray(net(Q, K, V)).astype(np.float64)
    ref = _ref_attn(Q, K, V, causal=True)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got)*np.linalg.norm(ref)+1e-30))
    print(f"causal SDPA (H={H},S={S},D={D}) on the native attention layer: cosine = {cos:.4f}")
    net.release()

    # KV-cache decode shape: one new query token attends over S cached keys/values
    qd, kc, vc = af.input((1, H, 1, D)), af.input((1, H, S, D)), af.input((1, H, S, D))
    dec = af.compile(af.sdpa(qd, kc, vc), opt=0)
    Qd = Q[:, :, :1, :].copy()                        # the single new-token query [1,H,1,D]
    out_shape = np.asarray(dec(Qd, K, V)).shape
    print(f"KV-cache decode shape (seq_q=1 over {S} cached): compiles + runs -> out {out_shape}")
    dec.release()
    print("\naf.sdpa reaches the ANE's dedicated attention hardware (ANECSDPALayerDesc) for")
    print("prefill (causal) and KV-cache decode - the basis of on-ANE GPT/LLaMA decoders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
