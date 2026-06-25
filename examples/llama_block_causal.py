"""A pre-norm LLaMA-style decoder block with native causal attention (af.sdpa is_causal) on the ANE."""
from __future__ import annotations
import math
import numpy as np
import aneforge as af


def build_causal_llama_block(S: int, D: int, H: int, Dff: int, seed: int = 0):
    """Return (graph_fn, weights). ``graph_fn(x_input)`` builds the block graph over an
    ``af.input((S, D))``; ``weights`` is the dict of fp32 numpy parameters."""
    rng = np.random.default_rng(seed)
    dh = D // H
    W = {
        "Wq": (rng.standard_normal((D, D)) * 0.1).astype(np.float32),
        "Wk": (rng.standard_normal((D, D)) * 0.1).astype(np.float32),
        "Wv": (rng.standard_normal((D, D)) * 0.1).astype(np.float32),
        "Wo": (rng.standard_normal((D, D)) * 0.1).astype(np.float32),
        "rn1": np.ones(D, np.float32), "rn2": np.ones(D, np.float32),
        "Wg": (rng.standard_normal((D, Dff)) * 0.1).astype(np.float32),
        "Wu": (rng.standard_normal((D, Dff)) * 0.1).astype(np.float32),
        "Wd": (rng.standard_normal((Dff, D)) * 0.1).astype(np.float32),
    }

    def heads(t):                                    # [S, D] -> [H, S, dh]
        return t.reshape(S, H, dh).transpose([1, 0, 2])

    def graph_fn(x):
        xn = x.rms_norm(W["rn1"])
        q, k, v = heads(xn @ W["Wq"]), heads(xn @ W["Wk"]), heads(xn @ W["Wv"])
        # NATIVE causal attention: [H,S,dh] -> [1,H,S,dh] for the SDPA layer, then back.
        a = af.sdpa(q.reshape(1, H, S, dh), k.reshape(1, H, S, dh), v.reshape(1, H, S, dh),
                    is_causal=True).reshape(H, S, dh)
        h = x + a.transpose([1, 0, 2]).reshape(S, D) @ W["Wo"]
        hn = h.rms_norm(W["rn2"])
        return h + ((hn @ W["Wg"]).silu() * (hn @ W["Wu"])) @ W["Wd"]

    return graph_fn, W


def numpy_reference(x, W, S, D, H, Dff):
    dh = D // H
    rms = lambda z, g, e=1e-5: z / np.sqrt((z ** 2).mean(-1, keepdims=True) + e) * g
    silu = lambda z: z / (1 + np.exp(-z))
    xn = rms(x, W["rn1"])
    q = (xn @ W["Wq"]).reshape(S, H, dh).transpose(1, 0, 2)
    k = (xn @ W["Wk"]).reshape(S, H, dh).transpose(1, 0, 2)
    v = (xn @ W["Wv"]).reshape(S, H, dh).transpose(1, 0, 2)
    sc = (q @ k.transpose(0, 2, 1)) / math.sqrt(dh)
    sc = sc + np.triu(np.full((S, S), -1e4, np.float32), 1)          # causal
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc = sc / sc.sum(-1, keepdims=True)
    a = (sc @ v).transpose(1, 0, 2).reshape(S, D)
    h = x + a @ W["Wo"]
    hn = rms(h, W["rn2"])
    return h + (silu(hn @ W["Wg"]) * (hn @ W["Wu"])) @ W["Wd"]


if __name__ == "__main__":
    S, D, H, Dff = 16, 64, 4, 128
    graph_fn, W = build_causal_llama_block(S, D, H, Dff)
    net = af.compile(graph_fn(af.input((S, D))))
    x = np.random.default_rng(1).standard_normal((S, D)).astype(np.float16)
    Y = np.asarray(net(x)).reshape(S, D)
    ref = numpy_reference(x.astype(np.float32), W, S, D, H, Dff)
    cos = float(Y.ravel() @ ref.ravel() / (np.linalg.norm(Y) * np.linalg.norm(ref) + 1e-9))
    print(f"native-causal LLaMA block on the ANE: cos vs numpy = {cos:.4f}  (S={S}, D={D}, H={H})")
    print("the causal mask ran on the ANE's native fused-attention layer (af.sdpa is_causal).")
