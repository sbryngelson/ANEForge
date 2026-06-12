"""A native-causal LLaMA decoder block on the ANE (af.sdpa is_causal in a full block)."""
from __future__ import annotations
import math
import numpy as np
import pytest
from examples.llama_block_causal import build_causal_llama_block, numpy_reference
import aneforge as af


def _ane():
    try:
        from aneforge._runtime import _find_dylib; _find_dylib(); return True
    except Exception:
        return False


requires_ane = pytest.mark.skipif(not _ane(), reason="ANE/e5rt dylib unavailable")


@requires_ane
@pytest.mark.parametrize("S,D,H,Dff", [(6, 16, 2, 32), (8, 32, 4, 64)])
def test_causal_llama_block_matches_numpy(S, D, H, Dff):
    graph_fn, W = build_causal_llama_block(S, D, H, Dff, seed=0)
    net = af.compile(graph_fn(af.input((S, D))))
    x = np.random.default_rng(1).standard_normal((S, D)).astype(np.float16)
    Y = np.asarray(net(x)).reshape(S, D)
    ref = numpy_reference(x.astype(np.float32), W, S, D, H, Dff)
    cos = float(Y.ravel().astype(np.float64) @ ref.ravel().astype(np.float64)
                / (np.linalg.norm(Y) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.99, cos
    assert np.isfinite(Y).all()


@requires_ane
def test_kv_cache_decode_step():
    # one autoregressive DECODE step: the new token attends to cached K/V (+ its own new k/v)
    # via the decode-shape SDPA (seq_q=1, seq_kv=cache+1), in a full LLaMA block.
    rng = np.random.default_rng(0); D, H, Dff, Sc = 16, 2, 32, 5; dh = D // H
    W = {k: (rng.standard_normal(s) * 0.1).astype(np.float32) for k, s in
         {"Wq": (D, D), "Wk": (D, D), "Wv": (D, D), "Wo": (D, D),
          "Wg": (D, Dff), "Wu": (D, Dff), "Wd": (Dff, D)}.items()}
    W["rn1"], W["rn2"] = np.ones(D, np.float32), np.ones(D, np.float32)

    def step(x, Kc, Vc):
        xn = x.rms_norm(W["rn1"])
        qn = (xn @ W["Wq"]).reshape(1, H, dh).transpose([1, 0, 2])
        kn = (xn @ W["Wk"]).reshape(1, H, dh).transpose([1, 0, 2])
        vn = (xn @ W["Wv"]).reshape(1, H, dh).transpose([1, 0, 2])
        Kf, Vf = af.concat([Kc, kn], axis=1), af.concat([Vc, vn], axis=1)
        a = af.sdpa(qn.reshape(1, H, 1, dh), Kf.reshape(1, H, Sc + 1, dh),
                    Vf.reshape(1, H, Sc + 1, dh)).reshape(H, 1, dh)
        h = x + a.transpose([1, 0, 2]).reshape(1, D) @ W["Wo"]
        hn = h.rms_norm(W["rn2"])
        return h + ((hn @ W["Wg"]).silu() * (hn @ W["Wu"])) @ W["Wd"]

    net = af.compile(step(af.input((1, D)), af.input((H, Sc, dh)), af.input((H, Sc, dh))))
    x = rng.standard_normal((1, D)).astype(np.float16)
    Kc = rng.standard_normal((H, Sc, dh)).astype(np.float16); Vc = rng.standard_normal((H, Sc, dh)).astype(np.float16)
    Y = np.asarray(net(x, Kc, Vc)).reshape(1, D)
    rms = lambda z, g, e=1e-5: z / np.sqrt((z ** 2).mean(-1, keepdims=True) + e) * g
    silu = lambda z: z / (1 + np.exp(-z))
    xn = rms(x.astype(np.float32), W["rn1"])
    qn = (xn @ W["Wq"]).reshape(1, H, dh).transpose(1, 0, 2)
    kn = (xn @ W["Wk"]).reshape(1, H, dh).transpose(1, 0, 2); vn = (xn @ W["Wv"]).reshape(1, H, dh).transpose(1, 0, 2)
    Kf = np.concatenate([Kc.astype(np.float32), kn], 1); Vf = np.concatenate([Vc.astype(np.float32), vn], 1)
    sc = (qn @ Kf.transpose(0, 2, 1)) / math.sqrt(dh)
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc = sc / sc.sum(-1, keepdims=True)
    a = (sc @ Vf).transpose(1, 0, 2).reshape(1, D)
    h = x.astype(np.float32) + a @ W["Wo"]; hn = rms(h, W["rn2"])
    ref = h + (silu(hn @ W["Wg"]) * (hn @ W["Wu"])) @ W["Wd"]
    cos = float(Y.ravel().astype(np.float64) @ ref.ravel().astype(np.float64)
                / (np.linalg.norm(Y) * np.linalg.norm(ref) + 1e-12))
    assert cos > 0.99, cos


@requires_ane
def test_ane_generate_matches_numpy():
    # end-to-end autoregressive generation on the ANE == a numpy reference (token-for-token).
    from examples.gpt_generate_ane import TinyDecoderANE
    m = TinyDecoderANE(seed=0)
    rms = lambda z, g, e=1e-5: z / np.sqrt((z ** 2).mean(-1, keepdims=True) + e) * g
    silu = lambda z: z / (1 + np.exp(-z))

    def np_generate(prompt, n):
        H, dh, D, W = m.H, m.dh, m.D, m.W
        seq, out = list(prompt), []
        for _ in range(n):
            X = m.emb[seq]; S = len(seq); xn = rms(X, m.rn1)
            q = (xn @ W["Wq"]).reshape(S, H, dh).transpose(1, 0, 2)
            k = (xn @ W["Wk"]).reshape(S, H, dh).transpose(1, 0, 2)
            v = (xn @ W["Wv"]).reshape(S, H, dh).transpose(1, 0, 2)
            sc = (q @ k.transpose(0, 2, 1)) / math.sqrt(dh) + np.triu(np.full((S, S), -1e4, np.float32), 1)
            sc = np.exp(sc - sc.max(-1, keepdims=True)); sc = sc / sc.sum(-1, keepdims=True)
            a = (sc @ v).transpose(1, 0, 2).reshape(S, D)
            h = X + a @ W["Wo"]; hn = rms(h, m.rn2)
            h = h + (silu(hn @ W["Wg"]) * (hn @ W["Wu"])) @ W["Wd"]
            tok = int(np.argmax(h[-1] @ m.uemb)); out.append(tok); seq.append(tok)
        return out

    prompt = [3, 7, 1]
    assert m.generate(prompt, 4) == np_generate(prompt, 4)


@requires_ane
def test_generate_fixed_cache_matches_per_step():
    # fixed-max cache (ONE decode compile + runtime position mask) == per-step growing cache.
    from examples.gpt_generate_ane import TinyDecoderANE
    m = TinyDecoderANE(seed=0)
    assert m.generate_fixed([3, 7, 1], 4, max_len=12) == m.generate([3, 7, 1], 4)


@requires_ane
def test_generate_resident_matches_per_step():
    # RESIDENT KV-cache (share_buffer: the masked positional write is in-graph and the cache
    # output is aliased onto its own input, so the cache never round-trips to the host) ==
    # the per-step growing cache. Productizes the zero-copy KV-cache crack.
    from examples.gpt_generate_ane import TinyDecoderANE
    m = TinyDecoderANE(seed=0)
    assert m.generate_resident([3, 7, 1], 4, max_len=12) == m.generate([3, 7, 1], 4)
