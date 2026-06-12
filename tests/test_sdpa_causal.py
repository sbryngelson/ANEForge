"""Native ANE causal SDPA: the fused-attention layer's optional additive-mask (5th) bottom.
Validated against softmax(QKᵀ·scale + mask)·V at the bridge entry. The high-level af.sdpa
is_causal path is not wired yet (it raises), so it is asserted to fail loudly, not silently."""
from __future__ import annotations
import math
import numpy as np
import pytest
import aneforge as af


def _ane():
    try:
        from aneforge._runtime import _find_dylib; _find_dylib(); return True
    except Exception:
        return False


requires_ane = pytest.mark.skipif(not _ane(), reason="ANE/e5rt dylib unavailable")
rng = np.random.default_rng(0)


def _ref(Q, K, V, scale, causal):
    s = (Q.astype(np.float32) @ K.astype(np.float32).swapaxes(-1, -2)) * scale
    if causal:
        S = Q.shape[2]
        s = s + np.triu(np.full((S, S), -1e4, np.float32), 1)
    s = np.exp(s - s.max(-1, keepdims=True)); s = s / s.sum(-1, keepdims=True)
    return s @ V.astype(np.float32)


def _cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@requires_ane
def test_af_sdpa_is_causal_native_end_to_end():
    # af.sdpa(is_causal=True) is native end-to-end: the causal sdpa stays on the native bridge
    # route (the optimizer won't decompose it) and the mask rides the 5th SDPA bottom.
    for H, S, D in [(1, 4, 8), (2, 8, 16)]:
        scale = 1.0 / math.sqrt(D)
        Q = rng.standard_normal((1, H, S, D)).astype(np.float16)
        K = rng.standard_normal((1, H, S, D)).astype(np.float16)
        V = rng.standard_normal((1, H, S, D)).astype(np.float16)
        net = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)),
                                 af.input((1, H, S, D)), is_causal=True))
        Y = np.asarray(net(Q, K, V)).reshape(1, H, S, D)
        assert _cos(Y, _ref(Q, K, V, scale, True)) > 0.99      # causal-masked attention
        assert _cos(Y, _ref(Q, K, V, scale, False)) < 0.95     # and NOT the unmasked result


@requires_ane
def test_native_sdpa_additive_mask_bridge():
    # the native fused-attention layer's optional 5th 'mask' bottom (validated on M1)
    from aneforge._bridges.ane_sdpa_fused import sdpa_fused
    H, S, D = 1, 4, 8
    scale = 1.0 / math.sqrt(D)
    Q = rng.standard_normal((1, H, S, D)).astype(np.float16)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    causal = np.triu(np.full((S, S), -1e4, np.float32), 1)
    Y = sdpa_fused(Q, K, V, scale=scale, mask=causal)
    assert _cos(Y, _ref(Q, K, V, scale, True)) > 0.99     # matches causal-masked attention
    assert _cos(Y, _ref(Q, K, V, scale, False)) < 0.95    # and is NOT the unmasked result


@requires_ane
@pytest.mark.parametrize("H,Sq,Skv,D", [(1, 1, 4, 8), (2, 1, 8, 16), (2, 3, 8, 16)])
def test_sdpa_kv_cache_decode_shape(H, Sq, Skv, D):
    # KV-cache DECODE shape: Sq query tokens attend to Skv cached K/V (seq_q != seq_kv).
    # The native SDPA validator allows it ("K,V same seq" + "Q,K same embed" — no Q-seq constraint).
    scale = 1.0 / math.sqrt(D)
    Q = rng.standard_normal((1, H, Sq, D)).astype(np.float16)
    K = rng.standard_normal((1, H, Skv, D)).astype(np.float16)
    V = rng.standard_normal((1, H, Skv, D)).astype(np.float16)
    net = af.compile(af.sdpa(af.input((1, H, Sq, D)), af.input((1, H, Skv, D)), af.input((1, H, Skv, D))))
    Y = np.asarray(net(Q, K, V)).reshape(1, H, Sq, D)
    s = (Q.astype(np.float32) @ K.astype(np.float32).swapaxes(-1, -2)) * scale
    s = np.exp(s - s.max(-1, keepdims=True)); s = s / s.sum(-1, keepdims=True)
    assert _cos(Y, s @ V.astype(np.float32)) > 0.99
