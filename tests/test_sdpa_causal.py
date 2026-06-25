"""Native ANE causal SDPA + additive-mask bottom and query-tiled decomposition, vs softmax(QK^T*scale + mask)*V."""
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
  # The native SDPA validator allows it ("K,V same seq" + "Q,K same embed" - no Q-seq constraint).
  scale = 1.0 / math.sqrt(D)
  Q = rng.standard_normal((1, H, Sq, D)).astype(np.float16)
  K = rng.standard_normal((1, H, Skv, D)).astype(np.float16)
  V = rng.standard_normal((1, H, Skv, D)).astype(np.float16)
  net = af.compile(af.sdpa(af.input((1, H, Sq, D)), af.input((1, H, Skv, D)), af.input((1, H, Skv, D))))
  Y = np.asarray(net(Q, K, V)).reshape(1, H, Sq, D)
  s = (Q.astype(np.float32) @ K.astype(np.float32).swapaxes(-1, -2)) * scale
  s = np.exp(s - s.max(-1, keepdims=True)); s = s / s.sum(-1, keepdims=True)
  assert _cos(Y, s @ V.astype(np.float32)) > 0.99


@requires_ane
def test_af_sdpa_query_tiled_decomposition():
  # At large seq the native layer is unreliable, so af.sdpa decomposes. The decomposition
  # query-tiles: it materializes [tile, Skv] score tiles instead of the full [Sq, Skv],
  # which is exact (each tile attends to all keys) and far faster on the ANE. Verify both
  # the plain path and the runtime additive mask (sliced per query tile).
  H, S, D = 4, 1500, 64                          # S >= 512 -> decomposes -> tiles the query
  scale = 1.0 / math.sqrt(D)
  Q = rng.standard_normal((1, H, S, D)).astype(np.float16)
  K = rng.standard_normal((1, H, S, D)).astype(np.float16)
  V = rng.standard_normal((1, H, S, D)).astype(np.float16)

  out = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)), af.input((1, H, S, D))))(Q, K, V)
  assert tuple(np.asarray(out).reshape(1, H, S, D).shape) == (1, H, S, D)
  assert _cos(out, _ref(Q, K, V, scale, False)) > 0.99

  M = rng.standard_normal((1, 1, S, S)).astype(np.float16)
  outm = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)),
                            af.input((1, H, S, D)), attn_mask=af.input((1, 1, S, S))))(Q, K, V, M)
  s = (Q.astype(np.float32) @ K.astype(np.float32).swapaxes(-1, -2)) * scale + M.astype(np.float32)
  s = np.exp(s - s.max(-1, keepdims=True)); s = s / s.sum(-1, keepdims=True)
  assert _cos(outm, s @ V.astype(np.float32)) > 0.99


@requires_ane
def test_af_mha_query_tiled():
  # af.mha query-tiles its per-head score matrix at large S (the same fission as af.sdpa);
  # verify the tiled path is exact against a numpy multi-head-attention reference.
  S, D, H = 1500, 256, 8                          # S >= 512 -> tiles the query
  dh = D // H
  w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float16)   # noqa: E731
  Wq, bq, Wk, Wv, bv, Wo, bo = w(D, D), w(D), w(D, D), w(D, D), w(D), w(D, D), w(D)
  x = rng.standard_normal((S, D)).astype(np.float16)
  out = np.asarray(af.compile(af.mha(af.input((S, D)), Wq, bq, Wk, None, Wv, bv, Wo, bo, H))(x))

  xf = x.astype(np.float32)
  def lin(W, b): return xf @ W.astype(np.float32).T + (0.0 if b is None else b.astype(np.float32))
  def hd(t): return t.reshape(S, H, dh).transpose(1, 0, 2)
  q, k, v = hd(lin(Wq, bq)), hd(lin(Wk, None)), hd(lin(Wv, bv))
  s = (q @ k.transpose(0, 2, 1)) * dh ** -0.5
  a = np.exp(s - s.max(-1, keepdims=True)); a = a / a.sum(-1, keepdims=True)
  o = (a @ v).transpose(1, 0, 2).reshape(S, D)
  assert _cos(out, o @ Wo.astype(np.float32).T + bo.astype(np.float32)) > 0.99


@requires_ane
def test_af_cross_attention_query_tiled():
  # cross_attention query-tiles when query and context are both long (S >= 768, T >= 512),
  # materializing [tile, T] score blocks instead of the full [H, S, T]; verify exact
  # against a numpy cross-attention reference. (Small-T stays single-shot; same result.)
  S, T, D, H = 768, 512, 256, 8                   # S>=768 and T>=512 -> tiles the query
  dh = D // H
  w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float16)   # noqa: E731
  Wq, Wk, Wv, Wo = w(D, D), w(D, D), w(D, D), w(D, D)
  x = rng.standard_normal((S, D)).astype(np.float16)
  ctx = rng.standard_normal((T, D)).astype(np.float16)
  out = np.asarray(af.compile(af.cross_attention(af.input((S, D)), af.input((T, D)),
                                                 Wq, Wk, Wv, Wo, H))(x, ctx))

  def lin(t, W): return t.astype(np.float32) @ W.astype(np.float32).T
  def hd(t, n): return t.reshape(n, H, dh).transpose(1, 0, 2)
  q, k, v = hd(lin(x, Wq), S), hd(lin(ctx, Wk), T), hd(lin(ctx, Wv), T)
  s = (q @ k.transpose(0, 2, 1)) * dh ** -0.5
  a = np.exp(s - s.max(-1, keepdims=True)); a = a / a.sum(-1, keepdims=True)
  o = (a @ v).transpose(1, 0, 2).reshape(S, D)
  assert _cos(out, o @ Wo.astype(np.float32).T) > 0.99
