"""Broader corpus part 2: multi-op compositions and batch variation at scale, vs numpy fp32."""
from __future__ import annotations

import numpy as np

import aneforge as af
from _corpus import Case
from _helpers import f16
# Reuse test_shapes' rng (seed 7) so drawn inputs/goldens stay byte-identical.
from test_shapes import np_conv, np_group_norm, np_softmax, rng


def _lin(z, W, b):
  z = z.astype(np.float32)
  out = z @ W.astype(np.float32).T
  return out + b.astype(np.float32) if b is not None else out


# ---- multi-head attention block at scale -------------------------------------
def _mha_case(S, D, H):
  dh = D // H
  Wq, bq = f16(rng, D, D, scale=1/np.sqrt(D)), f16(rng, D, scale=0.1)
  Wk, bk = f16(rng, D, D, scale=1/np.sqrt(D)), f16(rng, D, scale=0.1)
  Wv, bv = f16(rng, D, D, scale=1/np.sqrt(D)), f16(rng, D, scale=0.1)
  Wo, bo = f16(rng, D, D, scale=1/np.sqrt(D)), f16(rng, D, scale=0.1)
  x = f16(rng, S, D, scale=1.0)

  def build(xt):
    return af.mha(xt, Wq, bq, Wk, bk, Wv, bv, Wo, bo, n_heads=H)

  def ref(xa):
    q = _lin(xa, Wq, bq).reshape(S, H, dh).transpose(1, 0, 2)
    k = _lin(xa, Wk, bk).reshape(S, H, dh).transpose(1, 0, 2)
    v = _lin(xa, Wv, bv).reshape(S, H, dh).transpose(1, 0, 2)
    a = np_softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(dh), -1) @ v
    return _lin(a.transpose(1, 0, 2).reshape(S, D), Wo, bo)

  return Case(f"mha_S{S}_D{D}_H{H}", "broad", build, ref, [x], tol=0.04)


# ---- MLP / FFN at large width (batch via leading dim) ------------------------
def _mlp_case(N, Din, Dff, Dout):
  W1, b1 = f16(rng, Dff, Din, scale=1/np.sqrt(Din)), f16(rng, Dff, scale=0.1)
  W2, b2 = f16(rng, Dout, Dff, scale=1/np.sqrt(Dff)), f16(rng, Dout, scale=0.1)
  x = f16(rng, N, Din, scale=1.0)

  def build(xt):
    return xt.linear(W1, b1).gelu().linear(W2, b2)

  def ref(xa):
    from math import erf
    h = _lin(xa, W1, b1)
    h = 0.5 * h * (1 + np.vectorize(erf)(h / np.sqrt(2)))
    return _lin(h, W2, b2)

  return Case(f"mlp_N{N}_{Din}x{Dff}x{Dout}", "broad", build, ref, [x], tol=0.04, int8_ok=True)


# ---- conv -> group_norm -> relu -> conv residual (UNet-like) at scale --------
def _conv_block_case(C, HW):
  g = f16(rng, C, pos=True); b = f16(rng, C, scale=0.1)
  W1 = f16(rng, C, C, 3, 3, scale=1/np.sqrt(C*9))
  W2 = f16(rng, C, C, 3, 3, scale=1/np.sqrt(C*9))
  x = f16(rng, 1, C, HW, HW, scale=1.0)

  def build(xt):
    h = af.conv(xt, W1, pad=1)
    h = h.group_norm(g, b, num_groups=min(32, C)).relu()
    return xt + af.conv(h, W2, pad=1)

  def ref(xa):
    h = np_conv(xa, W1, pad=1)
    h = np_group_norm(h, g, b, min(32, C))
    h = np.maximum(h, 0.0)
    return xa.astype(np.float32) + np_conv(h, W2, pad=1)

  return Case(f"conv_block_C{C}_{HW}x{HW}", "broad", build, ref, [x], tol=0.05)


# ---- batched matmul at scale -------------------------------------------------
def _bmm_case(B, M, K, N):
  a = f16(rng, B, M, K, scale=1.0); b = f16(rng, B, K, N, scale=1/np.sqrt(K))

  def build(at, bt):
    return at @ bt

  def ref(aa, ba):
    return aa.astype(np.float32) @ ba.astype(np.float32)

  return Case(f"bmm_B{B}_{M}x{K}x{N}", "broad", build, ref, [a, b], tol=0.03)


# ---- wide deep fused chain (fusion at scale) ---------------------------------
def _deep_chain_case(D, depth):
  Ws = [f16(rng, D, D, scale=1/np.sqrt(D)) for _ in range(depth)]
  x = f16(rng, 8, D, scale=1.0)

  def build(xt):
    h = xt
    for i, W in enumerate(Ws):
      h = h.linear(W, None)
      h = h.gelu() if i % 2 == 0 else h.relu()
    return h

  def ref(xa):
    from math import erf
    h = xa.astype(np.float32)
    for i, W in enumerate(Ws):
      h = h @ W.astype(np.float32).T
      h = 0.5 * h * (1 + np.vectorize(erf)(h / np.sqrt(2))) if i % 2 == 0 else np.maximum(h, 0.0)
    return h

  # tol=0.2: deep fp16 chain drifts ~0.15 (depth 12, D=512); a regression blows past 0.2.
  return Case(f"deep_chain_D{D}_d{depth}", "broad", build, ref, [x], tol=0.2)


# ---- composition ops: native op arch-gated, function reachable by decomposition --
def _cumsum_case(M, N):
  x = f16(rng, M, N, scale=1.0)
  return Case(f"cumsum_{M}x{N}", "broad",
              lambda xt: xt.cumsum(),
              lambda xa: np.cumsum(xa.astype(np.float32), axis=-1),
              [x], tol=0.02)   # cumsum = x@triu_ones matmul; fp16 matmul band


def _gather_case(M, N, idx, axis):
  x = f16(rng, M, N, scale=1.0)
  return Case(f"gather_ax{axis}_{M}x{N}", "broad",
              lambda xt: af.gather(xt, idx, axis=axis),
              lambda xa: np.take(xa.astype(np.float32), idx, axis=axis),
              [x], exact=True)   # slice_by_size + concat: bit-exact


CASES = [
  _cumsum_case(4, 64), _cumsum_case(8, 256),
  _gather_case(16, 32, [5, 1, 0, 5, 3, 7, 15, 2], 0),
  _gather_case(8, 16, [2, 2, 0, 1], 1),
  _mha_case(128, 256, 4), _mha_case(512, 512, 8),
  _mlp_case(8, 512, 2048, 512), _mlp_case(64, 768, 3072, 768),
  _conv_block_case(64, 64), _conv_block_case(256, 32),
  _bmm_case(8, 128, 64, 128), _bmm_case(16, 64, 256, 64),
  _deep_chain_case(512, 12),
]
