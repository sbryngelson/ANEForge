"""Adversarial / corner cases for the aneforge corpus (extreme shapes, deep chains, graph cuts, multi-input, fp16-vs-int8)."""
from __future__ import annotations

import numpy as np

from _corpus import Case, run_corpus  # noqa: E402
import aneforge as af  # noqa: E402

rng = np.random.default_rng(23)


def f16(*shape, scale=1.0, pos=False):
  a = rng.standard_normal(shape).astype(np.float32) * scale
  if pos: a = np.abs(a) + 0.5
  return a.astype(np.float16)


def np_relu(x):
  return np.maximum(x, 0.0)


def np_softmax(x, axis=-1):
  x = x.astype(np.float32)
  x = x - x.max(axis, keepdims=True)
  e = np.exp(x)
  return e / e.sum(axis, keepdims=True)


# extreme-but-legal shapes
def _tall_skinny():
  # [4096, 1]: huge leading axis, trailing axis == 1
  x = f16(4096, 1)

  def build(xt):
    return (xt * 2.0).relu() + xt

  def ref(xa):
    return np_relu(xa * 2.0) + xa

  return Case("tall_skinny_4096x1", "corner", build, ref, [x], tol=0.02)


def _wide_vector_matmul():
  # contract a wide K dimension: [1, 4096] @ [4096, 8]
  K = 4096
  x = f16(1, K, scale=0.05)
  W = f16(K, 8, scale=0.02)

  def build(xt):
    return xt @ W

  def ref(xa):
    return xa @ W.astype(np.float32)

  # wide-K accumulation under fp16 inputs: modest tol
  return Case("wide_K_matmul_1x4096x8", "corner", build, ref, [x], tol=0.04, int8_ok=True, int8_tol=0.08)


def _singleton_reduce():
  # reduce over an axis of size 1 (no-op reduction) mixed with a real one
  x = f16(1, 1, 64)

  def build(xt):
    return xt.sum(0).mean(2)   # axis-0 size1, then real reduce over 64

  def ref(xa):
    return xa.sum(0, keepdims=True).mean(2, keepdims=True)

  return Case("singleton_axis_reduce", "corner", build, ref, [x], tol=0.02)


def _big_channel_conv():
  # 1x1 conv with a large channel count (256->256)
  C = 256
  W = f16(C, C, 1, 1, scale=0.03)
  x = f16(1, C, 4, 4, scale=0.5)

  def build(xt):
    return af.conv(xt, W).relu()

  def ref(xa):
    xa = xa.astype(np.float32); Wf = W.astype(np.float32).reshape(C, C)
    N, _, H, Wd = xa.shape
    out = np.einsum("oc,nchw->nohw", Wf, xa)
    return np_relu(out)

  return Case("conv1x1_256ch", "corner", build, ref, [x], tol=0.05, int8_ok=True, int8_tol=0.10)


# very deep chain fused into one program
def _deep_chain():
  # 32 ops fused into ONE program; sin/cos/tanh + *1.4 is magnitude-preserving (O(1)) so relerr stays meaningful
  x = f16(4, 16, scale=1.0)
  n = 32

  def build(xt):
    h = xt
    for i in range(n):
      r = i % 4
      if r == 0:
        h = h.sin()
      elif r == 1:
        h = h.cos()
      elif r == 2:
        h = h.tanh()
      else:
        h = h * 1.4
    return h

  def ref(xa):
    h = xa.astype(np.float32)
    for i in range(n):
      r = i % 4
      if r == 0:
        h = np.sin(h)
      elif r == 1:
        h = np.cos(h)
      elif r == 2:
        h = np.tanh(h)
      else:
        h = h * 1.4
    return h

  return Case("deep_chain_32ops", "corner", build, ref, [x], tol=0.03)


# mixed fused-MIL + netplist-bridge (graph cut -> SegmentedModel)
def _conv_argmax_cut():
  # conv -> relu -> reduce -> argmax (netplist cut); exercises the SegmentedModel split
  W = f16(4, 3, 3, 3, scale=0.2)
  x = f16(1, 3, 8, 8)

  def build(xt):
    h = af.conv(xt, W, pad=1).relu()     # [1,4,8,8]
    h = h.mean(2).reshape(4, 8)          # [4,8]
    return h.argmax(axis=-1)             # [4,1] indices

  def ref(xa):
    h = np_relu(_np_conv(xa, W, pad=1))
    h = h.mean(2).reshape(4, 8)
    return h.astype(np.float32).argmax(1, keepdims=True)

  return Case("conv_to_argmax_cut", "corner", build, ref, [x], exact=True)


def _np_conv(x, w, b=None, stride=1, pad=0):
  x = x.astype(np.float32); w = w.astype(np.float32)
  N, Cin, H, W = x.shape
  Cout, _, kH, kW = w.shape
  if pad:
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
  Hout = (x.shape[2] - kH) // stride + 1
  Wout = (x.shape[3] - kW) // stride + 1
  out = np.zeros((N, Cout, Hout, Wout), np.float32)
  for i in range(Hout):
    for j in range(Wout):
      patch = x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
      out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
  if b is not None:
    out += b.astype(np.float32).reshape(1, -1, 1, 1)
  return out


def _sdpa_sandwich():
  # linear -> sdpa (native cut) -> linear; exercises a netplist cut in the MIDDLE of a fused graph
  H, S, Dh = 2, 6, 8
  D = H * Dh
  Wi = f16(D, D, scale=0.2)
  Wo = f16(D, D, scale=0.2)
  x = f16(S, D, scale=0.5)

  def build(xt):
    h = xt.linear(Wi)                          # [S, D]
    # build q,k,v from the same projection (toy), shape [1,H,S,Dh]
    q = h.reshape(1, S, H, Dh).transpose([0, 2, 1, 3])
    k = q
    v = q
    a = af.sdpa(q, k, v)                        # native cut, [1,H,S,Dh]
    o = a.transpose([0, 2, 1, 3]).reshape(S, D)
    return o.linear(Wo)

  def ref(xa):
    h = xa @ Wi.astype(np.float32).T            # [S,D]
    q = h.reshape(1, S, H, Dh).transpose(0, 2, 1, 3)  # [1,H,S,Dh]
    scale = 1.0 / np.sqrt(Dh)
    scores = (q @ q.transpose(0, 1, 3, 2)) * scale
    a = np_softmax(scores, -1) @ q
    o = a.transpose(0, 2, 1, 3).reshape(S, D)
    return o @ Wo.astype(np.float32).T

  return Case("sdpa_region_cut_region", "corner", build, ref, [x], tol=0.04)


# multi-input graphs
def _multi_input():
  a = f16(4, 8); b = f16(4, 8); c = f16(8, 5, scale=0.3)

  def build(at, bt, ct):
    h = (at + bt).relu()
    return h @ ct

  def ref(aa, ba, ca):
    h = np_relu(aa + ba)
    return h @ ca.astype(np.float32)

  return Case("multi_input_3", "corner", build, ref, [a, b, c], tol=0.02)


def _multi_input_residual():
  # two image inputs combined, conv, with a third bias-vector input
  x1 = f16(1, 3, 8, 8); x2 = f16(1, 3, 8, 8)
  W = f16(6, 3, 3, 3, scale=0.2)

  def build(t1, t2):
    h = af.maximum(t1, t2)
    return af.conv(h, W, pad=1).relu()

  def ref(a1, a2):
    h = np.maximum(a1, a2)
    return np_relu(_np_conv(h, W, pad=1))

  return Case("multi_input_image_max", "corner", build, ref, [x1, x2], tol=0.03)


# same graph fp16 vs int8 (int8_ok cases below are run both ways)
def _int8_linear_deep():
  D = 64
  Ws = [f16(D, D, scale=0.12) for _ in range(3)]
  x = f16(8, D, scale=0.5)

  def build(xt):
    h = xt
    for W in Ws:
      h = h.linear(W).relu()
    return h

  def ref(xa):
    h = xa.astype(np.float32)
    for W in Ws:
      h = np_relu(h @ W.astype(np.float32).T)
    return h

  return Case("int8_vs_fp16_deep_linear", "corner", build, ref, [x],
              tol=0.04, int8_ok=True, int8_tol=0.10)


CASES = [
  _tall_skinny(),
  _wide_vector_matmul(),
  _singleton_reduce(),
  _big_channel_conv(),
  _deep_chain(),
  _conv_argmax_cut(),
  _sdpa_sandwich(),
  _multi_input(),
  _multi_input_residual(),
  _int8_linear_deep(),
]


if __name__ == "__main__":
  import sys
  _, code = run_corpus(CASES)
  sys.exit(code)
