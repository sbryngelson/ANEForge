"""NN-block cases for the aneforge corpus: realistic network fragments an
optimizer must not break - conv stacks, a transformer encoder block, a small
CNN->GAP->fc classifier, group/batch norm, upsample+conv (SR-style), and
conv_transpose.

Each case ships its own numpy golden reference. ``CASES`` is imported by
tests/run_corpus.py; this file also runs standalone.
"""
from __future__ import annotations

import numpy as np

from _corpus import Case, run_corpus  # noqa: E402
import aneforge as af  # noqa: E402

rng = np.random.default_rng(7)


def f16(*shape, scale=1.0, pos=False):
  a = rng.standard_normal(shape).astype(np.float32) * scale
  if pos: a = np.abs(a) + 0.5
  return a.astype(np.float16)


# numpy reference primitives
def np_conv(x, w, b=None, stride=1, pad=0):
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


def np_relu(x):
  return np.maximum(x, 0.0)


def np_gelu(x):
  from scipy import special  # noqa
  return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))


def _gelu_no_scipy(x):
  # erf via math.erf, vectorized (matches aneforge's exact-gelu MIL lowering)
  import math
  return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


try:
  import scipy  # noqa
  GELU = np_gelu
except Exception:  # noqa: BLE001
  GELU = _gelu_no_scipy


def np_layer_norm(x, g, b, eps=1e-5):
  x = x.astype(np.float32)
  mu = x.mean(-1, keepdims=True)
  var = x.var(-1, keepdims=True)
  return (x - mu) / np.sqrt(var + eps) * g.astype(np.float32) + b.astype(np.float32)


def np_rms_norm(x, g, eps=1e-5):
  x = x.astype(np.float32)
  ms = (x * x).mean(-1, keepdims=True)
  return x / np.sqrt(ms + eps) * g.astype(np.float32)


def np_softmax(x, axis=-1):
  x = x.astype(np.float32)
  x = x - x.max(axis, keepdims=True)
  e = np.exp(x)
  return e / e.sum(axis, keepdims=True)


def np_group_norm(x, g, b, groups, eps=1e-5):
  x = x.astype(np.float32)
  N, C, H, W = x.shape
  xg = x.reshape(N, groups, C // groups, H, W)
  mu = xg.mean((2, 3, 4), keepdims=True)
  var = xg.var((2, 3, 4), keepdims=True)
  xg = (xg - mu) / np.sqrt(var + eps)
  x = xg.reshape(N, C, H, W)
  return x * g.astype(np.float32).reshape(1, C, 1, 1) + b.astype(np.float32).reshape(1, C, 1, 1)


def np_batch_norm(x, g, b, m, v, eps=1e-5):
  x = x.astype(np.float32)
  r = lambda t: t.astype(np.float32).reshape(1, -1, 1, 1)
  return (x - r(m)) / np.sqrt(r(v) + eps) * r(g) + r(b)


def np_conv_transpose(x, w, stride=1, pad=0):
  x = x.astype(np.float32); w = w.astype(np.float32)
  N, Cin, H, W = x.shape
  _, Cout, kH, kW = w.shape
  Hout = (H - 1) * stride + kH - 2 * pad
  Wout = (W - 1) * stride + kW - 2 * pad
  full = np.zeros((N, Cout, (H - 1) * stride + kH, (W - 1) * stride + kW), np.float32)
  for ci in range(Cin):
    for i in range(H):
      for j in range(W):
        full[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW] += \
            x[:, ci:ci+1, i, j][:, :, None, None] * w[ci][None]
  if pad:
    full = full[:, :, pad:pad+Hout, pad:pad+Wout]
  return full


def np_upsample_nn(x, scale):
  return np.repeat(np.repeat(x.astype(np.float32), scale, axis=2), scale, axis=3)


# cases
def _conv_stack():
  W1 = f16(8, 3, 3, 3, scale=0.2); b1 = f16(8, scale=0.1)
  W2 = f16(16, 8, 3, 3, scale=0.2); b2 = f16(16, scale=0.1)
  x = f16(1, 3, 16, 16)

  def build(xt):
    h = af.conv(xt, W1, pad=1, bias=b1).relu()
    h = af.conv(h, W2, stride=2, pad=1, bias=b2).relu()
    return h

  def ref(xa):
    h = np_relu(np_conv(xa, W1, b1, pad=1))
    h = np_relu(np_conv(h, W2, b2, stride=2, pad=1))
    return h

  return Case("conv_stack_bias_relu", "nn", build, ref, [x], tol=0.03, int8_ok=True)


def _cnn_classifier():
  W1 = f16(8, 3, 3, 3, scale=0.2); b1 = f16(8, scale=0.1)
  W2 = f16(16, 8, 3, 3, scale=0.2); b2 = f16(16, scale=0.1)
  Wfc = f16(16, 10, scale=0.2)  # [in=16, out=10] for x @ W
  x = f16(1, 3, 16, 16)

  def build(xt):
    h = af.conv(xt, W1, pad=1, bias=b1).relu()
    h = af.conv(h, W2, stride=2, pad=1, bias=b2).relu()
    h = h.mean((2, 3)).reshape(1, 16)   # GAP
    return h @ Wfc

  def ref(xa):
    h = np_relu(np_conv(xa, W1, b1, pad=1))
    h = np_relu(np_conv(h, W2, b2, stride=2, pad=1))
    h = h.mean((2, 3)).reshape(1, 16)
    return h @ Wfc.astype(np.float32)

  return Case("cnn_gap_fc_classifier", "nn", build, ref, [x], tol=0.03, int8_ok=True)


def _transformer_encoder():
  S, D, H, Dff = 8, 16, 2, 32
  Wq = f16(D, D, scale=0.2); bq = f16(D, scale=0.1)
  Wk = f16(D, D, scale=0.2); bk = f16(D, scale=0.1)
  Wv = f16(D, D, scale=0.2); bv = f16(D, scale=0.1)
  Wo = f16(D, D, scale=0.2); bo = f16(D, scale=0.1)
  g1 = f16(D, pos=True); b1 = f16(D, scale=0.1)
  g2 = f16(D, pos=True); b2 = f16(D, scale=0.1)
  Wg = f16(2 * Dff, D, scale=0.15); bg = f16(2 * Dff, scale=0.1)   # GEGLU
  Wp = f16(D, Dff, scale=0.15); bp = f16(D, scale=0.1)             # FFN out
  x = f16(S, D, scale=1.0)

  def build(xt):
    h = xt.layer_norm(g1, b1)
    attn = af.mha(h, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H)
    xr = xt + attn                                  # residual 1
    h2 = xr.layer_norm(g2, b2)
    ff = af.geglu(h2, Wg, bg).linear(Wp, bp)        # GEGLU FFN
    return xr + ff                                  # residual 2

  def ref(xa):
    def lin(z, W, b):
      return z @ W.astype(np.float32).T + b.astype(np.float32)
    h = np_layer_norm(xa, g1, b1)
    dh = D // H
    q = lin(h, Wq, bq).reshape(S, H, dh).transpose(1, 0, 2)
    k = lin(h, Wk, bk).reshape(S, H, dh).transpose(1, 0, 2)
    v = lin(h, Wv, bv).reshape(S, H, dh).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(dh)
    a = np_softmax(scores, -1) @ v                  # [H,S,dh]
    o = a.transpose(1, 0, 2).reshape(S, D)
    attn = lin(o, Wo, bo)
    xr = xa + attn
    h2 = np_layer_norm(xr, g2, b2)
    Wgf = Wg.astype(np.float32); bgf = bg.astype(np.float32)
    val = h2 @ Wgf[:Dff].T + bgf[:Dff]
    gate = h2 @ Wgf[Dff:].T + bgf[Dff:]
    geglu = val * GELU(gate)
    ff = geglu @ Wp.astype(np.float32).T + bp.astype(np.float32)
    return xr + ff

  return Case("transformer_encoder_block", "nn", build, ref, [x], tol=0.04)


def _rms_norm_block():
  D = 32
  g = f16(D, pos=True)
  Wp = f16(D, D, scale=0.2)
  x = f16(6, D)

  def build(xt):
    return (xt.rms_norm(g) @ Wp).silu()

  def ref(xa):
    h = np_rms_norm(xa, g)
    h = h @ Wp.astype(np.float32)
    return h / (1 + np.exp(-h))  # silu

  return Case("rms_norm_linear_silu", "nn", build, ref, [x], tol=0.03)


def _group_norm_block():
  C = 8
  g = f16(C, pos=True); b = f16(C, scale=0.1)
  W = f16(C, C, 3, 3, scale=0.2)
  x = f16(1, C, 12, 12)

  def build(xt):
    h = xt.group_norm(g, b, num_groups=4)
    return af.conv(h, W, pad=1).relu()

  def ref(xa):
    h = np_group_norm(xa, g, b, groups=4)
    return np_relu(np_conv(h, W, pad=1))

  return Case("group_norm_conv_relu", "nn", build, ref, [x], tol=0.04)


def _batch_norm_block():
  C = 8
  g = f16(C, pos=True); b = f16(C, scale=0.1)
  m = f16(C, scale=0.5); v = f16(C, pos=True)
  W = f16(16, C, 3, 3, scale=0.2)
  x = f16(1, C, 12, 12)

  def build(xt):
    h = af.batch_norm(xt, g, b, m, v, eps=1e-3)
    return af.conv(h, W, pad=1).relu()

  def ref(xa):
    h = np_batch_norm(xa, g, b, m, v, eps=1e-3)
    return np_relu(np_conv(h, W, pad=1))

  return Case("batch_norm_conv_relu", "nn", build, ref, [x], tol=0.04)


def _upsample_conv_sr():
  Cin, Cout = 4, 3
  W = f16(Cout, Cin, 3, 3, scale=0.2); b = f16(Cout, scale=0.1)
  x = f16(1, Cin, 8, 8)

  def build(xt):
    h = xt.upsample(scale=2)
    return af.conv(h, W, pad=1, bias=b).relu()

  def ref(xa):
    h = np_upsample_nn(xa, 2)
    return np_relu(np_conv(h, W, b, pad=1))

  return Case("upsample_conv_sr", "nn", build, ref, [x], tol=0.03, int8_ok=True)


def _pixel_shuffle_sr():
  # SR-style sub-pixel conv: conv to C*r*r channels, then pixel_shuffle.
  r = 2
  Cin, Cmid = 4, 3
  W = f16(Cmid * r * r, Cin, 3, 3, scale=0.2); b = f16(Cmid * r * r, scale=0.1)
  x = f16(1, Cin, 8, 8)

  def build(xt):
    h = af.conv(xt, W, pad=1, bias=b)
    return af.pixel_shuffle(h, r)

  def ref(xa):
    h = np_conv(xa, W, b, pad=1)  # [1, 12, 8, 8]
    N, C2, H, Wd = h.shape
    C = C2 // (r * r)
    return h.reshape(N, C, r, r, H, Wd).transpose(0, 1, 4, 2, 5, 3).reshape(N, C, H * r, Wd * r)

  return Case("pixel_shuffle_subpixel_sr", "nn", build, ref, [x], tol=0.03)


def _conv_transpose_decoder():
  Cin, Cout = 4, 3
  W = f16(Cin, Cout, 2, 2, scale=0.2)  # [Cin, Cout, kH, kW]
  x = f16(1, Cin, 8, 8)

  def build(xt):
    return af.conv_transpose(xt, W, stride=2).relu()

  def ref(xa):
    return np_relu(np_conv_transpose(xa, W, stride=2))

  return Case("conv_transpose_decoder", "nn", build, ref, [x], tol=0.03)


# new primitives: norms / vision / einsum
def _instance_norm_block():
  C = 8
  g = f16(C, scale=0.5, pos=True); b = f16(C, scale=0.3)
  x = f16(1, C, 8, 8)

  def build(xt):
    return af.instance_norm(xt, g, b).relu()

  def ref(xa):
    xf = xa.astype(np.float32)
    mu = xf.mean((2, 3), keepdims=True); var = xf.var((2, 3), keepdims=True)
    xn = (xf - mu) / np.sqrt(var + float(np.float16(1e-5)))
    out = xn * g.astype(np.float32).reshape(1, C, 1, 1) + b.astype(np.float32).reshape(1, C, 1, 1)
    return np_relu(out)

  return Case("instance_norm_block", "nn", build, ref, [x], tol=0.03)


def _lrn_block():
  N, C, H, W = 1, 8, 4, 4
  x = (rng.standard_normal((N, C, H, W)) + 0.5).astype(np.float16)

  def build(xt):
    return af.local_response_norm(xt, size=3, alpha=1e-4, beta=0.75, k=1.0)

  def ref(xa):
    xf = xa.astype(np.float32); size = 3
    alpha, bf, kf = float(np.float16(1e-4)), float(np.float16(0.75)), 1.0
    out = np.zeros_like(xf); pad = size // 2
    xp = np.pad(xf, ((0, 0), (pad, pad), (0, 0), (0, 0)))
    for c in range(C):
      s = (xp[:, c:c + size] ** 2).sum(1)
      out[:, c] = xf[:, c] / ((kf + alpha / size * s) ** bf)
    return out

  return Case("local_response_norm_block", "nn", build, ref, [x], tol=0.03)


def _einsum_native_block():
  B, C, H, W1, W2 = 1, 4, 3, 6, 5
  Wb = f16(B, W1, H, W2, scale=0.3)
  x = f16(B, C, H, W1)

  def build(xt):
    return af.einsum_native("nchw,nwhu->nchu", xt, Wb).relu()

  def ref(xa):
    return np_relu(np.einsum('bchw,bwhu->bchu', xa.astype(np.float32), Wb.astype(np.float32)))

  return Case("einsum_native_block", "nn", build, ref, [x], tol=0.03)


def _space_depth_roundtrip():
  x = f16(1, 4, 8, 8)

  def build(xt):
    h = af.space_to_depth(xt, 2)      # (1, 16, 4, 4)
    return af.depth_to_space(h, 2)    # (1, 4, 8, 8) - identity

  def ref(xa):
    return xa.astype(np.float32)

  # the reorder is value-preserving; a sub-ULP fp16 residual (~1e-5) keeps it from
  # being bit-exact, so gate on a tight relerr rather than exact equality.
  return Case("space_depth_roundtrip", "nn", build, ref, [x], tol=1e-4)


def _crop_resize_block():
  x = f16(1, 3, 8, 8)

  def build(xt):
    h = af.crop(xt, 1, 1, 1, 1)               # (1,3,6,6)
    return af.resize_nearest_neighbor(h, 12, 12)

  def ref(xa):
    xf = xa.astype(np.float32)[:, :, 1:7, 1:7]
    out = np.zeros((1, 3, 12, 12), np.float32)
    for oh in range(12):
      for ow in range(12):
        out[0, :, oh, ow] = xf[0, :, (oh * 6) // 12, (ow * 6) // 12]
    return out

  return Case("crop_resize_nn_block", "nn", build, ref, [x], tol=0.02)


def _upsample_bilinear_block():
  x = f16(1, 3, 4, 4)

  def build(xt):
    return af.upsample_bilinear(xt, 2)        # half-pixel bilinear

  def ref(xa):
    xf = xa.astype(np.float32)
    N, C, H, W = xf.shape; Hn, Wn = H * 2, W * 2
    out = np.zeros((N, C, Hn, Wn), np.float32)
    for i in range(Hn):
      yi = (i + 0.5) * H / Hn - 0.5
      y0 = int(np.floor(yi)); dy = yi - y0
      y0c = max(0, min(y0, H - 1)); y1c = max(0, min(y0 + 1, H - 1))
      for j in range(Wn):
        xj = (j + 0.5) * W / Wn - 0.5
        x0 = int(np.floor(xj)); dx = xj - x0
        x0c = max(0, min(x0, W - 1)); x1c = max(0, min(x0 + 1, W - 1))
        out[:, :, i, j] = (xf[:, :, y0c, x0c] * (1 - dy) * (1 - dx)
                           + xf[:, :, y1c, x0c] * dy * (1 - dx)
                           + xf[:, :, y0c, x1c] * (1 - dy) * dx
                           + xf[:, :, y1c, x1c] * dy * dx)
    return out

  return Case("upsample_bilinear_block", "nn", build, ref, [x], tol=0.02)


CASES = [
  _conv_stack(),
  _cnn_classifier(),
  _transformer_encoder(),
  _rms_norm_block(),
  _group_norm_block(),
  _batch_norm_block(),
  _upsample_conv_sr(),
  _pixel_shuffle_sr(),
  _conv_transpose_decoder(),
  _instance_norm_block(),
  _lrn_block(),
  _einsum_native_block(),
  _space_depth_roundtrip(),
  _crop_resize_block(),
  _upsample_bilinear_block(),
]


if __name__ == "__main__":
  import sys
  _, code = run_corpus(CASES)
  sys.exit(code)
