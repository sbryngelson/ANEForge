"""Qwen3.5 / Qwen3-Next hybrid support. Registers a `gated_deltanet` token-mixer (linear attention with a
resident conv + recurrent state) into the `llm` registries, so a hybrid model is a per-layer plan mixing
`gated_deltanet` and `attention` layers.

Canonical weight keys (from the adapter): in_norm, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d
[conv_dim, K], neg_exp_A [nv] (= -exp(A_log)), dt_bias, ssm_norm, out_proj, + SwiGLU MLP. DeltaNet dims are
in `cfg.extra`: nk/nv (key/value heads), dk/dv (head dims), conv_k."""
from __future__ import annotations
import numpy as np

from .graph import input as _input, _const, concat as _concat
from . import llm


def _deltanet_decode(x, w, cfg, ls, ctx, M):
  """Single-token Gated-DeltaNet decode. Owns a resident conv state [conv_dim, K-1] and recurrent state
  [nv, dk, dv]; ignores the per-position `ctx` (it carries its own state). Returns (hidden+residual,
  [(conv_out,conv_in), (rec_out,rec_in)]) for the runner to alias resident via `share_buffer`."""
  e = cfg.extra
  nk, nv, dk, dv, K = e["nk"], e["nv"], e["dk"], e["dv"], e["conv_k"]
  kd, vd, CD, g3 = nk * dk, nv * dv, nk * dk * 2 + nv * dv, nv // nk
  cs = _input((CD, K - 1)); S = _input((nv, dk, dv))
  xn = x.rms_norm(w["in_norm"], cfg.norm_eps)
  qkv = xn.linear(w["in_proj_qkv"]).reshape((CD, 1))
  z = xn.linear(w["in_proj_z"]).reshape((nv, dv))
  b = xn.linear(w["in_proj_b"]).reshape((1, nv)); a = xn.linear(w["in_proj_a"]).reshape((1, nv))
  win = _concat([cs, qkv], axis=1)                               # [CD, K]: oldest..newest
  conv = (win * _const(w["conv1d"])).sum([1]).reshape((1, CD)).silu()
  cs_out = win.slice_by_size([0, 1], [CD, K - 1])                # drop the oldest -> new conv state
  q = conv.slice_by_size([0, 0], [1, kd]).reshape((nk, dk))
  k = conv.slice_by_size([0, kd], [1, kd]).reshape((nk, dk))
  v = conv.slice_by_size([0, 2 * kd], [1, vd]).reshape((nv, dv))
  beta = b.sigmoid().reshape((nv, 1, 1))
  gt = ((a + _const(w["dt_bias"])).softplus() * _const(w["neg_exp_A"])).exp().reshape((nv, 1, 1))  # exp(-exp(A_log)*softplus(a+dt))
  rep = _const(np.repeat(np.eye(nk, dtype=np.float32), g3, 0))   # GQA: nk -> nv heads
  q = (rep @ q).l2_norm(-1, 1e-6) * (dk ** -0.5); k = (rep @ k).l2_norm(-1, 1e-6)
  q = q.reshape((nv, 1, dk)); k = k.reshape((nv, 1, dk)); v = v.reshape((nv, 1, dv))
  S1 = S * gt                                                    # decay first (transformers/qwen3.5)
  kv = k @ S1; delta = (v - kv) * beta
  Sout = S1 + (k.transpose([0, 2, 1]) @ delta)
  o = (q @ Sout).reshape((nv, dv))
  o = o.rms_norm(w["ssm_norm"], cfg.norm_eps) * z.silu()         # RMSNormGated, then SwiGLU gate
  return x + o.reshape((1, vd)).linear(w["out_proj"]), [(cs_out, cs), (Sout, S)]


llm.DECODE_MIXERS["gated_deltanet"] = _deltanet_decode
