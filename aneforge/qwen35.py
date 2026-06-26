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
from .llm import _repeat_kv3


def _rotate_matrix(dh: int, rotary_dim: int, interleaved: bool) -> np.ndarray:
  """The [dh, dh] matrix P for matmul-rope `x*cos + (x@P)*sin` over the first `rotary_dim` dims (pass-through
  elsewhere, where cos=1/sin=0). `interleaved` (GPT-J) rotates pairs (2p, 2p+1); otherwise (NeoX/HF) rotates
  the two halves (i, i+rotary_dim/2)."""
  rd = rotary_dim or dh; half = rd // 2; P = np.zeros((dh, dh), np.float32)
  if interleaved:
    for p in range(half): P[2 * p + 1, 2 * p] = -1.0; P[2 * p, 2 * p + 1] = 1.0
  else:
    for i in range(half): P[i + half, i] = -1.0; P[i, i + half] = 1.0
  return P


def _gated_attn_decode(x, w, cfg, ls, ctx, M):
  """Single-token gated attention against a resident KV-cache (Qwen3.5): RMSNorm -> q_proj (q + output gate)
  /k/v -> QK-norm -> partial interleaved RoPE (matmul form) -> GQA causal SDPA -> out * sigmoid(gate) -> o_proj.
  Returns (hidden+residual, [(Kout,Kin),(Vout,Vin)])."""
  H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.dh
  Kin = _input((KV, M, dh)); Vin = _input((KV, M, dh))
  oh, inv, mask, cosp, sinp = ctx["oh"], ctx["inv"], ctx["mask"], ctx["cosp"], ctx["sinp"]
  P = _const(_rotate_matrix(dh, cfg.rotary_dim, cfg.rope_interleaved))
  xn = x.rms_norm(w["in_norm"], cfg.norm_eps)
  qg = xn.linear(w["wq"]).reshape((H, dh * 2))                  # q_proj outputs query + output gate
  q = qg.slice_by_size([0, 0], [H, dh]).rms_norm(w["q_norm"], cfg.norm_eps)
  gate = qg.slice_by_size([0, dh], [H, dh]).reshape((1, H * dh))
  k = xn.linear(w["wk"]).reshape((KV, dh)).rms_norm(w["k_norm"], cfg.norm_eps)
  v = xn.linear(w["wv"]).reshape((KV, dh))
  q = (q * cosp + (q @ P) * sinp).reshape((H, 1, dh))           # matmul-rope (interleaved partial)
  k = (k * cosp + (k @ P) * sinp).reshape((KV, 1, dh))
  Kout = Kin * inv + k * oh; Vout = Vin * inv + v.reshape((KV, 1, dh)) * oh
  Kr, Vr = _repeat_kv3(Kout, H // KV), _repeat_kv3(Vout, H // KV)
  sc = ((q @ Kr.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
  a = (sc @ Vr).reshape((1, H * dh)) * gate.sigmoid()
  return x + a.linear(w["wo"]), [(Kout, Kin), (Vout, Vin)]


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
llm.DECODE_MIXERS["gated_attention"] = _gated_attn_decode


def adapt(c, sd, prefix: str = ""):
  """Adapter for a Qwen3.5 / Qwen3-Next hybrid TEXT model: emit (LlamaConfig with the interleaved per-layer
  plan, canonical weights) from the HF text config `c` and a numpy state_dict `sd`. `prefix` is the layers
  prefix in `sd` ('' for Qwen3_5TextModel, 'model.language_model.' for the VL checkpoint). Canonicalizes the
  Qwen3.5 `(1+w)` RMSNorms (bake +1) and bakes `-exp(A_log)` so the runtime stays uniform."""
  n = int(c.num_hidden_layers); interval = int(getattr(c, "full_attention_interval", 4))
  hd = int(getattr(c, "head_dim", 0) or 0)
  is_attn = lambda L: (L + 1) % interval == 0
  cfg = llm.LlamaConfig(
    dim=c.hidden_size, n_layers=n, n_heads=c.num_attention_heads, n_kv_heads=c.num_key_value_heads,
    ffn_dim=c.intermediate_size, vocab=c.vocab_size, rope_base=float(getattr(c, "rope_theta", 1e4)),
    norm_eps=float(c.rms_norm_eps), head_dim=hd, rotary_dim=int(hd * getattr(c, "partial_rotary_factor", 1.0)),
    layers=[llm.LayerSpec(mixer="gated_attention" if is_attn(L) else "gated_deltanet") for L in range(n)],
    extra={"nk": c.linear_num_key_heads, "nv": c.linear_num_value_heads, "dk": c.linear_key_head_dim,
           "dv": c.linear_value_head_dim, "conv_k": c.linear_conv_kernel_dim})
  g = lambda k: sd[prefix + k]; p1 = lambda k: 1.0 + sd[prefix + k]      # p1: canonicalize (1+w) norm
  w = {"embed": g("embed_tokens.weight"), "final_norm": p1("norm.weight"),
       "lm_head": sd.get("lm_head.weight", sd.get(prefix + "embed_tokens.weight")), "layers": []}
  for L in range(n):
    q = f"layers.{L}."
    lw = {"in_norm": p1(q + "input_layernorm.weight"), "mlp_norm": p1(q + "post_attention_layernorm.weight"),
          "wgate": g(q + "mlp.gate_proj.weight"), "wup": g(q + "mlp.up_proj.weight"), "wdown": g(q + "mlp.down_proj.weight")}
    if is_attn(L):
      a = q + "self_attn."
      lw |= {"wq": g(a + "q_proj.weight"), "wk": g(a + "k_proj.weight"), "wv": g(a + "v_proj.weight"),
             "wo": g(a + "o_proj.weight"), "q_norm": p1(a + "q_norm.weight"), "k_norm": p1(a + "k_norm.weight")}
    else:
      d = q + "linear_attn."
      lw |= {"in_proj_qkv": g(d + "in_proj_qkv.weight"), "in_proj_z": g(d + "in_proj_z.weight"),
             "in_proj_a": g(d + "in_proj_a.weight"), "in_proj_b": g(d + "in_proj_b.weight"),
             "conv1d": np.squeeze(g(d + "conv1d.weight")), "neg_exp_A": -np.exp(g(d + "A_log")),
             "dt_bias": g(d + "dt_bias"), "ssm_norm": g(d + "norm.weight"), "out_proj": g(d + "out_proj.weight")}
    w["layers"].append(lw)
  return cfg, w
