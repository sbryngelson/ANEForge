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
  # GQA k/q heads -> v heads. GGUF/llama.cpp uses ggml_repeat (tile: v-head h -> k-head h%nk); transformers
  # uses repeat_interleave (h//g3). The two disagree, and tile is what the GGUF weights are baked for.
  eye = np.eye(nk, dtype=np.float32)
  rep = _const(np.tile(eye, (g3, 1)) if e.get("gqa_repeat") == "tile" else np.repeat(eye, g3, 0))
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


def load_gguf(path: str, n_layers: int | None = None, compress: str | None = None,
              resid_scale: float = 32.0) -> "llm.LlamaPrefill":
  """Load a Qwen3.5 / Qwen3-Next hybrid GGUF (arch `qwen35`) for pure-ANE decode, dequantizing each tensor to
  fp16. The hybrid plan (a `gated_attention` layer every `full_attention_interval`, `gated_deltanet` else) and
  the DeltaNet dims (nk/nv/dk/dv/conv_k) are read from the GGUF metadata. The reader's `.data` is PyTorch
  [out,in] order, so projections need no transpose. Unlike the HF `adapt`, the GGUF already bakes the (1+w)
  RMSNorms and stores `ssm_a = -exp(A_log)`, so both are used as-is. Layers stream + free like the MoE loader
  (64 layers fp16 ~54GB > RAM).

  `resid_scale` shrinks the residual stream by a constant S to keep it in fp16 range: this 27B's deep-layer
  activation outliers exceed fp16's 65504 max and overflow to inf (llama.cpp dodges it with fp32 activations).
  Every read of the residual goes through a scale-invariant rms_norm (each sublayer input + the final norm),
  so scaling `embed` and every residual-writing projection (wo, out_proj, wdown) by 1/S leaves the logits
  unchanged while keeping the hidden state ~S smaller."""
  import gguf
  import mmap as _mmap
  from gguf.quants import dequantize
  from .moe import _LazyLayers
  r = gguf.GGUFReader(path)
  meta = {f.name: f for f in r.fields.values()}
  arch = next(k.split(".")[0] for k in meta if k.endswith(".block_count"))
  def sc(key, default=None):
    f = meta.get(arch + "." + key)
    return f.parts[f.data[-1]][0] if f is not None else default
  tn = {t.name: t for t in r.tensors}
  def get(name, dtype: np.typing.DTypeLike = np.float16):       # dequantize -> fp16 (halves resident memory)
    t = tn[name]; d = np.asarray(t.data)
    if t.tensor_type != gguf.GGMLQuantizationType.F32:
      d = dequantize(d, t.tensor_type)
    return d.astype(dtype)

  n_total = int(sc("block_count", 0)); n = n_total if n_layers is None else min(n_layers, n_total)
  dim, heads = int(sc("embedding_length", 0)), int(sc("attention.head_count", 0))
  interval = int(sc("full_attention_interval", 4)); is_attn = lambda L: (L + 1) % interval == 0
  nk, dk, conv_k = int(sc("ssm.group_count", 0)), int(sc("ssm.state_size", 0)), int(sc("ssm.conv_kernel", 0))
  nv = int(tn["blk.0.ssm_a"].data.shape[0]); dv = int(sc("ssm.inner_size", 0)) // nv
  cfg = llm.LlamaConfig(
    dim=dim, n_layers=n, n_heads=heads, n_kv_heads=int(sc("attention.head_count_kv", heads)),
    ffn_dim=int(sc("feed_forward_length", 0)), vocab=int(tn["token_embd.weight"].data.shape[0]),
    rope_base=float(sc("rope.freq_base", 1e4)), norm_eps=float(sc("attention.layer_norm_rms_epsilon", 1e-6)),
    head_dim=int(sc("attention.key_length", 0)), rotary_dim=int(sc("rope.dimension_count", 0)),
    layers=[llm.LayerSpec(mixer="gated_attention" if is_attn(L) else "gated_deltanet") for L in range(n)],
    extra={"nk": nk, "nv": nv, "dk": dk, "dv": dv, "conv_k": conv_k, "gqa_repeat": "tile"})  # GGUF: ggml_repeat tiles

  def _drop_mmap():                                             # release the GGUF's resident pages (low warmup peak)
    mm = getattr(r.data, "_mmap", None)
    if mm is not None:
      try: mm.madvise(_mmap.MADV_DONTNEED)
      except (AttributeError, OSError): pass

  S = np.float16(1.0 / resid_scale)                            # residual-writing projections carry the 1/S factor
  def layer(L):                                                # materialize one hybrid layer's fp16 weights
    b = f"blk.{L}."
    d = {"in_norm": get(b + "attn_norm.weight"), "mlp_norm": get(b + "post_attention_norm.weight"),
         "wgate": get(b + "ffn_gate.weight"), "wup": get(b + "ffn_up.weight"), "wdown": get(b + "ffn_down.weight") * S}
    if is_attn(L):
      d |= {"wq": get(b + "attn_q.weight"), "wk": get(b + "attn_k.weight"), "wv": get(b + "attn_v.weight"),
            "wo": get(b + "attn_output.weight") * S, "q_norm": get(b + "attn_q_norm.weight"), "k_norm": get(b + "attn_k_norm.weight")}
    else:                                                       # GGUF: norms baked (1+w), ssm_a = -exp(A_log), conv1d [CD,K]
      d |= {"in_proj_qkv": get(b + "attn_qkv.weight"), "in_proj_z": get(b + "attn_gate.weight"),
            "in_proj_a": get(b + "ssm_alpha.weight"), "in_proj_b": get(b + "ssm_beta.weight"),
            "conv1d": get(b + "ssm_conv1d.weight", np.float32), "neg_exp_A": get(b + "ssm_a", np.float32),
            "dt_bias": get(b + "ssm_dt.bias", np.float32), "ssm_norm": get(b + "ssm_norm.weight"),
            "out_proj": get(b + "ssm_out.weight") * S}
    _drop_mmap()
    return d

  # embed (host gather) and lm_head (host matmul) stay fp32: numpy has no fp16 BLAS, so an fp16 lm_head at
  # vocab ~248k makes per-token logits pathologically slow. embed carries 1/S to match the scaled residual.
  w = {"embed": get("token_embd.weight", np.float32) / resid_scale, "final_norm": get("output_norm.weight"),
       "lm_head": get("output.weight", np.float32), "layers": _LazyLayers(layer, n)}
  return llm.LlamaPrefill(cfg, w, compress=compress)
