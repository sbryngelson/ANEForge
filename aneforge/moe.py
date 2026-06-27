"""Sparse Mixture-of-Experts on the ANE: registers a `moe` MLP into the `llm` registries and loads Qwen-MoE
GGUFs. Routing matches HF (softmax over experts -> top-k -> optional renorm); every expert is baked resident
and computed densely (one batched matmul over the expert axis), then weighted by the top-k (others get 0).
Dense is the right trade here -- decode is latency-bound, and the hidden state stays on-chip (no host
round-trip for routing).

Canonical per-layer weights: mlp_norm, router [E,dim], wgate_exps/wup_exps/wdown_exps [E,ffn,dim] (down stored
transposed for the batched `act @ wdown`); Qwen2-MoE adds a shared expert (wgate_sh/wup_sh/wdown_sh/shared_gate)
and QKV biases. cfg.extra: n_experts, n_experts_per_tok, norm_topk_prob."""
from __future__ import annotations
import numpy as np

from .graph import _const, concat as _concat, select
from . import llm
from .llm import LayerSpec, LlamaConfig

_MAX_DIM = 65536          # ANE per-op dimension cap; the all-expert gate/up matmul tiles under it


def _topk_gate(probs, k, E, T):
  """probs [T,E] masked to the top-k per row (0 elsewhere), WITHOUT the native `topk`/`sort` ops -- those are
  graph cuts that don't compose (two MoE layers in one program fail to compile). Peel the row-max k times with
  `amax`+`select` instead; exact for distinct logits. Caller renormalizes."""
  neg = _const(np.full((T, E), -3.0e4, np.float16)); one = _const(np.ones((T, E), np.float16))
  chosen = _const(np.zeros((T, E), np.float16)); rem = probs
  for _ in range(k):
    hit = rem.greater_equal(rem.amax([1]).reshape(T, 1))         # this round's row-max position
    chosen = select(hit, one, chosen); rem = select(hit, neg, rem)   # mark it, then drop it for the next round
  return probs * chosen


def _experts_gate(hn, wexps, E, ffn, dim, T):
  """hn @ wexps[e].T over all experts -> [T, E, ffn], tiled so each matmul's output (g*ffn) stays under _MAX_DIM."""
  g = max(1, min(E, _MAX_DIM // ffn))
  parts = [hn.linear(wexps[s:s + g].reshape(min(g, E - s) * ffn, dim)).reshape(T, min(g, E - s), ffn)
           for s in range(0, E, g)]
  return parts[0] if len(parts) == 1 else _concat(parts, axis=1)


def _moe(h, w, cfg, ls):
  """Top-k sparse MoE SwiGLU + residual. Stateless -- one builder for prefill [S,dim] and decode [1,dim]. All
  experts computed densely and combined by the renormalized top-k weights (== exact top-k routing)."""
  e = cfg.extra; E, k, dim = e["n_experts"], e["n_experts_per_tok"], cfg.dim
  ffn, T = w["wgate_exps"].shape[1], h.shape[0]
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  gate_w = _topk_gate(hn.linear(w["router"]).softmax(-1), k, E, T)
  if e.get("norm_topk_prob", True):
    gate_w = gate_w / gate_w.sum([1]).reshape(T, 1)
  gate, up = _experts_gate(hn, w["wgate_exps"], E, ffn, dim, T), _experts_gate(hn, w["wup_exps"], E, ffn, dim, T)
  out = (gate.silu() * up).transpose([1, 0, 2]).bmm_weight(w["wdown_exps"])   # [E,T,ffn] @ [E,ffn,dim] (quantizable)
  combined = (out * gate_w.transpose([1, 0]).reshape(E, T, 1)).sum([0]).reshape(T, dim)
  if "wgate_sh" in w:                                            # Qwen2-MoE always-on, sigmoid-gated shared expert
    sh = (hn.linear(w["wgate_sh"]).silu() * hn.linear(w["wup_sh"])).linear(w["wdown_sh"])
    combined = combined + sh * hn.linear(w["shared_gate"]).sigmoid()
  return h + combined


llm.PREFILL_MLPS["moe"] = _moe
llm.DECODE_MLPS["moe"] = _moe


def _moe_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Qwen3-MoE HF adapter: attention layers as usual, sparse FFN layers carry the fused expert weights
  (experts.gate_up_proj [E,2*ffn,dim], experts.down_proj [E,dim,ffn]) + a router, laid out for `_moe`.
  `decoder_sparse_step`/`mlp_only_layers` pick MoE vs plain-SwiGLU layers."""
  n, E, k = int(c.num_hidden_layers), int(c.num_experts), int(c.num_experts_per_tok)
  moe_ffn = int(c.moe_intermediate_size)
  qk = "model.layers.0.self_attn.q_norm.weight" in sd
  step = int(getattr(c, "decoder_sparse_step", 1) or 1)
  mlp_only = set(getattr(c, "mlp_only_layers", []) or [])
  is_moe = [(L not in mlp_only) and ((L + 1) % step == 0) for L in range(n)]
  specs = [LayerSpec(mixer="attention", mlp=("moe" if is_moe[L] else "swiglu"), qk_norm=qk) for L in range(n)]
  cfg = LlamaConfig(dim=c.hidden_size, n_layers=n, n_heads=c.num_attention_heads,
                    n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                    ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                    rope_base=float(getattr(c, "rope_theta", 10000.0)), norm_eps=float(c.rms_norm_eps),
                    head_dim=int(getattr(c, "head_dim", 0) or 0), layers=specs,
                    extra={"n_experts": E, "n_experts_per_tok": k,
                           "norm_topk_prob": bool(getattr(c, "norm_topk_prob", True))})
  w = {"embed": sd["model.embed_tokens.weight"], "final_norm": sd["model.norm.weight"],
       "lm_head": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]), "layers": []}
  for L in range(n):
    p = f"model.layers.{L}."
    lw = {"wq": sd[p + "self_attn.q_proj.weight"], "wk": sd[p + "self_attn.k_proj.weight"],
          "wv": sd[p + "self_attn.v_proj.weight"], "wo": sd[p + "self_attn.o_proj.weight"],
          "attn_norm": sd[p + "input_layernorm.weight"], "mlp_norm": sd[p + "post_attention_layernorm.weight"]}
    if qk:
      lw["q_norm"] = sd[p + "self_attn.q_norm.weight"]; lw["k_norm"] = sd[p + "self_attn.k_norm.weight"]
    if is_moe[L]:
      gu = sd[p + "mlp.experts.gate_up_proj"]                                  # [E, 2*ffn, dim] (out = gate||up)
      lw["router"] = sd[p + "mlp.gate.weight"]
      lw["wgate_exps"] = np.ascontiguousarray(gu[:, :moe_ffn, :])
      lw["wup_exps"] = np.ascontiguousarray(gu[:, moe_ffn:, :])
      lw["wdown_exps"] = np.ascontiguousarray(sd[p + "mlp.experts.down_proj"].transpose(0, 2, 1))  # ->[E,ffn,dim]
    else:
      lw["wgate"] = sd[p + "mlp.gate_proj.weight"]; lw["wup"] = sd[p + "mlp.up_proj.weight"]
      lw["wdown"] = sd[p + "mlp.down_proj.weight"]
    w["layers"].append(lw)
  return cfg, w


llm.ADAPTERS.append((lambda c: bool(getattr(c, "num_experts", 0)), _moe_adapter))


class _LazyLayers:
  """Per-layer weights materialized on demand and freed once a decode chunk has baked them, so the full model
  never holds every layer's fp16 at once (48L fp16 ~60GB > RAM). `_decoder` calls `free(i)` after each chunk
  compiles; re-materializing (e.g. recompiling for a new context length) just re-reads the GGUF."""
  def __init__(self, materialize, n: int):
    self._mk = materialize; self._n = n; self._cache: dict = {}

  def __len__(self) -> int: return self._n

  def __getitem__(self, i: int) -> dict:
    if i not in self._cache: self._cache[i] = self._mk(i)
    return self._cache[i]

  def free(self, i: int) -> None: self._cache.pop(i, None)


def load_gguf(path: str, n_layers: int | None = None, compress: str | None = None) -> "llm.LlamaPrefill":
  """Load a Qwen-MoE GGUF (Qwen3-MoE e.g. 30B-A3B, or Qwen2-MoE e.g. Qwen1.5-MoE-A2.7B), dequantizing each
  tensor to fp16. Arch features are detected from the tensors (QK-norm, QKV biases + shared expert, GQA,
  head_dim). The reader returns PyTorch [out,in] order, so the only re-layout is ffn_down_exps -> [E,ffn,dim].
  `n_layers` truncates the model (the full 30B is ~60GB fp16 > RAM); layers load lazily and free after baking."""
  import gguf
  from gguf.quants import dequantize
  r = gguf.GGUFReader(path)
  meta = {f.name: f for f in r.fields.values()}
  arch = next(k.split(".")[0] for k in meta if k.endswith(".block_count"))
  def sc(key, default=None):
    f = meta.get(arch + "." + key)
    return f.parts[f.data[-1]][0] if f is not None else default
  tn = {t.name: t for t in r.tensors}
  has = lambda name: name in tn

  def get(name, dtype: np.typing.DTypeLike = np.float16):    # dequantize -> fp16 (halves resident memory)
    t = tn[name]; d = np.asarray(t.data)
    if t.tensor_type != gguf.GGMLQuantizationType.F32:
      d = dequantize(d, t.tensor_type)
    return d.astype(dtype)
  opt = lambda name: get(name) if has(name) else None

  n_total = int(sc("block_count", 0)); n = n_total if n_layers is None else min(n_layers, n_total)
  E, k = int(sc("expert_count", 0)), int(sc("expert_used_count", 0))
  dim, vocab = int(sc("embedding_length", 0)), int(tn["token_embd.weight"].data.shape[0])
  heads = int(sc("attention.head_count", 0))
  moe_ffn = int(tn["blk.0.ffn_gate_exps.weight"].shape[1])   # GGUF [dim, ffn, E] -> ffn (no metadata key on Qwen1.5)
  qk = has("blk.0.attn_q_norm.weight")                       # Qwen3 has QK-norm; Qwen2 doesn't
  shared = has("blk.0.ffn_gate_shexp.weight")                # Qwen2-MoE shared expert
  cfg = llm.LlamaConfig(
    dim=dim, n_layers=n, n_heads=heads, n_kv_heads=int(sc("attention.head_count_kv", heads)),
    ffn_dim=moe_ffn, vocab=vocab, rope_base=float(sc("rope.freq_base", 10000.0)),
    norm_eps=float(sc("attention.layer_norm_rms_epsilon", 1e-6)), head_dim=int(sc("attention.key_length") or dim // heads),
    layers=[llm.LayerSpec(mixer="attention", mlp="moe", qk_norm=qk) for _ in range(n)],
    extra={"n_experts": E, "n_experts_per_tok": k, "norm_topk_prob": arch != "qwen2moe"})  # Qwen1.5 doesn't renorm

  import mmap as _mmap
  def _drop_mmap():                                          # release the GGUF's resident pages (low warmup peak)
    mm = getattr(r.data, "_mmap", None)
    if mm is not None:
      try: mm.madvise(_mmap.MADV_DONTNEED)
      except (AttributeError, OSError): pass

  def layer(L):                                             # materialize one layer's fp16 weights on demand
    b = f"blk.{L}."
    d = {"wq": get(b + "attn_q.weight"), "wk": get(b + "attn_k.weight"), "wv": get(b + "attn_v.weight"),
         "wo": get(b + "attn_output.weight"), "attn_norm": get(b + "attn_norm.weight"),
         "mlp_norm": get(b + "ffn_norm.weight"), "router": get(b + "ffn_gate_inp.weight"),
         "wgate_exps": get(b + "ffn_gate_exps.weight"), "wup_exps": get(b + "ffn_up_exps.weight"),
         "wdown_exps": np.ascontiguousarray(get(b + "ffn_down_exps.weight").transpose(0, 2, 1))}  # ->[E,ffn,dim]
    for key, nm in (("q_norm", "attn_q_norm.weight"), ("k_norm", "attn_k_norm.weight"),
                    ("q_bias", "attn_q.bias"), ("k_bias", "attn_k.bias"), ("v_bias", "attn_v.bias")):
      v = opt(b + nm)                                        # QK-norm (Qwen3) / QKV biases (Qwen2) -- only if present
      if v is not None: d[key] = v
    if shared:                                               # Qwen2-MoE shared expert (SwiGLU FFN + sigmoid gate)
      d["wgate_sh"] = get(b + "ffn_gate_shexp.weight"); d["wup_sh"] = get(b + "ffn_up_shexp.weight")
      d["wdown_sh"] = get(b + "ffn_down_shexp.weight"); d["shared_gate"] = get(b + "ffn_gate_inp_shexp.weight").reshape(1, dim)
    _drop_mmap()
    return d

  # embed (host gather) and lm_head (host matmul) stay fp32: numpy has no fp16 BLAS, so an fp16 lm_head at
  # vocab 151936 makes per-token logits pathologically slow.
  w = {"embed": get("token_embd.weight", np.float32), "final_norm": get("output_norm.weight"),
       "lm_head": get("output.weight", np.float32), "layers": _LazyLayers(layer, n)}
  return llm.LlamaPrefill(cfg, w, compress=compress)
