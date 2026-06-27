"""Qwen3-MoE (sparse Mixture-of-Experts) support. Registers a `moe` MLP into the `llm` registries.

Routing follows the HF Qwen3-MoE convention: softmax over all experts, take the top-k, then renormalize the
selected weights. On the ANE every expert is baked resident and computed *densely* (one batched matmul over
the expert axis), then weighted by the top-k router -- experts outside the top-k get weight 0. This trades
FLOPs for simplicity, which is the right trade here: the ANE decode is latency-bound (computing all experts
costs ~the same as a few), and the dense path keeps the hidden state on-chip with no per-layer host
round-trip for the data-dependent routing.

Canonical weights (from the adapter): mlp_norm [dim], router [n_experts, dim], wgate_exps/wup_exps
[n_experts, ffn, dim], wdown_exps [n_experts, ffn, dim] (down stored transposed [E, ffn, dim] so the
per-expert projection is a single batched `act @ wdown`). MoE dims live in cfg.extra: n_experts,
n_experts_per_tok, norm_topk_prob."""
from __future__ import annotations
import numpy as np

from .graph import _const, concat as _concat, select
from . import llm
from .llm import LayerSpec, LlamaConfig

_MAX_DIM = 65536          # ANE per-op dimension cap; the all-expert gate/up matmul must tile under it


def _topk_gate(probs, k, E, T):
  """Top-k routing weights WITHOUT a graph cut. The native `topk`/`sort` ops are graph cuts that don't
  compose -- two of them in one program (i.e. two MoE layers) fail to compile -- so build the top-k mask by
  peeling the row-max k times with plain reductions (`amax`) and `select`. Returns probs masked to the top-k
  (0 elsewhere); the caller renormalizes. Selection matches exact top-k for distinct logits."""
  neg = _const(np.full((T, E), -3.0e4, np.float16))
  one = _const(np.ones((T, E), np.float16)); chosen = _const(np.zeros((T, E), np.float16))
  rem = probs
  for _ in range(k):
    hit = rem.greater_equal(rem.amax([1]).reshape(T, 1))   # [T,E] bool: this round's row-max position
    chosen = select(hit, one, chosen)                      # accumulate the selected experts
    rem = select(hit, neg, rem)                            # drop it before the next round
  return probs * chosen                                    # keep probs at the top-k, 0 elsewhere


def _experts_gate(hn, wexps, E, ffn, dim, T):
  """All experts' projection hn @ wexps[e].T -> [T, E, ffn], tiled over experts so each matmul's output
  dimension (g*ffn) stays under the ANE's _MAX_DIM."""
  g = max(1, min(E, _MAX_DIM // ffn))
  parts = [hn.linear(wexps[s:s + g].reshape(min(g, E - s) * ffn, dim)).reshape(T, min(g, E - s), ffn)
           for s in range(0, E, g)]
  return parts[0] if len(parts) == 1 else _concat(parts, axis=1)


def _moe(h, w, cfg, ls):
  """Top-k sparse MoE SwiGLU MLP with its residual. Stateless -- one builder serves both prefill ([S, dim])
  and decode ([1, dim]). All experts are computed densely (batched) and combined by the renormalized top-k
  routing weights, so the output equals top-k routing exactly (non-selected experts contribute 0)."""
  e = cfg.extra
  E, k, dim = e["n_experts"], e["n_experts_per_tok"], cfg.dim
  ffn = w["wgate_exps"].shape[1]
  T = h.shape[0]
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)                       # [T, dim]
  probs = hn.linear(w["router"]).softmax(-1)                         # [T, E] softmax over all experts
  gate_w = _topk_gate(probs, k, E, T)                                # probs masked to top-k (cut-free)
  if e.get("norm_topk_prob", True):
    gate_w = gate_w / gate_w.sum([1]).reshape(T, 1)                  # renormalize over the selected k
  gate = _experts_gate(hn, w["wgate_exps"], E, ffn, dim, T)          # [T, E, ffn] (tiled over experts)
  up = _experts_gate(hn, w["wup_exps"], E, ffn, dim, T)
  act = (gate.silu() * up).transpose([1, 0, 2])                      # [E, T, ffn]
  out = act @ _const(np.ascontiguousarray(w["wdown_exps"]))          # [E,T,ffn] @ [E,ffn,dim] -> [E, T, dim]
  combined = (out * gate_w.transpose([1, 0]).reshape(E, T, 1)).sum([0]).reshape(T, dim)   # weight + combine
  return h + combined                                                          # residual -> [T, dim]


llm.PREFILL_MLPS["moe"] = _moe
llm.DECODE_MLPS["moe"] = _moe


def _moe_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Adapter for Qwen3-MoE: attention layers as usual; sparse FFN layers carry stacked expert weights and a
  router. Reads the fused HF expert layout -- experts.gate_up_proj [E, dim, 2*ffn] (gate||up) and
  experts.down_proj [E, ffn, dim] -- and lays them out for `_moe` (wgate_exps/wup_exps [E, ffn, dim] for the
  per-expert linears, wdown_exps [E, ffn, dim] for the batched matmul). `decoder_sparse_step`/`mlp_only_layers`
  pick which layers are MoE vs plain SwiGLU."""
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
      gu = sd[p + "mlp.experts.gate_up_proj"]                                    # [E, 2*ffn, dim] (out=gate||up)
      lw["router"] = sd[p + "mlp.gate.weight"]                                   # [E, dim]
      lw["wgate_exps"] = np.ascontiguousarray(gu[:, :moe_ffn, :])                # [E, ffn, dim] (Linear out,in)
      lw["wup_exps"] = np.ascontiguousarray(gu[:, moe_ffn:, :])                  # [E, ffn, dim]
      lw["wdown_exps"] = np.ascontiguousarray(sd[p + "mlp.experts.down_proj"].transpose(0, 2, 1))  # [E,dim,ffn]->[E,ffn,dim]
    else:
      lw["wgate"] = sd[p + "mlp.gate_proj.weight"]; lw["wup"] = sd[p + "mlp.up_proj.weight"]
      lw["wdown"] = sd[p + "mlp.down_proj.weight"]
    w["layers"].append(lw)
  return cfg, w


llm.ADAPTERS.append((lambda c: bool(getattr(c, "num_experts", 0)), _moe_adapter))


class _LazyLayers:
  """Per-layer weights materialized on demand and freed once a decode chunk has baked them, so the full model
  never holds every layer's fp16 at once (48L fp16 ~60GB > RAM). `LlamaPrefill._decoder` calls `free(i)` after
  each chunk compiles; re-materialization (e.g. recompiling for a new context length) just re-reads the GGUF."""
  def __init__(self, materialize, n: int):
    self._mk = materialize; self._n = n; self._cache: dict = {}

  def __len__(self) -> int: return self._n

  def __getitem__(self, i: int) -> dict:
    if i not in self._cache: self._cache[i] = self._mk(i)
    return self._cache[i]

  def free(self, i: int) -> None: self._cache.pop(i, None)


def load_gguf(path: str, n_layers: int | None = None, compress: str | None = None) -> "llm.LlamaPrefill":
  """Load a Qwen3-MoE GGUF (e.g. Qwen3-30B-A3B Q4_K_M) for ANE inference, dequantizing each tensor to fp16.

  GGUF stores weights transposed-and-reversed; the `gguf` reader already returns them in PyTorch [out, in]
  order, so the only re-layout is ffn_down_exps [E, dim, ffn] -> [E, ffn, dim] for the batched matmul.
  `n_layers` truncates to the first N decoder layers -- the full 30B is ~60GB in fp16 (> RAM), but a few
  layers fit, which is enough to measure whether MoE decode is latency-bound (dense viable) or
  bandwidth-bound. Norms/router are F32 in the file; everything is cast to fp16 to halve resident memory."""
  import gguf
  from gguf.quants import dequantize
  r = gguf.GGUFReader(path)
  meta = {f.name: f for f in r.fields.values()}
  arch = next(k.split(".")[0] for k in meta if k.endswith(".block_count"))
  sc = lambda key: meta[arch + "." + key].parts[meta[arch + "." + key].data[-1]][0]
  tn = {t.name: t for t in r.tensors}

  def get(name, dtype: np.typing.DTypeLike = np.float16):   # dequantize -> fp16 (halves resident memory)
    t = tn[name]; d = np.asarray(t.data)
    if t.tensor_type != gguf.GGMLQuantizationType.F32:
      d = dequantize(d, t.tensor_type)
    return d.astype(dtype)

  n_total = int(sc("block_count")); n = n_total if n_layers is None else min(n_layers, n_total)
  E, k = int(sc("expert_count")), int(sc("expert_used_count"))
  dim, vocab = int(sc("embedding_length")), tn["token_embd.weight"].data.shape[0]
  cfg = llm.LlamaConfig(
    dim=dim, n_layers=n, n_heads=int(sc("attention.head_count")), n_kv_heads=int(sc("attention.head_count_kv")),
    ffn_dim=int(sc("expert_feed_forward_length")), vocab=int(vocab), rope_base=float(sc("rope.freq_base")),
    norm_eps=float(sc("attention.layer_norm_rms_epsilon")), head_dim=int(sc("attention.key_length")),
    layers=[llm.LayerSpec(mixer="attention", mlp="moe", qk_norm=True) for _ in range(n)],
    extra={"n_experts": E, "n_experts_per_tok": k, "norm_topk_prob": True})
  def layer(L):                                            # materialize one layer's fp16 weights on demand
    b = f"blk.{L}."
    return {
      "wq": get(b + "attn_q.weight"), "wk": get(b + "attn_k.weight"), "wv": get(b + "attn_v.weight"),
      "wo": get(b + "attn_output.weight"), "q_norm": get(b + "attn_q_norm.weight"),
      "k_norm": get(b + "attn_k_norm.weight"), "attn_norm": get(b + "attn_norm.weight"),
      "mlp_norm": get(b + "ffn_norm.weight"), "router": get(b + "ffn_gate_inp.weight"),
      "wgate_exps": get(b + "ffn_gate_exps.weight"), "wup_exps": get(b + "ffn_up_exps.weight"),
      "wdown_exps": np.ascontiguousarray(get(b + "ffn_down_exps.weight").transpose(0, 2, 1))}  # [E,dim,ffn]->[E,ffn,dim]

  # embed (host gather) and lm_head (host matmul) stay fp32: numpy has no BLAS for fp16, so an fp16 lm_head
  # at vocab 151936 makes per-token logits pathologically slow. Layers are LAZY (_LazyLayers): the decoder
  # frees each layer's fp16 after its chunk bakes, so the full 48L (~60GB fp16) never resides all at once.
  w = {"embed": get("token_embd.weight", np.float32), "final_norm": get("output_norm.weight"),
       "lm_head": get("output.weight", np.float32), "layers": _LazyLayers(layer, n)}
  return llm.LlamaPrefill(cfg, w, compress=compress)
