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

from .graph import _const, select, topk
from . import llm
from .llm import LayerSpec, LlamaConfig


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
  thr = topk(probs, k).slice_by_size([0, k - 1], [T, 1])            # [T, 1]: the k-th largest prob (cutoff)
  gate_w = select(probs.greater_equal(thr), probs, _const(np.zeros((T, E), np.float16)))   # zero out non-top-k
  if e.get("norm_topk_prob", True):
    gate_w = gate_w / gate_w.sum([1]).reshape(T, 1)                  # renormalize over the selected k
  gate = hn.linear(w["wgate_exps"].reshape(E * ffn, dim)).reshape(T, E, ffn)   # all experts, one matmul
  up = hn.linear(w["wup_exps"].reshape(E * ffn, dim)).reshape(T, E, ffn)
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
