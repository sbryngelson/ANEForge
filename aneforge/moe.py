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
