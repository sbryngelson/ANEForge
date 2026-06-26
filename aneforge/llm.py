"""LLM prefill on the Apple Neural Engine. A config-driven Llama/Qwen-style decoder
(RMSNorm + RoPE + grouped-query causal attention + SwiGLU) that loads real Hugging Face
weights and prefills a prompt as ONE fused ANE program -- the compute-bound phase where the
Neural Engine is most energy-efficient. Matches HF logits; see `examples/llm_prefill.py`."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .graph import Tensor, input as _input, concat as _concat, _const
from . import _compile


@dataclass
class LlamaConfig:
  """A Llama/Qwen-class decoder config (the fields the prefill graph needs). `head_dim` defaults to
  `dim // n_heads` but is explicit for Qwen3 (where it differs)."""
  dim: int; n_layers: int; n_heads: int; n_kv_heads: int; ffn_dim: int; vocab: int
  rope_base: float = 10000.0; norm_eps: float = 1e-5; head_dim: int = 0

  @property
  def dh(self) -> int: return self.head_dim or self.dim // self.n_heads


def rope_tables(seq: int, dh: int, base: float = 10000.0):
  """Precompute the [seq, dh] cos/sin rotation tables (HF Llama layout: freqs duplicated over the two halves)."""
  inv = 1.0 / (base ** (np.arange(0, dh, 2) / dh))
  pos = np.arange(seq)[:, None] * inv[None, :]
  emb = np.concatenate([pos, pos], -1)
  return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)


def rope(x: Tensor, cos, sin) -> Tensor:
  """Apply rotary position embedding to `x` [.., seq, dh]: `x*cos + rotate_half(x)*sin`."""
  dh = x.shape[-1]; half = dh // 2; rank = len(x.shape)
  s = list(x.shape); s[-1] = half
  x1 = x.slice_by_size([0] * rank, s)
  b2 = [0] * rank; b2[-1] = half
  x2 = x.slice_by_size(b2, s)
  return x * _const(cos) + _concat([x2 * -1.0, x1], axis=rank - 1) * _const(sin)


def _repeat_kv(k: Tensor, g: int) -> Tensor:
  """Grouped-query repeat: [1, KV, S, dh] -> [1, KV*g, S, dh], each kv head repeated `g` times (via a 0/1 expansion matmul; any `g`)."""
  if g == 1: return k
  _, KV, S, dh = k.shape
  M = np.repeat(np.eye(KV, dtype=np.float16), g, axis=0)        # [KV*g, KV]: M[h, kv] = 1 iff kv == h // g
  k2 = k.reshape(KV, S * dh).transpose([1, 0])                  # [S*dh, KV]
  return k2.linear(M).transpose([1, 0]).reshape(1, KV * g, S, dh)


def _causal_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  """Causal self-attention on q,k,v [1,H,S,dh] as the decomposed `softmax(q@kᵀ·scale + causal_mask)@v`.
  For prefill this stays in ONE fused program (compute-bound big matmuls — what the ANE excels at);
  the native fused-attention layer is avoided here because its per-layer graph cut dominates prefill cost."""
  S, dh = q.shape[2], q.shape[3]
  mask = np.triu(np.full((S, S), -1e4, np.float16), 1)     # causal: keys after the query are masked out
  return ((q @ k.transpose([0, 1, 3, 2])) * (1.0 / dh ** 0.5) + _const(mask)).softmax(-1) @ v


def prefill_block(x: Tensor, w: dict, cfg: LlamaConfig, cos, sin) -> Tensor:
  """One pre-norm Llama/Qwen decoder block over a prompt `x` [S, dim]: RMSNorm -> QKV (+ Qwen3 QK-norm)
  -> RoPE -> causal attention -> SwiGLU MLP, with residuals."""
  H, KV, dh, S = cfg.n_heads, cfg.n_kv_heads, cfg.dh, x.shape[0]
  xn = x.rms_norm(w["attn_norm"], cfg.norm_eps)
  q = xn.linear(w["wq"]).reshape(S, H, dh); k = xn.linear(w["wk"]).reshape(S, KV, dh); v = xn.linear(w["wv"]).reshape(S, KV, dh)
  if "q_norm" in w:                                 # Qwen3 QK-norm: per-head RMSNorm over head_dim before RoPE
    q = q.reshape(S * H, dh).rms_norm(w["q_norm"], cfg.norm_eps).reshape(S, H, dh)
    k = k.reshape(S * KV, dh).rms_norm(w["k_norm"], cfg.norm_eps).reshape(S, KV, dh)
  q = q.transpose([1, 0, 2]).reshape(1, H, S, dh); k = k.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  v = v.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  q, k = rope(q, cos, sin), rope(k, cos, sin)
  k, v = _repeat_kv(k, H // KV), _repeat_kv(v, H // KV)
  a = _causal_attn(q, k, v).reshape(H, S, dh).transpose([1, 0, 2]).reshape(S, H * dh)
  h = x + a.linear(w["wo"])
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"])


class LlamaPrefill:
  """A Llama/Qwen decoder compiled for prefill on the ANE. `compile(seq)` builds the graph for a fixed
  prompt length; `prefill(token_ids)` returns the next-token logits. Weights are a dict of numpy arrays
  (see `from_pretrained`)."""
  def __init__(self, cfg: LlamaConfig, weights: dict):
    self.cfg = cfg; self.w = weights; self._net = None; self._seq = 0

  def compile(self, seq: int):
    cfg = self.cfg
    cos, sin = rope_tables(seq, cfg.dh, cfg.rope_base)
    x = _input((seq, cfg.dim))
    for lw in self.w["layers"]:
      x = prefill_block(x, lw, cfg, cos, sin)
    self._net = _compile.compile(x.rms_norm(self.w["final_norm"], cfg.norm_eps))   # ANE -> all positions' hidden [seq, dim]
    self._seq = seq
    return self

  def _hidden(self, token_ids):
    """Run the transformer layers on the ANE; return the final hidden states [S, dim]."""
    if self._net is None or len(token_ids) != self._seq:
      self.compile(len(token_ids))
    net = self._net; assert net is not None
    emb = self.w["embed"][np.asarray(token_ids)]           # host embedding gather -> [S, dim]
    return np.asarray(net(emb.astype(np.float16))).astype(np.float32)

  def _logits(self, hidden_row):
    return hidden_row @ np.asarray(self.w["lm_head"]).T    # host lm_head (vocab can exceed the ANE per-op dim limit)

  def prefill(self, token_ids):
    """Prefill `token_ids` and return the next-token logits [1, vocab] (the transformer runs on the ANE)."""
    return self._logits(self._hidden(token_ids)[-1])[None]

  def generate(self, token_ids, max_new_tokens=40, eos_id=None):
    """Greedy autoregressive generation. Prompt + generated tokens fit one fixed-length ANE program (length
    `len(prompt)+max_new_tokens`); each step re-runs it and reads the next-token logits at the current position.
    Returns the generated token ids (prompt excluded)."""
    prompt = [int(t) for t in token_ids]; n = len(prompt)
    full = n + max_new_tokens
    self.compile(full)
    toks = prompt + [0] * max_new_tokens                   # pad to the fixed length (causal mask ignores future positions)
    out = []
    for cur in range(n, full):
      nxt = int(self._logits(self._hidden(toks)[cur - 1]).argmax())   # next token from the last real position
      out.append(nxt)
      if nxt == eos_id: break
      toks[cur] = nxt
    return out


def _weights_from_state_dict(sd, cfg: LlamaConfig) -> dict:
  """Map a Llama/Qwen Hugging Face state_dict (numpy arrays) into the prefill weight layout."""
  w = {"embed": sd["model.embed_tokens.weight"], "final_norm": sd["model.norm.weight"],
       "lm_head": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]), "layers": []}
  for L in range(cfg.n_layers):
    p = f"model.layers.{L}."
    lw = {
      "wq": sd[p + "self_attn.q_proj.weight"], "wk": sd[p + "self_attn.k_proj.weight"],
      "wv": sd[p + "self_attn.v_proj.weight"], "wo": sd[p + "self_attn.o_proj.weight"],
      "wgate": sd[p + "mlp.gate_proj.weight"], "wup": sd[p + "mlp.up_proj.weight"],
      "wdown": sd[p + "mlp.down_proj.weight"], "attn_norm": sd[p + "input_layernorm.weight"],
      "mlp_norm": sd[p + "post_attention_layernorm.weight"]}
    if p + "self_attn.q_norm.weight" in sd:         # Qwen3 QK-norm
      lw["q_norm"] = sd[p + "self_attn.q_norm.weight"]; lw["k_norm"] = sd[p + "self_attn.k_norm.weight"]
    w["layers"].append(lw)
  return w


def _cfg_from_hf(c) -> LlamaConfig:
  return LlamaConfig(dim=c.hidden_size, n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                     n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                     ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                     rope_base=float(getattr(c, "rope_theta", 10000.0)), norm_eps=float(c.rms_norm_eps),
                     head_dim=int(getattr(c, "head_dim", 0) or 0))


def from_pretrained(name: str) -> LlamaPrefill:
  """Load a Llama/Qwen-class model from Hugging Face and prepare it for ANE prefill."""
  from transformers import AutoModelForCausalLM
  hf = AutoModelForCausalLM.from_pretrained(name)
  cfg = _cfg_from_hf(hf.config)
  sd = {k: v.detach().float().numpy() for k, v in hf.state_dict().items()}
  return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg))
