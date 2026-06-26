"""LLMs on the Apple Neural Engine. A config-driven Llama/Qwen decoder
(RMSNorm + RoPE + grouped-query causal attention + SwiGLU) that loads Hugging Face weights and
runs prefill and KV-cache decode as fused ANE programs. Matches HF logits; see `docs/llm.md`."""
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
  """Apply rotary position embedding to `x` [.., seq, dh]: `x*cos + rotate_half(x)*sin`. `cos`/`sin` may be
  numpy tables (prefill, baked as constants) or graph Tensors (decode, a runtime per-position row)."""
  dh = x.shape[-1]; half = dh // 2; rank = len(x.shape)
  s = list(x.shape); s[-1] = half
  x1 = x.slice_by_size([0] * rank, s)
  b2 = [0] * rank; b2[-1] = half
  x2 = x.slice_by_size(b2, s)
  c = cos if isinstance(cos, Tensor) else _const(cos); sn = sin if isinstance(sin, Tensor) else _const(sin)
  return x * c + _concat([x2 * -1.0, x1], axis=rank - 1) * sn


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


def _repeat_kv3(k: Tensor, g: int) -> Tensor:
  """Grouped-query repeat for the decode cache: [KV, M, dh] -> [KV*g, M, dh] (0/1 expansion matmul)."""
  if g == 1: return k
  KV, M, dh = k.shape
  Mx = np.repeat(np.eye(KV, dtype=np.float16), g, axis=0)
  return k.reshape(KV, M * dh).transpose([1, 0]).linear(Mx).transpose([1, 0]).reshape(KV * g, M, dh)


def _decode_block(x: Tensor, w: dict, cfg: LlamaConfig, Kin, Vin, oh, inv, mask, cosp, sinp):
  """One decoder block for a single new token, attending to a resident KV-cache. `x` [1, dim]; `Kin`/`Vin`
  [KV, M, dh] are the cache; `oh`/`inv` [1, M, 1] write the new K/V at the current position; `cosp`/`sinp`
  [1, dh] are RoPE for that position. Returns (hidden, Kout, Vout) where Kout/Vout become the resident cache."""
  H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.dh
  xn = x.rms_norm(w["attn_norm"], cfg.norm_eps)
  q = xn.linear(w["wq"]).reshape(H, dh); k = xn.linear(w["wk"]).reshape(KV, dh); v = xn.linear(w["wv"]).reshape(KV, dh)
  if "q_norm" in w:
    q = q.rms_norm(w["q_norm"], cfg.norm_eps); k = k.rms_norm(w["k_norm"], cfg.norm_eps)
  q = rope(q.reshape(H, 1, dh), cosp, sinp)        # [H, 1, dh]
  k = rope(k.reshape(KV, 1, dh), cosp, sinp); v = v.reshape(KV, 1, dh)
  Kout = Kin * inv + k * oh; Vout = Vin * inv + v * oh        # masked positional write into the cache
  Kr, Vr = _repeat_kv3(Kout, H // KV), _repeat_kv3(Vout, H // KV)
  sc = ((q @ Kr.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)   # [H, 1, M]
  a = (sc @ Vr).reshape(H, 1, dh).transpose([1, 0, 2]).reshape(1, H * dh)
  h = x + a.linear(w["wo"])
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"]), Kout, Vout


class LlamaPrefill:
  """A Llama/Qwen decoder compiled for prefill on the ANE. `compile(seq)` builds the graph for a fixed
  prompt length; `prefill(token_ids)` returns the next-token logits. Weights are a dict of numpy arrays
  (see `from_pretrained`)."""
  def __init__(self, cfg: LlamaConfig, weights: dict, compress: str | None = None):
    self.cfg = cfg; self.w = weights; self._net = None; self._seq = 0; self._dec = None
    self.compress = compress       # None=fp16, or "int8"/"int4"/"blockwise" to quantize the ANE weights

  def compile(self, seq: int):
    cfg = self.cfg
    cos, sin = rope_tables(seq, cfg.dh, cfg.rope_base)
    x = _input((seq, cfg.dim))
    for lw in self.w["layers"]:
      x = prefill_block(x, lw, cfg, cos, sin)
    self._net = _compile.compile(x.rms_norm(self.w["final_norm"], cfg.norm_eps),   # ANE -> all positions' hidden [seq, dim]
                                 compress=self.compress)
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

  def _decoder(self, M):
    """Build (once, cached) the resident-KV-cache decode-step program for a max sequence length `M`: all layers
    run for one token attending to per-layer caches that stay on-device across `execute` via `share_buffer`."""
    if self._dec is not None and self._dec["M"] == M: return self._dec
    if self._dec is not None: self._dec["net"].release()
    from ._compile import compile_multi
    cfg = self.cfg; KV, dh, D, L = cfg.n_kv_heads, cfg.dh, cfg.dim, cfg.n_layers
    x = _input((1, D)); oh = _input((1, M, 1)); inv = _input((1, M, 1)); mask = _input((1, 1, M))
    cosp = _input((1, dh)); sinp = _input((1, dh))
    Kin = [_input((KV, M, dh)) for _ in range(L)]; Vin = [_input((KV, M, dh)) for _ in range(L)]
    h = x; Kout = []; Vout = []
    for lw, ki, vi in zip(self.w["layers"], Kin, Vin):
      h, ko, vo = _decode_block(h, lw, cfg, ki, vi, oh, inv, mask, cosp, sinp)
      Kout.append(ko); Vout.append(vo)
    h = h.rms_norm(self.w["final_norm"], cfg.norm_eps)
    net = compile_multi([h] + Kout + Vout, compress=self.compress)
    inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
    for ki, vi, ko, vo in zip(Kin, Vin, Kout, Vout):           # alias each layer's cache output -> its input (resident)
      net.prog.share_buffer(0, om[ko], 0, inm[id(ki)]); net.prog.share_buffer(0, om[vo], 0, inm[id(vi)])
    cos_t, sin_t = rope_tables(M, dh, cfg.rope_base)
    ports = {"x": inm[id(x)], "oh": inm[id(oh)], "inv": inm[id(inv)], "mask": inm[id(mask)],
             "cosp": inm[id(cosp)], "sinp": inm[id(sinp)], "h": om[h], "kv": [(inm[id(ki)], inm[id(vi)]) for ki, vi in zip(Kin, Vin)]}
    self._dec = {"M": M, "net": net, "p": ports, "cos": cos_t, "sin": sin_t}
    return self._dec

  def warmup(self, max_len):
    """Compile and cache the decode program for context length `max_len` (the one-time cost) so the next
    `generate` streams immediately. Returns self."""
    self._decoder(int(max_len)); return self

  def generate(self, token_ids, max_new_tokens=40, eos_id=None, max_len=None, on_token=None):
    """Greedy autoregressive generation with a resident KV-cache. The decode program (compiled once and
    cached) runs all layers for one token attending to caches that stay on the ANE across steps, so each step
    feeds only the token embedding + a position one-hot. `on_token(id)` is called per token (streaming).
    Returns the generated token ids."""
    cfg = self.cfg; KV, dh = cfg.n_kv_heads, cfg.dh; f16 = np.float16
    prompt = [int(t) for t in token_ids]; M = int(max_len or len(prompt) + max_new_tokens)
    d = self._decoder(M); net = d["net"]; p = d["p"]
    for ki, vi in p["kv"]:                                      # reset the resident cache for this generation
      net.prog.set_input(ki, np.zeros((KV, M, dh), f16)); net.prog.set_input(vi, np.zeros((KV, M, dh), f16))
    def step(tok, pos):
      ohv = np.zeros((1, M, 1), f16); ohv[0, pos, 0] = 1.0
      invv = np.ones((1, M, 1), f16); invv[0, pos, 0] = 0.0
      mv = np.full((1, 1, M), -1e4, f16); mv[..., :pos + 1] = 0.0
      net.prog.set_input(p["x"], np.asarray(self.w["embed"][tok])[None].astype(f16))
      net.prog.set_input(p["oh"], ohv); net.prog.set_input(p["inv"], invv); net.prog.set_input(p["mask"], mv)
      net.prog.set_input(p["cosp"], d["cos"][pos][None]); net.prog.set_input(p["sinp"], d["sin"][pos][None])
      net.prog.execute()
      return np.asarray(net.prog.read_output(p["h"])).reshape(cfg.dim).astype(np.float32)
    for pos, tok in enumerate(prompt[:-1]): step(tok, pos)      # prefill the cache (discard hidden)
    cur = prompt[-1]; pos = len(prompt) - 1; out = []
    for _ in range(max_new_tokens):                            # decode
      if pos >= M - 1: break                                   # cache full
      nxt = int(self._logits(step(cur, pos)).argmax()); out.append(nxt)
      if nxt == eos_id: break
      if on_token is not None: on_token(nxt)                   # stream the token
      cur = nxt; pos += 1
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


def from_pretrained(name: str, compress: str | None = None) -> LlamaPrefill:
  """Load a Llama/Qwen-class model from Hugging Face for ANE inference. `compress` ("int8"/"int4"/"blockwise")
  quantizes the ANE weights."""
  from transformers import AutoModelForCausalLM
  hf = AutoModelForCausalLM.from_pretrained(name)
  cfg = _cfg_from_hf(hf.config)
  sd = {k: v.detach().float().numpy() for k, v in hf.state_dict().items()}
  return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg), compress=compress)
