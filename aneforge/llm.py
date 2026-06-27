"""LLMs on the Apple Neural Engine. A config-driven decoder (RMSNorm + RoPE + grouped-query causal
attention + SwiGLU) that loads Hugging Face weights and runs prefill and KV-cache decode as fused ANE
programs. Matches HF logits; see `docs/llm.md`.

The runner is model-agnostic: a `LlamaConfig` carries a per-layer plan (`LayerSpec`) naming a token-mixer
and an MLP from the builder registries below, so supporting a new architecture is an *adapter* that emits
the plan + canonical weights — not new branches in the hot path."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .graph import Tensor, input as _input, concat as _concat, _const
from . import _compile


@dataclass
class LayerSpec:
  """Per-layer plan: which token-mixer and MLP to build (registry keys), plus mixer flags. The runner
  dispatches on these rather than sniffing the weight dict."""
  mixer: str = "attention"      # key into PREFILL_MIXERS / DECODE_MIXERS
  mlp: str = "swiglu"           # key into PREFILL_MLPS / DECODE_MLPS
  qk_norm: bool = False          # per-head RMSNorm on Q/K before RoPE (Qwen3)


@dataclass
class LlamaConfig:
  """A decoder config (the fields the graph needs). `head_dim` defaults to `dim // n_heads` but is explicit
  for Qwen3 (where it differs). `layers` is the optional per-layer plan; when empty the runner falls back to
  all-attention with QK-norm detected from the weights (back-compat)."""
  dim: int; n_layers: int; n_heads: int; n_kv_heads: int; ffn_dim: int; vocab: int
  rope_base: float = 10000.0; norm_eps: float = 1e-5; head_dim: int = 0
  layers: list[LayerSpec] = field(default_factory=list)
  extra: dict = field(default_factory=dict)   # arch-specific params for non-default mixers (e.g. DeltaNet head dims)

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


# --- token-mixer + MLP builders (registry values). Each includes its own residual add. ---

def _attn_prefill(x: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec, cos, sin) -> Tensor:
  """Pre-norm attention mixer over a prompt `x` [S, dim]: RMSNorm -> QKV (+optional QK-norm) -> RoPE ->
  causal attention -> o_proj, with the residual."""
  H, KV, dh, S = cfg.n_heads, cfg.n_kv_heads, cfg.dh, x.shape[0]
  xn = x.rms_norm(w["attn_norm"], cfg.norm_eps)
  q = xn.linear(w["wq"]).reshape(S, H, dh); k = xn.linear(w["wk"]).reshape(S, KV, dh); v = xn.linear(w["wv"]).reshape(S, KV, dh)
  if ls.qk_norm:                                   # Qwen3 QK-norm: per-head RMSNorm over head_dim before RoPE
    q = q.reshape(S * H, dh).rms_norm(w["q_norm"], cfg.norm_eps).reshape(S, H, dh)
    k = k.reshape(S * KV, dh).rms_norm(w["k_norm"], cfg.norm_eps).reshape(S, KV, dh)
  q = q.transpose([1, 0, 2]).reshape(1, H, S, dh); k = k.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  v = v.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  q, k = rope(q, cos, sin), rope(k, cos, sin)
  k, v = _repeat_kv(k, H // KV), _repeat_kv(v, H // KV)
  a = _causal_attn(q, k, v).reshape(H, S, dh).transpose([1, 0, 2]).reshape(S, H * dh)
  return x + a.linear(w["wo"])


def _swiglu_prefill(h: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec) -> Tensor:
  """SwiGLU MLP with its residual: `h + down(silu(gate(norm(h))) * up(norm(h)))`."""
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"])


def prefill_block(x: Tensor, w: dict, cfg: LlamaConfig, cos, sin, ls: LayerSpec | None = None) -> Tensor:
  """One pre-norm decoder block (mixer + MLP). Back-compatible: with no `ls`, QK-norm is detected from `w`."""
  ls = ls or LayerSpec(qk_norm="q_norm" in w)
  return PREFILL_MLPS[ls.mlp](PREFILL_MIXERS[ls.mixer](x, w, cfg, ls, cos, sin), w, cfg, ls)


def _repeat_kv3(k: Tensor, g: int) -> Tensor:
  """Grouped-query repeat for the decode cache: [KV, M, dh] -> [KV*g, M, dh] (0/1 expansion matmul)."""
  if g == 1: return k
  KV, M, dh = k.shape
  Mx = np.repeat(np.eye(KV, dtype=np.float16), g, axis=0)
  return k.reshape(KV, M * dh).transpose([1, 0]).linear(Mx).transpose([1, 0]).reshape(KV * g, M, dh)


def _attn_decode(x: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec, ctx: dict, M: int):
  """Single-token attention against a resident KV-cache. Creates its own Kin/Vin state inputs and reads the
  shared per-position inputs from `ctx` (oh/inv/mask/cosp/sinp). Returns (hidden+residual, state_pairs) where
  `state_pairs` is [(out, in), ...] for the runner to alias resident via `share_buffer`."""
  H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.dh
  Kin = _input((KV, M, dh)); Vin = _input((KV, M, dh))
  oh, inv, mask, cosp, sinp = ctx["oh"], ctx["inv"], ctx["mask"], ctx["cosp"], ctx["sinp"]
  xn = x.rms_norm(w["attn_norm"], cfg.norm_eps)
  q = xn.linear(w["wq"]).reshape(H, dh); k = xn.linear(w["wk"]).reshape(KV, dh); v = xn.linear(w["wv"]).reshape(KV, dh)
  if ls.qk_norm:
    q = q.rms_norm(w["q_norm"], cfg.norm_eps); k = k.rms_norm(w["k_norm"], cfg.norm_eps)
  q = rope(q.reshape(H, 1, dh), cosp, sinp)        # [H, 1, dh]
  k = rope(k.reshape(KV, 1, dh), cosp, sinp); v = v.reshape(KV, 1, dh)
  Kout = Kin * inv + k * oh; Vout = Vin * inv + v * oh        # masked positional write into the cache
  Kr, Vr = _repeat_kv3(Kout, H // KV), _repeat_kv3(Vout, H // KV)
  sc = ((q @ Kr.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)   # [H, 1, M]
  a = (sc @ Vr).reshape(H, 1, dh).transpose([1, 0, 2]).reshape(1, H * dh)
  return x + a.linear(w["wo"]), [(Kout, Kin), (Vout, Vin)]


def _swiglu_decode(h: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec) -> Tensor:
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"])


PREFILL_MIXERS = {"attention": _attn_prefill}
PREFILL_MLPS = {"swiglu": _swiglu_prefill}
DECODE_MIXERS = {"attention": _attn_decode}
DECODE_MLPS = {"swiglu": _swiglu_decode}
# Per-position decode inputs a mixer may read from `ctx` (created per chunk, shared across its layers).
_DECODE_CTX = ("oh", "inv", "mask", "cosp", "sinp")


class LlamaPrefill:
  """A decoder compiled for the ANE. `compile(seq)` builds the prefill graph for a fixed prompt length;
  `prefill(token_ids)` returns next-token logits; `generate(...)` does resident-KV-cache decode. Weights are
  a dict of numpy arrays (see `from_pretrained`)."""
  def __init__(self, cfg: LlamaConfig, weights: dict, compress: str | None = None):
    self.cfg = cfg; self.w = weights; self._net = None; self._seq = 0; self._dec = None
    self.compress = compress       # None=fp16, or "int8"/"int4"/"blockwise" to quantize the ANE weights
    self._chunk_bytes = 1.6e9      # max fp16 weight bytes per decode program (under the ~2GB ANE ceiling)

  def _spec(self, i: int) -> LayerSpec:
    """The plan for layer `i`: the config's per-layer plan, or an attention fallback with QK-norm sniffed
    from the weights (back-compat for callers that pass no `layers`)."""
    if self.cfg.layers: return self.cfg.layers[i]
    return LayerSpec(qk_norm="q_norm" in self.w["layers"][i])

  def compile(self, seq: int):
    cfg = self.cfg
    cos, sin = rope_tables(seq, cfg.dh, cfg.rope_base)
    x = _input((seq, cfg.dim))
    for i, lw in enumerate(self.w["layers"]):
      ls = self._spec(i)
      x = PREFILL_MIXERS[ls.mixer](x, lw, cfg, ls, cos, sin)
      x = PREFILL_MLPS[ls.mlp](x, lw, cfg, ls)
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

  def _layer_chunks(self):
    """Group layers into contiguous chunks whose baked weights stay under the ANE single-program ceiling
    (~2GB; measured ~1.5GB OK, ~3.4GB fails). int8/int4 weights are smaller, so more layers fit per chunk."""
    per = sum(int(np.asarray(v).size) for v in self.w["layers"][0].values()) * 2   # fp16 bytes / layer
    if self.compress in ("int8", "blockwise"): per //= 2
    elif self.compress == "int4": per //= 4
    n = max(1, min(self.cfg.n_layers, int(self._chunk_bytes // max(per, 1))))
    return [range(i, min(i + n, self.cfg.n_layers)) for i in range(0, self.cfg.n_layers, n)]

  def _decoder(self, M):
    """Build (once, cached) the resident-state decode-step program(s) for a max sequence length `M`. Layers
    are split into chunks under the ANE single-program weight ceiling (`_layer_chunks`); each mixer keeps its
    resident state (KV cache, ...) on the ANE across `execute` via `share_buffer`, and the hidden state chains
    chunk->chunk. Mixers own their state, so the runner is agnostic to the mixer type."""
    if self._dec is not None and self._dec["M"] == M: return self._dec
    if self._dec is not None:
      for c in self._dec["chunks"]: c["net"].release()
    from ._compile import compile_multi
    cfg = self.cfg; dh, D = cfg.dh, cfg.dim
    groups = self._layer_chunks(); chunks = []
    for gi, grp in enumerate(groups):
      x = _input((1, D))
      ctx = {"oh": _input((1, M, 1)), "inv": _input((1, M, 1)), "mask": _input((1, 1, M)),
             "cosp": _input((1, dh)), "sinp": _input((1, dh))}
      h = x; pairs = []
      for li in grp:
        ls = self._spec(li); lw = self.w["layers"][li]
        h, sp = DECODE_MIXERS[ls.mixer](h, lw, cfg, ls, ctx, M)
        pairs += sp
        h = DECODE_MLPS[ls.mlp](h, lw, cfg, ls)
      if gi == len(groups) - 1: h = h.rms_norm(self.w["final_norm"], cfg.norm_eps)   # final norm on the last chunk
      net = compile_multi([h] + [o for o, _ in pairs], compress=self.compress)
      inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
      for o, i in pairs:                                        # alias each state output -> its input (resident)
        net.prog.share_buffer(0, om[o], 0, inm[id(i)])
      p = {"x": inm[id(x)], "h": om[h], "states": {inm[id(i)]: i.shape for _, i in pairs}}
      for k, t in ctx.items():
        if id(t) in inm: p[k] = inm[id(t)]                      # only ports the chunk's mixers actually use
      chunks.append({"net": net, "p": p})
      if hasattr(self.w["layers"], "free"):                    # streamed weights: free this chunk's fp16 now it's baked
        for li in grp: self.w["layers"].free(li)
    cos_t, sin_t = rope_tables(M, dh, cfg.rope_base)
    self._dec = {"M": M, "chunks": chunks, "cos": cos_t, "sin": sin_t}
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
    cfg = self.cfg; f16 = np.float16
    prompt = [int(t) for t in token_ids]; M = int(max_len or len(prompt) + max_new_tokens)
    d = self._decoder(M); chunks = d["chunks"]
    for c in chunks:                                           # reset every mixer's resident state for this generation
      for name, shape in c["p"]["states"].items():
        c["net"].prog.set_input(name, np.zeros(shape, f16))
    def step(tok, pos):
      ohv = np.zeros((1, M, 1), f16); ohv[0, pos, 0] = 1.0
      invv = np.ones((1, M, 1), f16); invv[0, pos, 0] = 0.0
      mv = np.full((1, 1, M), -1e4, f16); mv[..., :pos + 1] = 0.0
      vals = {"oh": ohv, "inv": invv, "mask": mv, "cosp": d["cos"][pos][None], "sinp": d["sin"][pos][None]}
      h = np.asarray(self.w["embed"][tok])[None].astype(f16)
      for c in chunks:                                         # hidden flows chunk -> chunk (cheap [1,dim] round-trip)
        p = c["p"]; pr = c["net"].prog
        pr.set_input(p["x"], h)
        for k in _DECODE_CTX:
          if k in p: pr.set_input(p[k], vals[k])
        pr.execute(); h = np.asarray(pr.read_output(p["h"])).astype(f16)
      return h.reshape(cfg.dim).astype(np.float32)
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


def _dense_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Adapter for dense Llama/Qwen-class models: emit a `LlamaConfig` (with an explicit per-layer plan) and the
  canonical weight dict from a Hugging Face config + state_dict. New architectures get their own adapter."""
  n = int(c.num_hidden_layers)
  qk = "model.layers.0.self_attn.q_norm.weight" in sd
  cfg = LlamaConfig(dim=c.hidden_size, n_layers=n, n_heads=c.num_attention_heads,
                    n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                    ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                    rope_base=float(getattr(c, "rope_theta", 10000.0)), norm_eps=float(c.rms_norm_eps),
                    head_dim=int(getattr(c, "head_dim", 0) or 0),
                    layers=[LayerSpec(mixer="attention", mlp="swiglu", qk_norm=qk) for _ in range(n)])
  return cfg, _weights_from_state_dict(sd, cfg)


def _cfg_from_hf(c) -> LlamaConfig:
  """Back-compat: a `LlamaConfig` (no per-layer plan) from a Hugging Face config."""
  return LlamaConfig(dim=c.hidden_size, n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                     n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                     ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                     rope_base=float(getattr(c, "rope_theta", 10000.0)), norm_eps=float(c.rms_norm_eps),
                     head_dim=int(getattr(c, "head_dim", 0) or 0))


# Architecture adapters: (predicate(hf_config) -> bool, adapter(hf_config, state_dict) -> (cfg, weights)).
# First match wins; the dense Llama/Qwen path is the fallback. New architectures (MoE, hybrid) append here
# from their own module (e.g. aneforge.moe) so the loader stays arch-agnostic.
ADAPTERS: list = []


def from_pretrained(name: str, compress: str | None = None) -> LlamaPrefill:
  """Load a Llama/Qwen-class model from Hugging Face for ANE inference. `compress` ("int8"/"int4"/"blockwise")
  quantizes the ANE weights."""
  from transformers import AutoModelForCausalLM
  hf = AutoModelForCausalLM.from_pretrained(name)
  sd = {k: v.detach().float().numpy() for k, v in hf.state_dict().items()}
  adapt = next((fn for pred, fn in ADAPTERS if pred(hf.config)), _dense_adapter)
  cfg, weights = adapt(hf.config, sd)
  return LlamaPrefill(cfg, weights, compress=compress)
