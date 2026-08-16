"""LLMs on the Apple Neural Engine. A config-driven decoder (RMSNorm + RoPE + grouped-query causal
attention + SwiGLU/GeGLU MLP) that loads Hugging Face weights and runs prefill and KV-cache decode as fused ANE
programs. Matches HF logits; see `docs/llm.md`.

The runner is model-agnostic: a `LlamaConfig` carries a per-layer plan (`LayerSpec`) naming a token-mixer
and an MLP from the builder registries below, so supporting a new architecture is an *adapter* that emits
the plan + canonical weights - not new branches in the hot path."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np

from .graph import Tensor, input as _input, concat as _concat, _const
from . import _compile


class ModelType(str, Enum):
  GPT2 = "gpt2"
  GEMMA = "gemma"
  GEMMA2 = "gemma2"
  GEMMA3 = "gemma3"
  GEMMA3_TEXT = "gemma3_text"
  MISTRAL = "mistral"
  LLAMA = "llama"
  QWEN = "qwen"


# Gemma-2/3 reach the dense fallback (Llama-shaped) but need softcapping / GeGLU / (1+w)-norm it lacks;
# a name backstop for the softcapping feature check in _assert_dense_compatible.
_DENSE_INCOMPATIBLE = {ModelType.GEMMA2, ModelType.GEMMA3, ModelType.GEMMA3_TEXT}


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
  rotary_dim: int = 0; rope_interleaved: bool = False   # 0 = full RoPE (dh); interleaved = GPT-J pair layout
  rope_scaling: dict | None = None                       # HF rope_scaling, e.g. Llama-3.1 "llama3" freq rescaling
  layers: list[LayerSpec] = field(default_factory=list)
  extra: dict = field(default_factory=dict)   # arch-specific params for non-default mixers (e.g. DeltaNet head dims)
  norm_type: str = "rms"                        # "rms" (RMSNorm) or "layer" (LayerNorm with affine gamma/beta)
  rope: bool = True                             # True = RoPE (Llama/Qwen); False = learned wpe / no-RoPE (GPT-2)

  @property
  def dh(self) -> int: return self.head_dim or self.dim // self.n_heads


def _llama3_rope_scale(inv, s):
  """Llama-3.1+ RoPE frequency rescaling (HF `rope_type: "llama3"`): low-frequency components (long wavelengths)
  are divided by `factor`, high frequencies pass through, with a smooth blend between. Without it, long-range
  positions get the wrong phase -- short context works, long context degenerates. Operates on inv_freq."""
  factor, lo, hi = float(s["factor"]), float(s["low_freq_factor"]), float(s["high_freq_factor"])
  old = float(s["original_max_position_embeddings"])
  low_wl, high_wl = old / lo, old / hi
  wl = 2.0 * np.pi / inv
  scaled = np.where(wl > low_wl, inv / factor, inv)                       # low freq -> /factor; else unchanged
  smooth = (old / wl - lo) / (hi - lo)
  smoothed = (1.0 - smooth) * scaled / factor + smooth * scaled
  return np.where((wl >= high_wl) & (wl <= low_wl), smoothed, scaled)     # medium freq -> smooth blend


def rope_tables(seq: int, dh: int, base: float = 10000.0, rotary_dim: int = 0, interleaved: bool = False,
                scaling: dict | None = None):
  """Precompute the [seq, dh] cos/sin rotation tables. Default: HF Llama "neox" layout (full `dh`, freqs
  duplicated over the two halves). `rotary_dim < dh` rotates only the first `rotary_dim` dims and pads the rest
  with cos=1/sin=0 (partial RoPE). `interleaved=True` uses the GPT-J pair layout (dim 2p,2p+1 share freq p) for
  use with the matmul-rope `x*cos + (x@P)*sin`. `scaling` applies an HF rope_scaling (Llama-3.1 "llama3")."""
  rd = rotary_dim or dh
  inv = 1.0 / (base ** (np.arange(0, rd, 2) / rd))
  if scaling and scaling.get("rope_type") == "llama3":
    inv = _llama3_rope_scale(inv, scaling)
  pos = np.arange(seq)[:, None] * inv[None, :]               # [seq, rd/2]
  if interleaved:
    c = np.repeat(np.cos(pos), 2, axis=1); s = np.repeat(np.sin(pos), 2, axis=1)   # pair layout [seq, rd]
  else:
    emb = np.concatenate([pos, pos], -1); c = np.cos(emb); s = np.sin(emb)         # half-split [seq, rd]
  if rd < dh:                                                # pad pass-through dims (cos=1, sin=0)
    c = np.concatenate([c, np.ones((seq, dh - rd))], 1); s = np.concatenate([s, np.zeros((seq, dh - rd))], 1)
  return c.astype(np.float16), s.astype(np.float16)


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
  """Grouped-query repeat: [1, KV, S, dh] -> [1, KV*g, S, dh], each kv head repeated `g` times (a 0/1 expansion
  matmul on the KV axis; any `g`). S and dh stay separate axes -- flattening them into one (S*dh) trips the ANE
  65536 per-axis limit, capping context at S*dh <= 65536 (e.g. 512 at dh=128)."""
  if g == 1: return k
  _, KV, S, dh = k.shape
  Mx = np.repeat(np.eye(KV, dtype=np.float16), g, axis=0)       # [KV*g, KV]: Mx[h, kv] = 1 iff kv == h // g
  return k.reshape(KV, S, dh).transpose([1, 2, 0]).linear(Mx).transpose([2, 0, 1]).reshape(1, KV * g, S, dh)


def _causal_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  """Causal self-attention on q,k,v [1,H,S,dh] as the decomposed `softmax(q@k^T*scale + causal_mask)@v`.
  For prefill this stays in ONE fused program (compute-bound big matmuls - what the ANE excels at);
  the native fused-attention layer is avoided here because its per-layer graph cut dominates prefill cost."""
  S, dh = q.shape[2], q.shape[3]
  mask = np.triu(np.full((S, S), -1e4, np.float16), 1)     # causal: keys after the query are masked out
  return ((q @ k.transpose([0, 1, 3, 2])) * (1.0 / dh ** 0.5) + _const(mask)).softmax(-1) @ v


# --- token-mixer + MLP builders (registry values). Each includes its own residual add. ---

def _attn_prefill(x: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec, cos, sin) -> Tensor:
  """Pre-norm attention mixer over a prompt `x` [S, dim]: RMSNorm/LayerNorm -> QKV (+optional QK-norm) -> RoPE ->
  causal attention -> o_proj, with the residual."""
  H, KV, dh, S = cfg.n_heads, cfg.n_kv_heads, cfg.dh, x.shape[0]
  xn = (x.layer_norm(w["attn_norm_w"], w["attn_norm_b"], cfg.norm_eps)
        if cfg.norm_type == "layer" else x.rms_norm(w["attn_norm"], cfg.norm_eps))
  q = xn.linear(w["wq"], w.get("q_bias")).reshape(S, H, dh)        # QKV biases (Qwen2/GPT-2): None when absent
  k = xn.linear(w["wk"], w.get("k_bias")).reshape(S, KV, dh); v = xn.linear(w["wv"], w.get("v_bias")).reshape(S, KV, dh)
  if ls.qk_norm:                                   # Qwen3 QK-norm: per-head RMSNorm over head_dim before RoPE
    q = q.reshape(S * H, dh).rms_norm(w["q_norm"], cfg.norm_eps).reshape(S, H, dh)
    k = k.reshape(S * KV, dh).rms_norm(w["k_norm"], cfg.norm_eps).reshape(S, KV, dh)
  q = q.transpose([1, 0, 2]).reshape(1, H, S, dh); k = k.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  v = v.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  if cfg.rope:
    q, k = rope(q, cos, sin), rope(k, cos, sin)
  k, v = _repeat_kv(k, H // KV), _repeat_kv(v, H // KV)
  a = _causal_attn(q, k, v).reshape(H, S, dh).transpose([1, 0, 2]).reshape(S, H * dh)
  return x + a.linear(w["wo"], w.get("o_bias"))


def _attn_prefill_kv(x: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec, cos, sin):
  """Like `_attn_prefill`, but also emits the per-layer cache tensors (roped/raw k, raw v) at `[KV, S, dh]` -- the
  exact layout `_attn_decode`'s resident cache stores -- so a batched prefill can seed the decode KV cache.
  Returns (hidden+residual, (k_cache, v_cache))."""
  H, KV, dh, S = cfg.n_heads, cfg.n_kv_heads, cfg.dh, x.shape[0]
  xn = (x.layer_norm(w["attn_norm_w"], w["attn_norm_b"], cfg.norm_eps)
        if cfg.norm_type == "layer" else x.rms_norm(w["attn_norm"], cfg.norm_eps))
  q = xn.linear(w["wq"], w.get("q_bias")).reshape(S, H, dh)
  k = xn.linear(w["wk"], w.get("k_bias")).reshape(S, KV, dh); v = xn.linear(w["wv"], w.get("v_bias")).reshape(S, KV, dh)
  if ls.qk_norm:
    q = q.reshape(S * H, dh).rms_norm(w["q_norm"], cfg.norm_eps).reshape(S, H, dh)
    k = k.reshape(S * KV, dh).rms_norm(w["k_norm"], cfg.norm_eps).reshape(S, KV, dh)
  q = q.transpose([1, 0, 2]).reshape(1, H, S, dh); k = k.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  v = v.transpose([1, 0, 2]).reshape(1, KV, S, dh)
  if cfg.rope:
    q, k = rope(q, cos, sin), rope(k, cos, sin)
  kc, vc = k.reshape(KV, S, dh), v.reshape(KV, S, dh)            # cache layout: pre-repeat, k roped/raw, v raw
  kr, vr = _repeat_kv(k, H // KV), _repeat_kv(v, H // KV)
  a = _causal_attn(q, kr, vr).reshape(H, S, dh).transpose([1, 0, 2]).reshape(S, H * dh)
  return x + a.linear(w["wo"], w.get("o_bias")), (kc, vc)


def _swiglu(h: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec) -> Tensor:
  """SwiGLU MLP with its residual: `h + down(silu(gate(norm(h))) * up(norm(h)))`. Stateless and
  shape-agnostic, so one builder serves both prefill [S,dim] and decode [1,dim]."""
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"])


def _gelu_mlp(h: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec) -> Tensor:
  """GPT-2 style GELU MLP with residual: `h + proj(gelu_new(fc(norm(h))))`."""
  hn = (h.layer_norm(w["mlp_norm_w"], w["mlp_norm_b"], cfg.norm_eps)
        if cfg.norm_type == "layer" else h.rms_norm(w["mlp_norm"], cfg.norm_eps))
  g = hn.linear(w["wfc"], w.get("fc_bias"))
  inner = (g + g.pow(3.0) * 0.044715) * np.sqrt(2.0 / np.pi)
  act = (g * 0.5) * inner.tanh().adds(1.0)
  return h + act.linear(w["wproj"], w.get("proj_bias"))

def _gelu_tanh(x: Tensor) -> Tensor:
  """PyTorch 'tanh' GELU (Gemma's `gelu_pytorch_tanh`), built out of primitive math ops rather than the
  native `gelu`/`scaled_tanh` MIL ops: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`."""
  inner = (x + x.pow(3.0) * 0.044715) * np.sqrt(2.0 / np.pi)
  return (x * 0.5) * inner.tanh().adds(1.0)

def _geglu(h: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec) -> Tensor:
  """GeGLU MLP with its residual: `h + down(gelu(gate(norm(h))) * up(norm(h)))`. Used by Gemma."""
  hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  # Gemma uses the tanh-approximation gelu, not the native `gelu` MIL op (that's the true Erf/exact
  # variant and diverges from Gemma's actual activation). See #214
  return h + (_gelu_tanh(hn.linear(w["wgate"])) * hn.linear(w["wup"])).linear(w["wdown"])


def prefill_block(x: Tensor, w: dict, cfg: LlamaConfig, cos, sin, ls: LayerSpec | None = None) -> Tensor:
  """One pre-norm decoder block (mixer + MLP). Back-compatible: with no `ls`, QK-norm is detected from `w`."""
  ls = ls or LayerSpec(qk_norm="q_norm" in w)
  return PREFILL_MLPS[ls.mlp](PREFILL_MIXERS[ls.mixer](x, w, cfg, ls, cos, sin), w, cfg, ls)


def _repeat_kv3(k: Tensor, g: int) -> Tensor:
  """Grouped-query repeat for the decode cache: [KV, M, dh] -> [KV*g, M, dh] (0/1 expansion matmul on the KV
  axis). M and dh stay separate axes -- flattening to M*dh trips the ANE 65536 per-axis limit (context cap
  M*dh <= 65536, i.e. 512 at dh=128)."""
  if g == 1: return k
  Mx = np.repeat(np.eye(k.shape[0], dtype=np.float16), g, axis=0)
  return k.transpose([1, 2, 0]).linear(Mx).transpose([2, 0, 1])   # [KV,M,dh]->[M,dh,KV]->[M,dh,KV*g]->[KV*g,M,dh]


def _attn_decode(x: Tensor, w: dict, cfg: LlamaConfig, ls: LayerSpec, ctx: dict, M: int):
  """Single-token attention against a resident KV-cache. Creates its own Kin/Vin state inputs and reads the
  shared per-position inputs from `ctx` (oh/inv/mask/cosp/sinp). Returns (hidden+residual, state_pairs) where
  `state_pairs` is [(out, in), ...] for the runner to alias resident via `share_buffer`."""
  H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.dh
  Kin = _input((KV, M, dh)); Vin = _input((KV, M, dh))
  oh, inv, mask, cosp, sinp = ctx["oh"], ctx["inv"], ctx["mask"], ctx["cosp"], ctx["sinp"]
  xn = (x.layer_norm(w["attn_norm_w"], w["attn_norm_b"], cfg.norm_eps)
        if cfg.norm_type == "layer" else x.rms_norm(w["attn_norm"], cfg.norm_eps))
  q = xn.linear(w["wq"], w.get("q_bias")).reshape(H, dh)           # QKV biases (Qwen2/GPT-2): None when absent
  k = xn.linear(w["wk"], w.get("k_bias")).reshape(KV, dh); v = xn.linear(w["wv"], w.get("v_bias")).reshape(KV, dh)
  if ls.qk_norm:
    q = q.rms_norm(w["q_norm"], cfg.norm_eps); k = k.rms_norm(w["k_norm"], cfg.norm_eps)
  if cfg.rope:
    q = rope(q.reshape(H, 1, dh), cosp, sinp)        # [H, 1, dh]
    k = rope(k.reshape(KV, 1, dh), cosp, sinp); v = v.reshape(KV, 1, dh)
  else:
    q = q.reshape(H, 1, dh)
    k = k.reshape(KV, 1, dh); v = v.reshape(KV, 1, dh)
  Kout = Kin * inv + k * oh; Vout = Vin * inv + v * oh        # masked positional write into the cache
  Kr, Vr = _repeat_kv3(Kout, H // KV), _repeat_kv3(Vout, H // KV)
  sc = ((q @ Kr.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)   # [H, 1, M]
  a = (sc @ Vr).reshape(H, 1, dh).transpose([1, 0, 2]).reshape(1, H * dh)
  return x + a.linear(w["wo"], w.get("o_bias")), [(Kout, Kin), (Vout, Vin)]


PREFILL_MIXERS = {"attention": _attn_prefill}
PREFILL_MLPS = {"swiglu": _swiglu, "geglu": _geglu, "gelu_new": _gelu_mlp}
DECODE_MIXERS = {"attention": _attn_decode}
DECODE_MLPS = {"swiglu": _swiglu, "geglu": _geglu, "gelu_new": _gelu_mlp}
# Per-position decode inputs a mixer may read from `ctx` (created per chunk, shared across its layers).
_DECODE_CTX = ("oh", "inv", "mask", "cosp", "sinp")


class LlamaPrefill:
  """A decoder compiled for the ANE. `compile(seq)` builds the prefill graph for a fixed prompt length;
  `prefill(token_ids)` returns next-token logits; `generate(...)` does resident-KV-cache decode. Weights are
  a dict of numpy arrays (see `from_pretrained`). The host `lm_head` keeps a cached fp32 transpose
  (`_lmT`): for large vocabularies this is the runner's biggest host allocation (GPT-2 ~206 MB, a
  151936x4096 head ~2.5 GB), freed by `release()`."""
  def __init__(self, cfg: LlamaConfig, weights: dict, compress: str | None = None):
    self.cfg = cfg; self.w = weights; self._net = None; self._seq = 0; self._dec = None; self._pre = None
    self.compress = compress       # None=fp16, or "int8"/"int4"/"blockwise" to quantize the ANE weights
    self._chunk_bytes = 1.6e9      # max fp16 weight bytes per decode program (under the ~2GB ANE ceiling)
    self._lmT = None               # cached contiguous fp32 lm_head^T (host matmul); built on first use

  def release(self) -> None:
    """Release every compiled program (prefill, decode chunks, batched-prefill chunks), the cached host
    `_lmT` transpose, and the cached state, so a subsequent call transparently recompiles instead of
    replaying a freed program or stale buffers."""
    if self._net is not None:
      self._net.release()
    if self._dec is not None:
      for c in self._dec["chunks"]: c["net"].release()
    if self._pre is not None:
      for c in self._pre["chunks"]: c["net"].release()
    self._net, self._seq, self._dec, self._pre, self._lmT = None, 0, None, None, None

  def _check_positions(self, n: int) -> None:
    """Raise a clear error instead of an out-of-bounds `wpe` index/broadcast failure when `n` exceeds a
    learned-position-embedding model's position table (e.g. GPT-2's 1024/2048 `n_positions`)."""
    wpe = self.w.get("wpe")
    if wpe is not None and n > wpe.shape[0]:
      raise ValueError(f"sequence length {n} exceeds this model's {wpe.shape[0]} max positions")

  def _spec(self, i: int) -> LayerSpec:
    """The plan for layer `i`: the config's per-layer plan, or an attention fallback with QK-norm sniffed
    from the weights (back-compat for callers that pass no `layers`)."""
    if self.cfg.layers: return self.cfg.layers[i]
    return LayerSpec(qk_norm="q_norm" in self.w["layers"][i])

  def compile(self, seq: int):
    cfg = self.cfg
    cos, sin = (rope_tables(seq, cfg.dh, cfg.rope_base, cfg.rotary_dim, cfg.rope_interleaved, cfg.rope_scaling)
                if cfg.rope else (None, None))
    x = _input((seq, cfg.dim))
    for i, lw in enumerate(self.w["layers"]):
      ls = self._spec(i)
      x = PREFILL_MIXERS[ls.mixer](x, lw, cfg, ls, cos, sin)
      x = PREFILL_MLPS[ls.mlp](x, lw, cfg, ls)
    fn = (x.layer_norm(self.w["final_norm_w"], self.w["final_norm_b"], cfg.norm_eps)
          if cfg.norm_type == "layer" else x.rms_norm(self.w["final_norm"], cfg.norm_eps))
    self._net = _compile.compile(fn, compress=self.compress)
    self._seq = seq
    return self

  def _hidden(self, token_ids):
    """Run the transformer layers on the ANE; return the final hidden states [S, dim]."""
    self._check_positions(len(token_ids))
    if self._net is None or len(token_ids) != self._seq:
      self.compile(len(token_ids))
    net = self._net; assert net is not None
    ids = np.asarray(token_ids)
    emb = self.w["embed"][ids]           # host embedding gather -> [S, dim]
    if "wpe" in self.w:
      emb = emb + self.w["wpe"][:len(ids)]
    return np.asarray(net(emb.astype(np.float16))).astype(np.float32)

  def _logits(self, hidden_row):
    lm = self._lmT
    if lm is None:
      # Build once: NumPy has no fp16 BLAS, and a live `.T` view is strided, so a per-call
      # `hidden @ lm_head.T` re-converts the (fp16) weight to fp32 every token (~124ms on a
      # 50k vocab). A contiguous fp32 copy of the transpose is converted once and is ~10x faster.
      lm = self._lmT = np.ascontiguousarray(np.asarray(self.w["lm_head"]).T, np.float32)
    return hidden_row @ lm

  @staticmethod
  def _sample(logits, temperature, top_p, top_k):
    """Greedy argmax when `temperature <= 0` (default); else temperature-scale, keep the top-k logits and the
    top-p nucleus, and sample. Greedy can loop on sampling-tuned models, so chat demos pass the model's params."""
    if temperature <= 0.0: return int(np.argmax(logits))
    lg = logits.astype(np.float64) / temperature
    if top_k and 0 < top_k < lg.size:
      lg[lg < np.partition(lg, -top_k)[-top_k]] = -np.inf
    p = np.exp(lg - lg.max()); p /= p.sum()
    if 0.0 < top_p < 1.0:
      order = np.argsort(p)[::-1]; cum = np.cumsum(p[order])
      keep = order[:int(np.searchsorted(cum, top_p)) + 1]      # include the token that crosses top_p
      m = np.zeros_like(p); m[keep] = p[keep]; p = m / m.sum()
    return int(np.random.choice(p.size, p=p))

  def prefill(self, token_ids):
    """Prefill `token_ids` and return the next-token logits [1, vocab] (the transformer runs on the ANE)."""
    return self._logits(self._hidden(token_ids)[-1])[None]

  def _layer_chunks(self):
    """Group layers into contiguous chunks whose baked weights stay under the ANE single-program ceiling
    (~2GB; measured ~1.5GB OK, ~3.4GB fails). int8/int4 weights are smaller, so more layers fit per chunk."""
    per = sum(int(np.asarray(v).size) for v in self.w["layers"][0].values() if isinstance(v, (np.ndarray, list, tuple))) * 2   # fp16 bytes / layer
    if self.compress in ("int8", "blockwise"): per //= 2
    elif self.compress == "int4": per //= 4
    n = max(1, min(self.cfg.n_layers, int(self._chunk_bytes // max(per, 1))))
    return [range(i, min(i + n, self.cfg.n_layers)) for i in range(0, self.cfg.n_layers, n)]

  def _decoder(self, M):
    """Build (once, cached) the resident-state decode-step program(s) for a max sequence length `M`. Layers
    are split into chunks under the ANE single-program weight ceiling (`_layer_chunks`); each mixer keeps its
    resident state (KV cache, ...) on the ANE across `execute` via `share_buffer`, and the hidden state chains
    chunk->chunk. Mixers own their state, so the runner is agnostic to the mixer type."""
    self._check_positions(int(M))
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
      wpe_pos = None
      if gi == 0 and "wpe" in self.w:
        wpe_pos = _input((1, M))
        h = h + wpe_pos @ _const(self.w["wpe"][:M].astype(np.float16))
      for li in grp:
        ls = self._spec(li); lw = self.w["layers"][li]
        h, sp = DECODE_MIXERS[ls.mixer](h, lw, cfg, ls, ctx, M)
        pairs += sp
        h = DECODE_MLPS[ls.mlp](h, lw, cfg, ls)
      if gi == len(groups) - 1:
        h = (h.layer_norm(self.w["final_norm_w"], self.w["final_norm_b"], cfg.norm_eps)
             if cfg.norm_type == "layer" else h.rms_norm(self.w["final_norm"], cfg.norm_eps))
      net = compile_multi([h] + [o for o, _ in pairs], compress=self.compress)
      inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
      for o, i in pairs:                                        # alias each state output -> its input (resident)
        net.prog.share_buffer(0, om[o], 0, inm[id(i)])
      p = {"x": inm[id(x)], "h": om[h], "states": {inm[id(i)]: i.shape for _, i in pairs}}
      if wpe_pos is not None: p["wpe_pos"] = inm[id(wpe_pos)]
      for k, t in ctx.items():
        if id(t) in inm: p[k] = inm[id(t)]                      # only ports the chunk's mixers actually use
      chunks.append({"net": net, "p": p})
      if hasattr(self.w["layers"], "free"):                    # streamed weights: free this chunk's fp16 now it's baked
        for li in grp: self.w["layers"].free(li)
    cos_t, sin_t = (rope_tables(M, dh, cfg.rope_base, cfg.rotary_dim, cfg.rope_interleaved, cfg.rope_scaling)
                    if cfg.rope else (None, None))
    self._dec = {"M": M, "chunks": chunks, "cos": cos_t, "sin": sin_t}
    return self._dec

  def warmup(self, max_len):
    """Compile and cache the decode program for context length `max_len` (the one-time cost) so the next
    `generate` streams immediately. Returns self."""
    self._decoder(int(max_len)); return self

  def _prefiller(self, seq):
    """Build (once, cached) the CHUNKED batched-prefill program(s) for prompt length `seq`: layers split under
    the ANE program ceiling (like `_decoder`), each chunk a full-sequence forward emitting its attention layers'
    (roped/raw k, raw v) at `[KV, seq, dh]`, hidden chained chunk->chunk. Attention layers only. Cached by `seq`, so
    padding prompts to a fixed bucket length reuses one compile across calls (amortizing the compile cost)."""
    if self._pre is not None and self._pre["seq"] == seq: return self._pre
    if self._pre is not None:
      for c in self._pre["chunks"]: c["net"].release()
    from ._compile import compile_multi
    cfg = self.cfg
    cos, sin = (rope_tables(seq, cfg.dh, cfg.rope_base, cfg.rotary_dim, cfg.rope_interleaved, cfg.rope_scaling)
                if cfg.rope else (None, None))
    chunks = []
    for grp in self._layer_chunks():
      x = _input((seq, cfg.dim)); h = x; kvts = []
      for li in grp:
        ls = self._spec(li)
        if ls.mixer != "attention":
          raise NotImplementedError("batched prefill supports attention layers only")
        h, kv = _attn_prefill_kv(h, self.w["layers"][li], cfg, ls, cos, sin)
        h = PREFILL_MLPS[ls.mlp](h, self.w["layers"][li], cfg, ls)
        kvts.append((li, kv))
      if grp[-1] == cfg.n_layers - 1:
        h = (h.layer_norm(self.w["final_norm_w"], self.w["final_norm_b"], cfg.norm_eps)
             if cfg.norm_type == "layer" else h.rms_norm(self.w["final_norm"], cfg.norm_eps))
      net = compile_multi([h] + [t for _, kv in kvts for t in kv], compress=self.compress)
      om = dict(net.output_ports); inm = {id(t): n for t, n in net.input_ports}
      p = {"x": inm[id(x)], "h": om[h], "kv": [(li, om[kv[0]], om[kv[1]]) for li, kv in kvts]}
      chunks.append({"net": net, "p": p})
    self._pre = {"seq": seq, "chunks": chunks}
    return self._pre

  def _prefill_seed(self, token_ids, pad_to=None):
    """Batched prefill via the cached chunked prefiller: run the whole prompt in one pass per chunk (full causal
    attention -- the compute-bound matmuls the engine likes) instead of stepping the decode program per token.
    `pad_to` right-pads the prompt to a fixed bucket length so ONE prefiller compile is reused across prompts of
    different real length (causal attention ignores the trailing pads); the real length drives the returned
    logits and the cache seed. Returns (next_logits [1, vocab], kv_by_layer) in the decode cache layout."""
    f16 = np.float16; real = len(token_ids)
    seq = max(int(pad_to), real) if pad_to else real
    self._check_positions(seq)
    pre = self._prefiller(seq)
    ids = list(token_ids) + [0] * (seq - real)
    emb = self.w["embed"][np.asarray(ids)]
    if "wpe" in self.w:
      emb = emb + self.w["wpe"][:seq]
    h = emb.astype(f16)   # [seq, dim], right-padded
    kv_by_layer: list = [None] * self.cfg.n_layers
    for c in pre["chunks"]:
      pr = c["net"].prog; p = c["p"]
      pr.set_input(p["x"], h); pr.execute()
      h = np.asarray(pr.read_output(p["h"])).astype(f16)                      # hidden chained chunk -> chunk
      for li, kport, vport in p["kv"]:                                        # keep only the real positions
        kv_by_layer[li] = (np.asarray(pr.read_output(kport)).astype(f16)[:, :real, :],
                           np.asarray(pr.read_output(vport)).astype(f16)[:, :real, :])
    return self._logits(h[real - 1].astype(np.float32))[None], kv_by_layer

  def _seed_cache(self, d, kv_by_layer, seq):
    """Write a batched prefill's per-layer K/V into the resident decode cache buffers (positions [0:seq]), the
    same `set_input` path `generate` uses to zero them. Decode chunks group layers exactly as `_layer_chunks`."""
    M = d["M"]; f16 = np.float16; groups = self._layer_chunks()
    for gi, c in enumerate(d["chunks"]):
      ports, seeded = list(c["p"]["states"].keys()), []
      for li in groups[gi]:                                   # states ordered K_li, V_li per attention layer
        K, V = kv_by_layer[li]; KVh, _, dh = K.shape
        pk = np.zeros((KVh, M, dh), f16); pk[:, :seq, :] = K
        pv = np.zeros((KVh, M, dh), f16); pv[:, :seq, :] = V
        seeded += [pk, pv]
      for port, buf in zip(ports, seeded):
        c["net"].prog.set_input(port, buf)

  def generate(self, token_ids, max_new_tokens=40, eos_id=None, max_len=None, on_token=None,
               temperature=0.0, top_p=1.0, top_k=0, batched_prefill=True, prefill_pad=None, on_stage=None):
    """Autoregressive generation with a resident KV-cache. The decode program (compiled once and cached) runs
    all layers for one token attending to caches that stay on the ANE across steps, so each step feeds only the
    token embedding + a position one-hot. `on_token(id)` is called per token (streaming). Greedy by default
    (`temperature=0`); pass temperature/top_p/top_k to sample. Returns the generated token ids.

    `on_stage(name, elapsed)` receives per-token timing (seconds) when profiling: `embedding` and `layers`
    are the two halves of one decode step (the `step` stage times the whole call, so summing all stages
    double-counts `step`), then `lm_head` and `sample`. Only emitted while profiling is on."""
    cfg = self.cfg; f16 = np.float16
    profiling = on_stage is not None
    prompt = [int(t) for t in token_ids]; M = int(max_len or len(prompt) + max_new_tokens)
    eos = {int(eos_id)} if isinstance(eos_id, int) else {int(e) for e in (eos_id or ())}   # any-of stop set
    d = self._decoder(M); chunks = d["chunks"]
    for c in chunks:                                           # reset every mixer's resident state for this generation
      for name, shape in c["p"]["states"].items():
        c["net"].prog.set_input(name, np.zeros(shape, f16))
    def step(tok, pos):
      ohv = np.zeros((1, M, 1), f16); ohv[0, pos, 0] = 1.0
      invv = np.ones((1, M, 1), f16); invv[0, pos, 0] = 0.0
      mv = np.full((1, 1, M), -1e4, f16); mv[..., :pos + 1] = 0.0
      vals = {"oh": ohv, "inv": invv, "mask": mv}
      if d["cos"] is not None:
        vals["cosp"] = d["cos"][pos][None]; vals["sinp"] = d["sin"][pos][None]
      t0 = time.perf_counter() if profiling else 0.0
      emb = self.w["embed"][tok]
      h = np.asarray(emb)[None].astype(f16)
      wpe_pos = np.zeros((1, M), f16) if "wpe" in self.w else None
      if wpe_pos is not None: wpe_pos[0, pos] = 1.0
      if profiling: on_stage("embedding", time.perf_counter() - t0)
      t0 = time.perf_counter() if profiling else 0.0
      for c in chunks:                                         # hidden flows chunk -> chunk (cheap [1,dim] round-trip)
        p = c["p"]; pr = c["net"].prog
        pr.set_input(p["x"], h)
        for k in _DECODE_CTX:
          if k in p: pr.set_input(p[k], vals[k])
        if "wpe_pos" in p: pr.set_input(p["wpe_pos"], wpe_pos)
        pr.execute(); h = np.asarray(pr.read_output(p["h"])).astype(f16)
      if profiling: on_stage("layers", time.perf_counter() - t0)
      return h.reshape(cfg.dim).astype(np.float32)
    use_batched = (batched_prefill and len(prompt) > 1
                   and not hasattr(self.w["layers"], "free")     # streamed weights are freed by _decoder; can't re-bake
                   and all(self._spec(i).mixer == "attention" for i in range(cfg.n_layers)))
    out = []
    if use_batched:                                            # one-pass prefill, then seed the resident cache
      logits0, kv = self._prefill_seed(prompt, prefill_pad)
      self._seed_cache(d, kv, len(prompt))
      cur = self._sample(logits0[0], temperature, top_p, top_k); out.append(cur)
      if cur in eos: return out
      if on_token is not None: on_token(cur)
      pos = len(prompt)
    else:
      for p, tok in enumerate(prompt[:-1]): step(tok, p)       # token-by-token prefill (fallback / oracle)
      cur, pos = prompt[-1], len(prompt) - 1
    while len(out) < max_new_tokens:                           # decode
      if pos >= M - 1: break                                   # cache full
      t0 = time.perf_counter() if profiling else 0.0; hidden = step(cur, pos)
      if profiling: on_stage("step", time.perf_counter() - t0)
      t0 = time.perf_counter() if profiling else 0.0; logits = self._logits(hidden)
      if profiling: on_stage("lm_head", time.perf_counter() - t0)
      t0 = time.perf_counter() if profiling else 0.0; nxt = self._sample(logits, temperature, top_p, top_k)
      if profiling: on_stage("sample", time.perf_counter() - t0)
      out.append(nxt)
      if nxt in eos: break
      if on_token is not None: on_token(nxt)                   # stream the token
      cur, pos = nxt, pos + 1
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
                    rope_base=_rope_theta(c), norm_eps=float(c.rms_norm_eps),
                    head_dim=int(getattr(c, "head_dim", 0) or 0), rope_scaling=_rope_scaling(c),
                    layers=[LayerSpec(mixer="attention", mlp="swiglu", qk_norm=qk) for _ in range(n)])
  return cfg, _weights_from_state_dict(sd, cfg)


def _rope_theta(c) -> float:
  """HF RoPE base, reading the transformers-5.x `rope_parameters` dict with a legacy fallback."""
  rp = getattr(c, "rope_parameters", None)
  if isinstance(rp, dict) and "rope_theta" in rp:
    return float(rp["rope_theta"])
  return float(getattr(c, "rope_theta", 10000.0))


def _rope_scaling(c):
  """HF rope_scaling, but only when the runner implements the type (Llama-3.1 "llama3")."""
  s = getattr(c, "rope_scaling", None)
  return s if isinstance(s, dict) and s.get("rope_type") == "llama3" else None


def _gemma_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Adapter for Gemma decoders. The runner's graph is Llama-shaped, so the differences are baked
  into the weights: embeddings are scaled by sqrt(dim) at input, the FFN is GeGLU (gelu_pytorch_tanh,
  via `LayerSpec(mlp="geglu")`), and the RMSNorm weights use the `(1 + weight)` convention."""
  n = int(c.num_hidden_layers)
  cfg = LlamaConfig(dim=c.hidden_size, n_layers=n, n_heads=c.num_attention_heads,
                    n_kv_heads=int(getattr(c, "num_key_value_heads", c.num_attention_heads)),
                    ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                    rope_base=_rope_theta(c), norm_eps=float(c.rms_norm_eps),
                    head_dim=int(getattr(c, "head_dim", 0) or 0),
                    layers=[LayerSpec(mixer="attention", mlp="geglu") for _ in range(n)])
  w = _weights_from_state_dict(sd, cfg)
  w["embed"] = w["embed"] * np.sqrt(c.hidden_size)        # HF: inputs_embeds * hidden_size**0.5
  for lw in w["layers"]:                                   # HF GemmaRMSNorm: output * (1.0 + weight)
    lw["attn_norm"] = lw["attn_norm"] + 1.0
    lw["mlp_norm"] = lw["mlp_norm"] + 1.0
  w["final_norm"] = w["final_norm"] + 1.0
  return cfg, w


def _mistral_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Adapter for Mistral decoders: Llama-shaped (same state_dict names) with grouped-query attention
  and a sliding window. With a resident KV cache the window is the cache length, so decode stays
  exact while `max_len` does not exceed `sliding_window`; the limit is surfaced in `cfg.extra`."""
  n = int(c.num_hidden_layers)
  cfg = LlamaConfig(dim=c.hidden_size, n_layers=n, n_heads=c.num_attention_heads,
                    n_kv_heads=int(getattr(c, "num_key_value_heads", c.num_attention_heads)),
                    ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                    rope_base=_rope_theta(c), norm_eps=float(c.rms_norm_eps),
                    head_dim=int(getattr(c, "head_dim", 0) or 0),
                    layers=[LayerSpec(mixer="attention", mlp="swiglu") for _ in range(n)])
  if getattr(c, "sliding_window", None):
    cfg.extra["sliding_window"] = int(c.sliding_window)
  return cfg, _weights_from_state_dict(sd, cfg)


def _cfg_from_hf(c) -> LlamaConfig:
  """Back-compat: a `LlamaConfig` (no per-layer plan) from a Hugging Face config."""
  return LlamaConfig(dim=c.hidden_size, n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                     n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                     ffn_dim=c.intermediate_size, vocab=c.vocab_size,
                     rope_base=_rope_theta(c), norm_eps=float(c.rms_norm_eps),
                     head_dim=int(getattr(c, "head_dim", 0) or 0), rope_scaling=_rope_scaling(c))


def _gpt2_adapter(c, sd) -> tuple[LlamaConfig, dict]:
  """Adapter for GPT-2 causal LM: LayerNorm + learned-positional wpe + tied lm_head + gelu_new."""
  if getattr(c, "scale_attn_by_inverse_layer_idx", False) or getattr(c, "reorder_and_upcast_attn", False):
    raise ValueError(
      "load_gpt2: this checkpoint sets scale_attn_by_inverse_layer_idx/reorder_and_upcast_attn, which "
      "aneforge's GPT-2 adapter does not implement, so it would silently produce wrong logits.")
  n = int(getattr(c, "n_layer", getattr(c, "num_hidden_layers", 12)))
  dim = int(getattr(c, "n_embd", getattr(c, "hidden_size", 768)))
  heads = int(getattr(c, "n_head", getattr(c, "num_attention_heads", 12)))
  ffn_dim = int(getattr(c, "n_inner", None) or getattr(c, "intermediate_size", None) or 4 * dim)
  vocab = int(getattr(c, "vocab_size", 50257))
  eps = float(getattr(c, "layer_norm_epsilon", getattr(c, "layer_norm_eps", 1e-5)))
  cfg = LlamaConfig(dim=dim, n_layers=n, n_heads=heads, n_kv_heads=heads, ffn_dim=ffn_dim, vocab=vocab,
                    norm_eps=eps, norm_type="layer", rope=False,
                    layers=[LayerSpec(mixer="attention", mlp="gelu_new") for _ in range(n)])
  w = {
    "embed": sd["transformer.wte.weight"], "wpe": sd["transformer.wpe.weight"],
    "final_norm_w": sd["transformer.ln_f.weight"], "final_norm_b": sd["transformer.ln_f.bias"],
    "lm_head": sd.get("lm_head.weight", sd["transformer.wte.weight"]), "layers": []}
  for i in range(n):
    p = f"transformer.h.{i}."
    Wqkv = sd[p + "attn.c_attn.weight"].T   # Conv1D [in, out] -> transposed [3D, D]
    bqkv = sd[p + "attn.c_attn.bias"]
    lw = {
      "attn_norm_w": sd[p + "ln_1.weight"], "attn_norm_b": sd[p + "ln_1.bias"],
      "wq": Wqkv[:dim], "q_bias": bqkv[:dim],
      "wk": Wqkv[dim:2 * dim], "k_bias": bqkv[dim:2 * dim],
      "wv": Wqkv[2 * dim:3 * dim], "v_bias": bqkv[2 * dim:3 * dim],
      "wo": sd[p + "attn.c_proj.weight"].T, "o_bias": sd[p + "attn.c_proj.bias"],
      "mlp_norm_w": sd[p + "ln_2.weight"], "mlp_norm_b": sd[p + "ln_2.bias"],
      "wfc": sd[p + "mlp.c_fc.weight"].T, "fc_bias": sd[p + "mlp.c_fc.bias"],
      "wproj": sd[p + "mlp.c_proj.weight"].T, "proj_bias": sd[p + "mlp.c_proj.bias"],
    }
    w["layers"].append(lw)
  return cfg, w


# Architecture adapters: (predicate(hf_config) -> bool, adapter(hf_config, sd) -> (cfg, weights)); first match
# wins, dense Llama/Qwen is the fallback. New archs append here (e.g. aneforge.moe) so the loader stays generic.
ADAPTERS: list = [
  (lambda c: getattr(c, "model_type", None) == ModelType.GPT2, _gpt2_adapter),
  (lambda c: getattr(c, "model_type", None) == ModelType.GEMMA, _gemma_adapter),
  (lambda c: getattr(c, "model_type", None) == ModelType.MISTRAL, _mistral_adapter),
]


def _assert_dense_compatible(c) -> None:
  """Raise if a config reaches the dense fallback but carries semantics the dense path silently drops.

  Feature-based first (`*_logit_softcapping`, which real Gemma-2/3 checkpoints set), so genuine
  Llama-shaped finetunes with an unusual `model_type` are not rejected; the name set is a backstop."""
  mt = getattr(c, "model_type", "?")
  cap = getattr(c, "attn_logit_softcapping", None) or getattr(c, "final_logit_softcapping", None)
  if cap or mt in _DENSE_INCOMPATIBLE:
    why = "logit softcapping" if cap else "architecture-specific semantics"
    raise ValueError(
      f"load_llm: {mt!r} needs {why} that aneforge's dense loader does not implement, so the dense "
      f"path would silently produce wrong logits. Supported: dense Llama/Qwen/Mistral, Gemma "
      f"(model_type 'gemma'), and MoE. See docs/llm.md.")


def from_pretrained(name: str, compress: str | None = None) -> LlamaPrefill:
  """Load a decoder LM from Hugging Face for ANE inference: dense Llama/Qwen/Mistral, Gemma, and MoE
  (an unsupported arch is rejected, not silently mis-loaded). `compress` ("int8"/"int4"/"blockwise")
  quantizes the ANE weights."""
  import torch
  from transformers import AutoModelForCausalLM
  hf = AutoModelForCausalLM.from_pretrained(name)
  adapt = next((fn for pred, fn in ADAPTERS if pred(hf.config)), _dense_adapter)
  if adapt is _dense_adapter:            # fail fast, before the state-dict copy, on an unsupported arch
    _assert_dense_compatible(hf.config)
  # fp16 state dict: the weights bake to fp16 (or quantized) for the ANE anyway, so an fp32 copy only
  # wastes memory (~28 GB for a 7B, enough to OOM a 32 GB Mac). Same rounding the bake does, so unchanged.
  sd = {k: v.detach().to(torch.float16).numpy() for k, v in hf.state_dict().items()}
  cfg, weights = adapt(hf.config, sd)
  return LlamaPrefill(cfg, weights, compress=compress)
