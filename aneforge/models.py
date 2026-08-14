"""Pretrained-model loaders (`load`, `load_resnet18`) and trainable-graph
builders (`group_norm_train`, `conv_block`, `cifar_cnn`). See docs/developer/models.md."""
from __future__ import annotations

from typing import cast

import numpy as np

from .graph import Tensor, concat, conv, input, mha, sdpa, space_to_depth
from .autograd import conv2d, conv_param, parameter
from ._compile import Model, SegmentedModel, MultiModel, compile, compile_multi

_NORM_CACHE: dict[int, Model | SegmentedModel] = {}


def _l2_normalizer(D: int) -> Model | SegmentedModel:
  """Cached fused-ANE program L2-normalizing a [1, D] vector over its last axis."""
  net = _NORM_CACHE.get(D)
  if net is None: net = _NORM_CACHE[D] = compile(input((1, D)).l2_norm(axis=-1))
  return net


def load(name: str, int8: bool = False, pooling: str = "mean") -> "Encoder":
  """Load a BERT-family sentence encoder from HF weights as an ANE embedder; `pooling` in mean/cls/max."""
  return Encoder(name, int8=int8, pooling=pooling)


def load_resnet18(int8: bool = False, compress: str | None = None,
                  compress_atol: float = 0.05, build_dir: str | None = None) -> "Vision":
  """Load torchvision ResNet-18 (ImageNet) as a fused ANE classifier; BatchNorm folded into the preceding conv at load."""
  return Vision(int8=int8, compress=compress, compress_atol=compress_atol, build_dir=build_dir)


class Vision:
  def __init__(self, int8: bool = False, compress: str | None = None,
               compress_atol: float = 0.05, build_dir: str | None = None) -> None:
    import torchvision  # lazy
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    self.sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    self.int8 = int8
    self.compress = compress
    self.compress_atol = compress_atol
    self.build_dir = build_dir
    self._model = self._build()

  def _fold(self, conv_key: str, bn: str):
    """Fold BatchNorm(`bn`) into conv(`conv_key`) -> (weight, bias)."""
    W = self.sd[conv_key + ".weight"]
    g, b = self.sd[bn + ".weight"], self.sd[bn + ".bias"]
    mu, var = self.sd[bn + ".running_mean"], self.sd[bn + ".running_var"]
    sc = g / np.sqrt(var + 1e-5)
    return (W * sc[:, None, None, None]).astype(np.float32), (b - mu * sc).astype(np.float32)

  def _block(self, x: Tensor, prefix: str, stride: int, downsample: bool) -> Tensor:
    w1, b1 = self._fold(prefix + ".conv1", prefix + ".bn1")
    w2, b2 = self._fold(prefix + ".conv2", prefix + ".bn2")
    out = conv(x, w1, stride=stride, pad=1, bias=b1).relu()
    out = conv(out, w2, stride=1, pad=1, bias=b2)
    idn = x
    if downsample:
      wd, bd = self._fold(prefix + ".downsample.0", prefix + ".downsample.1")
      idn = conv(x, wd, stride=stride, pad=0, bias=bd)
    return (out + idn).relu()

  def _build(self) -> Model | SegmentedModel:
    x = input((1, 3, 224, 224))
    w, b = self._fold("conv1", "bn1")
    h = conv(x, w, stride=2, pad=3, bias=b).relu().max_pool(3, stride=2, pad=1)
    for name, stride in [("layer1", 1), ("layer2", 2), ("layer3", 2), ("layer4", 2)]:
      for i in range(2):
        h = self._block(h, f"{name}.{i}", stride if i == 0 else 1,
                        downsample=(i == 0 and name != "layer1"))
    h = h.mean((2, 3)).reshape(1, 512)
    out = h.linear(self.sd["fc.weight"], self.sd["fc.bias"])
    return compile(out, int8=self.int8, compress=self.compress,
                   compress_atol=self.compress_atol, build_dir=self.build_dir)

  def release(self) -> None: self._model.release()

  def __call__(self, image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3: image = image[None]
    return self._model(image)            # [1, 1000] logits

  @property
  def n_ops(self) -> int: return self._model.n_ops


_BERT_KEYS = {
  "Wq": "attention.self.query.weight", "bq": "attention.self.query.bias",
  "Wk": "attention.self.key.weight", "bk": "attention.self.key.bias",
  "Wv": "attention.self.value.weight", "bv": "attention.self.value.bias",
  "Wo": "attention.output.dense.weight", "bo": "attention.output.dense.bias",
  "ln1w": "attention.output.LayerNorm.weight", "ln1b": "attention.output.LayerNorm.bias",
  "Wi": "intermediate.dense.weight", "bi": "intermediate.dense.bias",
  "Wd": "output.dense.weight", "bd": "output.dense.bias",
  "ln2w": "output.LayerNorm.weight", "ln2b": "output.LayerNorm.bias",
}


class Encoder:
  _POOL = ("mean", "cls", "max")

  def __init__(self, name: str, int8: bool = False, pooling: str = "mean") -> None:
    if pooling not in self._POOL:
      raise ValueError(f"pooling must be one of {self._POOL}, got {pooling!r}")
    self.pooling = pooling
    from transformers import AutoConfig, AutoModel, AutoTokenizer  # lazy
    cfg = AutoConfig.from_pretrained(name)
    self.tok = AutoTokenizer.from_pretrained(name)
    sd = AutoModel.from_pretrained(name).state_dict()
    g = lambda k: sd[k].detach().numpy().astype(np.float32)
    self.D, self.H = cfg.hidden_size, cfg.num_attention_heads
    self.L, self.eps, self.int8 = cfg.num_hidden_layers, cfg.layer_norm_eps, int8
    self.word = g("embeddings.word_embeddings.weight")
    self.pos = g("embeddings.position_embeddings.weight")
    self.typ = g("embeddings.token_type_embeddings.weight")
    self.eln_w, self.eln_b = g("embeddings.LayerNorm.weight"), g("embeddings.LayerNorm.bias")
    self.layers = [{k: g(f"encoder.layer.{i}." + v) for k, v in _BERT_KEYS.items()}
                   for i in range(self.L)]
    self._cache: dict[int, Model | SegmentedModel] = {}

  def _build(self, S: int) -> Model | SegmentedModel:
    h = input((S, self.D))
    for w in self.layers:
      attn = mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"], w["Wo"], w["bo"], self.H)
      h = (h + attn).layer_norm(w["ln1w"], w["ln1b"], self.eps)
      ff = h.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
      h = (h + ff).layer_norm(w["ln2w"], w["ln2b"], self.eps)
    return compile(h, int8=self.int8)

  def _embed(self, ids: np.ndarray) -> np.ndarray:
    """Host-side token + position + type embedding lookup, then LayerNorm."""  # gather is not an ANE op
    e = self.word[ids] + self.pos[np.arange(len(ids))] + self.typ[0]
    m = e.mean(-1, keepdims=True)
    v = ((e - m) ** 2).mean(-1, keepdims=True)
    return ((e - m) / np.sqrt(v + self.eps) * self.eln_w + self.eln_b).astype(np.float32)

  def __call__(self, texts, normalize: bool = True) -> np.ndarray:
    if isinstance(texts, str): texts = [texts]
    vecs = []
    for t in texts:
      ids = np.asarray(self.tok(t)["input_ids"], dtype=np.int64)
      net = self._cache.get(len(ids)) or self._cache.setdefault(len(ids), self._build(len(ids)))
      states = net(self._embed(ids))               # [S, D] per-token states on the ANE
      if self.pooling == "cls":
        v = states[0]
      elif self.pooling == "max":
        v = states.max(0)
      else:
        v = states.mean(0)
      if normalize:
        v = _l2_normalizer(self.D)(v.reshape(1, self.D))[0]
      vecs.append(v)
    return np.asarray(vecs, dtype=np.float32)


class CrossEncoder:
  """Reranker: scores (query, passage) pairs with a BERT-family sequence-classification
  model, the transformer running on the ANE. Mirrors `sentence_transformers.CrossEncoder`:
  `CrossEncoder(name).predict([(query, passage), ...])` returns one relevance score per pair
  (raw logits -- order is what a reranker needs). Higher is more relevant."""

  def __init__(self, name: str, int8: bool = False) -> None:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # lazy
    cfg = AutoConfig.from_pretrained(name)
    self.tok = AutoTokenizer.from_pretrained(name)
    sd = AutoModelForSequenceClassification.from_pretrained(name).state_dict()
    g = lambda k: sd[k].detach().numpy().astype(np.float32)
    # The base transformer sits under a model-specific prefix (bert / roberta / ...).
    marker = ".embeddings.word_embeddings.weight"
    pref = next((k[: -len(marker)] for k in sd if k.endswith(marker)), None)
    if pref is None or f"{pref}.pooler.dense.weight" not in sd or "classifier.weight" not in sd:
      raise ValueError(
        f"CrossEncoder: {name!r} is not a supported reranker -- it needs a BERT-family "
        "sequence-classification head (token-type embeddings, a pooler, and a classifier). "
        "RoBERTa/DistilBERT-style heads are not handled yet.")
    self.D, self.H = cfg.hidden_size, cfg.num_attention_heads
    self.L, self.eps, self.int8 = cfg.num_hidden_layers, cfg.layer_norm_eps, int8
    self.word = g(f"{pref}.embeddings.word_embeddings.weight")
    self.pos = g(f"{pref}.embeddings.position_embeddings.weight")
    self.typ = g(f"{pref}.embeddings.token_type_embeddings.weight")
    self.eln_w, self.eln_b = g(f"{pref}.embeddings.LayerNorm.weight"), g(f"{pref}.embeddings.LayerNorm.bias")
    self.layers = [{k: g(f"{pref}.encoder.layer.{i}." + v) for k, v in _BERT_KEYS.items()}
                   for i in range(self.L)]
    self.pool_w, self.pool_b = g(f"{pref}.pooler.dense.weight"), g(f"{pref}.pooler.dense.bias")
    self.cls_w, self.cls_b = g("classifier.weight"), g("classifier.bias")
    self._cache: dict[int, Model | SegmentedModel] = {}

  def _build(self, S: int) -> Model | SegmentedModel:
    h = input((S, self.D))
    for w in self.layers:
      attn = mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"], w["Wo"], w["bo"], self.H)
      h = (h + attn).layer_norm(w["ln1w"], w["ln1b"], self.eps)
      ff = h.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
      h = (h + ff).layer_norm(w["ln2w"], w["ln2b"], self.eps)
    return compile(h, int8=self.int8)

  def _embed(self, ids: np.ndarray, typ_ids: np.ndarray) -> np.ndarray:
    """Host-side token + position + segment embedding lookup, then LayerNorm."""
    e = self.word[ids] + self.pos[np.arange(len(ids))] + self.typ[typ_ids]
    m = e.mean(-1, keepdims=True)
    v = ((e - m) ** 2).mean(-1, keepdims=True)
    return ((e - m) / np.sqrt(v + self.eps) * self.eln_w + self.eln_b).astype(np.float32)

  def predict(self, pairs) -> np.ndarray:
    """`pairs` is a (query, passage) tuple or a list of them; returns a relevance score each."""
    if pairs and isinstance(pairs[0], str):     # a single (query, passage) pair
      pairs = [pairs]
    scores = []
    for query, passage in pairs:
      enc = self.tok(query, passage, truncation=True)
      ids = np.asarray(enc["input_ids"], dtype=np.int64)
      typ = np.asarray(enc.get("token_type_ids") or [0] * len(ids), dtype=np.int64)
      net = self._cache.get(len(ids)) or self._cache.setdefault(len(ids), self._build(len(ids)))
      cls = net(self._embed(ids, typ))[0]                              # [CLS] state, on the ANE
      pooled = np.tanh(cls @ self.pool_w.T + self.pool_b)              # BERT pooler
      logits = pooled @ self.cls_w.T + self.cls_b                      # [num_labels]
      scores.append(float(logits[0]))
    return np.asarray(scores, dtype=np.float32)


def load_vit(name: str, int8: bool = False) -> "ViT":
  """Load a Hugging Face ViT image classifier (ViTForImageClassification) as a fused ANE program."""
  return ViT(name, int8=int8)


def _row0(h: Tensor) -> Tensor:
  """Row 0 of h [M, D] -> [1, D] via a one-hot picker matmul (a bare row slice is walled on the ANE)."""
  M, _ = h.shape
  return h.transpose([1, 0]).linear(np.eye(1, M, dtype=np.float32)).transpose([1, 0])


def _vit_layer_keys(sd: dict, pref: str) -> tuple[str, dict]:
  """(layer-prefix format, key map) for a HF ViT encoder, handling both the modern
  (`<base>.layers.{i}.attention.q_proj`, `mlp.fc1/fc2`) and the legacy
  (`<base>.encoder.layer.{i}.attention.attention.query`, `intermediate.dense`) namings."""
  if f"{pref}.layers.0.attention.q_proj.weight" in sd:
    return (pref + ".layers.{i}.",
            {"q": "attention.q_proj", "k": "attention.k_proj", "v": "attention.v_proj",
             "o": "attention.o_proj", "ln1": "layernorm_before", "ln2": "layernorm_after",
             "fc1": "mlp.fc1", "fc2": "mlp.fc2"})
  return (pref + ".encoder.layer.{i}.",
          {"q": "attention.attention.query", "k": "attention.attention.key",
           "v": "attention.attention.value", "o": "attention.output.dense",
           "ln1": "layernorm_before", "ln2": "layernorm_after",
           "fc1": "intermediate.dense", "fc2": "output.dense"})


class ViT:
  """Vision Transformer image classifier from Hugging Face, running on the ANE.
  `ViT(name)(image)` returns logits [1, num_labels]; `.classify(image)` returns top labels.
  Scope: ViT-family classifiers with a CLS token and a pre-norm encoder (ViTForImageClassification
  and compatible DeiT/BEiT-style models)."""

  def __init__(self, name: str, int8: bool = False) -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification  # lazy
    cfg = AutoConfig.from_pretrained(name)
    self.proc = AutoImageProcessor.from_pretrained(name)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in AutoModelForImageClassification.from_pretrained(name).state_dict().items()}
    marker = ".embeddings.cls_token"
    pref = next((k[: -len(marker)] for k in sd if k.endswith(marker)), None)
    if pref is None or "classifier.weight" not in sd:
      raise ValueError(f"load_vit: {name!r} is not a ViT-family image classifier with a CLS token "
                       "(needs <base>.embeddings.cls_token and a classifier head).")
    self.int8 = int8
    self.D, self.H, self.L = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers
    self.P, self.img, self.eps = cfg.patch_size, cfg.image_size, cfg.layer_norm_eps
    self.id2label = getattr(cfg, "id2label", {}) or {}
    self._pref, self._g = pref, (lambda k: sd[k])
    self._lfmt, self._km = _vit_layer_keys(sd, pref)
    n = (self.img // self.P) ** 2                                      # number of patches
    self._cls = sd[f"{pref}.embeddings.cls_token"].reshape(1, self.D)
    self._pos = sd[f"{pref}.embeddings.position_embeddings"].reshape(n + 1, self.D)
    self._model = self._build(n)

  def _build(self, n: int) -> Model | SegmentedModel:
    g, pref, D, eps, H = self._g, self._pref, self.D, self.eps, self.H
    x = input((1, 3, self.img, self.img)); cls = input((1, D)); pos = input((n + 1, D))
    # a strided PxP patch conv is walled on the ANE -> space_to_depth(P) + a 1x1 conv
    w_pe = np.ascontiguousarray(
      g(f"{pref}.embeddings.patch_embeddings.projection.weight").transpose(0, 2, 3, 1)).reshape(D, -1, 1, 1)
    h = conv(space_to_depth(x, self.P), w_pe,
             bias=g(f"{pref}.embeddings.patch_embeddings.projection.bias"))
    patches = h.reshape(1, D, n).transpose([0, 2, 1]).reshape(n, D)
    seq = concat([cls, patches], axis=0) + pos
    km = self._km
    for i in range(self.L):
      p = self._lfmt.format(i=i)
      xn = seq.layer_norm(g(p + km["ln1"] + ".weight"), g(p + km["ln1"] + ".bias"), eps)
      attn = mha(xn,
                 g(p + km["q"] + ".weight"), g(p + km["q"] + ".bias"),
                 g(p + km["k"] + ".weight"), g(p + km["k"] + ".bias"),
                 g(p + km["v"] + ".weight"), g(p + km["v"] + ".bias"),
                 g(p + km["o"] + ".weight"), g(p + km["o"] + ".bias"), H)
      seq = seq + attn
      yn = seq.layer_norm(g(p + km["ln2"] + ".weight"), g(p + km["ln2"] + ".bias"), eps)
      y = yn.linear(g(p + km["fc1"] + ".weight"), g(p + km["fc1"] + ".bias")).gelu()
      seq = seq + y.linear(g(p + km["fc2"] + ".weight"), g(p + km["fc2"] + ".bias"))
    seq = seq.layer_norm(g(f"{pref}.layernorm.weight"), g(f"{pref}.layernorm.bias"), eps)
    return compile(_row0(seq).linear(g("classifier.weight"), g("classifier.bias")), int8=self.int8)

  def _pixels(self, image) -> np.ndarray:
    if isinstance(image, np.ndarray) and image.ndim == 4:
      return image.astype(np.float32)
    return np.asarray(self.proc(image, return_tensors="np")["pixel_values"], np.float32)

  def __call__(self, image) -> np.ndarray:
    return np.asarray(self._model(self._pixels(image), self._cls, self._pos), np.float32)

  def classify(self, image, top_k: int = 5):
    """Top-k (label, logit) for an image (PIL, path, or preprocessed pixel array)."""
    logits = self(image)[0]
    return [(self.id2label.get(int(i), str(int(i))), float(logits[i]))
            for i in np.argsort(-logits)[:top_k]]

  def release(self) -> None: self._model.release()

  @property
  def n_ops(self) -> int: return self._model.n_ops


def load_gpt2(name: str, int8: bool = False, max_layers: int | None = None) -> "GPT2":
  """Load a Hugging Face GPT-2 causal LM as fused ANE programs: a pre-norm transformer
  (native causal SDPA) plus the tied lm_head tiled along vocab. `max_layers` trims the
  stack (the compile-fallback knob examples use when the full model will not fit)."""
  return GPT2(name, int8=int8, max_layers=max_layers)


def _gpt2_layers(sd: dict, L: int, D: int, Dff: int) -> list[dict]:
  """Per-layer weight maps from a HF GPT-2 state dict. GPT-2 projections are Conv1D
  (`[in, out]`): transpose each so `.linear()` (x @ W.T with W `[out, in]`) consumes it,
  and split the c_attn rows into q/k/v. LayerNorm and embedding tensors are NOT transposed."""
  layers = []
  for i in range(L):
    p = f"transformer.h.{i}."
    Wqkv = sd[p + "attn.c_attn.weight"].T                # [3D, D]
    bqkv = sd[p + "attn.c_attn.bias"]
    layers.append({
      "ln1w": sd[p + "ln_1.weight"], "ln1b": sd[p + "ln_1.bias"],
      "Wq": Wqkv[:D], "bq": bqkv[:D],
      "Wk": Wqkv[D:2 * D], "bk": bqkv[D:2 * D],
      "Wv": Wqkv[2 * D:3 * D], "bv": bqkv[2 * D:3 * D],
      "Wo": sd[p + "attn.c_proj.weight"].T, "bo": sd[p + "attn.c_proj.bias"],
      "ln2w": sd[p + "ln_2.weight"], "ln2b": sd[p + "ln_2.bias"],
      "Wi": sd[p + "mlp.c_fc.weight"].T, "bi": sd[p + "mlp.c_fc.bias"],          # [Dff, D]
      "Wd": sd[p + "mlp.c_proj.weight"].T, "bd": sd[p + "mlp.c_proj.bias"],      # [D, Dff]
    })
  return layers


def _gelu_new(x: Tensor) -> Tensor:
  """GPT-2's tanh-approximated GELU, composed from native ops (mirrors the ONNX
  `approximate="tanh"` handler, aneforge/onnx.py:352-360)."""
  inner = (x + x.pow(3.0) * 0.044715) * np.sqrt(2.0 / np.pi)
  return (x * 0.5) * inner.tanh().adds(1.0)


# smallest supported matmul output dim on family 3 (A13-A15, M1/M2); see bench/decode_measurement.py:130-135
_LMHEAD_TILE = 16384


def _lm_head_tiles(h: Tensor, wte: np.ndarray) -> list[Tensor]:
  """Tied lm_head logits = h @ wte.T, tiled along vocab so no matmul output dim exceeds
  `_LMHEAD_TILE`. Returns a LIST of tiles, never a concatenated [S, vocab] tensor: the
  concat itself exceeds the family-3 cap (the #183 lesson). The tiles are the head's
  output ports; stitching is host-side via `_logits_from`."""
  V = wte.shape[0]
  if V <= _LMHEAD_TILE:
    return [h.linear(wte)]
  return [h.linear(wte[i:i + _LMHEAD_TILE]) for i in range(0, V, _LMHEAD_TILE)]


def _logits_from(net: MultiModel | Model, out) -> np.ndarray:
  """Reassemble a tiled lm_head result into [S, vocab] host-side (the tiles are what the
  ANE produced; concatenating here does not change the numerics)."""
  if not isinstance(net, MultiModel):
    return np.asarray(out, np.float32)
  return np.concatenate([np.asarray(out[name], np.float32) for _, name in net.output_ports], axis=1)


class GPT2:
  """GPT-2 causal LM from Hugging Face, running on the ANE as two fused programs per
  sequence length: the pre-norm transformer (native causal SDPA; final hidden states out)
  and the tied lm_head tiled along vocab (tiles as output ports). Activations fp16;
  `int8=True` streams the weights per-channel int8. `GPT2(name)(ids)` -> logits [S, vocab];
  `.generate(prompt, K)` greedy-decodes K tokens by re-running the forward on the growing
  sequence (no KV cache: recompute per length, cached per length)."""

  def __init__(self, name: str, int8: bool = False, max_layers: int | None = None) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # lazy
    cfg = AutoConfig.from_pretrained(name)
    if cfg.model_type != "gpt2":
      raise ValueError(f"load_gpt2: {name!r} is not a GPT-2 model (model_type={cfg.model_type!r})")
    self.tok = AutoTokenizer.from_pretrained(name)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in AutoModelForCausalLM.from_pretrained(name).state_dict().items()}
    self.D, self.H, self.L = cfg.n_embd, cfg.n_head, cfg.num_hidden_layers
    self.dh = self.D // self.H
    self.Dff = cfg.n_inner or 4 * self.D
    self.eps = float(getattr(cfg, "layer_norm_epsilon", getattr(cfg, "layer_norm_eps", 1e-5)))
    self.int8 = int8
    self.wte = sd["transformer.wte.weight"]          # [V, D]; also the tied lm_head
    self.wpe = sd["transformer.wpe.weight"]          # [n_positions, D]
    self.lnf_w, self.lnf_b = sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]
    self.layers = _gpt2_layers(sd, self.L, self.D, self.Dff)
    if max_layers is not None:
      if not 1 <= max_layers <= self.L:
        raise ValueError(f"load_gpt2: max_layers must be in [1, {self.L}], got {max_layers}")
      self.layers = self.layers[:max_layers]
    self._cache: dict[int, tuple[Model | SegmentedModel, MultiModel | Model]] = {}

  def _build(self, S: int) -> Tensor:
    """The pre-norm transformer -> ln_f hidden states [S, D]. Causal SDPA is native here
    only while S < 512 (sdpa's reliable regime); pick short prompts and small K."""
    h = input((S, self.D))
    for w in self.layers:
      xn = h.layer_norm(w["ln1w"], w["ln1b"], self.eps)
      q = xn.linear(w["Wq"], w["bq"]).reshape(1, S, self.H, self.dh).transpose([0, 2, 1, 3])
      k = xn.linear(w["Wk"], w["bk"]).reshape(1, S, self.H, self.dh).transpose([0, 2, 1, 3])
      v = xn.linear(w["Wv"], w["bv"]).reshape(1, S, self.H, self.dh).transpose([0, 2, 1, 3])
      o = sdpa(q, k, v, is_causal=True).transpose([0, 2, 1, 3]).reshape(S, self.D)
      h = h + o.linear(w["Wo"], w["bo"])
      yn = h.layer_norm(w["ln2w"], w["ln2b"], self.eps)
      h = h + _gelu_new(yn.linear(w["Wi"], w["bi"])).linear(w["Wd"], w["bd"])
    return h.layer_norm(self.lnf_w, self.lnf_b, self.eps)

  def _compile(self, S: int) -> tuple[Model | SegmentedModel, MultiModel | Model]:
    """The two fused programs for sequence length S: the transformer (hidden [S, D] out;
    carries the native-SDPA graph cuts) and the head (pure matmul; one program when the
    vocab fits a single tile, tiled ports otherwise -- compile_multi rejects SDPA cuts)."""
    tiles = _lm_head_tiles(input((S, self.D)), self.wte.astype(np.float16))
    head = cast(MultiModel | Model,
                compile(tiles[0], int8=self.int8) if len(tiles) == 1 else compile_multi(tiles, int8=self.int8))
    return cast(Model | SegmentedModel, compile(self._build(S), int8=self.int8)), head

  def _embed(self, ids: np.ndarray) -> np.ndarray:
    """Host-side token + position embedding lookup (gather is not an ANE op)."""
    return (self.wte[ids] + self.wpe[:len(ids)]).astype(np.float32)

  def __call__(self, ids) -> np.ndarray:
    """1-D token ids -> logits [S, vocab]; the program pair for this S compiles on first use."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
      raise ValueError(f"GPT2.__call__ expects 1-D token ids, got shape {ids.shape}")
    S = ids.shape[0]
    net, head = self._cache.get(S) or self._cache.setdefault(S, self._compile(S))
    return _logits_from(head, head(np.asarray(net(self._embed(ids)), np.float32)))

  def generate(self, prompt: str, max_new_tokens: int = 16) -> list[int]:
    """Greedy-decode `max_new_tokens` tokens: re-run the forward on the growing sequence
    (no KV cache -- recompute per length; the per-S program cache keeps each step cheap)."""
    base = np.asarray(self.tok.encode(prompt), dtype=np.int64)
    ids = base
    for _ in range(max_new_tokens):
      ids = np.concatenate([ids, [int(self(ids)[-1].argmax())]])
    return [int(t) for t in ids[len(base):]]

  def release(self) -> None:
    for net, head in self._cache.values():
      net.release(); head.release()
    self._cache.clear()


def group_norm_train(x, gamma, beta, groups: int, eps: float = 1e-5):
  """Any-batch GroupNorm with trainable affine, built from VJP-bearing primitives; `x` is [N,C,H,W], gamma/beta [1,C,1,1]."""
  N, C, H, W = x.shape
  if C % groups:
    raise ValueError(f"group_norm_train: channels {C} not divisible by groups {groups}")
  M = (C // groups) * H * W
  xg = x.reshape(N, groups, M)
  xc = xg - xg.mean((2,))
  var = xc.square().mean((2,))
  xn = (xc * var.adds(float(eps)).rsqrt()).reshape(N, C, H, W)
  return xn * gamma + beta


def conv_block(x, conv_w, gamma, beta, groups: int, pool: int = 0):
  """conv2d(pad=1) -> GroupNorm(train) -> ReLU -> optional max_pool(pool); `pool=0` skips pooling."""
  h = conv2d(x, conv_w, pad=1)
  h = group_norm_train(h, gamma, beta, groups).relu()
  return h.max_pool(pool) if pool else h


def _he(rng, shape):
  """He/Kaiming-normal init; fan_in is layout-dependent (conv: trailing dims, 2-D fc: leading dim)."""
  fan_in = shape[0] if len(shape) == 2 else int(np.prod(shape[1:]))
  return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def cifar_cnn(batch: int, widths=(32, 64, 128), groups: int = 8, classes: int = 10, seed: int = 0):
  """Build the CIFAR-10 CNN graph; returns (x_input, logits, params) with params in fixed trainable order."""
  rng = np.random.default_rng(seed)
  w0, w1, w2 = widths
  x = input((batch, 3, 32, 32))
  cW1 = conv_param(_he(rng, (w0, 3, 3, 3)))
  cW2 = conv_param(_he(rng, (w1, w0, 3, 3)))
  cW3 = conv_param(_he(rng, (w2, w1, 3, 3)))
  g1 = parameter(np.ones((1, w0, 1, 1), np.float32)); b1 = parameter(np.zeros((1, w0, 1, 1), np.float32))
  g2 = parameter(np.ones((1, w1, 1, 1), np.float32)); b2 = parameter(np.zeros((1, w1, 1, 1), np.float32))
  g3 = parameter(np.ones((1, w2, 1, 1), np.float32)); b3 = parameter(np.zeros((1, w2, 1, 1), np.float32))
  Wfc = parameter(_he(rng, (w2, classes))); bfc = parameter(np.zeros((1, classes), np.float32))

  h = conv_block(x, cW1, g1, b1, groups, pool=2)     # -> [B, w0, 16, 16]
  h = conv_block(h, cW2, g2, b2, groups, pool=2)     # -> [B, w1,  8,  8]
  h = conv_block(h, cW3, g3, b3, groups, pool=0)     # -> [B, w2,  8,  8]
  h = h.mean((2, 3)).reshape(batch, w2)              # global average pool -> [B, w2]
  logits = (h @ Wfc) + bfc                           # [B, classes]
  params = [cW1, g1, b1, cW2, g2, b2, cW3, g3, b3, Wfc, bfc]
  return x, logits, params
