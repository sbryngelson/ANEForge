"""Pretrained-model loaders (`load`, `load_resnet18`) and trainable-graph
builders (`group_norm_train`, `conv_block`, `cifar_cnn`). See docs/developer/models.md."""
from __future__ import annotations

import numpy as np

from .graph import Tensor, concat, conv, input, mha, space_to_depth
from .autograd import conv2d, conv_param, parameter
from ._compile import Model, SegmentedModel, compile

_NORM_CACHE: dict[int, Model | SegmentedModel] = {}


def _l2_normalizer(D: int) -> Model | SegmentedModel:
  """Cached fused-ANE program L2-normalizing a [1, D] vector over its last axis."""
  net = _NORM_CACHE.get(D)
  if net is None: net = _NORM_CACHE[D] = compile(input((1, D)).l2_norm(axis=-1))
  return net


def load(name: str, int8: bool = False, pooling: str = "mean") -> "Encoder":
  """Load a BERT-family sentence encoder from HF weights as an ANE embedder; `pooling` in mean/cls/max."""
  return Encoder(name, int8=int8, pooling=pooling)


# ResNet stage layouts, straight from the torchvision factories: blocks per stage and the residual
# block shape. Bottleneck expands its output 4x, which is what makes its stage-1 shortcut projected.
_RESNETS: dict[int, tuple[str, tuple[int, int, int, int]]] = {
  18: ("basic", (2, 2, 2, 2)),
  34: ("basic", (3, 4, 6, 3)),
  50: ("bottleneck", (3, 4, 6, 3)),
  101: ("bottleneck", (3, 4, 23, 3)),
}


def load_resnet(name_or_depth: int | str = 18, int8: bool = False, compress: str | None = None,
                compress_atol: float = 0.05, build_dir: str | None = None,
                weights: str = "IMAGENET1K_V1") -> "Vision":
  """Load a torchvision ResNet (18/34/50/101, ImageNet) as a fused ANE classifier; BatchNorm folded
  into the preceding conv at load. `name_or_depth` takes 50, "50" or "resnet50"."""
  return Vision(name_or_depth, int8=int8, compress=compress, compress_atol=compress_atol,
                build_dir=build_dir, weights=weights)


def load_resnet18(int8: bool = False, compress: str | None = None,
                  compress_atol: float = 0.05, build_dir: str | None = None) -> "Vision":
  """Load torchvision ResNet-18 (ImageNet) as a fused ANE classifier; BatchNorm folded into the preceding conv at load."""
  return Vision(int8=int8, compress=compress, compress_atol=compress_atol, build_dir=build_dir)


def _resnet_depth(name_or_depth: int | str) -> int:
  """Accept 50, "50" or "resnet50"."""
  s = str(name_or_depth).lower().removeprefix("resnet")
  if not s.isdigit() or int(s) not in _RESNETS:
    raise ValueError(f"load_resnet: unsupported ResNet {name_or_depth!r}; "
                     f"supported depths are {sorted(_RESNETS)}")
  return int(s)


class Vision:
  def __init__(self, name_or_depth: int | str = 18, int8: bool = False, compress: str | None = None,
               compress_atol: float = 0.05, build_dir: str | None = None,
               weights: str = "IMAGENET1K_V1") -> None:
    import torchvision  # lazy
    self.depth = _resnet_depth(name_or_depth)
    self.block, self.stages = _RESNETS[self.depth]
    m = getattr(torchvision.models, f"resnet{self.depth}")(weights=weights).eval()
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

  def _shortcut(self, x: Tensor, prefix: str, stride: int) -> Tensor:
    """The residual branch: a projection when the block carries one, otherwise the identity.

    Presence is read from the weights rather than derived from a rule. torchvision projects when
    `stride != 1 or inplanes != planes * expansion`, so with Bottleneck even stage 1 is projected
    (64 -> 256 at stride 1) while with BasicBlock it is not. A "stage 1 never downsamples" rule holds
    for BasicBlock only, and would silently drop layer1's projection on ResNet-50/101."""
    if prefix + ".downsample.0.weight" not in self.sd:
      return x
    wd, bd = self._fold(prefix + ".downsample.0", prefix + ".downsample.1")
    return conv(x, wd, stride=stride, pad=0, bias=bd)

  def _block(self, x: Tensor, prefix: str, stride: int) -> Tensor:
    """BasicBlock: 3x3 -> 3x3, stride on the first conv."""
    w1, b1 = self._fold(prefix + ".conv1", prefix + ".bn1")
    w2, b2 = self._fold(prefix + ".conv2", prefix + ".bn2")
    out = conv(x, w1, stride=stride, pad=1, bias=b1).relu()
    out = conv(out, w2, stride=1, pad=1, bias=b2)
    return (out + self._shortcut(x, prefix, stride)).relu()

  def _bottleneck(self, x: Tensor, prefix: str, stride: int) -> Tensor:
    """Bottleneck: 1x1 -> 3x3 -> 1x1 with a 4x expansion. The stride sits on the 3x3, not on the
    first 1x1: that is torchvision's ResNet V1.5 variant, and putting it on conv1 instead changes
    which pixels survive."""
    w1, b1 = self._fold(prefix + ".conv1", prefix + ".bn1")
    w2, b2 = self._fold(prefix + ".conv2", prefix + ".bn2")
    w3, b3 = self._fold(prefix + ".conv3", prefix + ".bn3")
    out = conv(x, w1, stride=1, pad=0, bias=b1).relu()
    out = conv(out, w2, stride=stride, pad=1, bias=b2).relu()
    out = conv(out, w3, stride=1, pad=0, bias=b3)
    return (out + self._shortcut(x, prefix, stride)).relu()

  def _build(self) -> Model | SegmentedModel:
    x = input((1, 3, 224, 224))
    w, b = self._fold("conv1", "bn1")
    h = conv(x, w, stride=2, pad=3, bias=b).relu().max_pool(3, stride=2, pad=1)
    block = self._bottleneck if self.block == "bottleneck" else self._block
    for name, stride, n in zip(("layer1", "layer2", "layer3", "layer4"), (1, 2, 2, 2), self.stages):
      for i in range(n):
        h = block(h, f"{name}.{i}", stride if i == 0 else 1)
    feat = self.sd["fc.weight"].shape[1]          # 512 for BasicBlock, 2048 once expanded 4x
    h = h.mean((2, 3)).reshape(1, feat)
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
