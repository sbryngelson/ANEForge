"""Pretrained-model loaders (`load` and `CrossEncoder` for BERT/RoBERTa encoders and rerankers,
`load_resnet`/`load_vit` image classifiers, `load_gpt2` text generation) and trainable-graph
builders (`group_norm_train`, `conv_block`, `cifar_cnn`). See docs/developer/models.md."""
from __future__ import annotations


import numpy as np

from .graph import Tensor, concat, conv, cross_attention, input, mha, space_to_depth, _const
from .autograd import conv2d, conv_param, parameter
from ._compile import Model, SegmentedModel, MultiModel, compile, compile_multi
from . import _targets

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

_DISTILBERT_KEYS = {
  "Wq": "attention.q_lin.weight", "bq": "attention.q_lin.bias",
  "Wk": "attention.k_lin.weight", "bk": "attention.k_lin.bias",
  "Wv": "attention.v_lin.weight", "bv": "attention.v_lin.bias",
  "Wo": "attention.out_lin.weight", "bo": "attention.out_lin.bias",
  "ln1w": "sa_layer_norm.weight", "ln1b": "sa_layer_norm.bias",
  "Wi": "ffn.lin1.weight", "bi": "ffn.lin1.bias",
  "Wd": "ffn.lin2.weight", "bd": "ffn.lin2.bias",
  "ln2w": "output_layer_norm.weight", "ln2b": "output_layer_norm.bias",
}


def _encoder_layer_spec(model_type: str) -> tuple[str, dict[str, str]]:
  """Return the layer prefix and weight-key map for a supported encoder family."""
  if model_type == "distilbert":
    return "transformer.layer.{i}.", _DISTILBERT_KEYS
  return "encoder.layer.{i}.", _BERT_KEYS


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
    m = input((1, S, S))                     # additive key-padding mask: 0 on real keys, -1e4 on padded ones
    for w in self.layers:
      attn = mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"], w["Wo"], w["bo"], self.H, mask=m)
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
    enc = [np.asarray(self.tok(t)["input_ids"], dtype=np.int64) for t in texts]
    # Pad the whole batch to its longest sequence and compile ONE program for that length -- the padded
    # keys are masked out per text, so a corpus of varied lengths shares one program instead of one each.
    S = max((len(ids) for ids in enc), default=1)
    net = self._cache.get(S) or self._cache.setdefault(S, self._build(S))
    pad_id = self.tok.pad_token_id or 0
    vecs = []
    for ids in enc:
      n = len(ids)
      padded = np.full(S, pad_id, dtype=np.int64); padded[:n] = ids
      mask = np.zeros((1, S, S), dtype=np.float32); mask[0, :, n:] = -1e4   # mask the padded key columns
      states = net(self._embed(padded), mask)[:n]  # real-token states on the ANE (pads masked + dropped)
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


def _position_ids(ids: np.ndarray, pad_id: int | None) -> np.ndarray:
  """Position ids as the reference implementations build them.

  BERT counts from 0. RoBERTa counts from `padding_idx + 1` and skips pads, i.e. HF's
  `create_position_ids_from_input_ids`: `cumsum(ids != pad) * mask + pad`. That offset is why a
  RoBERTa position table has `max_position_embeddings + 2` rows; counting from 0 reads the wrong row
  for every token and silently shifts the whole sequence."""
  if pad_id is None:
    return np.arange(len(ids))
  mask = (ids != pad_id).astype(np.int64)
  return np.cumsum(mask) * mask + pad_id


def _seqcls_head(
  sd: dict, pref: str | None, model_type: str | None = None
) -> tuple[str, str, str, str] | None:
  """Return (dense_w, dense_b, out_w, out_b) classification-head state-dict keys.

  BERT is `pooler.dense -> tanh -> classifier`; RoBERTa is `RobertaClassificationHead`, i.e.
  `classifier.dense -> tanh -> classifier.out_proj`, with no pooler at all. Same arithmetic on token 0,
  different names. DistilBERT uses `pre_classifier -> ReLU -> classifier` and has no pooler.
  Returns None when neither shape is present.

  RoBERTa is identified by its head, *not* by a missing token-type table: XLM-R does ship
  `token_type_embeddings` (a single row, always index 0), so absence identifies nothing."""
  if model_type == "distilbert" and "pre_classifier.weight" in sd and "classifier.weight" in sd:
    return ("pre_classifier.weight", "pre_classifier.bias", "classifier.weight", "classifier.bias")
  if "classifier.dense.weight" in sd and "classifier.out_proj.weight" in sd:
    return ("classifier.dense.weight", "classifier.dense.bias",
            "classifier.out_proj.weight", "classifier.out_proj.bias")
  if pref is not None and f"{pref}.pooler.dense.weight" in sd and "classifier.weight" in sd:
    return (f"{pref}.pooler.dense.weight", f"{pref}.pooler.dense.bias",
            "classifier.weight", "classifier.bias")
  return None


def _seqcls_activation(values: np.ndarray, model_type: str) -> np.ndarray:
  """Apply the activation used between the sequence-classification head layers."""
  return np.maximum(values, 0) if model_type == "distilbert" else np.tanh(values)


def _token_type_ids(enc: dict, length: int, has_token_type_embeddings: bool) -> np.ndarray:
  """Return segment ids, ignoring tokenizer output when the model has no segment table."""
  if not has_token_type_embeddings:
    return np.zeros(length, dtype=np.int64)
  values = enc.get("token_type_ids")
  return np.asarray(values if values is not None else [0] * length, dtype=np.int64)


class CrossEncoder:
  """Reranker: scores (query, passage) pairs with a BERT-, RoBERTa-, or DistilBERT-family sequence-classification
  model, the transformer running on the ANE. Mirrors `sentence_transformers.CrossEncoder`:
  `CrossEncoder(name).predict([(query, passage), ...])` returns one relevance score per pair
  (raw logits -- order is what a reranker needs). Higher is more relevant.

  BERT and RoBERTa share the encoder graph and `_BERT_KEYS`; DistilBERT uses the same graph
  operations with a family-specific key map. The families differ only in weight selection and
  host-side head plumbing, rather than requiring separate graph code paths:

  - **Head.** See `_seqcls_head`: BERT/RoBERTa use tanh; DistilBERT uses ReLU.
  - **Position ids.** See `_position_ids`: RoBERTa counts from `padding_idx + 1`; BERT and DistilBERT count from 0."""

  def __init__(self, name: str, int8: bool = False) -> None:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # lazy
    cfg = AutoConfig.from_pretrained(name)
    self.tok = AutoTokenizer.from_pretrained(name)
    sd = AutoModelForSequenceClassification.from_pretrained(name).state_dict()
    g = lambda k: sd[k].detach().numpy().astype(np.float32)
    # The base transformer sits under a model-specific prefix (bert / roberta / ...).
    marker = ".embeddings.word_embeddings.weight"
    pref = next((k[: -len(marker)] for k in sd if k.endswith(marker)), None)
    head = _seqcls_head(sd, pref, cfg.model_type)
    if pref is None or head is None:
      raise ValueError(
        f"CrossEncoder: {name!r} is not a supported reranker -- it needs either a BERT-family head "
        "(a pooler plus `classifier`) or a RoBERTa-family one (`classifier.dense` plus "
        "`classifier.out_proj`), or a DistilBERT-family head (`pre_classifier` plus `classifier`).")
    roberta = head[0].startswith("classifier.")
    layer_prefix, layer_keys = _encoder_layer_spec(cfg.model_type)
    self.D, self.H = cfg.hidden_size, cfg.num_attention_heads
    # DistilBertConfig omits layer_norm_eps; its reference implementation uses 1e-12.
    self.L, self.eps, self.int8 = cfg.num_hidden_layers, getattr(cfg, "layer_norm_eps", 1e-12), int8
    self.pad_id = int(cfg.pad_token_id) if roberta else None   # None -> positions count from 0
    self.word = g(f"{pref}.embeddings.word_embeddings.weight")
    self.pos = g(f"{pref}.embeddings.position_embeddings.weight")
    typ_key = f"{pref}.embeddings.token_type_embeddings.weight"
    self._has_typ = typ_key in sd
    self.typ = g(typ_key) if self._has_typ else np.zeros((1, self.D), np.float32)
    self.eln_w, self.eln_b = g(f"{pref}.embeddings.LayerNorm.weight"), g(f"{pref}.embeddings.LayerNorm.bias")
    self.layers = [{k: g(f"{pref}.{layer_prefix.format(i=i)}" + v) for k, v in layer_keys.items()}
                   for i in range(self.L)]
    self.pool_w, self.pool_b, self.cls_w, self.cls_b = (g(k) for k in head)
    self._model_type = cfg.model_type
    self._cache: dict[int, Model | SegmentedModel] = {}

  def _build(self, S: int) -> Model | SegmentedModel:
    h = input((S, self.D))
    m = input((1, S, S))                     # additive key-padding mask: 0 on real keys, -1e4 on padded ones
    for w in self.layers:
      attn = mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"], w["Wo"], w["bo"], self.H, mask=m)
      h = (h + attn).layer_norm(w["ln1w"], w["ln1b"], self.eps)
      ff = h.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
      h = (h + ff).layer_norm(w["ln2w"], w["ln2b"], self.eps)
    return compile(h, int8=self.int8)

  def _embed(self, ids: np.ndarray, typ_ids: np.ndarray) -> np.ndarray:
    """Host-side token + position + segment embedding lookup, then LayerNorm."""
    e = self.word[ids] + self.pos[_position_ids(ids, self.pad_id)] + self.typ[typ_ids]
    m = e.mean(-1, keepdims=True)
    v = ((e - m) ** 2).mean(-1, keepdims=True)
    return ((e - m) / np.sqrt(v + self.eps) * self.eln_w + self.eln_b).astype(np.float32)

  def predict(self, pairs) -> np.ndarray:
    """`pairs` is a (query, passage) tuple or a list of them; returns a relevance score each."""
    if pairs and isinstance(pairs[0], str):     # a single (query, passage) pair
      pairs = [pairs]
    enc = [self.tok(q, p, truncation=True) for q, p in pairs]
    ids = [np.asarray(e["input_ids"], dtype=np.int64) for e in enc]
    # Pad the batch to its longest pair and compile ONE program; padded keys are masked per pair, so a
    # query's candidates (and the next query) share one program instead of one per pair length.
    S = max((len(i) for i in ids), default=1)
    net = self._cache.get(S) or self._cache.setdefault(S, self._build(S))
    pad_id = self.pad_id or 0
    scores = []
    for e, tid in zip(enc, ids):
      n = len(tid)
      typ = _token_type_ids(e, n, self._has_typ)
      padded = np.full(S, pad_id, dtype=np.int64); padded[:n] = tid
      typ_p = np.zeros(S, dtype=np.int64); typ_p[:n] = typ
      mask = np.zeros((1, S, S), dtype=np.float32); mask[0, :, n:] = -1e4   # mask the padded key columns
      cls = net(self._embed(padded, typ_p), mask)[0]                   # [CLS] / <s> state (row 0), on the ANE
      pooled = _seqcls_activation(cls @ self.pool_w.T + self.pool_b, self._model_type)
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


def _whisper_layers(sd: dict, prefix: str, n: int) -> list[dict]:
  """Per-layer numpy weights for the Whisper encoder/decoder graphs. HF linear weights are [out, in]
  and used as-is by `.linear()`. Whisper's k_proj carries no bias, so bk/Cbk are omitted. A decoder
  layer additionally carries the cross-attention set (C*) and its layer norm (cln)."""
  out = []
  for i in range(n):
    p = f"model.{prefix}.layers.{i}."
    g = lambda k: sd[p + k]
    w = {"Wq": g("self_attn.q_proj.weight"), "bq": g("self_attn.q_proj.bias"),
         "Wk": g("self_attn.k_proj.weight"),
         "Wv": g("self_attn.v_proj.weight"), "bv": g("self_attn.v_proj.bias"),
         "Wo": g("self_attn.out_proj.weight"), "bo": g("self_attn.out_proj.bias"),
         "ln1w": g("self_attn_layer_norm.weight"), "ln1b": g("self_attn_layer_norm.bias"),
         "Wi": g("fc1.weight"), "bi": g("fc1.bias"), "Wd": g("fc2.weight"), "bd": g("fc2.bias"),
         "ln2w": g("final_layer_norm.weight"), "ln2b": g("final_layer_norm.bias")}
    if prefix == "decoder":
      w.update({"CWq": g("encoder_attn.q_proj.weight"), "Cbq": g("encoder_attn.q_proj.bias"),
                "CWk": g("encoder_attn.k_proj.weight"),
                "CWv": g("encoder_attn.v_proj.weight"), "Cbv": g("encoder_attn.v_proj.bias"),
                "CWo": g("encoder_attn.out_proj.weight"), "Cbo": g("encoder_attn.out_proj.bias"),
                "cln_w": g("encoder_attn_layer_norm.weight"), "cln_b": g("encoder_attn_layer_norm.bias")})
    out.append(w)
  return out


def _gelu_new(x: Tensor) -> Tensor:
  """GPT-2's tanh-approximated GELU, composed from native ops (mirrors the ONNX
  `approximate="tanh"` handler, aneforge/onnx.py:352-360)."""
  inner = (x + x.pow(3.0) * 0.044715) * np.sqrt(2.0 / np.pi)
  return (x * 0.5) * inner.tanh().adds(1.0)


def _lm_head_tiles(h: Tensor, wte: np.ndarray) -> list[Tensor]:
  """Tied lm_head logits = h @ wte.T, tiled along vocab so no matmul output dim exceeds the
  target family's max tensor dimension. Returns a LIST of tiles, never a concatenated [S, vocab]
  tensor: the concat itself exceeds the family-3 cap (the #183 lesson). The tiles are the head's
  output ports; stitching is host-side via `_logits_from`."""
  V = wte.shape[0]
  tile = _targets.limit("max_tensor_dim", _targets.detect_family())
  if V <= tile:
    return [h.linear(wte)]
  return [h.linear(wte[i:i + tile]) for i in range(0, V, tile)]


def _logits_from(net: MultiModel | Model, out) -> np.ndarray:
  """Reassemble a tiled lm_head result into [S, vocab] host-side (the tiles are what the
  ANE produced; concatenating here does not change the numerics)."""
  if not isinstance(net, MultiModel):
    return np.asarray(out, np.float32)
  return np.concatenate([np.asarray(out[name], np.float32) for _, name in net.output_ports], axis=1)


class GPT2:
  """GPT-2 causal LM from Hugging Face, running on the ANE via the unified LLM runner with
  resident KV-cache decode, LayerNorm, and learned positional embeddings. Activations fp16;
  `int8=True` quantizes weights per-channel int8. `GPT2(name)(ids)` -> logits [S, vocab];
  `.generate(prompt, K)` autoregressively generates K tokens using the resident KV cache."""

  def __init__(self, name: str, int8: bool = False, max_layers: int | None = None) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # lazy
    from .llm import _gpt2_adapter, LlamaPrefill, ModelType
    cfg_hf = AutoConfig.from_pretrained(name)
    if cfg_hf.model_type != ModelType.GPT2:
      raise ValueError(f"load_gpt2: {name!r} is not a GPT-2 model (model_type={cfg_hf.model_type!r})")
    self.tok = AutoTokenizer.from_pretrained(name)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in AutoModelForCausalLM.from_pretrained(name).state_dict().items()}
    cfg, weights = _gpt2_adapter(cfg_hf, sd)
    if max_layers is not None:
      if not 1 <= max_layers <= cfg.n_layers:
        raise ValueError(f"load_gpt2: max_layers must be in [1, {cfg.n_layers}], got {max_layers}")
      cfg.n_layers = max_layers
      cfg.layers = cfg.layers[:max_layers]
      weights["layers"] = weights["layers"][:max_layers]
    self.cfg = cfg
    self.w = weights
    self.D, self.H, self.L = cfg.dim, cfg.n_heads, cfg.n_layers
    self.dh = cfg.dh
    self.Dff = cfg.ffn_dim
    self.eps = cfg.norm_eps
    self.int8 = int8
    self.wte = weights["embed"]
    self.wpe = weights["wpe"]
    self.runner = LlamaPrefill(cfg, weights, compress="int8" if int8 else None)

  def __call__(self, ids) -> np.ndarray:
    """1-D token ids -> logits [S, vocab]."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
      raise ValueError(f"GPT2.__call__ expects 1-D token ids, got shape {ids.shape}")
    hidden = self.runner._hidden(ids)
    return hidden @ np.asarray(self.w["lm_head"]).T

  def generate(self, prompt: str, max_new_tokens: int = 16, **kwargs) -> list[int]:
    """Autoregressive generation with resident KV-cache."""
    base = np.asarray(self.tok.encode(prompt), dtype=np.int64)
    return self.runner.generate(base, max_new_tokens=max_new_tokens, **kwargs)

  def generate_text(self, prompt: str, max_new_tokens: int = 16) -> str:
    """Greedy-decode and return the newly generated tokens as text."""
    return self.tok.decode(self.generate(prompt, max_new_tokens))

  def release(self) -> None:
    self.runner.release()


def _quick_gelu(x: Tensor) -> Tensor:
  """CLIP's QuickGELU activation: x * sigmoid(1.702 * x)."""
  return x * (x * 1.702).sigmoid()


def _causal_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  """Causal self-attention on q,k,v [1,H,S,dh] as decomposed softmax(q@k^T*scale + mask)@v.
  Stays in ONE fused ANE program without graph cuts."""
  S, dh = q.shape[2], q.shape[3]
  mask = np.triu(np.full((S, S), -1e4, np.float32), 1)
  return ((q @ k.transpose([0, 1, 3, 2])) * (1.0 / np.sqrt(dh)) + _const(mask.astype(np.float16))).softmax(-1) @ v


def load_clip(name: str = "openai/clip-vit-base-patch32", int8: bool = False) -> "CLIP":
  """Load a Hugging Face CLIP dual-encoder model (CLIPModel) for zero-shot image/text classification on the ANE."""
  return CLIP(name, int8=int8)


class CLIP:
  """CLIP dual-encoder model from Hugging Face (CLIPModel), running both vision and text encoders on the ANE.
  `.encode_image(image)` -> [1, proj_dim] normalized image embedding.
  `.encode_text(texts)` -> [N, proj_dim] normalized text embeddings.
  `.classify(image, candidate_labels)` -> sorted list of (label, probability)."""

  def __init__(self, name: str = "openai/clip-vit-base-patch32", int8: bool = False) -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer, CLIPModel  # lazy
    cfg = AutoConfig.from_pretrained(name)
    if getattr(cfg, "model_type", None) != "clip":
      raise ValueError(f"load_clip: {name!r} is not a CLIP model (model_type={getattr(cfg, 'model_type', None)!r})")
    self.tok = AutoTokenizer.from_pretrained(name)
    try:
      self.proc = AutoImageProcessor.from_pretrained(name)
    except Exception:
      self.proc = None
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in CLIPModel.from_pretrained(name).state_dict().items()}
    self.int8 = int8
    v_cfg, t_cfg = cfg.vision_config, cfg.text_config
    self.Dv, self.Hv, self.Lv = v_cfg.hidden_size, v_cfg.num_attention_heads, v_cfg.num_hidden_layers
    self.Pv, self.img, self.v_eps = v_cfg.patch_size, v_cfg.image_size, v_cfg.layer_norm_eps
    self.Dt, self.Ht, self.Lt = t_cfg.hidden_size, t_cfg.num_attention_heads, t_cfg.num_hidden_layers
    self.t_eps = t_cfg.layer_norm_eps
    self.St = int(getattr(t_cfg, "max_position_embeddings", 77))
    self.proj_dim = getattr(cfg, "projection_dim", 512)
    self.logit_scale = float(np.exp(sd.get("logit_scale", np.log(100.0))))

    # Vision weights & embeddings
    self.nv = (self.img // self.Pv) ** 2
    self._cls_v = sd["vision_model.embeddings.class_embedding"].reshape(1, self.Dv)
    self._pos_v = sd["vision_model.embeddings.position_embedding.weight"].reshape(self.nv + 1, self.Dv)
    self._w_pe_v = np.ascontiguousarray(
      sd["vision_model.embeddings.patch_embedding.weight"].transpose(0, 2, 3, 1)).reshape(self.Dv, -1, 1, 1)

    # Text weights & embeddings needed at runtime
    self.token_embed = sd["text_model.embeddings.token_embedding.weight"]
    self.pos_embed = sd["text_model.embeddings.position_embedding.weight"]
    self.text_proj = sd["text_projection.weight"]

    self._v_model = self._build_vision(sd)
    self._t_model = self._build_text(sd, self.St)

  def _build_vision(self, sd: dict) -> Model | SegmentedModel:
    Dv, Hv, Lv, Pv, img, eps = self.Dv, self.Hv, self.Lv, self.Pv, self.img, self.v_eps
    nv = self.nv
    x_img = input((1, 3, img, img)); cls_in = input((1, Dv)); pos_in = input((nv + 1, Dv))
    h = conv(space_to_depth(x_img, Pv), self._w_pe_v, bias=None)
    patches = h.reshape(1, Dv, nv).transpose([0, 2, 1]).reshape(nv, Dv)
    seq = concat([cls_in, patches], axis=0) + pos_in
    seq = seq.layer_norm(sd["vision_model.pre_layrnorm.weight"], sd["vision_model.pre_layrnorm.bias"], eps)
    for i in range(Lv):
      p = f"vision_model.encoder.layers.{i}."
      xn = seq.layer_norm(sd[p + "layer_norm1.weight"], sd[p + "layer_norm1.bias"], eps)
      attn = mha(xn,
                 sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.q_proj.bias"],
                 sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.k_proj.bias"],
                 sd[p + "self_attn.v_proj.weight"], sd[p + "self_attn.v_proj.bias"],
                 sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"], Hv)
      seq = seq + attn
      yn = seq.layer_norm(sd[p + "layer_norm2.weight"], sd[p + "layer_norm2.bias"], eps)
      mlp = _quick_gelu(yn.linear(sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"])).linear(
        sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"])
      seq = seq + mlp
    cls_out = _row0(seq).layer_norm(sd["vision_model.post_layernorm.weight"], sd["vision_model.post_layernorm.bias"], eps)
    v_proj = cls_out.linear(sd["visual_projection.weight"])
    return compile(v_proj.l2_norm(axis=-1), int8=self.int8)

  def _build_text(self, sd: dict, S: int) -> Model | SegmentedModel:
    Dt, Ht, Lt, eps = self.Dt, self.Ht, self.Lt, self.t_eps
    dht = Dt // Ht
    x_txt = input((S, Dt))
    seq = x_txt
    for i in range(Lt):
      p = f"text_model.encoder.layers.{i}."
      xn = seq.layer_norm(sd[p + "layer_norm1.weight"], sd[p + "layer_norm1.bias"], eps)
      q = xn.linear(sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.q_proj.bias"]).reshape(1, S, Ht, dht).transpose([0, 2, 1, 3])
      k = xn.linear(sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.k_proj.bias"]).reshape(1, S, Ht, dht).transpose([0, 2, 1, 3])
      v = xn.linear(sd[p + "self_attn.v_proj.weight"], sd[p + "self_attn.v_proj.bias"]).reshape(1, S, Ht, dht).transpose([0, 2, 1, 3])
      attn = _causal_attn(q, k, v).transpose([0, 2, 1, 3]).reshape(S, Dt)
      seq = seq + attn.linear(sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"])
      yn = seq.layer_norm(sd[p + "layer_norm2.weight"], sd[p + "layer_norm2.bias"], eps)
      mlp = _quick_gelu(yn.linear(sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"])).linear(
        sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"])
      seq = seq + mlp
    seq = seq.layer_norm(sd["text_model.final_layer_norm.weight"], sd["text_model.final_layer_norm.bias"], eps)
    return compile(seq, int8=self.int8)

  def _pixels(self, image) -> np.ndarray:
    if isinstance(image, np.ndarray) and image.ndim == 4 and image.shape[1:] == (3, self.img, self.img):
      return image.astype(np.float32)
    if self.proc is not None:
      return np.asarray(self.proc(images=image, return_tensors="np")["pixel_values"], np.float32)
    # Fallback normalization: PIL if present, else pure-numpy resampling
    try:
      import PIL.Image
      if not isinstance(image, PIL.Image.Image):
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 3: arr = arr.transpose(1, 2, 0)
        u8 = np.asarray(arr if arr.max() > 1.0 else arr * 255.0, dtype=np.uint8)
        image = PIL.Image.fromarray(u8)
      image = image.resize((self.img, self.img), PIL.Image.Resampling.BICUBIC)
      arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    except Exception:
      arr = np.asarray(image, dtype=np.float32)
      if arr.ndim == 3 and arr.shape[-1] == 3: arr = arr.transpose(2, 0, 1)
      if arr.ndim == 2: arr = np.repeat(arr[None], 3, axis=0)
      if arr.shape[1:] != (self.img, self.img):
        H, W = arr.shape[1], arr.shape[2]
        r = np.linspace(0, H - 1, self.img).astype(int)
        c = np.linspace(0, W - 1, self.img).astype(int)
        arr = arr[:, r[:, None], c[None, :]]
      if arr.max() > 1.0: arr /= 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    return ((arr - mean) / std)[None].astype(np.float32)

  def encode_image(self, image) -> np.ndarray:
    """Encode an image -> L2-normalized embedding vector [1, proj_dim]."""
    px = self._pixels(image)
    return np.asarray(self._v_model(px, self._cls_v, self._pos_v), np.float32)

  def encode_text(self, texts: str | list[str]) -> np.ndarray:
    """Encode text string(s) -> L2-normalized embedding vectors [N, proj_dim]."""
    if isinstance(texts, str): texts = [texts]
    ids = np.asarray(self.tok(texts, padding="max_length", max_length=self.St, truncation=True, return_tensors="np")["input_ids"], dtype=np.int64)
    vecs = []
    for row in ids:
      eot_pos = int(row.argmax())
      emb = self.token_embed[row] + self.pos_embed[:self.St]
      hidden = np.asarray(self._t_model(emb), np.float32)
      eot = hidden[eot_pos:eot_pos + 1]
      proj = eot @ self.text_proj.T
      norm_proj = proj / np.linalg.norm(proj, axis=-1, keepdims=True)
      vecs.append(norm_proj[0])
    return np.asarray(vecs, dtype=np.float32)

  def classify(self, image, candidate_labels: list[str]) -> list[tuple[str, float]]:
    """Zero-shot image classification: score image against candidate labels, returning sorted (label, prob) pairs."""
    img_feat = self.encode_image(image)
    txt_feat = self.encode_text(candidate_labels)
    logits = (img_feat @ txt_feat.T * self.logit_scale)[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    ranked = sorted(zip(probs, candidate_labels), key=lambda p: p[0], reverse=True)
    return [(label, float(prob)) for prob, label in ranked]

  def release(self) -> None:
    self._v_model.release()
    self._t_model.release()


def load_whisper(name: str = "openai/whisper-base.en") -> "Whisper":
  """Load a Hugging Face Whisper speech-to-text model with both towers (audio encoder + text
  decoder) running on the ANE. `Whisper(name).transcribe(audio)` -> greedy English transcript."""
  return Whisper(name)


class Whisper:
  """OpenAI Whisper (encoder-decoder ASR) on the ANE. The audio encoder is one fused program run
  once per clip; the text decoder is one fused program per greedy step (recompute-per-step, no KV
  cache yet). Host-side log-mel and tokenization only. Default `whisper-base.en`. English, greedy,
  no timestamps. `.transcribe(audio)` -> str; `.encode(audio)` -> audio features [1500, 512]."""

  MAX_DEC = 128                                                # padded decoder length for the demo

  def __init__(self, name: str = "openai/whisper-base.en") -> None:
    from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor  # lazy
    cfg = AutoConfig.from_pretrained(name)
    if getattr(cfg, "model_type", None) != "whisper":
      raise ValueError(f"load_whisper: {name!r} is not a Whisper model (model_type={cfg.model_type!r})")
    self.proc = WhisperProcessor.from_pretrained(name)
    sd = {k: v.detach().numpy().astype(np.float16)
          for k, v in WhisperForConditionalGeneration.from_pretrained(name).state_dict().items()}
    self.D, self.H = cfg.d_model, cfg.encoder_attention_heads
    self.n_mels, self.vocab = cfg.num_mel_bins, cfg.vocab_size
    self.sot = [cfg.decoder_start_token_id, 50362]            # <|startoftranscript|>, <|notimestamps|>
    self.eot = cfg.eos_token_id                               # <|endoftext|>
    g = lambda k: sd[k]
    self._enc_conv1 = g("model.encoder.conv1.weight").reshape(self.D, self.n_mels, 1, 3)
    self._enc_conv1_b = g("model.encoder.conv1.bias")
    self._enc_conv2 = g("model.encoder.conv2.weight").reshape(self.D, self.D, 1, 3)
    self._enc_conv2_b = g("model.encoder.conv2.bias")
    self._enc_pos = g("model.encoder.embed_positions.weight")           # [1500, D] sinusoidal
    self._enc_lnw, self._enc_lnb = g("model.encoder.layer_norm.weight"), g("model.encoder.layer_norm.bias")
    self._enc_layers = _whisper_layers(sd, "encoder", cfg.encoder_layers)
    self._dec_tok = g("model.decoder.embed_tokens.weight")              # [vocab, D], tied to proj_out
    self._dec_pos = g("model.decoder.embed_positions.weight")          # [max_target_positions, D] learned
    self._dec_lnw, self._dec_lnb = g("model.decoder.layer_norm.weight"), g("model.decoder.layer_norm.bias")
    self._dec_layers = _whisper_layers(sd, "decoder", cfg.decoder_layers)
    self._encoder = self._build_encoder()
    self._decoder = None                                       # built lazily in Task 3

  def _build_encoder(self) -> Model | SegmentedModel:
    mel = input((1, self.n_mels, 1, 3000))                    # log-mel as a 1xT "image" for the 1D convs
    h = conv(mel, self._enc_conv1, pad=(0, 0, 1, 1), bias=self._enc_conv1_b).gelu()      # [1, D, 1, 3000]
    h = conv(h, self._enc_conv2, stride=2, pad=(0, 0, 1, 1), bias=self._enc_conv2_b).gelu()  # [1, D, 1, 1500]
    h = h.reshape(self.D, 1500).transpose([1, 0]) + self._enc_pos    # [1500, D] + sinusoidal positions
    for w in self._enc_layers:
      a = mha(h.layer_norm(w["ln1w"], w["ln1b"]), w["Wq"], w["bq"], w["Wk"], None,
              w["Wv"], w["bv"], w["Wo"], w["bo"], self.H)
      h = h + a
      f = h.layer_norm(w["ln2w"], w["ln2b"]).linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
      h = h + f
    return compile(h.layer_norm(self._enc_lnw, self._enc_lnb))

  def _features(self, audio: np.ndarray) -> np.ndarray:
    """Host log-mel [n_mels, 3000] via the Whisper feature extractor (no ANE)."""
    audio = np.asarray(audio, dtype=np.float32)
    mel = self.proc.feature_extractor(audio, sampling_rate=16000, return_tensors="np")["input_features"]
    return np.asarray(mel, np.float32).reshape(self.n_mels, 3000)

  def encode(self, audio: np.ndarray) -> np.ndarray:
    """Audio features [1500, D] on the ANE for a 16 kHz mono clip (truncated/padded to 30 s)."""
    mel = self._features(audio).reshape(1, self.n_mels, 1, 3000)
    return np.asarray(self._encoder(mel), np.float32)

  def _build_decoder(self) -> MultiModel:
    S = self.MAX_DEC
    emb = input((S, self.D))                                  # token+position embedding (host gather)
    mask = input((1, S, S))                                   # causal + key-padding additive mask
    audio = input((1500, self.D))                             # encoder features (cross-attn K/V)
    h = emb
    for w in self._dec_layers:
      s = mha(h.layer_norm(w["ln1w"], w["ln1b"]), w["Wq"], w["bq"], w["Wk"], None,
              w["Wv"], w["bv"], w["Wo"], w["bo"], self.H, mask=mask)
      h = h + s
      c = cross_attention(h.layer_norm(w["cln_w"], w["cln_b"]), audio,
                          w["CWq"], w["CWk"], w["CWv"], w["CWo"], self.H,
                          bq=w["Cbq"], bk=None, bv=w["Cbv"], bo=w["Cbo"])
      h = h + c
      f = h.layer_norm(w["ln2w"], w["ln2b"]).linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
      h = h + f
    h = h.layer_norm(self._dec_lnw, self._dec_lnb)
    tiles = _lm_head_tiles(h, self._dec_tok)                  # tied head -> logits [S, vocab] (tiled by family)
    return compile_multi(tiles)

  def transcribe(self, audio: np.ndarray) -> str:
    """Greedy English transcript of a 16 kHz mono clip, both towers on the ANE."""
    if self._decoder is None:
      self._decoder = self._build_decoder()
    feats = self.encode(audio)
    S = self.MAX_DEC
    causal = np.triu(np.full((S, S), -1e4, np.float32), 1)    # query i attends to keys <= i
    ids = list(self.sot)
    while len(ids) < S:
      n = len(ids)
      emb = np.zeros((S, self.D), np.float32)
      emb[:n] = self._dec_tok[ids] + self._dec_pos[:n]        # token + learned positional embedding
      out = self._decoder(emb, causal.reshape(1, S, S), feats)
      logits = _logits_from(self._decoder, out)               # [S, vocab]
      nxt = int(np.argmax(logits[n - 1]))
      ids.append(nxt)
      if nxt == self.eot:
        break
    return self.proc.tokenizer.decode(ids[len(self.sot):], skip_special_tokens=True)

  def release(self) -> None:
    self._encoder.release()
    if self._decoder is not None:
      self._decoder.release()


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
