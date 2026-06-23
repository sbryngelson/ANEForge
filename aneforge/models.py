"""Pretrained-model loaders: `load` (BERT-family sentence encoders) and
`load_resnet18` (torchvision ImageNet classifier). Each builds an aneforge graph
from the real weights and compiles it to a fused ANE program. Heavy deps
(transformers / torchvision) are imported lazily so the core stays light.

Also ships the trainable-graph builders used with the on-ANE autograd:
`group_norm_train` (any-batch GroupNorm with trainable affine), `conv_block`
(conv -> GroupNorm -> ReLU -> optional max-pool), and `cifar_cnn` (a full
CIFAR-10 CNN returning the input, logits, and trainable parameter list)."""
from __future__ import annotations

import numpy as np

from .graph import Tensor, conv, input, mha
from .autograd import conv2d, conv_param, parameter
from ._compile import Model, compile

_NORM_CACHE: dict[int, Model] = {}


def _l2_normalizer(D: int) -> Model:
    """A tiny cached fused-ANE program that L2-normalizes a [1, D] vector over its
    last axis (the verified reduce_l2_norm + real_div path)."""
    net = _NORM_CACHE.get(D)
    if net is None:
        net = _NORM_CACHE[D] = compile(input((1, D)).l2_norm(axis=-1))
    return net


def load(name: str, int8: bool = False, pooling: str = "mean") -> "Encoder":
    """Load a BERT-family sentence encoder from HF weights as an ANE embedder.

        embed = af.load("sentence-transformers/all-MiniLM-L6-v2")
        vecs  = embed(["hello world", "the cat sat"])   # [2, D], L2-normalised

    Tokenisation + embedding lookup run on the host (gather is not an ANE op); the
    transformer layers run on the ANE as fused programs (cached per sequence
    length); pooling + normalise run on the host / ANE.

    `pooling` selects how the per-token states reduce to one vector: "mean" (the
    default, MiniLM / E5), "cls" (the first token, BGE / GTE), or "max". A model's
    correct mode is in its sentence-transformers config; `aneforge.sentence_transformers`
    reads it for you.
    """
    return Encoder(name, int8=int8, pooling=pooling)


def load_resnet18(int8: bool = False, compress: str | None = None,
                  compress_atol: float = 0.05, build_dir: str | None = None) -> "Vision":
    """Load torchvision ResNet-18 (ImageNet) as a fused ANE classifier.

        clf = af.load_resnet18()
        logits = clf(image)        # [1,3,224,224] -> [1,1000]
        clf = af.load_resnet18(compress="int4")   # 4-bit LUT weights

    BatchNorm is folded into the preceding conv at load, so the ANE graph is pure
    conv/relu/pool/add/fc. Conv is the ANE's strongest workload. `compress` picks
    the weight encoding (see `af.compile`); `build_dir` keeps the packed program
    on disk (its `weights.bin` is the packed-model size).
    """
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

    def _build(self) -> Model:
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

    def release(self) -> None:
        self._model.release()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3:
            image = image[None]
        return self._model(image)            # [1, 1000] logits

    @property
    def n_ops(self) -> int:
        return self._model.n_ops


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
        self._cache: dict[int, Model] = {}

    def _build(self, S: int) -> Model:
        h = input((S, self.D))
        for w in self.layers:
            attn = mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"], w["Wo"], w["bo"], self.H)
            h = (h + attn).layer_norm(w["ln1w"], w["ln1b"], self.eps)
            ff = h.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
            h = (h + ff).layer_norm(w["ln2w"], w["ln2b"], self.eps)
        return compile(h, int8=self.int8)

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        """Host-side token + position + type embedding lookup, then LayerNorm."""
        e = self.word[ids] + self.pos[np.arange(len(ids))] + self.typ[0]
        m = e.mean(-1, keepdims=True)
        v = ((e - m) ** 2).mean(-1, keepdims=True)
        return ((e - m) / np.sqrt(v + self.eps) * self.eln_w + self.eln_b).astype(np.float32)

    def __call__(self, texts, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            ids = np.asarray(self.tok(t)["input_ids"], dtype=np.int64)
            net = self._cache.get(len(ids)) or self._cache.setdefault(len(ids), self._build(len(ids)))
            states = net(self._embed(ids))               # [S, D] per-token states on the ANE
            if self.pooling == "cls":                    # first token (BERT [CLS])
                v = states[0]
            elif self.pooling == "max":
                v = states.max(0)
            else:
                v = states.mean(0)                       # mean over all (unpadded) tokens
            if normalize:
                # final L2-normalize runs on the ANE (fused reduce_l2_norm + real_div)
                v = _l2_normalizer(self.D)(v.reshape(1, self.D))[0]
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float32)


def group_norm_train(x, gamma, beta, groups: int, eps: float = 1e-5):
    """GroupNorm built from primitives so it works at ANY batch N (the stock
    Tensor.group_norm op is batch-1 only) and so the affine `gamma`/`beta` are real
    trainable parameters. `x` is [N, C, H, W]; `gamma`/`beta` are `[1, C, 1, 1]`
    parameter Tensors. Normalizes per-(group, sample) over the C/groups*H*W elements,
    then applies the affine. Every op here (reshape, mean, square, rsqrt, adds, mul,
    add) has a VJP, so input/gamma/beta gradients all run on the ANE. Mirrors the
    group_norm VJP math (aneforge/autograd.py:425)."""
    N, C, H, W = x.shape
    if C % groups:
        raise ValueError(f"group_norm_train: channels {C} not divisible by groups {groups}")
    M = (C // groups) * H * W
    xg = x.reshape(N, groups, M)
    xc = xg - xg.mean((2,))                       # per-(sample,group) center
    var = xc.square().mean((2,))                  # per-(sample,group) variance
    xn = (xc * var.adds(float(eps)).rsqrt()).reshape(N, C, H, W)
    return xn * gamma + beta


def conv_block(x, conv_w, gamma, beta, groups: int, pool: int = 0):
    """conv2d(pad=1) -> GroupNorm(train) -> ReLU -> optional max_pool(pool).
    `conv_w` is a conv_param; `gamma`/`beta` are [1,Cout,1,1] params; `pool=0`
    means no pooling. Returns the block output Tensor."""
    h = conv2d(x, conv_w, pad=1)
    h = group_norm_train(h, gamma, beta, groups).relu()
    return h.max_pool(pool) if pool else h


def _he(rng, shape):
    """He/Kaiming-normal init. fan_in is layout-dependent: a conv weight
    [Cout, Cin, kH, kW] has fan_in = Cin*kH*kW (prod of the trailing dims), while a
    2-D fc weight [in, out] has fan_in = in (the leading dim)."""
    fan_in = shape[0] if len(shape) == 2 else int(np.prod(shape[1:]))
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def cifar_cnn(batch: int, widths=(32, 64, 128), groups: int = 8, classes: int = 10, seed: int = 0):
    """Build the CIFAR-10 CNN graph. Returns (x_input, logits, params) where params is
    the trainable list in a fixed order. Architecture (per the design spec):
      block1 conv 3->w0  GN ReLU maxpool2   (32x32 -> 16x16)
      block2 conv w0->w1 GN ReLU maxpool2   (16x16 ->  8x8)
      block3 conv w1->w2 GN ReLU            ( 8x8)
      global-avg-pool over H,W -> fc(w2->classes)
    """
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
