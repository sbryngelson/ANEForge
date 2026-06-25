# Pretrained models

ANEForge ships pretrained-model loaders that build an ANEForge graph from real weights and compile it to a fused ANE program, plus the trainable-graph builders used with the [on-ANE autograd](autograd.md). Heavy dependencies (transformers, torchvision) are imported lazily so the core stays light. This page covers the host↔ANE weight layout decisions and the sentence-transformers drop-in.

## The host / ANE split

Not every operation belongs on the Neural Engine. The loaders split work deliberately:

- **Host:** tokenisation and the embedding lookup (`gather` is not an ANE op), plus pooling/normalise in some configurations.
- **ANE:** the transformer layers, compiled as fused programs and cached per sequence length. Conv-heavy classifiers run entirely on the ANE.

## load() — BERT-family sentence encoders

`af.load("sentence-transformers/all-MiniLM-L6-v2")` returns a callable embedder:

```python
embed = af.load("sentence-transformers/all-MiniLM-L6-v2")
vecs  = embed(["hello world", "the cat sat"])   # [2, D], L2-normalised
```

The transformer layers run on the ANE as fused programs (cached per sequence length); tokenisation + embedding lookup run on the host.

`pooling` selects how per-token states reduce to one vector:

| Mode | Default for |
|---|---|
| `"mean"` (default) | MiniLM, E5 |
| `"cls"` (first token) | BGE, GTE |
| `"max"` | — |

A model's correct mode lives in its sentence-transformers config; `aneforge.sentence_transformers` reads it for you (see below).

## load_resnet18() — torchvision ImageNet classifier

`af.load_resnet18()` loads torchvision ResNet-18 as a fused ANE classifier:

```python
clf    = af.load_resnet18()
logits = clf(image)                          # [1,3,224,224] -> [1,1000]
clf    = af.load_resnet18(compress="int4")   # 4-bit LUT weights
```

**BatchNorm is folded into the preceding conv at load**, so the ANE graph is pure conv/relu/pool/add/fc — conv is the ANE's strongest workload. `compress` picks the weight encoding (see `af.compile`); `build_dir` keeps the packed program on disk (its `weights.bin` is the packed-model size).

## Weight layout: He init is layout-dependent

`_he` (He/Kaiming-normal init) computes `fan_in` from the weight layout, which differs between conv and fc weights:

| Weight | Layout | `fan_in` |
|---|---|---|
| conv | `[Cout, Cin, kH, kW]` | `Cin*kH*kW` (product of trailing dims) |
| fc | `[in, out]` | `in` (the leading dim) |

Getting this wrong silently mis-scales the initial weights, so the layout dependence is explicit.

## Trainable-graph builders

These build graphs whose parameters are real trainable leaves, every op carrying a VJP so input/affine gradients all run on the ANE.

- **`group_norm_train`** — GroupNorm built from primitives so it works at *any* batch N (the stock `Tensor.group_norm` op is batch-1 only) and so the affine `gamma`/`beta` are real trainable parameters. `x` is `[N,C,H,W]`; `gamma`/`beta` are `[1,C,1,1]` parameter Tensors. Normalizes per-(group, sample) over the `C/groups*H*W` elements, then applies the affine. Mirrors the `group_norm` VJP math.
- **`conv_block`** — conv → GroupNorm → ReLU → optional max-pool.
- **`cifar_cnn`** — the full CIFAR-10 CNN, returning `(x_input, logits, params)` where `params` is the trainable list in a fixed order:

  ```
  block1  conv 3->w0   GN ReLU maxpool2   (32x32 -> 16x16)
  block2  conv w0->w1  GN ReLU maxpool2   (16x16 ->  8x8)
  block3  conv w1->w2  GN ReLU            ( 8x8)
  global-avg-pool over H,W -> fc(w2 -> classes)
  ```

## sentence-transformers drop-in

`aneforge.sentence_transformers.SentenceTransformer` is a **drop-in** that runs the encoder on the Neural Engine:

```python
from aneforge.sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb   = model.encode(["a query", "a passage"])   # [2, D] on the ANE
```

Design notes:

- The transformer layers run on the ANE as one fused e5rt program (cached per sequence length).
- **Only numpy + aneforge are needed** — the `sentence-transformers` package is *not* imported. This mirrors its `.encode` surface; it does not wrap it.
- Pooling mode and L2-normalise are read from the model's own config (below), so a mean-pooled model (MiniLM, E5) and a cls-pooled model (BGE, GTE) both come out correct.
- The `device` argument is accepted for signature parity and ignored — the encoder always runs on the Neural Engine.
- `encode()`'s `batch_size` is accepted for parity but does not change the result (the ANE path is fused per sequence length and cached).

### Parity claims

| Mode | Cosine vs reference | Size |
|---|---|---|
| default (fp16) | **~1.0** | — |
| `int8=True` | **~0.9999** | half the weight size (int8 streamed) |

Embeddings match the reference encoder at a fraction of the GPU's energy.

### Config-driven pooling / normalize

`_read_st_config` returns `(pooling_mode, has_normalize)` from the model's sentence-transformers config, defaulting to `("mean", False)` for a raw model with no such config:

- **Pooling** comes from `1_Pooling/config.json` (`pooling_mode_cls_token` / `pooling_mode_max_tokens` / `pooling_mode_mean_tokens`).
- **Normalize** is true if `modules.json` lists a `Normalize` module. A model that ships a Normalize module is L2-normalised regardless of `normalize_embeddings`, matching sentence-transformers.
