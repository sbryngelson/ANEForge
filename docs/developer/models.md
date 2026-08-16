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

## CrossEncoder() — reranker

`aneforge.sentence_transformers.CrossEncoder` scores `(query, passage)` pairs, mirroring `sentence_transformers.CrossEncoder`:

```python
from aneforge.sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = ce.predict([(query, passage) for passage in passages])   # higher = more relevant
```

It reuses the same encoder graph as `load()`, then applies the sequence-classification head host-side. It supports BERT-family and RoBERTa/XLM-R (e.g. bge-reranker) `AutoModelForSequenceClassification` heads, detected by head shape; DistilBERT-style heads aren't wired up yet.

## load_resnet() — torchvision ImageNet classifier

`af.load_resnet()` loads a torchvision ResNet as a fused ANE classifier. Depths 18, 34, 50 and 101 are supported; `af.load_resnet18()` stays as the shorthand for depth 18.

```python
clf    = af.load_resnet(50)                  # 50, "50" and "resnet50" all work
logits = clf(image)                          # [1,3,224,224] -> [1,1000]
clf    = af.load_resnet(18, compress="int4") # 4-bit LUT weights
```

**BatchNorm is folded into the preceding conv at load**, so the ANE graph is pure conv/relu/pool/add/fc — conv is the ANE's strongest workload. `compress` picks the weight encoding (see `af.compile`); `build_dir` keeps the packed program on disk (its `weights.bin` is the packed-model size).

18 and 34 are BasicBlock (3x3 -> 3x3); 50 and 101 are Bottleneck (1x1 -> 3x3 -> 1x1, 4x expansion). Two details are worth knowing if you touch this code:

- **The stride sits on the Bottleneck's 3x3, not on its first 1x1.** That is torchvision's ResNet V1.5 variant. Moving it keeps every tensor shape intact and still agrees on top-1, so shape checks will not catch the mistake.
- **A block's shortcut is projected or not according to its weights**, never according to its stage index. Bottleneck projects stage 1 (64 -> 256 at stride 1) while BasicBlock does not.

## load_vit() — Hugging Face ViT image classifier

`af.load_vit()` loads any HF `ViTForImageClassification` (and compatible DeiT/BEiT-style models with a CLS token) as a fused ANE classifier:

```python
vit    = af.load_vit("google/vit-base-patch16-224")
logits = vit(image)             # [1,3,H,W] -> [1,num_labels]
top    = vit.classify(image)    # top-k (label, logit)
```

A strided `PxP` patch conv is walled on the ANE, so patch embedding runs as `space_to_depth(P)` followed by a 1x1 conv; the CLS token's row is picked out of the encoder output via a one-hot picker matmul (a bare row slice is also walled). The loader auto-detects two HF layer namings: the modern one (`vit.layers.{i}.attention.q_proj`, `mlp.fc1/fc2`) and the legacy one (`vit.encoder.layer.{i}.attention.attention.query`, `intermediate.dense`).

## load_gpt2() — GPT-2 text generation

`af.load_gpt2()` loads a GPT-2-family checkpoint — the pre-norm, pure-LayerNorm decoder — as a fused ANE program:

```python
gpt2 = af.load_gpt2("gpt2-medium")
ids  = gpt2.generate("The future of artificial intelligence is", max_new_tokens=16)  # greedy; returns token ids
logits = gpt2(token_ids)                     # 1-D ids -> [S, vocab], for custom decoding
```

GPT-2 is the family that the LayerNorm-at-`D>=1024` fix unlocks (Llama/Qwen use RMSNorm, so `load_llm` never hit that wall). Each sequence length compiles as **two fused programs**: the pre-norm transformer (`ln_1 -> native causal SDPA -> residual; ln_2 -> Linear -> gelu_new -> Linear -> residual`, then `ln_f`) and the **tied lm_head tiled along the 50257 vocab** — a single matmul that wide exceeds the ANE's per-op dimension cap on the A13-A15 families, so it is emitted as vocab-sized output-port tiles (e.g. `[16384, 16384, 16384, 1105]`) and stitched host-side. Token + positional embedding lookup runs on the host (`gather` is not an ANE op).

Two limits worth knowing: decode has **no KV cache yet** — it recomputes the forward on the growing sequence (cached per length), so it validates correctness rather than throughput; and native causal SDPA is reliable only for `S < 512`, so use short prompts and small `max_new_tokens`. `examples/gpt2.py` compiles the full 24-layer `gpt2-medium` as one program and checks the greedy decode is token-identical to Hugging Face.

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
