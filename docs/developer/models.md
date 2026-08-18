# Pretrained models

ANEForge ships pretrained-model loaders that build an ANEForge graph from real weights and compile it to a fused ANE program, plus the trainable-graph builders used with the [on-ANE autograd](autograd.md). Heavy dependencies (transformers, torchvision) are imported lazily so the core stays light. This page covers the host↔ANE weight layout decisions and the sentence-transformers drop-in.

## The host / ANE split

Not every operation belongs on the Neural Engine. The loaders split work deliberately:

- **Host:** tokenisation and the embedding lookup (`gather` is not an ANE op), plus pooling/normalise in some configurations.
- **ANE:** the transformer layers, compiled as fused programs (a batch is padded to one length and shares a single program). Conv-heavy classifiers run entirely on the ANE.

## load() — BERT-family sentence encoders

`af.load("sentence-transformers/all-MiniLM-L6-v2")` returns a callable embedder:

```python
embed = af.load("sentence-transformers/all-MiniLM-L6-v2")
vecs  = embed(["hello world", "the cat sat"])   # [2, D], L2-normalised
```

The transformer layers run on the ANE: a batch is padded to its longest sequence and compiled as one fused program (padded keys are masked out); tokenisation + embedding lookup run on the host.

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

## load_clip() — CLIP zero-shot image/text classification

`af.load_clip()` loads any Hugging Face CLIP dual-encoder checkpoint (`CLIPModel`) to run both the Vision Transformer and the causal Text Transformer on the ANE:

```python
clip = af.load_clip("openai/clip-vit-base-patch32")
img_feat = clip.encode_image(image)             # [1, 3, 224, 224] -> [1, 512] L2-normalised
txt_feat = clip.encode_text(["a photo of a cat", "a photo of a dog"])  # [2, 512] L2-normalised
ranked   = clip.classify(image, ["a photo of a cat", "a photo of a dog"]) # [(label, prob), ...]
```

Both towers compile as fused ANE programs:
- **Vision:** Patch embedding via `space_to_depth(P)` + 1x1 conv, CLS token + positional embedding, pre-norm Transformer stack with `QuickGELU` (`x * sigmoid(1.702 * x)`), CLS pooling, visual projection, and in-graph L2 normalisation.
- **Text:** Causal self-attention with precomputed triangular mask, QuickGELU MLP, EOT token extraction, text projection, and in-graph L2 normalisation.

`examples/clip_zero_shot.py` demonstrates zero-shot image classification end-to-end on the ANE with ranking and probability comparison against Hugging Face PyTorch.

## load_whisper() — speech to text

`af.load_whisper()` loads a Hugging Face Whisper checkpoint (default `openai/whisper-base.en`) and runs both towers — the audio encoder and the autoregressive text decoder — on the ANE:

```python
w = af.load_whisper("openai/whisper-base.en")
text = w.transcribe(audio)         # 16 kHz mono float32 waveform -> greedy English transcript
feats = w.encode(audio)            # audio features [1500, 512] (the encoder alone)
```

- **Encoder** (one fused program, run once per clip): the two Whisper conv layers (the strided `conv2` runs directly on the ANE), sinusoidal positional embedding, six pre-norm blocks, final layer norm -> audio features `[1500, 512]`.
- **Decoder** (one fused single-token program with a resident KV cache): token + learned positional embedding (host gather), six pre-norm blocks of causal self-attention against a resident `[H, M, dh]` cache (the one-hot positional write the LLM runner uses) + cross-attention to the audio features + a GELU MLP, then the tied `lm_head`. Each layer's cross-attention K/V over the audio is computed once per clip and held resident, so decode never re-projects the 1500 audio frames. Whisper's `k_proj` carries no bias.
- Host-side only: the log-mel spectrogram (Whisper's `WhisperFeatureExtractor`) and tokenization, the same split as tokenization for the LLM loaders.
- Greedy decoding reads the start prompt and the logit suppressions (`suppress_tokens`, `begin_suppress_tokens`) from the checkpoint's generation config, so `transcribe` reproduces `generate` rather than assuming English ids. The resident cache holds up to `max_target_positions` (448) tokens, Whisper's own decode ceiling.

Scope: greedy, no timestamps; the `.en` default is English. `examples/whisper.py` transcribes a sample clip and validates the encoder features (cosine) and the greedy transcript against Hugging Face.

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

- The transformer layers run on the ANE as one fused e5rt program (a batch is padded to one length and shares it).
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

## RAG on the ANE (examples/rag_chat.py)

`examples/rag_chat.py` is a chat-with-your-docs demo where every stage runs on the
Neural Engine: embedding, reranking, and generation. There is no host-side model, and no
GPU or CPU inference path in the loop.

Run it against a folder of `.md`/`.txt` files:

```
python3 examples/rag_chat.py [path]   # path defaults to this repo's docs/
```

It walks `path`, chunks each file (`examples._rag.chunk_text`), embeds every chunk with
`SentenceTransformer`, and holds the vectors in memory. Each query is embedded, matched
by cosine similarity against the corpus (`top_k`), reranked with a `CrossEncoder`, packed
into a prompt (`pack_context`), and answered by a resident-KV-cache Qwen3-0.6B decode
(`aneforge.load_llm`) that streams tokens as they are produced.

Model stack:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- `Qwen/Qwen3-0.6B` for generation

All three are fixed and small enough to keep the demo's compile and load times short.

`MAX_LEN` is a fixed 512-token context (`ANSWER_TOKENS = 160` reserved for the answer, so
the packed prompt budget is 352 tokens). The decode program is compiled once via
`llm.warmup(MAX_LEN)` before the first question, so every subsequent query streams
immediately instead of paying a per-query compile cost.

`--energy` adds a per-query joules line (needs `sudo` and `powermetrics`):

```
sudo python3 examples/rag_chat.py --energy
```

It samples package power with `powermetrics` for the duration of one `Pipeline.answer`
call and reports the query's energy as `~NNN mJ this query, 0 GPU` -- 0 GPU because the
whole pipeline, including generation, never leaves the Neural Engine. Without `sudo` (or
without `powermetrics` on the machine), the flag falls back to running the query with no
energy figure rather than failing.

The demo works best on prose documents, but the corpus can be any mix of lengths: the ANE
encoder pads each batch to one length and masks the padding, so a whole corpus -- long
files or many tiny ones -- embeds through a single compiled program.
