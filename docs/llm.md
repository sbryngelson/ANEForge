# LLMs on the ANE

ANEForge runs decoder LLMs on the Apple Neural Engine - prefill and KV-cache decode, from Hugging
Face weights or GGUF, as fused ANE programs. Beyond dense Llama/Qwen, it runs sparse Mixture-of-Experts
and hybrid DeltaNet+attention models, with quantization and exact speculative decoding (sections below).

Prefill (processing the prompt) is compute-bound: a stack of matmuls over all prompt tokens at
once, which is the ANE's efficient regime. Decode (one token at a time) is memory-bound; it
works, but the energy advantage is in prefill.

## API

```python
import aneforge as af

model = af.load_llm("Qwen/Qwen3-0.6B")            # any Llama/Qwen-class HF model
text_ids = model.generate(prompt_ids, max_new_tokens=24)
logits = model.prefill(prompt_ids)                # next-token logits [1, vocab]
```

- `af.load_llm(name, compress=None)` - load an HF Llama/Qwen model for ANE inference.
  `compress="int8"`/`"int4"` quantizes the weights (see below).
- `model.generate(ids, max_new_tokens, eos_id)` - greedy generation on the ANE.
- `model.prefill(ids)` - prefill only; returns next-token logits.
- `af.LlamaPrefill(cfg, weights)` / `af.LlamaConfig` - build from numpy weights directly.
  `af.rope` / `af.prefill_block` are the building blocks.

### lm_head on the ANE

`ane_lm_head=True` (default `False`) runs the lm_head on the ANE as a `compile_multi` tiled along
vocab, instead of the host fp32 matmul. Pass it to `af.load_llm` or `af.LlamaPrefill`, or per call
as `generate(..., ane_lm_head=True)`. Measured on an M2 Pro (macOS 26.5.2, fp16), median of 3 runs:

| model | lm_head host | lm_head ANE | decode host | decode ANE |
|---|---|---|---|---|
| gpt2-medium (vocab 50257, 4 tiles) | 10.8 ms/tok | 3.4 ms/tok | 39.6 tok/s | 59.4 tok/s |
| Qwen3-0.6B (vocab 151936, 10 tiles) | 30.1 ms/tok | 8.2 ms/tok | 17.8 tok/s | 29.3 tok/s |

It also drops the `_lmT` host allocation (GPT-2: 206 MB, Qwen3-0.6B: 622 MB). The trade-off is fp16
matmul numerics instead of fp32, so greedy argmax can differ on close ties (it did not on either
model above, 64/64 tokens); and when the head is tied to `embed`, the ANE program bakes a second
copy of those weights. The default stays `False` for that reason. `warmup(max_len)` compiles the
head too when the flag is set on the model, keeping the compile out of the first token.

## Prefill and decode

`generate()` prefills the prompt, then runs a greedy decode loop with a resident KV-cache: each
layer's K/V stays on the ANE across steps via `share_buffer`, so a decode step feeds only the new
token's embedding and a position one-hot. The decode program compiles once and is reused.

A single ANE program holds ~2 GB of baked weights, so for larger models the layers are split into
chunks under that ceiling; each chunk keeps its KV resident and the hidden state chains chunk ->
chunk (a `[1, dim]` round-trip). This is automatic - Qwen3-0.6B is one chunk; Qwen3-8B is nine.

On Qwen3-0.6B: ~8,600 prompt-tok/s prefill, ~75 tok/s decode. On Qwen3-8B (36 layers, 16 GB fp16,
9 chunks): ~7.5 tok/s decode. Both produce correct text - e.g. *"The capital of France is"* ->
*"Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ..."*

`examples/llm_chat.py` is an interactive streaming chat; `examples/llm_prefill.py` is a benchmark
(prompt-tok/s, and `--energy` for ANE joules/token via `powermetrics`).

## Speculative decoding

A small *draft* model proposes K tokens; the large *target* verifies all K in a single forward. This
is a natural fit for the ANE: decode is latency-bound, so `verify(K) ~ verify(1)` - measured 1.05x for
K=5 on Qwen3-8B - and the draft's proposals are checked almost for free. The output is **exact**: the
same tokens plain greedy decode would produce.

```python
from aneforge.speculative import spec_generate
out = spec_generate(target, draft, prompt_ids, max_new_tokens=40)   # target, draft = af.load_llm(...)
```

Measured 2.28x on Qwen3-8B with a Qwen3-0.6B draft (7.4 -> 16.8 tok/s), using a tiled on-ANE lm_head.
Profiling one round: ~64% is the target verify (a single 8B forward - the irreducible floor), ~36% the
draft, and ~0 orchestration overhead. `examples/spec_chat.py` is an interactive speculative chat.

## Quantized weights

`compress="int8"` quantizes the ANE matmul weights to per-channel int8 (`"int4"` = 4-bit LUT),
dequantized on the ANE - halving (int8) or quartering (int4) the weight bytes. Measured honestly,
the tradeoffs are narrow:

- Speed: roughly neutral. Decode is latency-bound (the ANE's array is idle at one token/step), not
  weight-bandwidth-bound, so fewer weight bytes barely move tok/s - Qwen3-8B was 7.9 (int8) vs 7.5
  (fp16) tok/s, with *fewer* chunks.
- Accuracy: faithful on small models (Qwen3-0.6B int8 is byte-identical to fp16) but per-channel
  int8 degrades deep models - Qwen3-8B int8 collapses into repetition. Run large models in fp16.
- Memory: the one real win (half the weight bytes). But fp16 + segmented decode already fits models
  up to RAM, so int8 only matters for weights that exceed RAM in fp16 - and the accuracy has to be
  solved first. The robust path is quantized *storage* dequantized to fp16 *compute* (not int8
  compute), which this option does not yet do.

Pass `--int8` to the examples to try it (best on small models).

## Mixture-of-Experts

Sparse MoE decoders run from a GGUF - Qwen3-MoE (e.g. 30B-A3B) and Qwen2-MoE (e.g. Qwen1.5-MoE-A2.7B):

```python
from aneforge.moe import load_gguf
model = load_gguf("Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf", compress="int8")
```

The router and *every* expert run on the ANE: a softmax over the experts, a top-k mask, then one
batched matmul over all experts weighted by the top-k (non-selected experts contribute 0). The top-k
mask is built from `amax` + `select` rather than the native `topk` op, which is a graph cut that won't
compose across layers. Qwen2-MoE's shared expert and QKV biases are detected from the weights. The
math matches HF `Qwen3MoeForCausalLM` (cosine >0.99), and the full 24-layer Qwen1.5-MoE-A2.7B decodes
coherently end-to-end on the ANE at int8 (~2 tok/s). `examples/moe_chat.py` is an interactive MoE chat.

Unlike the dense 8B, MoE decode at 30B scale is **weight-bandwidth-bound**: ~13.5 ms per MoE layer
(~89 GB/s), because each token reads all of the experts' weights. So here int8 *does* help (~1.3x), and
the experts' sparsity (4-8 of 60-128 active) is the real lever - but exploiting it needs a host-side
FFN split (measured ~4.6x in a prototype), not the dense on-ANE path. The 30B-A3B is currently gated
only by compile-time RAM on a 52 GB machine (it fits and runs on more).

## Hybrid models (Qwen3.5)

Qwen3.5 / Qwen3-Next interleave gated **DeltaNet** (linear-attention) layers with gated full-attention
layers. Both mixers run on the ANE: DeltaNet carries a resident causal-conv state and a recurrent
`[heads, dk, dv]` state across decode steps (a decay-first recurrence); attention carries the usual
resident KV cache. A per-layer plan (`LayerSpec`) names the mixer for each layer, so the runner stays
architecture-agnostic - it dispatches on the plan, not the weight shapes.

The real **Qwen3.5-27B** (64 layers, 48 DeltaNet + 16 attention) decodes coherently end-to-end on a pure
ANE from its GGUF at int8 - no host fallback:

```python
from aneforge.qwen35 import load_gguf
m = load_gguf("~/Models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf", compress="int8")
m.warmup(512)                       # compiles the segmented decode (cached under ~/Models)
print(m.generate(prompt_ids, max_new_tokens=64))
```

`examples/qwen35_chat.py` is an interactive chat over this loader. Two details are specific to the GGUF:
the DeltaNet GQA key/query heads tile onto the value heads (`ggml_repeat`, not `repeat_interleave` - they
disagree, and the GGUF is baked for tiling), and the residual stream is scaled by a constant to keep fp16
from overflowing on this model's large deep-layer activations (exact, since every read goes through a
scale-invariant RMSNorm).

Fidelity: an fp32 reference forward matches llama.cpp's logits exactly (correlation 1.0 at every position),
but the on-device int8-weight/fp16-compute decode is coherent yet lower-fidelity than llama.cpp's
fp32-accumulated Q4 - harder prompts come out shallower. The cost is the fp16 compute, not the weights:
per-channel int8 weights alone move the logit correlation by ~0.0001, but this model's large deep-layer
activation outliers leave it fp16-bound at this depth. `compress="blockwise"` adds per-block scales but
helps little and compiles much slower.

## What's inside

A pre-norm Llama/Qwen decoder block on the ANE:
`RMSNorm -> QKV -> RoPE -> grouped-query causal attention -> SwiGLU MLP`, with residuals.

- RoPE: `x*cos + rotate_half(x)*sin`, from sin/cos/slice/concat (no new hardware op).
- GQA: KV heads repeated to the query-head count via a 0/1 expansion matmul.
- Causal attention: decomposed `softmax(Q@K^T*scale + mask)@V` in one fused program. The native
  fused-attention op is avoided for prefill - its per-layer graph cut dominates, and the
  big-matmul path is ~170x faster.

## Correctness

Prefill matches Hugging Face's forward pass - next-token logits at cosine ~1.0 and identical
argmax - validated on-device (`tests/test_llm.py`). On Qwen3-0.6B (28 layers), *"The capital of
France is"* -> *" Paris"*, the token HF predicts.

Qwen3 specifics handled: a separate `head_dim` (!= `dim/n_heads`), QK-norm (per-head RMSNorm on Q/K
before RoPE), and a large vocab - the layers run on the ANE; the lm_head projection runs on host
(the vocab can exceed the ANE's per-op dimension limit).

## Scope

Decoder LLMs that run today: dense Llama/Qwen (RMSNorm + RoPE + GQA + SwiGLU), **Mistral**
(GQA + sliding window), **Gemma** (scaled embeddings + GeGLU + `(1+w)` RMSNorm), sparse **Mixture-of-Experts**
(Qwen3-MoE, Qwen2-MoE), and **hybrid** DeltaNet+attention (Qwen3.5). Runtime features: prefill + resident
KV-cache decode, automatic segmentation past the ~2 GB program ceiling, int8/int4 weights, and exact
speculative decoding. Static prompt length per compiled graph; the lm_head projection runs on host.
