# LLMs on the ANE

ANEForge runs a Llama/Qwen-class decoder on the Apple Neural Engine — prefill and KV-cache
decode, from Hugging Face weights, as fused ANE programs.

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

- `af.load_llm(name, compress=None)` — load an HF Llama/Qwen model for ANE inference.
  `compress="int8"`/`"int4"` quantizes the weights (see below).
- `model.generate(ids, max_new_tokens, eos_id)` — greedy generation on the ANE.
- `model.prefill(ids)` — prefill only; returns next-token logits.
- `af.LlamaPrefill(cfg, weights)` / `af.LlamaConfig` — build from numpy weights directly.
  `af.rope` / `af.prefill_block` are the building blocks.

## Prefill and decode

`generate()` prefills the prompt, then runs a greedy decode loop with a resident KV-cache: each
layer's K/V stays on the ANE across steps via `share_buffer`, so a decode step feeds only the new
token's embedding and a position one-hot. The decode program compiles once and is reused.

A single ANE program holds ~2 GB of baked weights, so for larger models the layers are split into
chunks under that ceiling; each chunk keeps its KV resident and the hidden state chains chunk →
chunk (a `[1, dim]` round-trip). This is automatic — Qwen3-0.6B is one chunk; Qwen3-8B is nine.

On Qwen3-0.6B: ~8,600 prompt-tok/s prefill, ~75 tok/s decode. On Qwen3-8B (36 layers, 16 GB fp16,
9 chunks): ~7.5 tok/s decode. Both produce correct text — e.g. *"The capital of France is"* →
*"Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ..."*

`examples/llm_chat.py` is an interactive streaming chat; `examples/llm_prefill.py` is a benchmark
(prompt-tok/s, and `--energy` for ANE joules/token via `powermetrics`).

## Quantized weights

`compress="int8"` quantizes the ANE matmul weights to per-channel int8 (`"int4"` = 4-bit LUT),
dequantized on the ANE — halving (int8) or quartering (int4) the weight bytes. Measured honestly,
the tradeoffs are narrow:

- Speed: roughly neutral. Decode is latency-bound (the ANE's array is idle at one token/step), not
  weight-bandwidth-bound, so fewer weight bytes barely move tok/s — Qwen3-8B was 7.9 (int8) vs 7.5
  (fp16) tok/s, with *fewer* chunks.
- Accuracy: faithful on small models (Qwen3-0.6B int8 is byte-identical to fp16) but per-channel
  int8 degrades deep models — Qwen3-8B int8 collapses into repetition. Run large models in fp16.
- Memory: the one real win (half the weight bytes). But fp16 + segmented decode already fits models
  up to RAM, so int8 only matters for weights that exceed RAM in fp16 — and the accuracy has to be
  solved first. The robust path is quantized *storage* dequantized to fp16 *compute* (not int8
  compute), which this option does not yet do.

Pass `--int8` to the examples to try it (best on small models).

## What's inside

A pre-norm Llama/Qwen decoder block on the ANE:
`RMSNorm → QKV → RoPE → grouped-query causal attention → SwiGLU MLP`, with residuals.

- RoPE: `x·cos + rotate_half(x)·sin`, from sin/cos/slice/concat (no new hardware op).
- GQA: KV heads repeated to the query-head count via a 0/1 expansion matmul.
- Causal attention: decomposed `softmax(Q@Kᵀ·scale + mask)@V` in one fused program. The native
  fused-attention op is avoided for prefill — its per-layer graph cut dominates, and the
  big-matmul path is ~170× faster.

## Correctness

Prefill matches Hugging Face's forward pass — next-token logits at cosine ~1.0 and identical
argmax — validated on-device (`tests/test_llm.py`). On Qwen3-0.6B (28 layers), *"The capital of
France is"* → *" Paris"*, the token HF predicts.

Qwen3 specifics handled: a separate `head_dim` (≠ `dim/n_heads`), QK-norm (per-head RMSNorm on Q/K
before RoPE), and a large vocab — the layers run on the ANE; the lm_head projection runs on host
(the vocab can exceed the ANE's per-op dimension limit).

## Scope

Llama/Qwen-class decoders (RMSNorm + RoPE + GQA + SwiGLU). Static prompt length per compiled graph.
