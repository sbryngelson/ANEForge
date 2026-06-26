# LLM prefill on the ANE

Prefill — processing the prompt — is the **compute-bound** phase of LLM inference: a stack of
large matmuls over all prompt tokens at once. That is exactly what the Apple Neural Engine is
good at, and where it is most **energy-efficient**. ANEForge runs the whole prefill as **one
fused ANE program** for a Llama/Qwen-class decoder, loaded from real Hugging Face weights.

(Decode — one token at a time — is memory-bound and dispatch-floor-limited; that is the ANE's
weak phase. The energy win is in prefill.)

## API

```python
import aneforge as af

model = af.load_llm("Qwen/Qwen3-0.6B")             # any Llama/Qwen-class HF model
text_ids = model.generate(prompt_ids, max_new_tokens=24)   # generate text on the ANE
logits = model.prefill(prompt_ids)                 # or just prefill -> next-token logits [1, vocab]
```

- `af.load_llm(name, compress=None)` — load a Hugging Face Llama/Qwen model and prepare it for ANE inference.
  `compress="int8"` (or `"int4"`) quantizes the ANE weights — see **Quantized weights** below.
- `model.generate(ids, max_new_tokens, eos_id)` — greedy autoregressive generation, on the ANE.
- `model.prefill(ids)` — prefill only; returns next-token logits.
- `af.LlamaPrefill(cfg, weights)` / `af.LlamaConfig` — build one directly from numpy weights.
  `af.rope` / `af.prefill_block` are the building blocks.

## Prefill and decode

An LLM writes one token at a time. **Prefill** reads the whole prompt in one parallel pass
(compute-bound — the ANE's fast, energy-efficient phase). **Decode** then emits each following
token one at a time. `generate()` does both: prefill the prompt, then a greedy decode loop with
a **resident on-device KV-cache** — each layer's K/V stays on the ANE across steps (via
`share_buffer`), so a decode step feeds only the new token's embedding + a position one-hot and
runs one fused program over all layers. The decode program is compiled once and reused.

On **Qwen3-0.6B** this decodes at **~75 tok/s** (after a one-time ~6 s decoder compile),
prefills at **~8,600 prompt-tok/s**, and produces correct text:
*"The capital of France is"* → *"Paris. The capital of Italy is Rome. The capital of Spain is
Madrid. The capital of Portugal is Lisbon. ..."*

[`examples/llm_chat.py`](https://github.com/sbryngelson/ANEForge/blob/main/examples/llm_chat.py)
is an interactive chat that streams a reply token-by-token on the ANE; a runnable benchmark
(prompt-tokens/sec, plus `--energy` for ANE joules/token via `powermetrics`) is
[`examples/llm_prefill.py`](https://github.com/sbryngelson/ANEForge/blob/main/examples/llm_prefill.py).

## Quantized weights

`af.load_llm(name, compress="int8")` quantizes the on-ANE matmul weights to **per-channel int8**
(or `"int4"` for 4-bit LUT palettization), dequantized on the ANE. This halves (int8) or quarters
(int4) the weight bytes, which does two things:

- **Memory** — the dominant, certain win. fp16 weights are 2 bytes/param; a model that does not
  fit in fp16 can fit in int8. This is what makes larger models runnable.
- **Decode speed** — *scales with model size*. Decode reads every weight per token; on a small
  model (e.g. Qwen3-0.6B) decode is latency/under-utilization-bound, so int8 is roughly neutral
  (~1.1×) and text is byte-identical. As weights grow, decode becomes weight-bandwidth-bound and
  int8 approaches ~2× — that is where it pays off.

int8 is faithful: on Qwen3-0.6B, greedy `generate` produces text **identical** to fp16. Try it
with `examples/llm_chat.py <model> --int8`.

## What's inside

A pre-norm Llama/Qwen decoder block, all on the ANE:
`RMSNorm → QKV → RoPE → grouped-query causal attention → SwiGLU MLP`, with residuals.

- **RoPE** (rotary position embedding) — `x·cos + rotate_half(x)·sin`, built from `sin`/`cos`/
  `slice`/`concat` (no new hardware op).
- **Grouped-query attention** — KV heads are repeated to the query-head count via a 0/1
  expansion matmul (any group size).
- **Causal attention** — the decomposed `softmax(Q@Kᵀ·scale + causal_mask)@V`, kept in one
  fused program. (The native fused-attention layer is *avoided* for prefill: its per-layer
  graph cut dominates the cost, where the fused big-matmul path is ~170× faster.)

## Correctness

The prefill output matches Hugging Face's own forward pass — next-token logits at **cosine
~1.0** and an identical argmax — validated on-device (`tests/test_llm.py`). On a real
**Qwen3-0.6B** (28 layers), *"The capital of France is"* → **" Paris"**, the same token
HuggingFace predicts, at **~8,600 prompt-tokens/sec** prefill.

Qwen3 specifics are handled: a separate `head_dim` (≠ `dim/n_heads`), **QK-norm** (per-head
RMSNorm on Q/K before RoPE), and a large vocab — the transformer layers run on the ANE and
the lm_head vocab projection (a single matvec, and the vocab can exceed the ANE's per-op
dimension limit) is done on host.

## Scope

Llama/Qwen-class decoders (RMSNorm + RoPE + GQA + SwiGLU). Prefill (prompt processing) is the
target; KV-cache decode is a separate path. Static prompt length per compiled graph.
