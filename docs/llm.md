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

model = af.load_llm("HuggingFaceTB/SmolLM-135M")   # any Llama/Qwen-class HF model
logits = model.prefill(token_ids)                  # prefill on the ANE -> next-token logits [1, vocab]
```

- `af.load_llm(name)` — load a Hugging Face Llama/Qwen model and prepare it for ANE prefill.
- `af.LlamaPrefill(cfg, weights)` / `af.LlamaConfig` — build one directly from numpy weights.
- `model.compile(seq)` builds the fused graph for a fixed prompt length; `model.prefill(ids)`
  runs it. `af.rope` / `af.prefill_block` are the building blocks.

A runnable benchmark (prompt-tokens/sec, plus `--energy` for ANE joules/token via
`powermetrics`) is [`examples/llm_prefill.py`](https://github.com/sbryngelson/ANEForge/blob/main/examples/llm_prefill.py).

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

The prefill output matches Hugging Face's own `LlamaForCausalLM` forward pass — next-token
logits at **cosine ~1.0** and an identical argmax — validated on-device (`tests/test_llm.py`).

## Scope

Llama/Qwen-class decoders (RMSNorm + RoPE + GQA + SwiGLU). Prefill (prompt processing) is the
target; KV-cache decode is a separate path. Static prompt length per compiled graph.
