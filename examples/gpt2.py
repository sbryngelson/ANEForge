"""GPT-2 medium forward + resident KV-cache decode on the Apple Neural Engine (aneforge), from real
Hugging Face weights, validated vs an fp32 reference, with LayerNorm, learned positional embeddings,
and the tied lm_head tiled along vocab (#183, #203, #215).
Run: python3 examples/gpt2.py"""
import sys, time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

NAME = "gpt2-medium"
PROMPT = "The future of artificial intelligence is"
K = 16


def main():
    _common.head("GPT-2 medium on the Apple Neural Engine (aneforge)")
    print("config: gpt2-medium | 24 layers x 1024 dim x 16 heads | vocab 50257 | "
          "LayerNorm eps 1e-5 | gelu_new | tied lm_head | resident KV-cache decode")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    ids = np.asarray(tok.encode(PROMPT), dtype=np.int64)
    print(f"prompt: {PROMPT!r} -> {len(ids)} tokens | K={K} greedy tokens\n")

    print("loading model via unified LLM runner ...")
    model = af.load_llm(NAME)
    print(f"model loaded: {model.cfg.n_layers} layers x {model.cfg.dim} dim (norm_type={model.cfg.norm_type}, rope={model.cfg.rope})")

    # 1. Validate prefill logits against Hugging Face reference
    print("\n1. VALIDATION (ANE prefill vs HF fp32 reference):")
    with torch.no_grad():
        ref_logits = hf(torch.tensor(ids[None])).logits[0, -1].numpy()
    ane_logits = model.prefill(ids)[0]
    cos = float(ane_logits @ ref_logits / (np.linalg.norm(ane_logits) * np.linalg.norm(ref_logits) + 1e-9))
    top_ane = int(ane_logits.argmax())
    top_ref = int(ref_logits.argmax())
    print(f"  logit cosine (last row) = {cos:.5f}")
    print(f"  ANE top-1 = {top_ane} ({tok.decode([top_ane])!r}) | ref top-1 = {top_ref} ({tok.decode([top_ref])!r})")
    forward_ok = cos > 0.99 and top_ane == top_ref

    # 2. Resident KV-cache decode warmup + greedy decode
    print("\n2. RESIDENT KV-CACHE DECODE:")
    print("  warming up decode program for context 128 ...", end="", flush=True)
    t_warm0 = time.perf_counter()
    model.warmup(128)
    t_warm = time.perf_counter() - t_warm0
    print(f" done ({t_warm:.2f}s)")

    # Decode on ANE with resident KV cache
    t0 = time.perf_counter()
    ane_toks = model.generate(ids, max_new_tokens=K, max_len=128, batched_prefill=True)
    dt = time.perf_counter() - t0
    tok_s = K / dt if dt > 0 else 0

    # Reference HF greedy decode
    with torch.no_grad():
        ref_toks = list(hf.generate(torch.tensor(ids[None]), max_new_tokens=K, do_sample=False)[0].numpy()[len(ids):])

    print(f"\n  ANE greedy: {tok.decode(ane_toks)!r}")
    print(f"  Ref greedy: {tok.decode(ref_toks)!r}")
    print(f"  Speed:      {dt / K * 1e3:.1f} ms/token ({tok_s:.2f} tok/s, compile excluded)")

    match_count = sum(int(a) == int(b) for a, b in zip(ane_toks, ref_toks))
    greedy_ok = list(ane_toks) == list(ref_toks) and match_count == K
    print(f"  Token match: {match_count}/{K} ({'PERFECT' if greedy_ok else 'MISMATCH'})")

    # 3. Long context test (>512 tokens)
    print("\n3. LONG CONTEXT TEST (>512 tokens):")
    long_prompt = (list(ids) * 100)[:520]
    print(f"  testing batched prefill + resident decode with {len(long_prompt)} tokens prompt ...", end="", flush=True)
    t_long0 = time.perf_counter()
    long_out = model.generate(long_prompt, max_new_tokens=4, max_len=600, batched_prefill=True)
    t_long = time.perf_counter() - t_long0
    long_ok = len(long_out) == 4
    print(f" done ({t_long:.2f}s)")
    print(f"  generated {len(long_out)} tokens at context >512: {long_out}")

    # Final verdict
    print("\n" + "=" * 50)
    print(f"Prefill validates (cos>0.99):   {forward_ok}")
    print(f"Greedy matches HF ({K}/{K}):     {greedy_ok}")
    print(f"Long context (>512) works:     {long_ok}")
    ok = forward_ok and greedy_ok and long_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 50)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
