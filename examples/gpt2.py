"""GPT-2 medium forward + greedy decode on the Apple Neural Engine (aneforge), from real Hugging Face
weights, validated vs an fp32 reference, with the tied lm_head tiled along vocab (#183).
Run: python3 examples/gpt2.py

Decode strategy: the greedy loop runs every step through ONE compiled program of length
S_MAX = len(prompt) + K, with the growing sequence padded to S_MAX at the END. Under causal
attention a trailing pad changes nothing on the real rows (positions > j are masked, and
LayerNorm is row-wise), so the last real row equals the per-length recompute exactly - and
the program compiles once instead of once per length."""
import sys, time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

NAME = "gpt2-medium"
PROMPT = "The future of artificial intelligence is"
K = 16
N_LAYERS = 24
DIM, HEADS = 1024, 16


def _block(hf, x, i):
    """HF block i on `x` (transformers 5.x returns a bare tensor; older versions a tuple)."""
    out = hf.transformer.h[i](x)
    return out[0] if isinstance(out, tuple) else out


def ref_logits(hf, ids, n_layers):
    """Last-row logits of the HF GPT-2 forward through the first `n_layers` blocks (fp32)."""
    import torch
    with torch.no_grad():
        x = hf.transformer.wte(torch.from_numpy(ids[None])) + hf.transformer.wpe(torch.arange(len(ids)))
        for i in range(n_layers):
            x = _block(hf, x, i)
        x = hf.transformer.ln_f(x)
        return hf.lm_head(x)[0, -1].numpy()


def compile_retry(g, S, tries=5, wait=5):
    """Compile the S-length program pair, retrying the flaky ANE compiler; re-raises other errors."""
    for a in range(tries):
        try:
            return g._compile(S)
        except RuntimeError as e:
            if "program_compile failed" not in str(e):
                raise
            print(f"  [compile] attempt {a + 1} failed, waiting {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"compile failed {tries} times")


def main():
    _common.head("GPT-2 medium on the Apple Neural Engine (aneforge)")
    print("config: gpt2-medium | 24 layers x 1024 dim x 16 heads | vocab 50257 | "
          "LayerNorm eps 1e-5 | gelu_new | tied lm_head")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    ids = np.asarray(tok.encode(PROMPT), dtype=np.int64)
    pad = int(tok.eos_token_id)
    s_max = len(ids) + K
    print(f"prompt: {PROMPT!r} -> {len(ids)} tokens | K={K} greedy tokens "
          f"| ONE program at S_MAX={s_max} (trailing-pad decode)")

    # try the full 24-layer model as ONE program; fall back to fewer layers (validated vs the
    # SAME truncated HF forward, exactly as examples/vit.py does)
    n_layers, g, why = N_LAYERS, None, ""
    for k in (N_LAYERS, 16, 12, 8, 4, 1):
        try:
            g = af.load_gpt2(NAME, max_layers=k)
            net, head = compile_retry(g, s_max)
            n_layers = k
            why = ("full 24-layer model compiled as one program" if k == N_LAYERS else
                   f"full 24-layer model did not compile as one program; using first {k} layers")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [compile] {k} layers failed ({type(e).__name__}: {str(e)[:90]}) -> trying fewer")
            g = None
    if g is None:
        print("FAIL: could not compile even 1 layer")
        return 1

    tile_sizes = [min(16384, g.wte.shape[0] - i) for i in range(0, g.wte.shape[0], 16384)]
    print(f"\nPATH: {why} (S_MAX={s_max})")
    print(f"forward: {net.n_ops} ops fused into 1 ANE program"
          + (f" (+{net.n_sdpa} native-SDPA sub-programs)" if getattr(net, "n_sdpa", 0) else ""))
    print(f"lm_head: tied to wte, tiled along vocab as {len(tile_sizes)} output ports {tile_sizes}")

    def padded(x):
        out = np.full(s_max, pad, np.int64)
        out[:len(x)] = x
        return out

    def last_logits(x):
        return np.asarray(g(padded(x))[len(x) - 1], np.float32)

    # sanity: a trailing-padded HF forward must equal the unpadded one on the last real row
    with torch.no_grad():
        x = hf.transformer.wte(torch.from_numpy(padded(ids)[None])) + hf.transformer.wpe(torch.arange(s_max))
        for i in range(n_layers):
            x = _block(hf, x, i)
        ref_pad = hf.lm_head(hf.transformer.ln_f(x))[0, len(ids) - 1].numpy()
    ref = ref_logits(hf, ids, n_layers)
    pad_ok = bool(np.allclose(ref_pad, ref, rtol=1e-4, atol=1e-3))
    print(f"pad-equivalence (HF padded row == HF unpadded row): {pad_ok}")
    if not pad_ok:
        print("FAIL: trailing-pad decode does not match the unpadded forward")
        return 1

    # validate the forward: last-row cosine vs HF fp32
    last = last_logits(ids)
    cos = float(last @ ref / (np.linalg.norm(last) * np.linalg.norm(ref)))
    top = int(last.argmax())
    print("\nVALIDATION (ANE forward vs HF fp32 reference, same layer count):")
    print(f"  logit cosine (last row) = {cos:.4f}")
    print(f"  ANE top-1 = {top} ({tok.decode([top])!r}) | ref top-1 = {int(ref.argmax())} "
          f"({tok.decode([int(ref.argmax())])!r})")
    forward_ok = cos > 0.99

    # greedy decode through the single S_MAX program (trailing-pad feed); compare token-id to token-id
    ane_steps, cur = [], ids
    t0 = time.perf_counter()
    for _ in range(K):
        lg = last_logits(cur)
        ane_steps.append(lg)
        cur = np.concatenate([cur, [int(lg.argmax())]])
    ane_toks = [int(t) for t in cur[len(ids):]]
    dt = time.perf_counter() - t0

    ref_steps, cur = [], ids
    for _ in range(K):
        lg = ref_logits(hf, cur, n_layers)
        ref_steps.append(lg)
        cur = np.concatenate([cur, [int(lg.argmax())]])
    ref_toks = [int(t) for t in cur[len(ids):]]

    print("\nGREEDY DECODE (K=%d tokens, one S_MAX program):" % K)
    match = 0
    for i, (ane_lg, ref_lg) in enumerate(zip(ane_steps, ref_steps)):
        order = np.argsort(ane_lg)[::-1][:2]
        m = int(ane_lg.argmax()) == int(ref_lg.argmax())
        match += m
        print(f"  tok {i + 1:2d}: ane={int(order[0])} ({tok.decode([int(order[0])])!r}) "
              f"ref={int(ref_lg.argmax())} ({tok.decode([int(ref_lg.argmax())])!r}) "
              f"match={m} gap={float(ane_lg[order[0]] - ane_lg[order[1]]):.4f}")
    print(f"  ane greedy: {tok.decode(ane_toks)}")
    print(f"  ref greedy: {tok.decode(ref_toks)}")
    print(f"  decode: {dt / K * 1e3:.1f} ms/token ({K / dt:.2f} tok/s, compiles excluded)")
    greedy_ok = ane_toks == ref_toks and match == K

    # verdict
    print(f"forward validates (cos>0.99): {forward_ok}")
    print(f"greedy matches HF ({K}/{K}):   {greedy_ok}")
    ok = forward_ok and greedy_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
