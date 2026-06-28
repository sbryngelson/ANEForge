"""Interactive chat with speculative decoding on the Apple Neural Engine. A small draft model proposes tokens
and the larger target verifies K at once -- on the ANE `verify(K) ~= verify(1)`, so this is a ~2x EXACT
speedup (the streamed reply is identical to plain greedy decode, just faster).

Run: python3 examples/spec_chat.py <target-model> [draft-model]
  e.g. python3 examples/spec_chat.py ~/Models/Qwen3-8B
The draft defaults to a cached Qwen3-0.6B; target and draft must share the tokenizer/vocab (both Qwen3)."""
import glob
import os
import sys
import time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import aneforge as af
from aneforge.speculative import spec_generate

MAX_LEN = 512        # fixed context so the verifier compiles once and every turn reuses it


def _cached_06b():
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3-0.6B*/snapshots/*"))
    return hits[0] if hits else None


def main():
    if len(sys.argv) < 2:
        print("usage: python3 examples/spec_chat.py <target-model> [draft-model]"); return 1
    target_path = sys.argv[1]
    draft_path = sys.argv[2] if len(sys.argv) > 2 else _cached_06b()
    if not draft_path:
        print("no draft model found -- pass one as the 2nd arg, or cache Qwen/Qwen3-0.6B"); return 1
    print(f"loading target {os.path.basename(target_path)} + draft {os.path.basename(str(draft_path))} ...")
    target = af.load_llm(target_path); draft = af.load_llm(draft_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target_path)
    print(f"target {target.cfg.n_layers}L dim {target.cfg.dim}; draft {draft.cfg.n_layers}L. Speculative decode on the ANE.")
    print("compiling the verifier (one-time; minutes for a big target) ...", end="", flush=True)
    spec_generate(target, draft, _common.encode_chat(tok, "hi"), max_new_tokens=1, max_len=MAX_LEN)   # build + cache the verifier
    print(" done.")
    print("type a message (Ctrl-D or 'exit' to quit). replies stream; speculative is exact (== greedy decode).\n")

    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print(); break
        if not prompt: continue
        if prompt in ("exit", "quit"): break
        ids = _common.encode_chat(tok, prompt)
        if len(ids) >= MAX_LEN - 16:
            print("ane> (prompt too long for the demo context)\n"); continue
        print("ane> ", end="", flush=True)
        count = 0
        def on_tok(t):
            nonlocal count; count += 1
            print(tok.decode([t]), end="", flush=True)
        t0 = time.perf_counter()
        spec_generate(target, draft, ids, max_new_tokens=MAX_LEN - len(ids) - 1, max_len=MAX_LEN,
                      eos_id=tok.eos_token_id, n_draft=4, on_token=on_tok)
        dt = time.perf_counter() - t0
        print(f"\n   [{count} tokens, {count / dt:.1f} tok/s -- speculative, identical to greedy]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
