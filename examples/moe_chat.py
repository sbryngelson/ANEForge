"""Interactive chat with a Qwen Mixture-of-Experts model decoding ENTIRELY on the ANE (no host FFN). Loads a
GGUF (Qwen3-MoE e.g. 30B-A3B, or Qwen2-MoE e.g. Qwen1.5-MoE-A2.7B) at int8 and runs the full model as a
segmented decode (one MoE layer per program); the first run compiles the chunks (cached under ~/Models).

Run: python3 examples/moe_chat.py [gguf-path] [tokenizer-path]
  e.g. python3 examples/moe_chat.py ~/Models/Qwen1.5-MoE-A2.7B-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf
No gguf-path picks the smallest MoE GGUF under ~/Models; the tokenizer is read from the GGUF (matching chat
template), overridable with a 2nd arg."""
import glob
import os
import sys
import time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
from aneforge.moe import load_gguf

MAX_LEN = 512


def _default_gguf():
    hits = glob.glob(os.path.expanduser("~/Models/*MoE*-GGUF/*.gguf")) + glob.glob(os.path.expanduser("~/Models/*-A*B-GGUF/*.gguf"))
    hits = [h for h in hits if "incomplete" not in h]
    return min(hits, key=os.path.getsize) if hits else None      # smallest = most likely to fit + compile fast


def main():
    gguf = sys.argv[1] if len(sys.argv) > 1 else _default_gguf()
    if not gguf:
        print("usage: python3 examples/moe_chat.py <gguf-path> [tokenizer-path]  (no MoE GGUF found under ~/Models)"); return 1
    print(f"loading {os.path.basename(gguf)} (int8, ANE) ...")
    m = load_gguf(gguf, compress="int8")
    m._chunk_bytes = 1.0e9                          # one MoE layer per program (two MoE blocks/program won't compile)
    from transformers import AutoTokenizer
    if len(sys.argv) > 2:                            # override tokenizer (HF dir)
        tok = AutoTokenizer.from_pretrained(sys.argv[2])
    else:                                           # read the model's own tokenizer from the GGUF (matching chat template)
        tok = AutoTokenizer.from_pretrained(os.path.dirname(gguf), gguf_file=os.path.basename(gguf))
    e = m.cfg.extra
    print(f"{m.cfg.n_layers}L dim {m.cfg.dim}, {e['n_experts']} experts / top-{e['n_experts_per_tok']}. "
          f"compiling {len(m._layer_chunks())} chunks (one-time; cached under ~/Models) ...", end="", flush=True)
    m.warmup(MAX_LEN)
    imend = tok.convert_tokens_to_ids("<|im_end|>")           # ChatML turn-end (cleaner stop than eos_token_id)
    eos = imend if isinstance(imend, int) and imend >= 0 else tok.eos_token_id
    print(" done.")
    print("type a message (Ctrl-D or 'exit' to quit). full MoE model, decoding on the ANE.\n")

    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print(); break
        if not prompt: continue
        if prompt in ("exit", "quit"): break
        ids = _common.encode_chat(tok, prompt)
        if len(ids) >= MAX_LEN - 32:
            print("ane> (prompt too long for the demo context)\n"); continue
        print("ane> ", end="", flush=True)
        count = 0
        def on_tok(t):
            nonlocal count; count += 1
            print(tok.decode([t]), end="", flush=True)
        t0 = time.perf_counter()
        m.generate(ids, max_new_tokens=MAX_LEN - len(ids) - 1, max_len=MAX_LEN, eos_id=eos, on_token=on_tok)
        dt = time.perf_counter() - t0
        print(f"\n   [{count} tokens, {count / dt:.1f} tok/s on the ANE]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
