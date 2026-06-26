"""Interactive chat with an LLM running on the Apple Neural Engine. Type a message and the model streams a
reply token-by-token, generated entirely on the ANE (resident KV-cache decode). Defaults to a cached
Qwen3-0.6B if present, else pass a Hugging Face model name. Run: python3 examples/llm_chat.py [hf-model]"""
import glob
import os
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import aneforge as af

MAX_LEN = 512        # fixed context so the decode program compiles once and every turn reuses it


def _default_model():
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3-0.6B*/snapshots/*"))
    return hits[0] if hits else None


def _encode(tok, prompt):
    """Apply the chat template (no 'thinking' block) so the model answers as an assistant; returns a list of ids."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        r = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:                                    # tokenizers without the enable_thinking kwarg
        r = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if isinstance(r, dict) or hasattr(r, "input_ids"): r = r["input_ids"]   # BatchEncoding -> ids
    if r and isinstance(r[0], (list, tuple)): r = r[0]                       # [[ids]] -> [ids]
    return [int(t) for t in r]


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    compress = "int8" if "--int8" in sys.argv else ("int4" if "--int4" in sys.argv else None)
    name = argv[0] if argv else _default_model()
    if not name:
        print("usage: python3 examples/llm_chat.py <hf-model-name> [--int8|--int4]   (no cached Qwen3-0.6B found)"); return 1
    print(f"loading {os.path.basename(str(name))}{' ('+compress+' weights)' if compress else ''} ...")
    model = af.load_llm(name, compress=compress)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    print("compiling the decoder for the ANE (one-time, ~6s) ...", end="", flush=True)
    model.warmup(MAX_LEN)                                 # compile the decode program now so every reply streams instantly
    print(" done.")
    print(f"ready: {model.cfg.n_layers}L dim {model.cfg.dim}, running on the Apple Neural Engine.")
    print("type a message (Ctrl-D or 'exit' to quit).\n")

    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print(); break
        if not prompt: continue
        if prompt in ("exit", "quit"): break
        ids = _encode(tok, prompt)
        if len(ids) >= MAX_LEN - 8:
            print("ane> (prompt too long for the demo context)\n"); continue
        print("ane> ", end="", flush=True)
        model.generate(ids, max_new_tokens=MAX_LEN - len(ids) - 1, max_len=MAX_LEN, eos_id=tok.eos_token_id,
                       on_token=lambda t: print(tok.decode([t]), end="", flush=True))
        print("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
