"""Interactive chat with a Qwen3.5 / Qwen3-Next hybrid model decoding ENTIRELY on the ANE. Loads a qwen35-arch
GGUF (e.g. Qwen3.5-27B) at int8 and runs the full hybrid stack -- interleaved gated-DeltaNet (linear-attention)
and gated full-attention layers -- as a segmented decode (a few layers per program); the first run compiles the
chunks (cached under ~/Models).

Run: python3 examples/qwen35_chat.py [gguf-path] [tokenizer-dir]
  e.g. python3 examples/qwen35_chat.py ~/Models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
No gguf-path picks the smallest qwen35 GGUF under ~/Models. The tokenizer is read from the sibling HF directory
(the GGUF dir minus '-GGUF'), since transformers can't yet build the tokenizer from a qwen35 GGUF; override with
a 2nd arg pointing at any directory holding tokenizer.json + chat_template."""
import glob
import os
import sys
import time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
from aneforge.qwen35 import load_gguf

MAX_LEN = 512


def _default_gguf():
  hits = glob.glob(os.path.expanduser("~/Models/*[Qq]wen3*-GGUF/*.gguf"))
  hits = [h for h in hits if "incomplete" not in h]
  return min(hits, key=os.path.getsize) if hits else None      # smallest = most likely to fit + compile fast


def _tokenizer_dir(gguf):
  """The HF tokenizer dir for a GGUF: the GGUF's parent with '-GGUF' stripped (Qwen3.5-27B-GGUF -> Qwen3.5-27B)."""
  d = os.path.dirname(gguf)
  cand = d[:-5] if d.endswith("-GGUF") else d
  return cand if os.path.exists(os.path.join(cand, "tokenizer.json")) else None


def _encode(tok, prompt):
  """Apply the chat template (no 'thinking' block); returns a list of token ids."""
  msgs = [{"role": "user", "content": prompt}]
  try:
    r = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
  except TypeError:
    r = tok.apply_chat_template(msgs, add_generation_prompt=True)
  if isinstance(r, dict) or hasattr(r, "input_ids"): r = r["input_ids"]
  if r and isinstance(r[0], (list, tuple)): r = r[0]
  return [int(t) for t in r]


def main():
  gguf = sys.argv[1] if len(sys.argv) > 1 else _default_gguf()
  if not gguf:
    print("usage: python3 examples/qwen35_chat.py <gguf-path> [tokenizer-dir]  (no qwen3* GGUF found under ~/Models)"); return 1
  tokdir = sys.argv[2] if len(sys.argv) > 2 else _tokenizer_dir(gguf)
  if not tokdir:
    print(f"no tokenizer found next to {gguf}; pass a tokenizer dir (with tokenizer.json) as the 2nd arg"); return 1
  print(f"loading {os.path.basename(gguf)} (int8, ANE) ...")
  m = load_gguf(gguf, compress="int8")
  m._chunk_bytes = 1.0e9                          # a couple of hybrid layers per program (under the ~2GB ANE ceiling)
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(tokdir)
  e = m.cfg.extra
  print(f"{m.cfg.n_layers}L dim {m.cfg.dim}, {e['nv']} DeltaNet value-heads, {m.cfg.n_heads}/{m.cfg.n_kv_heads} attn heads. "
        f"compiling {len(m._layer_chunks())} chunks (one-time; cached under ~/Models) ...", end="", flush=True)
  m.warmup(MAX_LEN)
  imend = tok.convert_tokens_to_ids("<|im_end|>")           # ChatML turn-end (cleaner stop than eos_token_id)
  eos = imend if isinstance(imend, int) and imend >= 0 else tok.eos_token_id
  samp = m.cfg.extra.get("sampling", {})                    # the model's own recommended sampling (from the GGUF);
  if not samp.get("temperature"): samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}  # greedy loops, so sample
  print(" done.")
  print("type a message (Ctrl-D or 'exit' to quit). full hybrid model, decoding on the ANE.\n")

  while True:
    try:
      prompt = input("you> ").strip()
    except EOFError:
      print(); break
    if not prompt: continue
    if prompt in ("exit", "quit"): break
    ids = _encode(tok, prompt)
    if len(ids) >= MAX_LEN - 32:
      print("ane> (prompt too long for the demo context)\n"); continue
    print("ane> ", end="", flush=True)
    n = [0]; t0 = time.perf_counter()
    m.generate(ids, max_new_tokens=MAX_LEN - len(ids) - 1, max_len=MAX_LEN, eos_id=eos,
               temperature=samp["temperature"], top_p=samp.get("top_p", 1.0), top_k=samp.get("top_k", 0),
               on_token=lambda t: (n.__setitem__(0, n[0] + 1), print(tok.decode([t]), end="", flush=True)))
    dt = time.perf_counter() - t0
    print(f"\n   [{n[0]} tokens, {n[0] / dt:.1f} tok/s on the ANE]\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
