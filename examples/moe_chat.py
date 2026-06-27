"""Interactive chat with a Qwen Mixture-of-Experts model decoding ENTIRELY on the Apple Neural Engine -- no
host/CPU FFN. Loads a GGUF (Qwen3-MoE e.g. 30B-A3B, or Qwen2-MoE e.g. Qwen1.5-MoE-A2.7B), dequantizes each
tensor to int8, and runs the full model as a segmented ANE decode (one MoE layer per program). The first run
compiles the chunks (cached under ~/Models/.aneforge-cache), so later runs start fast.

Run: python3 examples/moe_chat.py [gguf-path] [tokenizer-path]
  e.g. python3 examples/moe_chat.py ~/Models/Qwen1.5-MoE-A2.7B-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf
With no gguf-path it picks the smallest *MoE* GGUF under ~/Models. The tokenizer is read from the GGUF itself
(so the chat template + special tokens match the model); pass a 2nd arg to override with an HF tokenizer dir."""
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
    ids = _encode(tok, prompt)
    if len(ids) >= MAX_LEN - 32:
      print("ane> (prompt too long for the demo context)\n"); continue
    print("ane> ", end="", flush=True)
    n = [0]; t0 = time.perf_counter()
    m.generate(ids, max_new_tokens=MAX_LEN - len(ids) - 1, max_len=MAX_LEN, eos_id=eos,
               on_token=lambda t: (n.__setitem__(0, n[0] + 1), print(tok.decode([t]), end="", flush=True)))
    dt = time.perf_counter() - t0
    print(f"\n   [{n[0]} tokens, {n[0] / dt:.1f} tok/s on the ANE]\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
