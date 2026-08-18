"""Chat with a folder of docs, entirely on the Apple Neural Engine: SentenceTransformer
embeddings, a CrossEncoder reranker, and Qwen3-0.6B generation all run on the engine.

  python3 examples/rag_chat.py [path]        # path defaults to this repo's docs/
  sudo python3 examples/rag_chat.py --energy # add per-query joules (needs powermetrics)
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import aneforge / examples._rag; works as a script and when imported under pytest

import numpy as np

EMBED = "sentence-transformers/all-MiniLM-L6-v2"
RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM = "Qwen/Qwen3-0.6B"
MAX_LEN = 512
ANSWER_TOKENS = 160
TOP_K, TOP_N = 20, 4
CHUNK_TOK = 192          # token-window chunk size; uniform windows keep the Encoder to a few compiled programs
REPO_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

from examples._rag import Chunk, chunk_text, pack_context, top_k   # noqa: E402


def _read_corpus(path: str, tok) -> list[Chunk]:
  chunks: list[Chunk] = []
  for root, _, files in os.walk(path):
    for f in sorted(files):
      if f.endswith((".md", ".txt")):
        fp = os.path.join(root, f)
        text = open(fp, encoding="utf-8", errors="ignore").read()
        chunks += chunk_text(text, os.path.relpath(fp, path),
                             lambda s: tok.encode(s, add_special_tokens=False), tok.decode,
                             size=CHUNK_TOK, overlap=32)
  return chunks


def _sample_energy(fn):
  """Run fn() while sampling package power with powermetrics; return (result, millijoules).
  Returns (result, None) if powermetrics is unavailable or not run as root."""
  try:
    proc = subprocess.Popen(["powermetrics", "-i", "50", "--samplers", "cpu_power"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
  except (FileNotFoundError, PermissionError):
    return fn(), None
  t0 = time.perf_counter()
  result = fn()
  dt = time.perf_counter() - t0
  proc.terminate()
  out = proc.stdout.read() if proc.stdout else ""
  mw = [float(l.split(":")[1].strip().split()[0]) for l in out.splitlines() if "Package Power" in l]
  if not mw:
    return result, None
  return result, (sum(mw) / len(mw)) * dt      # avg mW * seconds = millijoules


class Pipeline:
  def __init__(self, embed, rerank, llm, tok, chunks, vecs):
    self.embed, self.rerank, self.llm, self.tok = embed, rerank, llm, tok
    self.chunks, self.vecs = chunks, vecs

  @classmethod
  def build(cls, path: str, llm_name: str = LLM) -> "Pipeline":
    from transformers import AutoTokenizer            # lazy
    from aneforge import load_llm
    from aneforge.sentence_transformers import CrossEncoder, SentenceTransformer
    tok = AutoTokenizer.from_pretrained(llm_name)
    chunks = _read_corpus(path, tok)
    if not chunks:
      raise SystemExit(f"rag_chat: no .md/.txt files found under {path!r}")
    embed = SentenceTransformer(EMBED)
    vecs = embed.encode([c.text for c in chunks], normalize_embeddings=True).astype(np.float32)
    llm = load_llm(llm_name); llm.warmup(MAX_LEN)
    return cls(embed, CrossEncoder(RERANK), llm, tok, chunks, vecs)

  def answer(self, query: str, on_token) -> dict:
    t = {}
    t0 = time.perf_counter()
    qv = self.embed.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    cand = top_k(qv, self.vecs, TOP_K)
    t["retrieve"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    scores = self.rerank.predict([(query, self.chunks[i].text) for i in cand])
    ranked = [self.chunks[cand[i]] for i in np.argsort(scores)[::-1][:TOP_N]]
    t["rerank"] = (time.perf_counter() - t0) * 1e3
    budget = MAX_LEN - ANSWER_TOKENS
    prompt = pack_context(ranked, query, budget, lambda s: len(self.tok.encode(s)))
    ids = self.tok.encode(prompt)[:budget]
    t0 = time.perf_counter()
    self.llm.generate(ids, max_new_tokens=ANSWER_TOKENS, max_len=MAX_LEN,
                      eos_id=self.tok.eos_token_id, on_token=on_token)
    t["generate"] = (time.perf_counter() - t0) * 1e3
    return {"stage_ms": t, "sources": sorted({c.source for c in ranked})}


def main(argv) -> int:
  args = [a for a in argv if not a.startswith("--")]
  path = args[0] if args else REPO_DOCS
  energy = "--energy" in argv
  print(f"indexing {path} on the Apple Neural Engine ...", flush=True)
  p = Pipeline.build(path)
  print(f"indexed {len(p.chunks)} chunks from {len({c.source for c in p.chunks})} files. "
        f"embeddings + reranker + Qwen3-0.6B all on the ANE.\nask a question (Ctrl-D to quit).\n")
  while True:
    try:
      q = input("> ").strip()
    except EOFError:
      print(); return 0
    if not q:
      continue
    print("ane> ", end="", flush=True)
    stream = lambda i: print(p.tok.decode([i]), end="", flush=True)
    if energy:
      timing, mj = _sample_energy(lambda: p.answer(q, on_token=stream))
      tail = f" | ~{mj:.0f} mJ this query, 0 GPU" if mj is not None else " | (energy: run with sudo)"
    else:
      timing, tail = p.answer(q, on_token=stream), ""
    ms = timing["stage_ms"]
    print(f"\n[retrieve {ms['retrieve']:.0f}ms | rerank {ms['rerank']:.0f}ms | "
          f"generate {ms['generate']:.0f}ms | all ANE | sources: {', '.join(timing['sources'])}{tail}]\n")


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
