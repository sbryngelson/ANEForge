"""Chat with a folder of docs, entirely on the Apple Neural Engine: SentenceTransformer
embeddings, a CrossEncoder reranker, and Qwen3-0.6B generation all run on the engine.

  python3 examples/rag_chat.py [path]                 # path defaults to this repo's docs/
  python3 examples/rag_chat.py ./docs --llm Qwen/Qwen3-8B   # swap the generation model
  sudo python3 examples/rag_chat.py --energy          # add per-query joules (needs powermetrics)

The corpus index is cached per folder, so a restart over unchanged docs is instant (no re-embedding).
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import warnings

# Keep Hugging Face *download* progress bars (the first run pulls ~1.5 GB); only the per-load bars and
# aneforge's per-call dispatch note are silenced below -- those are noise, a multi-minute download is not.
warnings.filterwarnings("ignore", "aneforge.compile: dispatch-floor")  # per-call dispatch notes are noise in a REPL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import aneforge / examples._rag; works as a script and when imported under pytest

import numpy as np

# Directories that are not part of a project's documentation -- skip them so the index is not polluted
# by version control, dependencies, or working notes (e.g. this repo's gitignored docs/superpowers/).
_SKIP_DIRS = {".git", ".github", "node_modules", "__pycache__", "superpowers"}

EMBED = "sentence-transformers/all-MiniLM-L6-v2"
RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM = "Qwen/Qwen3-0.6B"
MAX_LEN = 512
ANSWER_TOKENS = 160
TOP_K, TOP_N = 20, 4
REPO_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "aneforge_rag")   # cached corpus embeddings live here

from examples._rag import Chunk, chunk_text, pack_context, top_k   # noqa: E402


def _read_corpus(path: str) -> list[Chunk]:
  chunks: list[Chunk] = []
  for root, dirs, files in os.walk(path):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]   # prune noise dirs
    for f in sorted(files):
      if f.endswith((".md", ".txt")):
        fp = os.path.join(root, f)
        with open(fp, encoding="utf-8", errors="ignore") as fh:
          text = fh.read()
        chunks += chunk_text(text, os.path.relpath(fp, path))
  return chunks


def _corpus_signature(path: str, embed_name: str) -> str:
  """A hash over the embedding model and every doc's path/mtime/size -- the index cache key. Any edit,
  addition, or removal changes it, so a stale index is never reused."""
  parts = [embed_name]
  for root, dirs, files in os.walk(path):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]
    for f in sorted(files):
      if f.endswith((".md", ".txt")):
        fp = os.path.join(root, f); st = os.stat(fp)
        parts.append(f"{os.path.relpath(fp, path)}:{int(st.st_mtime)}:{st.st_size}")
  return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def _load_or_build_index(path: str, embed, embed_name: str):
  """Return (chunks, vectors), reading a cached index when the corpus is unchanged so a restart does not
  re-embed. Keyed by `_corpus_signature`, so any doc edit rebuilds it automatically."""
  cache = os.path.join(_CACHE_DIR, f"{_corpus_signature(path, embed_name)}.npz")
  if os.path.exists(cache):
    d = np.load(cache, allow_pickle=True)
    chunks = [Chunk(t, s) for t, s in zip(d["texts"].tolist(), d["sources"].tolist())]
    return chunks, d["vecs"]
  chunks = _read_corpus(path)
  if not chunks:
    raise SystemExit(f"rag_chat: no .md/.txt files found under {path!r}")
  vecs = embed.encode([c.text for c in chunks], normalize_embeddings=True).astype(np.float32)
  os.makedirs(_CACHE_DIR, exist_ok=True)
  np.savez(cache, texts=np.array([c.text for c in chunks], dtype=object),
           sources=np.array([c.source for c in chunks], dtype=object), vecs=vecs)
  return chunks, vecs


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
  proc.wait()
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
  def build(cls, path: str, llm_name: str = LLM, embed_name: str = EMBED,
            rerank_name: str = RERANK) -> "Pipeline":
    import transformers                               # lazy
    from transformers import AutoTokenizer
    from aneforge import load_llm
    from aneforge.sentence_transformers import CrossEncoder, SentenceTransformer
    transformers.logging.set_verbosity_error(); transformers.logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(llm_name)
    embed = SentenceTransformer(embed_name)
    chunks, vecs = _load_or_build_index(path, embed, embed_name)   # cached index -> instant restart
    llm = load_llm(llm_name); llm.warmup(MAX_LEN)
    return cls(embed, CrossEncoder(rerank_name), llm, tok, chunks, vecs)

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
    def _wrap(content):   # Qwen chat format so the model answers from context and stops at EOS (no rambling)
      return self.tok.apply_chat_template([{"role": "user", "content": content}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    budget = MAX_LEN - ANSWER_TOKENS
    content = pack_context(ranked, query, budget, lambda c: len(self.tok.encode(_wrap(c))))
    ids = self.tok.encode(_wrap(content))[:budget]
    t0 = time.perf_counter()
    self.llm.generate(ids, max_new_tokens=ANSWER_TOKENS, max_len=MAX_LEN,
                      eos_id=self.tok.eos_token_id, on_token=on_token)
    t["generate"] = (time.perf_counter() - t0) * 1e3
    return {"stage_ms": t, "sources": sorted({c.source for c in ranked})}


def main(argv) -> int:
  import argparse
  ap = argparse.ArgumentParser(description="Chat with a folder of docs, entirely on the Apple Neural Engine.")
  ap.add_argument("path", nargs="?", default=REPO_DOCS, help="folder of .md/.txt docs (default: this repo's docs/)")
  ap.add_argument("--energy", action="store_true", help="report per-query joules via powermetrics (needs sudo)")
  ap.add_argument("--llm", default=LLM, help=f"HF generation model (default: {LLM})")
  ap.add_argument("--embed", default=EMBED, help=f"HF embedding model (default: {EMBED})")
  ap.add_argument("--rerank", default=RERANK, help=f"HF reranker model (default: {RERANK})")
  a = ap.parse_args(argv)
  print(f"loading models and indexing {a.path} on the Apple Neural Engine ...\n"
        f"(first run downloads the models from Hugging Face -- ~1.5 GB for the defaults; cached for later runs)", flush=True)
  p = Pipeline.build(a.path, llm_name=a.llm, embed_name=a.embed, rerank_name=a.rerank)
  print(f"indexed {len(p.chunks)} chunks from {len({c.source for c in p.chunks})} files. "
        f"embeddings + reranker + {os.path.basename(a.llm)} all on the ANE.\nask a question (Ctrl-D to quit).\n")
  while True:
    try:
      q = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
      print(); return 0
    if not q:
      continue
    print("ane> ", end="", flush=True)
    stream = lambda i: print(p.tok.decode([i]), end="", flush=True)
    if a.energy:
      timing, mj = _sample_energy(lambda: p.answer(q, on_token=stream))
      tail = f" | ~{mj:.0f} mJ this query, 0 GPU" if mj is not None else " | (energy: run with sudo)"
    else:
      timing, tail = p.answer(q, on_token=stream), ""
    ms = timing["stage_ms"]
    print(f"\n[retrieve {ms['retrieve']:.0f}ms | rerank {ms['rerank']:.0f}ms | "
          f"generate {ms['generate']:.0f}ms | all ANE | sources: {', '.join(timing['sources'])}{tail}]\n")


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
