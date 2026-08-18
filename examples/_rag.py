"""Pure RAG orchestration for examples/rag_chat.py: chunking, vector search, prompt
packing. No ANE and no I/O here -- models and tokenizers are passed in as callables so
this is unit-tested off-device."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
  text: str
  source: str


def chunk_text(text: str, source: str, size: int = 800, overlap: int = 160) -> list[Chunk]:
  """Split `text` into overlapping character windows of `size`, stepping `size - overlap`.
  A document shorter than `size` is a single chunk; no text is dropped."""
  text = text.strip()
  if len(text) <= size:
    return [Chunk(text, source)] if text else []
  step = size - overlap
  chunks = []
  i = 0
  while i < len(text):
    chunk = text[i:i + size]
    chunks.append(Chunk(chunk, source))
    if i + size >= len(text):
      break
    i += step
  return chunks


def top_k(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> list[int]:
  """Indices of the top `k` corpus rows by cosine similarity to `query_vec`
  (dot product, since both sides are L2-normalized), highest first."""
  scores = corpus @ query_vec
  return np.argsort(scores)[::-1][:k].tolist()


PROMPT_TEMPLATE = (
  "Answer the question using only the context below. If the context does not contain the "
  "answer, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")


def pack_context(chunks: list[Chunk], query: str, budget: int, token_len) -> str:
  """Build the prompt from as many reranked chunks (best first) as fit in `budget` tokens.
  Always includes the query; if the first chunk alone overflows, truncate it to fit."""
  kept: list[str] = []
  for c in chunks:
    trial = "\n".join(kept + [c.text])
    if token_len(PROMPT_TEMPLATE.format(context=trial, question=query)) <= budget:
      kept.append(c.text)
  if not kept and chunks:                       # first chunk alone overflows: truncate it
    text = chunks[0].text
    while text and token_len(PROMPT_TEMPLATE.format(context=text, question=query)) > budget:
      text = text[: max(1, len(text) * 3 // 4)]
    kept = [text]
  return PROMPT_TEMPLATE.format(context="\n".join(kept), question=query)
