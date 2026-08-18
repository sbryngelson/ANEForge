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


def chunk_text(text, source, encode, decode, size: int = 192, overlap: int = 32) -> list[Chunk]:
  """Split `text` into windows of exactly `size` real tokens (back-extending the final window so it is
  also `size`), using injected `encode(str)->list` and `decode(list)->str`. A document with <= `size`
  tokens is a single chunk. Uniform-length windows keep the ANE embedder to a few compiled programs."""
  text = text.strip()
  if not text:
    return []
  ids = encode(text)
  if len(ids) <= size:
    return [Chunk(text, source)]
  step = size - overlap
  out = []
  for i in range(0, len(ids), step):
    w = ids[i:i + size]
    if len(w) < size:
      w = ids[-size:]
    out.append(Chunk(decode(w), source))
    if i + size >= len(ids):
      break
  return out


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
    while len(text) > 1 and token_len(PROMPT_TEMPLATE.format(context=text, question=query)) > budget:
      text = text[: max(1, len(text) * 3 // 4)]
    kept = [text]
  return PROMPT_TEMPLATE.format(context="\n".join(kept), question=query)
