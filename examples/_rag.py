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
