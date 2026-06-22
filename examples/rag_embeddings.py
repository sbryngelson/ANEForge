"""Use ANEForge embeddings in a RAG pipeline: a LangChain `Embeddings` adapter.

The embedding model runs on the Apple Neural Engine - `af.load` fuses the encoder's
transformer layers into one e5rt program - about 4-5x faster than the PyTorch GPU
path on Apple Silicon, at ~9x lower energy, and cosine 1.0000 against the fp32
reference. The adapter is a standard `langchain_core.embeddings.Embeddings`, so it
drops into any LangChain retriever or vector store.

    pip install "aneforge>=0.1.3" "transformers[torch]" langchain-core
    python3 examples/rag_embeddings.py
"""
from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings

import aneforge as af


class ANEEmbeddings(Embeddings):
    """LangChain embeddings backed by the Apple Neural Engine via ANEForge.

    `model` is any BERT-family sentence encoder on the HF hub (MiniLM, BGE, E5, ...).
    Pass `int8=True` to stream int8 weights (half the size, cosine ~0.9999)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", int8: bool = False):
        self._embed = af.load(model, int8=int8)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return np.asarray(self._embed(list(texts))).astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        return np.asarray(self._embed([text]))[0].astype(float).tolist()


def main():
    emb = ANEEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    corpus = [
        "The Apple Neural Engine accelerates neural networks at low power.",
        "Cats are independent animals that enjoy sleeping in the sun.",
        "Transformers process whole sequences in a single forward pass.",
    ]
    # embed_documents / embed_query are exactly what LangChain vector stores call.
    docs = np.asarray(emb.embed_documents(corpus))
    q = np.asarray(emb.embed_query("running ML efficiently on Apple hardware"))
    order = np.argsort(-(docs @ q))
    print("query -> nearest documents (cosine):")
    for rank, i in enumerate(order):
        print(f"  {rank + 1}. ({docs[i] @ q:.3f}) {corpus[i]}")


if __name__ == "__main__":
    main()
