from examples._rag import Chunk, chunk_text, top_k
import numpy as np


def test_chunk_text_windows_with_overlap():
  text = "x" * 2000
  chunks = chunk_text(text, "a.md", size=800, overlap=160)   # step = 640
  assert [len(c.text) for c in chunks] == [800, 800, 720]     # 0:800, 640:1440, 1280:2000
  assert all(c.source == "a.md" for c in chunks)
  assert "".join(dict.fromkeys("".join(c.text for c in chunks))) != ""  # covers all text


def test_chunk_text_short_doc_is_one_chunk():
  chunks = chunk_text("hello", "b.txt", size=800, overlap=160)
  assert chunks == [Chunk("hello", "b.txt")]


def test_top_k_orders_by_similarity_descending():
  corpus = np.array([[1, 0], [0, 1], [0.9, 0.1], [-1, 0], [0.7, 0.7]], dtype=np.float32)
  corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
  q = np.array([1, 0], dtype=np.float32)
  assert top_k(q, corpus, 3) == [0, 2, 4]        # cos: 1.0, 0.994, 0.707
  assert top_k(q, corpus, 99) == [0, 2, 4, 1, 3]  # k > N returns all, ordered
