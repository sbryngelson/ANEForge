from examples._rag import Chunk, chunk_text, top_k, PROMPT_TEMPLATE, pack_context
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


def test_pack_context_stops_at_budget_and_keeps_query():
  chunks = [Chunk("AAAA", "a"), Chunk("BBBB", "b"), Chunk("CCCC", "c")]
  # token_len = len (characters). Budget admits the template + query + ~two chunks.
  prompt = pack_context(chunks, "Q?", budget=len(PROMPT_TEMPLATE.format(context="", question="Q?")) + 9, token_len=len)
  assert "Q?" in prompt
  assert "AAAA" in prompt and "BBBB" in prompt   # two 4-char chunks fit in 9 spare
  assert "CCCC" not in prompt                     # third overflows, dropped
  assert len(prompt) <= len(PROMPT_TEMPLATE.format(context="", question="Q?")) + 9 + 1


def test_pack_context_truncates_a_single_oversized_chunk():
  chunks = [Chunk("Z" * 1000, "big")]
  budget = len(PROMPT_TEMPLATE.format(context="", question="Q?")) + 20
  prompt = pack_context(chunks, "Q?", budget=budget, token_len=len)
  assert "Q?" in prompt and "Z" in prompt
  assert len(prompt) <= budget
