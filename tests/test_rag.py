from examples._rag import Chunk, chunk_text


def test_chunk_text_windows_with_overlap():
  text = "x" * 2000
  chunks = chunk_text(text, "a.md", size=800, overlap=160)   # step = 640
  assert [len(c.text) for c in chunks] == [800, 800, 720]     # 0:800, 640:1440, 1280:2000
  assert all(c.source == "a.md" for c in chunks)
  assert "".join(dict.fromkeys("".join(c.text for c in chunks))) != ""  # covers all text


def test_chunk_text_short_doc_is_one_chunk():
  chunks = chunk_text("hello", "b.txt", size=800, overlap=160)
  assert chunks == [Chunk("hello", "b.txt")]
