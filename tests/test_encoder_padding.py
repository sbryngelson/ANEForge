"""The BERT encoder pads a batch to one length and masks the padding (mha `mask=`), so a varied-length
corpus compiles one program instead of one per length, and each text's embedding is unchanged by the
padding. Regression for the per-length-compile crash / padding corruption (issue #232)."""
from _helpers import requires_ane
from aneforge.sentence_transformers import SentenceTransformer


@requires_ane
def test_encoder_padding_is_invariant_and_bounds_programs():
  st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  texts = ["cat",
           "the neural engine runs fp16 compute",
           "a much longer sentence about compiling operator graphs to the apple neural engine"]
  batch = st.encode(texts, normalize_embeddings=True)         # padded to the longest -> one program
  alone = st.encode(["cat"], normalize_embeddings=True)[0]    # the same short text, no padding
  assert float(batch[0] @ alone) > 0.999                      # padding must not change the embedding

  enc = st._enc
  enc._cache.clear()
  st.encode([("w " * int(n)).strip() for n in (5, 40, 90, 150, 30, 200)], normalize_embeddings=True)
  assert len(enc._cache) == 1                                 # six distinct lengths -> a single compiled program
