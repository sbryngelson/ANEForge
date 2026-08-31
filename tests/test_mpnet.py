"""MPNet encoder support: relative-position bias + no token-type + position offset (aneforge.models)."""
import numpy as np

from aneforge.models import _mpnet_relative_bucket, _mpnet_position_bias
from _helpers import requires_ane


def test_mpnet_relative_bucket_matches_hf():
  """Our numpy bucketing must match HF's MPNet static method across near and far relative positions."""
  import torch
  from transformers.models.mpnet.modeling_mpnet import MPNetEncoder
  rp = np.arange(-140, 141)
  got = _mpnet_relative_bucket(rp)
  ref = MPNetEncoder.relative_position_bucket(torch.tensor(rp)).numpy()
  assert np.array_equal(got, ref), f"buckets diverge at {np.flatnonzero(got != ref)[:5]}"


def test_mpnet_position_bias_shape_and_symmetry():
  """The bias is [H, S, S]; the diagonal (relative position 0) shares one bucket across all query rows."""
  H, S = 4, 6
  rel = np.arange(32 * H, dtype=np.float32).reshape(32, H)   # buckets x heads
  bias = _mpnet_position_bias(rel, S)
  assert bias.shape == (H, S, S)
  diag = np.array([bias[:, i, i] for i in range(S)])         # every diagonal entry -> bucket 0
  assert np.allclose(diag, diag[0])


@requires_ane
def test_mpnet_dropin_matches_hf():
  """all-mpnet-base-v2 through the ANE drop-in matches HF mean-pooled + normalized embeddings."""
  import torch
  from transformers import AutoTokenizer, AutoModel
  from aneforge.sentence_transformers import SentenceTransformer
  name = "sentence-transformers/all-mpnet-base-v2"
  sents = ["Hello from the Neural Engine", "A cat sits on the mat", "Quantum computing is hard"]
  af = np.asarray(SentenceTransformer(name).encode(sents, normalize_embeddings=True))
  tok = AutoTokenizer.from_pretrained(name); hf = AutoModel.from_pretrained(name).eval()
  b = tok(sents, padding=True, truncation=True, return_tensors="pt")
  with torch.no_grad():
    out = hf(**b).last_hidden_state
  m = b["attention_mask"].unsqueeze(-1).float()
  ref = torch.nn.functional.normalize((out * m).sum(1) / m.sum(1), dim=1).numpy()
  cos = [float(af[i] @ ref[i] / (np.linalg.norm(af[i]) * np.linalg.norm(ref[i]) + 1e-9)) for i in range(len(sents))]
  assert min(cos) > 0.99, f"MPNet drop-in vs HF cosine {min(cos):.4f}"
