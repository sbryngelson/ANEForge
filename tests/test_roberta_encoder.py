"""RoBERTa-family encoders need HF's pad_id+1 position offset; 0-based positions wreck the embeddings."""
import numpy as np

from _helpers import requires_ane


@requires_ane
def test_distilroberta_dropin_matches_hf():
  """all-distilroberta-v1 (model_type roberta) through the drop-in matches HF mean-pooled embeddings.

  Without the pad_id+1 offset the positions are wrong and the cosine falls to ~0.65."""
  import torch
  from transformers import AutoTokenizer, AutoModel
  from aneforge.sentence_transformers import SentenceTransformer
  name = "sentence-transformers/all-distilroberta-v1"
  sents = ["Hello from the Neural Engine", "A cat sits on the mat", "Quantum computing is fun"]
  af = np.asarray(SentenceTransformer(name).encode(sents, normalize_embeddings=True))
  tok = AutoTokenizer.from_pretrained(name); hf = AutoModel.from_pretrained(name).eval()
  b = tok(sents, padding=True, truncation=True, return_tensors="pt")
  with torch.no_grad():
    out = hf(**b).last_hidden_state
  m = b["attention_mask"].unsqueeze(-1).float()
  ref = torch.nn.functional.normalize((out * m).sum(1) / m.sum(1), dim=1).numpy()
  cos = [float(af[i] @ ref[i] / (np.linalg.norm(af[i]) * np.linalg.norm(ref[i]) + 1e-9)) for i in range(len(sents))]
  assert min(cos) > 0.99, f"RoBERTa drop-in vs HF cosine {min(cos):.4f}"
