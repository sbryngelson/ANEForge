"""A drop-in `SentenceTransformer` that runs the encoder on the Apple Neural Engine.

    from aneforge.sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    emb = model.encode(["a query", "a passage"])     # [2, D] on the ANE

The transformer layers run on the ANE as one fused e5rt program (cached per
sequence length); embeddings match the reference to cosine ~1.0 at a fraction of
the GPU's energy. The pooling mode (mean / cls / max) and whether the output is
L2-normalised are read from the model's own sentence-transformers config, so a
mean-pooled model (MiniLM, E5) and a cls-pooled model (BGE, GTE) both come out
right. Only numpy + aneforge are needed; the sentence-transformers package is not
imported (this mirrors its `.encode` surface, it does not wrap it).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .models import Encoder


def _read_text(name: str, rel: str) -> str | None:
  """Read one config file from a local model dir or the HF hub, or None."""
  if os.path.isdir(name):
    path = os.path.join(name, rel)
    return open(path).read() if os.path.exists(path) else None
  from huggingface_hub import hf_hub_download  # lazy
  try: return open(hf_hub_download(name, rel)).read()
  except Exception: return None


def _read_st_config(name: str) -> tuple[str, bool]:
  """Return (pooling_mode, has_normalize) from the model's sentence-transformers
    config. Defaults to ("mean", False) for a raw model with no such config."""
  pooling = "mean"
  pc = _read_text(name, "1_Pooling/config.json")
  if pc:
    d = json.loads(pc)
    if d.get("pooling_mode_cls_token"): pooling = "cls"
    elif d.get("pooling_mode_max_tokens"): pooling = "max"
    elif d.get("pooling_mode_mean_tokens"): pooling = "mean"
  normalize = False
  mj = _read_text(name, "modules.json")
  if mj: normalize = any("Normalize" in m.get("type", "") for m in json.loads(mj))
  return pooling, normalize


class SentenceTransformer:
  """`sentence_transformers.SentenceTransformer`-compatible encoder on the ANE.

    `int8=True` streams int8 weights (half the size, cosine ~0.9999). The `device`
    argument is accepted for signature parity and ignored: the encoder always runs
    on the Neural Engine.
    """

  def __init__(self, model_name_or_path: str, *, int8: bool = False, device: str | None = None) -> None:
    self.pooling_mode, self._normalize_module = _read_st_config(model_name_or_path)
    self._enc = Encoder(model_name_or_path, int8=int8, pooling=self.pooling_mode)

  def encode(self, sentences, batch_size: int = 32, normalize_embeddings: bool = False,
             convert_to_numpy: bool = True, convert_to_tensor: bool = False, **kwargs):
    """Encode `sentences` (a str or list of str) to embeddings on the ANE.

        `batch_size` is accepted for parity; the ANE path is fused per sequence
        length and cached, so it does not change the result. A model that ships a
        Normalize module is L2-normalised regardless of `normalize_embeddings`,
        matching sentence-transformers.
        """
    single = isinstance(sentences, str)
    texts = [sentences] if single else list(sentences)
    normalize = self._normalize_module or normalize_embeddings
    out = self._enc(texts, normalize=normalize).astype(np.float32)
    out = out[0] if single else out
    if convert_to_tensor:
      import torch  # lazy
      return torch.from_numpy(np.ascontiguousarray(out))
    return out
