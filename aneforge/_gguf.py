"""Shared GGUF reader helpers for the model loaders (moe, qwen35). `GGUFView` opens a GGUF once and exposes
the accessors both loaders need: arch-scoped scalar metadata, raw fields by full key, tensors by name,
dequantize-to-fp16, and an mmap page-dropper to keep the warmup memory peak low."""
from __future__ import annotations
import mmap as _mmap
from typing import Any
import numpy as np


class GGUFView:
  """A thin view over a `gguf.GGUFReader`. `arch` is the model architecture (the `*.block_count` prefix)."""
  def __init__(self, path: str):
    import gguf
    from gguf.quants import dequantize
    self._F32 = gguf.GGMLQuantizationType.F32
    self._dequant = dequantize
    self.reader = r = gguf.GGUFReader(path)
    self.meta = {f.name: f for f in r.fields.values()}
    self.arch = next(k.split(".")[0] for k in self.meta if k.endswith(".block_count"))
    self.tn = {t.name: t for t in r.tensors}

  def field(self, key, default=None) -> Any:
    """A raw scalar metadata field by FULL key (e.g. 'general.sampling.temp'); `default` if absent."""
    f = self.meta.get(key)
    return f.parts[f.data[-1]][0] if f is not None else default

  def sc(self, key, default=None) -> Any:
    """An arch-scoped scalar field: `field` with the 'arch.' prefix added (e.g. sc('block_count'))."""
    return self.field(self.arch + "." + key, default)

  def has(self, name: str) -> bool:
    return name in self.tn

  def get(self, name: str, dtype: np.typing.DTypeLike = np.float16) -> np.ndarray:
    """Dequantize tensor `name` to a numpy array of `dtype` (fp16 by default, to halve resident memory)."""
    t = self.tn[name]; d = np.asarray(t.data)
    if t.tensor_type != self._F32:
      d = self._dequant(d, t.tensor_type)
    return d.astype(dtype)

  def drop_pages(self) -> None:
    """madvise(DONTNEED) the GGUF mmap so the dequantized layers don't also pin the raw file in RAM."""
    mm = getattr(self.reader.data, "_mmap", None)
    if mm is not None:
      try: mm.madvise(_mmap.MADV_DONTNEED)
      except (AttributeError, OSError): pass
