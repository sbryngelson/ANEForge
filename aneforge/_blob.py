"""BLOBFILE weight container: 64-byte header, then per blob a 64-byte descriptor (magic 0xDEADBEEF, dtype code, length, data offset) and raw bytes."""
from __future__ import annotations

import struct

import numpy as np

_HEADER = 64
_DESCRIPTOR = 64
# BLOBFILE container dtype codes (distinct from the MIL-proto DataType enum).
FP16, INT8 = 1, 4
UINT1 = 9   # 1-bit sparse mask
UINT4 = 11  # packed 4-bit LUT indices


def fp16_bytes(a: np.ndarray) -> bytes:
  return np.ascontiguousarray(a.astype(np.float16)).tobytes()


def quantize_per_row(W: np.ndarray) -> tuple[bytes, bytes]:
  """Per-output-channel symmetric int8 quant of [OUT,IN] -> (int8 bytes, fp16 scale bytes); reconstruct as `int8 * scale[:,None]`."""
  W = W.astype(np.float32)
  scale = np.clip(np.abs(W).max(axis=1, keepdims=True) / 127.0, 1e-8, None)
  q = np.round(W / scale).clip(-127, 127).astype(np.int8)
  return np.ascontiguousarray(q).tobytes(), fp16_bytes(scale[:, 0])


def palettize_lut4(W: np.ndarray) -> tuple[bytes, bytes]:
  """Per-tensor 4-bit LUT palettization of [OUT,IN] -> (packed-index bytes, fp16 codebook bytes). 16 centroids via deterministic Lloyd iterations; indices packed two-per-byte (low nibble first). Reconstruction `codebook[index]` matches constexpr_lut_to_dense."""
  flat = W.reshape(-1).astype(np.float32)
  edges = np.quantile(flat, np.linspace(0.0, 1.0, 17))
  centroids = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
  for _ in range(20):
    idx = np.abs(flat[:, None] - centroids[None, :]).argmin(axis=1)
    for k in range(16):
      sel = flat[idx == k]
      if sel.size: centroids[k] = sel.mean()
  idx = np.abs(flat[:, None] - centroids[None, :]).argmin(axis=1).astype(np.uint8)
  idx = idx.reshape(W.shape[0], -1)
  if idx.shape[1] % 2:                                   # pad odd row length with 0
    idx = np.pad(idx, ((0, 0), (0, 1)))
  packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
  return np.ascontiguousarray(packed).tobytes(), fp16_bytes(centroids)


def quantize_blockwise(W: np.ndarray, block_size: int = 32) -> tuple[bytes, bytes, int]:
  """Per-block symmetric int8 quant of [OUT,IN] -> (int8 data, fp16 scale [OUT,nblocks], nblocks). IN splits into contiguous `block_size`-column blocks; `block_size` clamps to the largest divisor of IN. Reconstructs as `data.reshape(OUT,nblocks,bs)*scale[:,:,None]`."""
  W = W.astype(np.float32)
  OUT, IN = W.shape
  bs = int(block_size)
  while bs > 1 and IN % bs: bs -= 1
  nblocks = IN // bs
  Wb = W.reshape(OUT, nblocks, bs)
  scale = np.clip(np.abs(Wb).max(axis=2) / 127.0, 1e-8, None)        # [OUT, nblocks]
  q = np.round(Wb / scale[:, :, None]).clip(-127, 127).astype(np.int8).reshape(OUT, IN)
  return np.ascontiguousarray(q).tobytes(), fp16_bytes(scale), nblocks


def sparsify(W: np.ndarray) -> tuple[bytes, bytes]:
  """Bitmask sparse encoding of [OUT,IN] -> (packed 1-bit mask, fp16 nonzero values), row-major (mask bit 1 = keep, LSB-first). Matches constexpr_sparse_to_dense; lossless but for fp16 rounding."""
  flat = W.reshape(-1)
  nz = flat != 0.0
  mask = np.packbits(nz.astype(np.uint8), bitorder="little")
  return np.ascontiguousarray(mask).tobytes(), fp16_bytes(flat[nz])


class BlobWriter:
  """Accumulates weight payloads; `add` returns each blob's descriptor offset (the MIL `BLOBFILE(offset=...)` value)."""

  def __init__(self) -> None:
    self._items: list[tuple[bytes, int]] = []

  def add(self, payload: bytes, code: int) -> int:
    offset = _HEADER + sum(_DESCRIPTOR + len(p) for p, _ in self._items)
    self._items.append((payload, code))
    return offset

  def build(self) -> bytes:
    parts = [struct.pack("<II", len(self._items), 2) + b"\0" * 56]
    cursor = _HEADER
    for payload, code in self._items:
      data_offset = cursor + _DESCRIPTOR
      parts.append(struct.pack("<IIQQ", 0xDEADBEEF, code, len(payload), data_offset) + b"\0" * 40)
      parts.append(payload)
      cursor = data_offset + len(payload)
    return b"".join(parts)
