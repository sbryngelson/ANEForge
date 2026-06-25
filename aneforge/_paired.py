"""Paired-fp16 ("double-fp16") extended precision, no fp32 in the compute path. A value is an unevaluated pair `(hi, lo)` carrying ~2x the fp16 significand via error-free transforms (TwoSum/TwoProduct); the compensated dot accumulates through `@ ones`, never reduce_sum."""
from __future__ import annotations

import numpy as np

from .graph import Tensor

# Dekker/Veltkamp split constant for fp16 (11-bit significand): 2^ceil(11/2)+1.
_SPLIT = float(np.float16(2 ** 6 + 1))


def _two_sum(a: Tensor, b: Tensor):
  """Knuth TwoSum: (s, e) with s = fl(a+b), a+b = s+e exactly. Pure fp16."""
  s = a + b
  bb = s - a
  e = (a - (s - bb)) + (b - bb)
  return s, e


def _split(t: Tensor):
  """Veltkamp split into hi + lo (each ~half the significand). Pure fp16."""
  c = t * _SPLIT
  hi = c - (c - t)
  lo = t - hi
  return hi, lo


def _two_prod(a: Tensor, b: Tensor):
  """Dekker TwoProduct (no FMA): (p, e) with p = fl(a*b), a*b = p+e exactly. Pure fp16."""
  p = a * b
  ah, al = _split(a)
  bh, bl = _split(b)
  e = ((ah * bh - p) + (ah * bl) + (al * bh)) + (al * bl)
  return p, e


def _renorm(s: Tensor, e: Tensor):
  """Renormalize (s, e) into a non-overlapping (hi, lo) so hi = fl(hi+lo). Pure fp16."""
  hi = s + e
  lo = e - (hi - s)
  return hi, lo


class Paired:
  """A value carried as an unevaluated fp16 pair `hi + lo` (double-fp16); construct via :func:`paired`."""

  __slots__ = ("hi", "lo")

  def __init__(self, hi: Tensor, lo: Tensor) -> None:
    if not isinstance(hi, Tensor) or not isinstance(lo, Tensor):
      raise TypeError("Paired(hi, lo) expects two aneforge Tensors")
    if hi.shape != lo.shape:
      raise ValueError(f"Paired: hi shape {hi.shape} != lo shape {lo.shape}")
    self.hi, self.lo = hi, lo

  @property
  def shape(self) -> tuple:
    return self.hi.shape

  # -- arithmetic (error-free transforms, all fp16) ---------------------- #
  def __add__(self, o: "Paired") -> "Paired":
    o = _as_paired(o)
    s, e = _two_sum(self.hi, o.hi)        # exact hi sum + its rounding
    e = e + (self.lo + o.lo)              # fold in the carried lows
    return Paired(*_renorm(s, e))

  def __sub__(self, o: "Paired") -> "Paired":
    o = _as_paired(o)
    s, e = _two_sum(self.hi, o.hi * -1.0)  # exact hi difference + its rounding
    e = e + (self.lo - o.lo)               # fold in the carried lows
    return Paired(*_renorm(s, e))

  def __mul__(self, o) -> "Paired":
    if isinstance(o, (int, float)): return Paired(self.hi * float(o), self.lo * float(o))  # scalar: scale both limbs
    o = _as_paired(o)
    p, e = _two_prod(self.hi, o.hi)
    e = e + (self.hi * o.lo + self.lo * o.hi)  # cross terms (lo*lo dropped, below ulp)
    return Paired(*_renorm(p, e))
  __rmul__ = __mul__

  def __neg__(self) -> "Paired":
    return Paired(self.hi * -1.0, self.lo * -1.0)

  def dot(self, o: "Paired", axis: int = -1) -> "Paired":
    """Compensated dot over `axis`: TwoProduct each element, accumulate products and error streams through `@ ones` (never reduce_sum). Returns a Paired reduced along `axis` (keepdims)."""
    o = _as_paired(o)
    if self.shape != o.shape: raise ValueError(f"dot: shape mismatch {self.shape} vs {o.shape}")
    ax = axis % len(self.shape)
    K = self.shape[ax]

    p, e = _two_prod(self.hi, o.hi)
    e = e + (self.hi * o.lo + self.lo * o.hi)

    # accumulate via matmul (wide accumulator): move `axis` last, contract with [K,1] ones
    def _accum(t: Tensor) -> Tensor:
      perm = [i for i in range(len(t.shape)) if i != ax] + [ax]
      tp = t.transpose(perm) if perm != list(range(len(t.shape))) else t
      ones = np.ones((K, 1), dtype=np.float16)
      acc = tp @ ones
      return acc

    sp, se = _accum(p), _accum(e)
    s, e2 = _two_sum(sp, se)   # recover the matmul's own fp16 output rounding
    return Paired(*_renorm(s, e2))

  # -- conversions ------------------------------------------------------- #
  def to_tensor(self) -> Tensor:
    """Best single-fp16 value: `hi + lo` as an explicit add so the compiler materializes lo (else a dead-code pass would regress to plain fp16)."""
    return self.hi + self.lo

  combine = to_tensor

  def __repr__(self) -> str:
    return f"Paired(hi={self.hi!r}, lo={self.lo!r})"


def _as_paired(o) -> Paired:
  if isinstance(o, Paired): return o
  if isinstance(o, Tensor): return paired(o)
  raise TypeError(f"expected Paired or Tensor, got {type(o).__name__}")


def paired(hi: Tensor, lo: Tensor | None = None) -> Paired:
  """Construct a :class:`Paired` (double-fp16) value. `af.paired(x)` splits a Tensor into a pair (lo=0); `af.paired(hi, lo)` wraps an already-split pair carrying sub-ulp info (regime B)."""
  if not isinstance(hi, Tensor):
    raise TypeError("af.paired(hi[, lo]) expects an aneforge Tensor for hi")
  if lo is None:
    # lo = x - x: structurally zero, emitted as ops so the pair is a real two-limb dataflow
    lo = hi - hi
  elif not isinstance(lo, Tensor): raise TypeError("af.paired(hi, lo): lo must be an aneforge Tensor")
  elif hi.shape != lo.shape: raise ValueError(f"af.paired: hi shape {hi.shape} != lo shape {lo.shape}")
  return Paired(hi, lo)
