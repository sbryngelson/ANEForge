"""Paired-fp16 ("double-fp16") arithmetic for aneforge - extended precision with
NO fp32 anywhere in the compute path.

A value is carried as an *unevaluated pair* `(hi, lo)` with `hi = fp16(x)` and
`lo = fp16(x - hi)`, so the pair represents `hi + lo` to ~2x the fp16
significand (~22 effective bits). Both `hi` and `lo` are ordinary aneforge
Tensors, so every Paired operation builds a pure-fp16 graph that compiles to fp16
MIL and runs on the ANE.

The arithmetic is the classic *error-free transforms*, every intermediate rounded
to fp16 (validated first in numpy then on-silicon in the reverse-engineering
corpus; this module reuses those exact algorithms):

  * TwoSum (Knuth)      - add/sub: `a+b = s + e` exactly, s = fl(a+b), all fp16.
  * TwoProduct (Dekker) - mul: `a*b = p + e` exactly via a Veltkamp split, all fp16.
  * compensated dot     - TwoProduct each element, then accumulate the product AND
                          the error streams. CRUCIAL: accumulate via `@ ones`
                          (matmul, whose ANE accumulator is WIDE >= fp32), never via
                          `reduce_sum` (whose ANE accumulator is NARROW fp16 and
                          re-injects the error the compensation just removed).

Why this beats plain fp16: the wall is *catastrophic cancellation* - a tiny result
formed from large nearly-equal quantities, where the fp16 rounding of the operands
and products (not the accumulation) swamps the signal. The lo terms capture exactly
that rounding, so carrying (hi, lo) recovers it.

COMPILER CAVEAT (verified): an aggressive algebraic simplifier could collapse
`hi + lo` back to `hi` (since `lo` is, in exact arithmetic, "just the rounding
of hi"), defeating the trick. On this ANE / e5rt it does NOT happen - the on-device
Paired results match the numpy-fp16 proof bit-for-bit-close (validation in
`examples/paired_fp16.py`). The transforms are opaque fp16 add/sub/mul chains with
no fp32 island for the compiler to "see through", and e5rt preserves them. If a
future toolchain fuses them away, the tell is the on-device relerr jumping back to
the plain-fp16 value; the demo asserts against that.

Op cost (fp16 ops/element, matching the envelope finding):
  * TwoSum / compensated add-sub : ~6 fp16 ops/elem
  * TwoProduct / compensated mul : ~17 fp16 ops/elem
  * compensated dot of length K  : ~17 fp16 ops/elem + 2 matmul-accumulates

API::

    import aneforge as af
    a = af.paired(af.input((1, D)))          # split an fp16 Tensor -> Paired (lo=0)
    a = af.paired(hi_tensor, lo_tensor)      # or supply (hi, lo) carrying sub-ulp bits
    d = (a - b).to_tensor()                  # compensated subtract, best-fp16 result
    s = (a + b)                              # Paired
    p = (a * b)                              # Paired (TwoProduct)
    dot = a.dot(b)                           # compensated contraction -> Paired
"""
from __future__ import annotations

import numpy as np

from .graph import Tensor

# Dekker/Veltkamp split constant for fp16 (11-bit significand): 2^ceil(11/2)+1.
# Same constant the numpy proof uses; fed as a scalar multiply (graph `muls`).
_SPLIT = float(np.float16(2 ** 6 + 1))


def _two_sum(a: Tensor, b: Tensor):
    """Knuth TwoSum on aneforge Tensors: returns (s, e) with s = fl(a+b) and, in
    exact arithmetic, a + b = s + e. Pure fp16 ops (~6 adds/subs)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _split(t: Tensor):
    """Veltkamp split of an fp16 Tensor into hi + lo (each ~half the significand).
    Pure fp16 (one scalar mul + two subs)."""
    c = t * _SPLIT
    hi = c - (c - t)
    lo = t - hi
    return hi, lo


def _two_prod(a: Tensor, b: Tensor):
    """Dekker TwoProduct (no FMA) on aneforge Tensors: returns (p, e) with p = fl(a*b)
    and, in exact arithmetic, a*b = p + e. Pure fp16 ops (~17)."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + (ah * bl) + (al * bh)) + (al * bl)
    return p, e


def _renorm(s: Tensor, e: Tensor):
    """Renormalize an overlapping (s, e) into a non-overlapping pair (hi, lo) so the
    Paired invariant hi = fl(hi+lo) holds. Pure fp16 (fast-two-sum)."""
    hi = s + e
    lo = e - (hi - s)
    return hi, lo


class Paired:
    """A value carried as an unevaluated fp16 pair `hi + lo` (double-fp16).

    `hi` and `lo` are aneforge `Tensor` graph nodes of the same shape; every
    operation builds more fp16 graph. Construct via :func:`paired` (the public
    `af.paired`), not directly, unless you hold matched (hi, lo) tensors."""

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
        if isinstance(o, (int, float)):        # scalar: scale both limbs
            return Paired(self.hi * float(o), self.lo * float(o))
        o = _as_paired(o)
        p, e = _two_prod(self.hi, o.hi)        # exact hi*hi product + its rounding
        # cross terms hi*lo + lo*hi captured in fp16 (lo*lo dropped - below fp16 ulp)
        e = e + (self.hi * o.lo + self.lo * o.hi)
        return Paired(*_renorm(p, e))
    __rmul__ = __mul__

    def __neg__(self) -> "Paired":
        return Paired(self.hi * -1.0, self.lo * -1.0)

    def dot(self, o: "Paired", axis: int = -1) -> "Paired":
        """Compensated dot / contraction over `axis` (the down_proj / accurate-sum
        case). TwoProduct each element, then accumulate the product AND error streams
        through `@ ones` - the WIDE matmul accumulator - never reduce_sum.

        Returns a Paired reduced along `axis` (keepdims). Pure fp16."""
        o = _as_paired(o)
        if self.shape != o.shape:
            raise ValueError(f"dot: shape mismatch {self.shape} vs {o.shape}")
        ax = axis % len(self.shape)
        K = self.shape[ax]

        # full TwoProduct of the two pairs, element-wise (products + captured error)
        p, e = _two_prod(self.hi, o.hi)
        e = e + (self.hi * o.lo + self.lo * o.hi)

        # accumulate via matmul (wide accumulator): move `axis` to last, contract it
        # with a [K,1] ones weight, then restore.
        def _accum(t: Tensor) -> Tensor:
            perm = [i for i in range(len(t.shape)) if i != ax] + [ax]
            tp = t.transpose(perm) if perm != list(range(len(t.shape))) else t
            ones = np.ones((K, 1), dtype=np.float16)
            acc = tp @ ones                    # [..., 1]  wide accumulate
            return acc                          # leading dims preserved, last dim == 1

        sp, se = _accum(p), _accum(e)
        # the matmul itself rounds its fp16 output; recover that with one more TwoSum
        s, e2 = _two_sum(sp, se)
        return Paired(*_renorm(s, e2))

    # -- conversions ------------------------------------------------------- #
    def to_tensor(self) -> Tensor:
        """Collapse the pair to its best single-fp16 approximation.

        That value is `hi` (the pair invariant is hi = fl(hi+lo), so fl(hi+lo)==hi).
        We return `hi + lo` - identical in fp16 to `hi` for a normalized pair, but
        written as an explicit add so the compiler must MATERIALIZE lo into the result
        (a guard against a dead-code pass dropping the lo computation; if elided the
        on-device error would regress to plain fp16)."""
        return self.hi + self.lo

    combine = to_tensor

    def __repr__(self) -> str:
        return f"Paired(hi={self.hi!r}, lo={self.lo!r})"


def _as_paired(o) -> Paired:
    if isinstance(o, Paired):
        return o
    if isinstance(o, Tensor):
        return paired(o)
    raise TypeError(f"expected Paired or Tensor, got {type(o).__name__}")


def paired(hi: Tensor, lo: Tensor | None = None) -> Paired:
    """Public constructor for a :class:`Paired` (double-fp16) value.

    `af.paired(x)`       - split an fp16 Tensor `x` into a pair (lo = 0). A genuine
                             fp16 input has no sub-ulp bits to carry, so the win comes
                             from the compensated *ops* capturing each operation's
                             rounding.
    `af.paired(hi, lo)`  - wrap an already-split pair (hi, lo) that carries sub-ulp
                             information (e.g. a residual or a value produced upstream
                             in higher working precision). The regime where paired-fp16
                             recovers the most (the CFG/regime-B case).

    Pure fp16: the split is `hi = x`, `lo = x - x` ( == 0 ) built as graph ops,
    so the result stays a live fp16 dataflow with no fp32 cast."""
    if not isinstance(hi, Tensor):
        raise TypeError("af.paired(hi[, lo]) expects an aneforge Tensor for hi")
    if lo is None:
        # lo = x - x: structurally zero, but emitted as ops so hi/lo share a shape and
        # the pair is a real two-limb dataflow (no Python-side fp32 collapse).
        lo = hi - hi
    elif not isinstance(lo, Tensor):
        raise TypeError("af.paired(hi, lo): lo must be an aneforge Tensor")
    elif hi.shape != lo.shape:
        raise ValueError(f"af.paired: hi shape {hi.shape} != lo shape {lo.shape}")
    return Paired(hi, lo)
