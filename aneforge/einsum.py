"""aneforge.einsum - a general `einsum(equation, *operands) -> Tensor` that
decomposes a numpy-style subscript equation into a sequence of aneforge graph ops
(transpose / reshape / matmul-bmm / broadcast-mul / reduce-sum) so the whole
contraction runs as ONE fused e5rt program on the Apple Neural Engine.

    import aneforge as af

    a = af.input((32, 64)); b = af.input((64, 16))
    c = af.einsum("ij,jk->ik", a, b)     # a plain matmul, on the ANE
    net = af.compile(c); out = net(A, B)

`af.einsum` IS this function (the package rebinds the name to it), so it is
callable directly; `from aneforge.einsum import einsum` works identically.

WHY THIS IS POSSIBLE (and where it stops)
-----------------------------------------
A two-operand einsum `A,B->C` partitions its indices into four classes:
  * BATCH  (B): appear in A, B and C        -> kept as outer/batch dims
  * CONTRACT (K): appear in A and B, not C  -> summed over (the dot dimension)
  * KEEP-A (M): appear in A and C, not B    -> rows of the output
  * KEEP-B (N): appear in B and C, not A    -> cols of the output
Align each operand to `[B..., M..., K...]` / `[B..., K..., N...]` by
transpose, collapse each group to a single axis by reshape, then a batched
matmul `[B,M,K] @ [B,K,N] -> [B,M,N]` does the contract-and-sum in ONE op
(the ANE matmul accumulator is wide, >=fp32, so the K-sum is clean). Reshape the
result back to the named output dims, then transpose to the requested order.

Indices in ONE operand but NOT in the output are summed FIRST with a reduce
(`af.sum`); output-only indices absent from the inputs are unsupported (einsum
cannot invent a dimension). For >2 operands we left-fold: contract the running
result with the next operand pairwise.

REDUCTION-ONLY and OUTER products are handled directly:
  * `ij->i` / `ij->` : pure reduce (sum over the dropped axes).
  * `i,j->ij`          : broadcast-mul (no contraction), an outer product.

WHAT IS *NOT* SUPPORTED (rejected with a clear error, never wrong results)
-------------------------------------------------------------------------
Any equation with a REPEATED index inside a single operand - diagonal / trace
extraction (`ii->i`, `ii->`, `...ii->...`). Reading a tensor's diagonal is a
GATHER, unsupported on the ANE (per the op-conformance finding), with no
matmul/reduce decomposition. These raise `EinsumUnsupported`. Ellipsis (`...`)
is also not implemented (v1).

Run the self-test (validates a battery vs `numpy.einsum` on the ANE):
    PYTHONPATH=. python3 aneforge/einsum.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter

import numpy as np

from aneforge.graph import Tensor


class EinsumUnsupported(NotImplementedError):
    """Raised for einsum equations that cannot be lowered to ANE matmul/reduce ops
    (diagonal/trace via repeated indices, ellipsis, invented output dims)."""


# --------------------------------------------------------------------------- #
# equation parsing                                                            #
# --------------------------------------------------------------------------- #

def _parse(equation: str, n_operands: int) -> tuple[list[str], str]:
    """Return (input_subscripts, output_subscript). Implied output is the set of
    indices appearing exactly once, in alphabetical order (numpy's rule)."""
    eq = equation.replace(" ", "")
    if "..." in eq:
        raise EinsumUnsupported("einsum: ellipsis '...' is not supported (v1)")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs, rhs = eq, None
    ins = lhs.split(",")
    if len(ins) != n_operands:
        raise ValueError(f"einsum: equation '{equation}' has {len(ins)} operand(s) "
                         f"but {n_operands} tensor(s) were given")
    for s in ins:
        for ch in s:
            if not ch.isalpha():
                raise ValueError(f"einsum: subscript '{s}' has a non-letter index {ch!r}")
    if rhs is None:
        counts = Counter("".join(ins))
        rhs = "".join(sorted(ch for ch, c in counts.items() if c == 1))
    else:
        for ch in rhs:
            if not ch.isalpha():
                raise ValueError(f"einsum: output '{rhs}' has a non-letter index {ch!r}")
        if len(set(rhs)) != len(rhs):
            raise EinsumUnsupported(f"einsum: repeated output index in '{rhs}' "
                                    f"(diagonal-write / broadcast-back needs gather; unsupported)")
        all_in = set("".join(ins))
        invented = [ch for ch in rhs if ch not in all_in]
        if invented:
            raise ValueError(f"einsum: output index {invented} not present in any input")
    return ins, rhs


def _reduce_repeats_check(sub: str) -> None:
    """A repeated index within one operand is a diagonal extraction (gather)."""
    if len(set(sub)) != len(sub):
        dup = sorted({ch for ch in sub if sub.count(ch) > 1})
        raise EinsumUnsupported(
            f"einsum: repeated index {dup} within operand '{sub}' is a diagonal/trace "
            f"extraction, which needs gather - unsupported on the ANE. "
            f"(Reject rather than return wrong results.)")


# --------------------------------------------------------------------------- #
# operand-level helpers                                                       #
# --------------------------------------------------------------------------- #

def _sum_to(t: Tensor, sub: str, keep: str) -> tuple[Tensor, str]:
    """Sum `t` over the axes whose index is not in `keep`, then drop the size-1
    reduced axes via reshape (numpy keepdims=False semantics)."""
    drop_axes = tuple(i for i, ch in enumerate(sub) if ch not in keep)
    if not drop_axes: return t, sub
    t = t.sum(drop_axes)                       # af.sum keeps dims (size 1)
    new_sub = "".join(ch for i, ch in enumerate(sub) if i not in drop_axes)
    # squeeze the reduced (size-1) axes out
    t = t.reshape(*[t.shape[i] for i in range(len(sub)) if i not in drop_axes] or [1])
    return t, new_sub


def _align(t: Tensor, sub: str, order: str) -> Tensor:
    """Transpose `t` so its indices appear in the index-order given by `order`
    (a permutation of the letters of `sub`)."""
    if list(sub) == list(order): return t
    perm = [sub.index(ch) for ch in order]
    return t.transpose(perm)


def _size(t: Tensor, sub: str, ch: str) -> int:
    return t.shape[sub.index(ch)]


# --------------------------------------------------------------------------- #
# two-operand contraction                                                     #
# --------------------------------------------------------------------------- #

def _contract_pair(a: Tensor, sa: str, b: Tensor, sb: str, out: str) -> tuple[Tensor, str]:
    """Contract operands a (sub sa) and b (sub sb) producing the indices in `out`
    (the subset of sa|sb that must survive this step). Returns (tensor, its_sub)."""
    _reduce_repeats_check(sa)
    _reduce_repeats_check(sb)

    set_a, set_b, set_out = set(sa), set(sb), set(out)

    # indices that this step must not keep can be summed away early if they live in
    # only one operand (a partial reduce); shared ones are the contraction.
    keep_a = set_a & (set_b | set_out)
    keep_b = set_b & (set_a | set_out)
    a, sa = _sum_to(a, sa, "".join(keep_a))
    b, sb = _sum_to(b, sb, "".join(keep_b))
    set_a, set_b = set(sa), set(sb)

    batch = [ch for ch in sa if ch in set_b and ch in set_out]              # B
    contract = [ch for ch in sa if ch in set_b and ch not in set_out]       # K
    keepM = [ch for ch in sa if ch not in set_b]                            # M (a-only, in out)
    keepN = [ch for ch in sb if ch not in set_a]                            # N (b-only, in out)

    # --- pure elementwise / outer (no contraction) ------------------------ #
    if not contract: return _broadcast_mul(a, sa, b, sb, batch, keepM, keepN)

    # --- batched matmul path ---------------------------------------------- #
    a = _align(a, sa, "".join(batch + keepM + contract))   # [B.., M.., K..]
    b = _align(b, sb, "".join(batch + contract + keepN))   # [B.., K.., N..]

    Bdims = [a.shape[i] for i in range(len(batch))]
    Mdims = [a.shape[len(batch) + i] for i in range(len(keepM))]
    Kdims = [a.shape[len(batch) + len(keepM) + i] for i in range(len(contract))]
    Ndims = [b.shape[len(batch) + len(contract) + i] for i in range(len(keepN))]
    Bsz = int(np.prod(Bdims)) if Bdims else 1
    Msz = int(np.prod(Mdims)) if Mdims else 1
    Ksz = int(np.prod(Kdims)) if Kdims else 1
    Nsz = int(np.prod(Ndims)) if Ndims else 1

    if batch:
        a3 = a.reshape(Bsz, Msz, Ksz); b3 = b.reshape(Bsz, Ksz, Nsz)
        c = a3 @ b3                                        # bmm -> [Bsz, Msz, Nsz]
    else:
        a2 = a.reshape(Msz, Ksz); b2 = b.reshape(Ksz, Nsz)
        c = a2 @ b2                                        # bmm 2D -> [Msz, Nsz]

    res_sub = "".join(batch + keepM + keepN)
    c = c.reshape(*(Bdims + Mdims + Ndims) or [1])
    return c, res_sub


def _broadcast_mul(a: Tensor, sa: str, b: Tensor, sb: str,
                   batch: list, keepM: list, keepN: list) -> tuple[Tensor, str]:
    """No shared contracted index: result = elementwise/outer product. We align
    both operands to [batch.., M.., N..] and broadcast-multiply (M absent in b,
    N absent in a appear as size-1 and broadcast)."""
    res = batch + keepM + keepN
    # build a-aligned tensor with size-1 placeholders for N dims
    a = _align(a, sa, "".join([c for c in res if c in set(sa)]))
    b = _align(b, sb, "".join([c for c in res if c in set(sb)]))
    a = _expand_to(a, [c for c in res if c in set(sa)], res,
                   {c: _len(a, [x for x in res if x in set(sa)], c) for c in sa})
    b = _expand_to(b, [c for c in res if c in set(sb)], res,
                   {c: _len(b, [x for x in res if x in set(sb)], c) for c in sb})
    c = a * b
    return c, "".join(res)


def _len(t: Tensor, present: list, ch: str) -> int:
    return t.shape[present.index(ch)]


def _expand_to(t: Tensor, present: list, target: list, _sizes) -> Tensor:
    """Reshape `t` (whose dims are `present`, a subsequence of `target`) to
    rank `len(target)` with size-1 at the missing positions, so it broadcasts."""
    if list(present) == list(target): return t
    shape = []
    pi = 0
    for ch in target:
        if pi < len(present) and present[pi] == ch:
            shape.append(t.shape[pi]); pi += 1
        else:
            shape.append(1)
    return t.reshape(*shape)


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #

def einsum(equation: str, *operands: Tensor) -> Tensor:
    """numpy-style `einsum` lowered to aneforge ops (matmul-reducible patterns).

    Supports: matmul (`ij,jk->ik`), batched matmul (`bij,bjk->bik`), transpose
    (`ij->ji`), outer product (`i,j->ij`), elementwise+reduce (`ij,ij->i`,
    `ij->i`, `ij->`), bilinear (`bi,ij,bj->b`), attention scores
    (`bhqd,bhkd->bhqk`), and general >2-operand left-fold contractions.

    Rejects (`EinsumUnsupported`): diagonal/trace via repeated indices in one
    operand (`ii->i`, `ii->`), repeated output indices, and ellipsis."""
    if not operands:
        raise ValueError("einsum: at least one operand is required")
    if not all(isinstance(o, Tensor) for o in operands):
        raise TypeError("einsum: all operands must be aneforge Tensors (build them with af.input)")

    ins, out = _parse(equation, len(operands))
    for s, t in zip(ins, operands):
        if len(s) != len(t.shape):
            raise ValueError(f"einsum: subscript '{s}' has {len(s)} indices but operand "
                             f"shape {t.shape} has {len(t.shape)} dims")
        _reduce_repeats_check(s)

    # ---- single operand: pure transpose / reduce ------------------------- #
    if len(operands) == 1:
        t, sub = operands[0], ins[0]
        t, sub = _sum_to(t, sub, out)            # drop summed indices
        if sub == out: return t
        return _align(t, sub, out)               # transpose to requested order

    # ---- multi-operand: left-fold pairwise ------------------------------- #
    cur, csub = operands[0], ins[0]
    for i in range(1, len(operands)):
        nxt, nsub = operands[i], ins[i]
        remaining = "".join(ins[i + 1:])
        # indices we must keep after this pair: anything in the final output OR
        # still needed by a later operand.
        survive = set(out) | set(remaining)
        step_out = "".join(dict.fromkeys(
            [ch for ch in csub + nsub if ch in survive]))
        cur, csub = _contract_pair(cur, csub, nxt, nsub, step_out)

    # final: sum away any leftover non-output indices, then order to `out`
    cur, csub = _sum_to(cur, csub, out)
    if csub != out: cur = _align(cur, csub, out)
    return cur


__all__ = ["EinsumUnsupported", "einsum"]


# --------------------------------------------------------------------------- #
# self-test: validate vs numpy.einsum on the ANE                              #
# --------------------------------------------------------------------------- #

def _selftest() -> int:
  import aneforge as af

  rng = np.random.default_rng(0)
  f16 = np.float16

  def R(*shape):
    return rng.standard_normal(shape).astype(f16)

  # (name, equation, [shapes...], tol)
  battery = [
    ("matmul",            "ij,jk->ik",        [(8, 16), (16, 12)]),
    ("matmul-implied",    "ij,jk",            [(8, 16), (16, 12)]),
    ("batched-matmul",    "bij,bjk->bik",     [(4, 8, 16), (4, 16, 12)]),
    ("transpose",         "ij->ji",           [(8, 16)]),
    ("transpose-3d",      "abc->cab",         [(3, 4, 5)]),
    ("outer",             "i,j->ij",          [(8,), (16,)]),
    ("matvec",            "ij,j->i",          [(8, 16), (16,)]),
    ("vecmat",            "i,ij->j",          [(8,), (8, 16)]),
    ("elem-reduce",       "ij,ij->i",         [(8, 16), (8, 16)]),
    ("dot",               "i,i->",            [(32,), (32,)]),
    ("full-reduce",       "ij->",             [(8, 16)]),
    ("row-reduce",        "ij->i",            [(8, 16)]),
    ("col-reduce",        "ij->j",            [(8, 16)]),
    ("bilinear",          "bi,ij,bj->b",      [(4, 8), (8, 8), (4, 8)]),
    ("attn-scores",       "bhqd,bhkd->bhqk",  [(2, 3, 5, 7), (2, 3, 6, 7)]),
    ("attn-context",      "bhqk,bhkd->bhqd",  [(2, 3, 5, 6), (2, 3, 6, 7)]),
    ("weighted-combo",    "bn,bnd->bd",       [(4, 6), (4, 6, 5)]),
    ("three-chain",       "ij,jk,kl->il",     [(6, 8), (8, 5), (5, 7)]),
    ("batch-elem-mul",    "bij,bij->bij",     [(2, 4, 5), (2, 4, 5)]),
    ("tensordot",         "ijk,kl->ijl",      [(3, 4, 5), (5, 6)]),
    ("contract-mid",      "ijk,jl->ikl",      [(3, 4, 5), (4, 6)]),
    ("partial-reduce",    "ijk,ik->i",        [(3, 4, 5), (3, 5)]),
    ("rand-A",            "abc,cd->abd",      [(2, 3, 4), (4, 5)]),
    ("rand-B",            "ij,ik->jk",        [(7, 4), (7, 5)]),
    ("rand-C",            "abcd,abce->abde",  [(2, 2, 3, 4), (2, 2, 3, 5)]),
  ]

  rejected = [
    ("diag",   "ii->i",   [(5, 5)]),
    ("trace",  "ii->",    [(5, 5)]),
    ("diag-batch", "bii->bi", [(3, 5, 5)]),
    ("ellipsis", "...ij->...ji", [(2, 3, 4)]),
    ("diag-write", "ij->ii", [(5, 5)]),
  ]

  print("SUPPORTED battery (relerr vs numpy.einsum on the ANE):")
  passes = 0
  for name, eq, shapes in battery:
    arrs = [R(*s) for s in shapes]
    try:
      ref = np.einsum(eq, *[a.astype(np.float32) for a in arrs])
      t_in = [af.input(s) for s in shapes]
      t_out = einsum(eq, *t_in)
      net = af.compile(t_out)
      got = net(*arrs)
      ref_a = np.atleast_1d(ref)
      got_a = np.atleast_1d(got).reshape(ref_a.shape)
      relerr = float(np.abs(got_a - ref_a).max() / (np.abs(ref_a).max() + 1e-6))
      ok = relerr < 0.02
      passes += ok
      print(f"  {name:18s} {eq:20s} {'OK ' if ok else 'FAIL'} relerr {relerr:.5f}  shape {tuple(np.shape(got))}")
    except Exception as e:  # noqa
      print(f"  {name:18s} {eq:20s} ERROR {type(e).__name__}: {e}")

  print(f"\n{passes}/{len(battery)} supported patterns correct on the ANE\n")

  print("REJECTED patterns (must raise, never compute wrong results):")
  rej_ok = 0
  for name, eq, shapes in rejected:
    t_in = [af.input(s) for s in shapes]
    try:
      einsum(eq, *t_in)
      print(f"  {name:14s} {eq:16s} FAIL (did NOT raise)")
    except EinsumUnsupported as e:
      rej_ok += 1
      print(f"  {name:14s} {eq:16s} rejected: {str(e).splitlines()[0][:70]}")
    except (ValueError, NotImplementedError) as e:
      rej_ok += 1
      print(f"  {name:14s} {eq:16s} rejected: {type(e).__name__}: {str(e)[:60]}")

  print(f"\n{rej_ok}/{len(rejected)} unsupported patterns correctly rejected")
  return 0 if (passes == len(battery) and rej_ok == len(rejected)) else 1


if __name__ == "__main__":
  import sys
  sys.exit(_selftest())
