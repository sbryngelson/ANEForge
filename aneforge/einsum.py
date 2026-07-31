"""aneforge.einsum - numpy-style `einsum` lowered to aneforge graph ops as one fused ANE program."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter

import numpy as np

from aneforge.graph import Tensor, _const


class EinsumUnsupported(NotImplementedError):
  """Raised for equations not lowerable to ANE matmul/reduce ops (diagonal-write, triple repeats, ellipsis)."""


# equation parsing

def _expand_ellipsis(equation: str, operands) -> str:
  """Rewrite '...' into explicit batch letters (right-aligned per operand rank) so the v1 parser
  never sees an ellipsis. Implicit output follows numpy's rule (ellipsis dims first, then the
  once-only letters alphabetically). Ellipsis dims must match exactly - size-1 broadcast between
  operands' ellipsis dims needs expand semantics the lowering does not have, and rejects."""
  eq = equation.replace(" ", "")
  if "..." not in eq: return eq
  lhs, rhs = eq.split("->") if "->" in eq else (eq, None)
  ins = lhs.split(",")
  if len(ins) != len(operands):
    raise ValueError(f"einsum: equation '{equation}' has {len(ins)} operand(s) "
             f"but {len(operands)} tensor(s) were given")
  ranks = []
  for s, op in zip(ins, operands):
    if "..." in s:
      e = len(op.shape) - len(s.replace("...", ""))
      if e < 0: raise ValueError(f"einsum: subscript '{s}' has more indices than operand rank {len(op.shape)}")
      ranks.append(e)
    else:
      ranks.append(0)
  E = max(ranks)
  used = set(eq.replace(".", "").replace(",", "").replace("->", ""))
  import string
  pool = [c for c in string.ascii_letters if c not in used]
  if len(pool) < E: raise EinsumUnsupported("einsum: not enough free letters to expand '...'")
  batch = "".join(pool[:E])
  sizes: dict[str, int] = {}
  new_ins = []
  for s, op, e in zip(ins, operands, ranks):
    bletters = batch[E - e:] if "..." in s else ""
    new_ins.append(s.replace("...", bletters))
    for ch, d in zip(bletters, op.shape[:e]):
      if ch in sizes and sizes[ch] != d:
        raise EinsumUnsupported(f"einsum: ellipsis dims disagree ({sizes[ch]} vs {d}) - "
                    "size-1 broadcast across '...' is unsupported")
      sizes[ch] = d
  if rhs is None:
    counts = Counter("".join(new_ins))
    singles = "".join(sorted(ch for ch, c in counts.items() if c == 1 and ch not in batch))
    rhs = batch + singles                       # numpy: ellipsis dims first, then once-only letters
  else:
    if E and "..." not in rhs:
      raise ValueError("einsum: inputs carry '...' dims but the output subscript omits '...' (numpy rejects this)")
    rhs = rhs.replace("...", batch)
  return ",".join(new_ins) + "->" + rhs


def _parse(equation: str, n_operands: int) -> tuple[list[str], str]:
  """Return (input_subscripts, output_subscript); implied output is indices appearing once, alphabetical (numpy's rule)."""
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
  """Invariant guard: operands reach the contraction with distinct indices (`_extract_diagonals` ran first)."""
  if len(set(sub)) != len(sub):
    dup = sorted({ch for ch in sub if sub.count(ch) > 1})
    raise EinsumUnsupported(
      f"einsum: repeated index {dup} within operand '{sub}' reached the contraction "
      f"undiagonalized (internal invariant).")


def _diag_pairs(sub: str) -> dict[str, tuple[int, int]]:
  """Map each index repeated exactly twice in `sub` to its two positions; a higher multiplicity rejects."""
  counts = Counter(sub)
  over = sorted(ch for ch, c in counts.items() if c > 2)
  if over:
    raise EinsumUnsupported(
      f"einsum: index {over!r} appears {counts[over[0]]} times in operand '{sub}'; only a "
      f"doubly-repeated index reduces to a mask-and-reduce diagonal - unsupported on the ANE.")
  return {ch: tuple(i for i, c in enumerate(sub) if c == ch) for ch, c in counts.items() if c == 2}


def _extract_diagonals(t: Tensor, sub: str) -> tuple[Tensor, str]:
  """Rewrite each doubly-repeated index in one operand as its diagonal, dropping the second axis.

  No gather needed: `(x * eye).sum(axis)` IS the diagonal, from ops the lowering already has. The baked
  identity zeroes every off-diagonal term, so each output element keeps exactly one survivor - the
  reduce adds only zeros and the extraction is bit-exact in fp16 (unlike a true multi-term sum)."""
  while True:
    pairs = _diag_pairs(sub)
    if not pairs: return t, sub
    ch, (p, q) = min(pairs.items())
    n = t.shape[p]
    if t.shape[q] != n:
      raise ValueError(f"einsum: repeated index {ch!r} in operand '{sub}' spans dims "
               f"{n} and {t.shape[q]}; a diagonal needs them equal")
    mshape = [n if i in (p, q) else 1 for i in range(len(sub))]   # eye over the pair, size-1 elsewhere
    t = (t * _const(np.eye(n, dtype=np.float16).reshape(mshape))).sum(q).squeeze(q)
    sub = sub[:q] + sub[q + 1:]


# operand-level helpers

def _sum_to(t: Tensor, sub: str, keep: str) -> tuple[Tensor, str]:
  """Sum `t` over axes whose index is not in `keep`, then squeeze the reduced axes (keepdims=False)."""
  drop_axes = tuple(i for i, ch in enumerate(sub) if ch not in keep)
  if not drop_axes: return t, sub
  t = t.sum(drop_axes)                       # af.sum keeps dims (size 1)
  new_sub = "".join(ch for i, ch in enumerate(sub) if i not in drop_axes)
  t = t.reshape(*[t.shape[i] for i in range(len(sub)) if i not in drop_axes] or [1])
  return t, new_sub


def _align(t: Tensor, sub: str, order: str) -> Tensor:
  """Transpose `t` so its indices appear in the order given by `order`."""
  if list(sub) == list(order): return t
  perm = [sub.index(ch) for ch in order]
  return t.transpose(perm)


def _size(t: Tensor, sub: str, ch: str) -> int:
  return t.shape[sub.index(ch)]


# two-operand contraction

def _contract_pair(a: Tensor, sa: str, b: Tensor, sb: str, out: str) -> tuple[Tensor, str]:
  """Contract operands a (sub sa) and b (sub sb) producing the indices in `out`; returns (tensor, its_sub)."""
  _reduce_repeats_check(sa)
  _reduce_repeats_check(sb)

  set_a, set_b, set_out = set(sa), set(sb), set(out)

  # single-operand indices not kept can be summed away early; shared ones are the contraction.
  keep_a = set_a & (set_b | set_out)
  keep_b = set_b & (set_a | set_out)
  a, sa = _sum_to(a, sa, "".join(keep_a))
  b, sb = _sum_to(b, sb, "".join(keep_b))
  set_a, set_b = set(sa), set(sb)

  batch = [ch for ch in sa if ch in set_b and ch in set_out]              # B
  contract = [ch for ch in sa if ch in set_b and ch not in set_out]       # K
  keepM = [ch for ch in sa if ch not in set_b]                            # M (a-only, in out)
  keepN = [ch for ch in sb if ch not in set_a]                            # N (b-only, in out)

  if not contract: return _broadcast_mul(a, sa, b, sb, batch, keepM, keepN)  # elementwise/outer

  # batched matmul path
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
  """No shared contracted index: align both to [batch.., M.., N..] and broadcast-multiply (outer product)."""
  res = batch + keepM + keepN
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
  """Reshape `t` (dims `present`, a subsequence of `target`) to rank len(target) with size-1 at missing positions."""
  if list(present) == list(target): return t
  shape = []
  pi = 0
  for ch in target:
    if pi < len(present) and present[pi] == ch:
      shape.append(t.shape[pi]); pi += 1
    else:
      shape.append(1)
  return t.reshape(*shape)


# public API

def einsum(equation: str, *operands: Tensor) -> Tensor:
  """numpy-style einsum lowered to aneforge ops (matmul-reducible patterns, incl. '...' batch
  dims and diagonal/trace extraction); rejects diagonal-WRITE and repeated output indices."""
  if not operands:
    raise ValueError("einsum: at least one operand is required")
  if not all(isinstance(o, Tensor) for o in operands):
    raise TypeError("einsum: all operands must be aneforge Tensors (build them with af.input)")

  equation = _expand_ellipsis(equation, operands)
  ins, out = _parse(equation, len(operands))
  for s, t in zip(ins, operands):
    if len(s) != len(t.shape):
      raise ValueError(f"einsum: subscript '{s}' has {len(s)} indices but operand "
               f"shape {t.shape} has {len(t.shape)} dims")

  # a doubly-repeated index within an operand is a diagonal: take it before contracting, so
  # everything downstream sees operands whose indices are distinct.
  operands, ins = zip(*(_extract_diagonals(t, s) for t, s in zip(operands, ins)))
  ins = list(ins)

  # single operand: pure transpose / reduce
  if len(operands) == 1:
    t, sub = operands[0], ins[0]
    t, sub = _sum_to(t, sub, out)            # drop summed indices
    if sub == out: return t
    return _align(t, sub, out)               # transpose to requested order

  # multi-operand: left-fold pairwise
  cur, csub = operands[0], ins[0]
  for i in range(1, len(operands)):
    nxt, nsub = operands[i], ins[i]
    remaining = "".join(ins[i + 1:])
    # keep anything in the final output or still needed by a later operand.
    survive = set(out) | set(remaining)
    step_out = "".join(dict.fromkeys(
      [ch for ch in csub + nsub if ch in survive]))
    cur, csub = _contract_pair(cur, csub, nxt, nsub, step_out)

  # final: sum away any leftover non-output indices, then order to `out`
  cur, csub = _sum_to(cur, csub, out)
  if csub != out: cur = _align(cur, csub, out)
  return cur


__all__ = ["EinsumUnsupported", "einsum"]


# self-test: validate vs numpy.einsum on the ANE

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
    ("diag",              "ii->i",            [(5, 5)]),
    ("trace",             "ii->",             [(5, 5)]),
    ("diag-batch",        "bii->bi",          [(3, 5, 5)]),
    ("diag-then-contract", "ii,ij->j",        [(5, 5), (5, 6)]),
    ("diag-both-operands", "ii,jj->ij",       [(4, 4), (5, 5)]),
    ("diag-middle",       "ibi->bi",          [(4, 3, 4)]),
    ("two-diag-pairs",    "iijj->ij",         [(3, 3, 4, 4)]),
    ("ellipsis",          "...ij->...ji",     [(2, 3, 4)]),   # supported since _expand_ellipsis landed
  ]

  rejected = [
    ("diag-write", "ij->ii", [(5, 5)]),
    ("triple-repeat", "iii->i", [(4, 4, 4)]),
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
