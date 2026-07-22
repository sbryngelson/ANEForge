"""Ellipsis ('...') support in aneforge.einsum: expansion to explicit batch letters."""
import numpy as np
import pytest

import aneforge as af
from aneforge.einsum import einsum, EinsumUnsupported, _expand_ellipsis
from _helpers import requires_ane

pytestmark = requires_ane  # the equivalence checks compile and dispatch to the ANE


def _check(eq, *shapes, seed=0):
  rng = np.random.default_rng(seed)
  vals = [rng.standard_normal(s).astype(np.float16) for s in shapes]
  out = einsum(eq, *[af.input(s) for s in shapes])
  got = np.asarray(af.compile(out)(*vals)).astype(np.float32)
  ref = np.einsum(eq, *[v.astype(np.float32) for v in vals])
  assert got.shape == ref.shape, f"{eq}: {got.shape} != {ref.shape}"
  err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6)
  assert err <= 5e-3, f"{eq}: relerr {err:.2e}"


def test_batched_matmul_ellipsis():
  _check("...ij,...jk->...ik", (2, 4, 5), (2, 5, 3))

def test_ellipsis_ranks_align_right():
  # one operand has no batch dims; its '...' expands to nothing
  _check("...i,i->...", (2, 3, 4), (4,))

def test_implicit_output_puts_batch_first():
  # numpy implicit mode: ellipsis dims lead the output
  _check("...ij,...jk", (2, 4, 5), (2, 5, 3))

def test_explicit_output_can_reduce_batch():
  _check("...i,...i->i", (3, 4), (3, 4))

def test_expansion_is_pure_rewrite():
  class _T:  # shape-only stand-in
    def __init__(s, sh): s.shape = sh
  eq = _expand_ellipsis("...ij,...jk->...ik", [_T((2, 4, 5)), _T((2, 5, 3))])
  assert "..." not in eq and eq.count(",") == 1 and "->" in eq

def test_mismatched_ellipsis_dims_reject():
  class _T:
    def __init__(s, sh): s.shape = sh
  with pytest.raises(EinsumUnsupported):
    _expand_ellipsis("...i,...i->...", [_T((2, 4)), _T((3, 4))])
