"""Diagonal extraction in aneforge.einsum: a doubly-repeated operand index via mask-and-reduce."""
import numpy as np
import pytest

import aneforge as af
from aneforge.einsum import einsum, EinsumUnsupported, _diag_pairs, _extract_diagonals
from _helpers import requires_ane


def _check(eq, *shapes, seed=0, tol=5e-3):
  """Compare against np.einsum on the ANE (fp32 reference, relative max error)."""
  rng = np.random.default_rng(seed)
  vals = [rng.standard_normal(s).astype(np.float16) for s in shapes]
  out = einsum(eq, *[af.input(s) for s in shapes])
  got = np.asarray(af.compile(out)(*vals)).astype(np.float32)
  ref = np.atleast_1d(np.einsum(eq, *[v.astype(np.float32) for v in vals]))
  got = got.reshape(ref.shape)
  err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6)
  assert err <= tol, f"{eq}: relerr {err:.2e}"
  return got, ref


# -- shape/rewrite level (off-device) ----------------------------------- #

def test_diag_pairs_finds_the_repeated_pair():
  assert _diag_pairs("ii") == {"i": (0, 1)}
  assert _diag_pairs("bii") == {"i": (1, 2)}
  assert _diag_pairs("ibi") == {"i": (0, 2)}
  assert _diag_pairs("iijj") == {"i": (0, 1), "j": (2, 3)}
  assert _diag_pairs("ijk") == {}

def test_extract_drops_one_axis_per_repeated_index():
  t, sub = _extract_diagonals(af.input((5, 5)), "ii")
  assert sub == "i" and t.shape == (5,)
  t, sub = _extract_diagonals(af.input((3, 3, 4, 4)), "iijj")
  assert sub == "ij" and t.shape == (3, 4)

def test_extract_is_a_noop_without_repeats():
  x = af.input((3, 4))
  t, sub = _extract_diagonals(x, "ij")
  assert t is x and sub == "ij"

@pytest.mark.parametrize("eq,shapes,want", [
  ("ii->i",     [(5, 5)],           (5,)),
  ("bii->bi",   [(3, 5, 5)],        (3, 5)),
  ("ibi->bi",   [(4, 3, 4)],        (3, 4)),
  ("iijj->ij",  [(3, 3, 4, 4)],     (3, 4)),
  ("ii,ij->j",  [(5, 5), (5, 6)],   (6,)),
  ("ii,jj->ij", [(4, 4), (5, 5)],   (4, 5)),
])
def test_output_shape_matches_numpy(eq, shapes, want):
  assert einsum(eq, *[af.input(s) for s in shapes]).shape == want


# -- rejects that must survive the rewrite ------------------------------ #

def test_triple_repeat_rejects():
  with pytest.raises(EinsumUnsupported, match="appears 3 times"):
    einsum("iii->i", af.input((4, 4, 4)))

def test_diagonal_write_still_rejects():
  with pytest.raises(EinsumUnsupported, match="repeated output index"):
    einsum("ij->ii", af.input((5, 5)))

def test_non_square_repeated_index_rejects():
  with pytest.raises(ValueError, match="spans dims"):
    einsum("ii->i", af.input((5, 6)))


# -- numerics on the ANE ------------------------------------------------ #

@requires_ane
@pytest.mark.parametrize("eq,shapes", [
  ("ii->i",             [(5, 5)]),
  ("ii->",              [(5, 5)]),
  ("bii->bi",           [(3, 5, 5)]),
  ("bii->b",            [(3, 5, 5)]),
  ("ibi->bi",           [(4, 3, 4)]),
  ("iijj->ij",          [(3, 3, 4, 4)]),
  ("ii,ij->j",          [(5, 5), (5, 6)]),
  ("ii,jj->ij",         [(4, 4), (5, 5)]),
  ("bii,bij->bj",       [(2, 4, 4), (2, 4, 5)]),
])
def test_matches_numpy_on_ane(eq, shapes):
  _check(eq, *shapes)

@requires_ane
def test_pure_diagonal_is_bit_exact():
  """The identity mask leaves exactly one survivor per output element, so the reduce adds only
  zeros: extraction introduces no fp16 rounding of its own."""
  for eq, shape in [("ii->i", (6, 6)), ("bii->bi", (3, 5, 5)), ("ibi->bi", (4, 3, 4))]:
    got, ref = _check(eq, shape)
    assert np.array_equal(got, ref), f"{eq}: expected bit-exact diagonal, max |d| {np.abs(got-ref).max()}"

@requires_ane
def test_trace_equals_numpy_trace():
  rng = np.random.default_rng(3)
  a = rng.standard_normal((8, 8)).astype(np.float16)
  got = float(np.asarray(af.compile(einsum("ii->", af.input((8, 8))))(a)).reshape(()))
  ref = float(np.trace(a.astype(np.float32)))
  assert abs(got - ref) / (abs(ref) + 1e-6) <= 5e-3, f"trace {got} vs {ref}"

@requires_ane
def test_diagonal_of_known_matrix():
  """A matrix whose diagonal is 1..n and whose off-diagonal is large: catches an extraction that
  leaks off-diagonal mass (a wrong axis, or a mask that is not the identity)."""
  n = 6
  a = np.full((n, n), 100.0, np.float16)
  a[np.diag_indices(n)] = np.arange(1, n + 1)
  got = np.asarray(af.compile(einsum("ii->i", af.input((n, n))))(a)).astype(np.float32).reshape(n)
  assert np.array_equal(got, np.arange(1, n + 1, dtype=np.float32)), got
