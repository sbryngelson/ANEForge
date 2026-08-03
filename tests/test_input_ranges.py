"""Declared input activation ranges, and the bound propagation that uses them (#155).

Build-level: no dispatch, so these run in CI without an ANE. Over-estimating a bound is always safe
(a loose bound declines a lossy variant more often, never wrongly keeps one), so the assertions here
check "bounded and correct-side-of-the-ceiling", not tightness.
"""
import numpy as np
import pytest

import aneforge as af
from aneforge import _optimize
from aneforge._compile import _act_max_abs, _propagate_max_abs
from aneforge._targets import Q4_X16_SAT
from aneforge.graph import _const

W = (np.random.default_rng(0).standard_normal((15, 16)) / 4).astype(np.float16)


def _keeps_int8(out) -> bool:
  return any(c.get("int8") for c in _optimize._variants(out, drop_unsafe=True))


# -- the declaration ---------------------------------------------------- #

def test_undeclared_input_is_unbounded():
  assert _propagate_max_abs(af.input((4, 8))) is None

def test_declared_input_reports_its_bound():
  assert _propagate_max_abs(af.input((4, 8), max_abs=6.0)) == 6.0

def test_zero_is_a_valid_bound():
  assert _propagate_max_abs(af.input((4, 8), max_abs=0.0)) == 0.0

@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_invalid_bound_rejects(bad):
  with pytest.raises(ValueError, match="non-negative"):
    af.input((4, 8), max_abs=bad)

def test_declaration_does_not_disturb_input_ordering():
  """`idx` drives feed order, so declaring a range must not shift it."""
  a, b = af.input((2, 2)), af.input((2, 2), max_abs=1.0)
  assert b.attrs["idx"] == a.attrs["idx"] + 1


# -- propagation -------------------------------------------------------- #

def test_nonexpanding_ops_carry_the_bound():
  x = af.input((4, 8), max_abs=6.0)
  for t in (x.relu(), x.abs(), x.reshape(8, 4), x.transpose([1, 0])):
    assert _propagate_max_abs(t) == 6.0

def test_saturating_activations_bound_an_undeclared_input():
  """The useful case: these are provably bounded whatever feeds them."""
  x = af.input((4, 8))
  assert _propagate_max_abs(x.sigmoid()) == 1.0
  assert _propagate_max_abs(x.tanh()) == 1.0
  assert _propagate_max_abs(x.softmax(-1)) == 1.0
  assert _propagate_max_abs(x.relu6()) == 6.0

def test_clip_bounds_by_its_limits():
  assert _propagate_max_abs(af.input((4, 8)).clip(-3.0, 2.0)) == 3.0

def test_scalar_ops_scale_and_shift_the_bound():
  x = af.input((4, 8), max_abs=6.0)
  assert _propagate_max_abs(x * 2.0) == 12.0
  assert _propagate_max_abs(x.adds(1.5)) == 7.5

def test_add_sums_both_bounds():
  x, y = af.input((4, 8), max_abs=6.0), af.input((4, 8), max_abs=2.0)
  assert _propagate_max_abs(x + y) == 8.0

def test_one_unbounded_operand_makes_the_sum_unbounded():
  assert _propagate_max_abs(af.input((4, 8), max_abs=6.0) + af.input((4, 8))) is None

def test_matmul_bound_uses_the_weight_row_sums():
  """|x @ W| <= max|x| * max_n sum_k |W[k,n]|, an upper bound, so it must not undershoot."""
  x = af.input((4, 15), max_abs=2.0)
  got = _propagate_max_abs(x @ W)
  worst = 2.0 * float(np.abs(W.astype(np.float32)).sum(axis=0).max())
  assert got is not None and got >= worst - 1e-3

def test_unmodelled_op_returns_none_rather_than_guessing():
  assert _propagate_max_abs(af.input((4, 8), max_abs=1.0).exp()) is None


# -- what it unblocks (the point of #155) ------------------------------- #

def test_declared_safe_range_keeps_int8_at_opt1():
  assert _keeps_int8(af.input((31, 15), max_abs=6.0) @ W)

def test_declared_range_above_the_ceiling_still_declines():
  assert not _keeps_int8(af.input((31, 15), max_abs=1e5) @ W)

def test_undeclared_input_still_declines():
  """The fail-closed default from #153 must survive: no declaration, no int8."""
  assert not _keeps_int8(af.input((31, 15)) @ W)

def test_saturating_op_recovers_int8_without_any_declaration():
  assert _keeps_int8(af.input((31, 15)).softmax(-1) @ W)

@pytest.mark.parametrize("declared,expect", [(Q4_X16_SAT, True), (Q4_X16_SAT + 2.0, False)])
def test_ceiling_boundary_on_a_declared_range(declared, expect):
  """A declaration exactly at the ceiling is safe; the next fp16 representable above is not."""
  assert _keeps_int8(af.input((31, 15), max_abs=declared)) is not None  # graph builds
  x = af.input((31, 15), max_abs=declared)
  assert (_act_max_abs(x @ np.eye(15, 16, dtype=np.float16)) <= Q4_X16_SAT) is expect


def test_const_fed_graph_still_bounded():
  assert _act_max_abs(_const((np.ones((31, 15)) * 10.0).astype(np.float16)) @ W) is not None
