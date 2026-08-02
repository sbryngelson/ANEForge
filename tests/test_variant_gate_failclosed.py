"""The variant accuracy gate must fail closed on a non-computable error (#153).

`relerr > tol` lets `nan` through, because every comparison with `nan` is False. A lossy variant
whose output saturated, or whose error could not be computed, would then be accepted on the strength
of an uncomputable number. `not (relerr <= tol)` rejects nan and inf alike.

Build-level and arithmetic only, so these run in CI without an ANE.
"""
import inspect

import numpy as np

from aneforge import _optimize


def _accepts(relerr: float, tol: float = 5e-3) -> bool:
  """The gate's decision, expressed the way the fixed code expresses it."""
  return bool(relerr <= tol)


def test_nan_is_rejected():
  """The bug: `nan > tol` is False, so the old form accepted it."""
  assert not (float("nan") > 5e-3), "documents why a `>` test is unsafe"
  assert not _accepts(float("nan"))

def test_inf_is_rejected():
  """A variant that saturated to inf has relerr inf against a finite baseline."""
  ref = np.array([[17321.0, 100.0]], np.float64)
  cur = np.array([[np.inf, 100.0]], np.float64)
  relerr = float(np.abs(cur - ref).max() / (float(np.abs(ref).max()) + 1e-9))
  assert relerr == float("inf")
  assert not _accepts(relerr)

def test_within_tolerance_still_accepted():
  assert _accepts(1e-4)
  assert _accepts(5e-3), "the boundary stays inclusive, as before"

def test_above_tolerance_still_rejected():
  assert not _accepts(5e-2)


def test_gate_source_is_fail_closed():
  """Pin the form itself: a future edit back to `relerr > tol` reintroduces the nan hole."""
  src = inspect.getsource(_optimize.measure)
  assert "not (relerr <= tol)" in src, "the accuracy gate must reject nan, not just large errors"
