"""Shared test helpers (plain importable module, NOT a test file).

  Consolidates fixtures that had drifted across the suite: the fp16 random-input
  factory, the one-hot static-index column selector, and the ANE-availability skip
  marker. Importing this module has no side effects beyond importing numpy/pytest.
  """
from __future__ import annotations

import numpy as np
import pytest

import aneforge as af


def f16(rng, *shape, scale=1.0, pos=False):
  """Random fp16 tensor of the given shape, drawn from ``rng`` (a numpy Generator).

    Superset of the per-file variants: ``scale`` multiplies a standard normal; when
    ``pos`` is set the values are made strictly positive via ``abs(.) + 0.5`` (the
    behaviour of the files that took a ``pos`` flag).
    """
  a = rng.standard_normal(shape).astype(np.float32) * scale
  if pos:
    a = np.abs(a) + 0.5
  return a.astype(np.float16)


def onehot_select(t: af.Tensor, i: int, w: int | None = None) -> af.Tensor:
  """Select element ``i`` of a ``[1, W]`` tensor as a ``[1, 1]`` tensor via a matmul
    against a folded one-hot column selector (a static-index trick that stays fused -
    no dynamic_slice/gather, so it does not cut the graph)."""
  W = w or t.shape[-1]
  sel = np.zeros((W, 1), np.float16)
  sel[i, 0] = 1.0
  return t @ sel.astype(np.float16)


def ane_available() -> bool:
  """True iff the ANE/e5rt dispatch dylib can be located (device tests can run)."""
  try:
    from aneforge._runtime import _find_dylib
    _find_dylib()
    return True
  except Exception:
    return False


requires_ane = pytest.mark.skipif(not ane_available(), reason="ANE/e5rt dylib unavailable")
