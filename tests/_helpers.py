"""Shared test helpers (plain importable module, NOT a test file)."""
from __future__ import annotations

import numpy as np
import pytest

import aneforge as af


def f16(rng, *shape, scale=1.0, pos=False):
  """Random fp16 tensor of the given shape; scale multiplies a normal, pos makes values positive via abs(.)+0.5."""
  a = rng.standard_normal(shape).astype(np.float32) * scale
  if pos:
    a = np.abs(a) + 0.5
  return a.astype(np.float16)


def onehot_select(t: af.Tensor, i: int, w: int | None = None) -> af.Tensor:
  """Select element i of a [1, W] tensor as [1, 1] via a folded one-hot column matmul (stays fused)."""
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
