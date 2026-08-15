"""Pytest coverage for the einsum self-test case inventory."""
import numpy as np
import pytest

import importlib

import aneforge as af

einsum_mod = importlib.import_module("aneforge.einsum")


def _inputs(shapes):
  return [af.input(shape) for shape in shapes]


@pytest.mark.parametrize("name,equation,shapes", einsum_mod._SELFTEST_SUPPORTED)
def test_einsum_selftest_supported_cases_lower_without_dispatch(name, equation, shapes):
  """Every supported self-test equation should build a graph with numpy's shape."""
  tensors = _inputs(shapes)

  out = einsum_mod.einsum(equation, *tensors)
  ref = np.einsum(equation, *[np.zeros(shape, np.float32) for shape in shapes])

  expected_shape = tuple(np.shape(ref)) or (1,)
  assert tuple(out.shape) == expected_shape, name


@pytest.mark.parametrize("name,equation,shapes", einsum_mod._SELFTEST_REJECTED)
def test_einsum_selftest_rejected_cases_still_raise(name, equation, shapes):
  """Unsupported self-test equations should keep failing before dispatch."""
  tensors = _inputs(shapes)

  with pytest.raises((einsum_mod.EinsumUnsupported, ValueError, NotImplementedError)):
    einsum_mod.einsum(equation, *tensors)
