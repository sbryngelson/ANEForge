"""erf's formulation, asserted off-device so CI runs it (test_special_trig is requires_ane)."""
import inspect

import numpy as np

from aneforge import special


def test_erf_is_not_built_as_one_minus_erfc():
  # 1 - erfc(x) cancels exactly where erf is wanted; erf must be its own
  # polynomial. Guards against a future "simplification" back to the naive form.
  src = inspect.getsource(special.erf)
  body = src.split('"""')[-1]        # skip the docstring, which mentions erfc on purpose
  assert "erfc" not in body
  assert "_ERF_P" in body


def test_erf_leading_coefficient_recovers_two_over_sqrt_pi():
  # erf(x) = 2/sqrt(pi) * x + O(x^3), so the low-order coefficient of the
  # erf(x)/x polynomial must be 2/sqrt(pi); catches a bad refit.
  assert np.isclose(special._ERF_P[0], 2.0 / np.sqrt(np.pi), rtol=1e-3)


def test_erf_is_exported():
  assert "erf" in special.__all__
