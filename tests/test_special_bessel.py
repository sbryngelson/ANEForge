"""bessel_j1 and bessel_i1 formulation checks, off-device so CI runs them."""
import inspect

import numpy as np

from aneforge import special


def test_bessel_j1_uses_own_coefficients():
  # j1 must use _J1, not _J0 (j0's coefficients).
  src = inspect.getsource(special.bessel_j1)
  assert "_J1" in src
  assert "_J0" not in src


def test_bessel_i1_uses_own_coefficients():
  # i1 must use _I1, not _I0 (i0's coefficients).
  src = inspect.getsource(special.bessel_i1)
  assert "_I1" in src
  assert "_I0" not in src


def test_bessel_j1_leading_coefficient_is_one():
  # The polynomial is for J1(x)/(x/2), so the leading coefficient must be 1.0.
  assert np.isclose(special._J1[0], 1.0)


def test_bessel_i1_leading_coefficient_is_one():
  # The polynomial is for I1(x)/(x/2), so the leading coefficient must be 1.0.
  assert np.isclose(special._I1[0], 1.0)


def test_bessel_j1_is_exported():
  assert "bessel_j1" in special.__all__


def test_bessel_i1_is_exported():
  assert "bessel_i1" in special.__all__