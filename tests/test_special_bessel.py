"""bessel_j1 and bessel_i1: off-device form checks + scipy-oracle accuracy (aneforge.special)."""
import inspect

import numpy as np

from aneforge import special


# -- form checks (off-device, CI-verifiable) ----------------------------------

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


# -- scipy-oracle accuracy (skipped if scipy absent) --------------------------

def test_bessel_j1_matches_scipy():
  sp = __import__("pytest").importorskip("scipy.special")
  import aneforge as af
  xs = np.linspace(0.0, 3.0, 128).astype(np.float16).reshape(1, -1)
  net = af.compile(special.bessel_j1(af.input(xs.shape)))
  out = np.asarray(net(xs)).astype(np.float32).ravel()
  ref = sp.j1(xs.astype(np.float32).ravel())
  rel = np.abs(out - ref) / (np.abs(ref) + 1e-6)
  assert rel.max() < 0.05, f"bessel_j1 relerr {rel.max():.3e} exceeds 5%"


def test_bessel_i1_matches_scipy():
  sp = __import__("pytest").importorskip("scipy.special")
  import aneforge as af
  xs = np.linspace(0.0, 3.75, 128).astype(np.float16).reshape(1, -1)
  net = af.compile(special.bessel_i1(af.input(xs.shape)))
  out = np.asarray(net(xs)).astype(np.float32).ravel()
  ref = sp.i1(xs.astype(np.float32).ravel())
  rel = np.abs(out - ref) / (np.abs(ref) + 1e-6)
  assert rel.max() < 0.05, f"bessel_i1 relerr {rel.max():.3e} exceeds 5%"
