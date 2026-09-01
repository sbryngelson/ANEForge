"""sin/cos as fused fp16 polynomials + the cos-free _const fix (aneforge.special)."""
import numpy as np

import aneforge as af
from aneforge import special
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE


def _run(fn, xs):
  net = af.compile(fn(af.input(xs.shape)))
  return np.asarray(net(xs.astype(np.float16))).astype(np.float32)


def test_const_does_not_use_cos():
  # the source must not build the constant out of cos (the A15+ op)
  import inspect
  src = inspect.getsource(special._const)
  assert ".cos()" not in src


def test_const_yields_the_constant_on_device():
  x = np.linspace(-2, 2, 64).astype(np.float16).reshape(1, 64)
  out = _run(lambda t: special._const(t, 3.0), x)
  assert np.allclose(out, 3.0, atol=2e-3)


def test_sin_decomposition_matches_numpy():
  x = np.linspace(-np.pi / 2, np.pi / 2, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.sin, x)
  assert np.abs(out - np.sin(x.astype(np.float32))).max() < 3e-3


def test_cos_decomposition_matches_numpy():
  x = np.linspace(-np.pi / 2, np.pi / 2, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.cos, x)
  assert np.abs(out - np.cos(x.astype(np.float32))).max() < 3e-3


def test_expm1_still_works_after_const_change():
  # regression: expm1 builds entirely on _const; it must survive the cos->exp swap
  x = np.linspace(-0.7, 0.7, 64).astype(np.float16).reshape(1, 64)
  out = _run(special.expm1, x)
  ref = np.expm1(x.astype(np.float32))
  assert np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6) < 5e-2


def test_log1p_still_works_after_const_change():
  x = np.linspace(-0.5, 1.0, 64).astype(np.float16).reshape(1, 64)
  out = _run(special.log1p, x)
  ref = np.log1p(x.astype(np.float32))
  assert np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6) < 5e-2


def test_erf_matches_scipy_on_device():
  import scipy.special as sp
  x = np.linspace(-2.0, 2.0, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.erf, x)
  assert np.abs(out - sp.erf(x.astype(np.float32))).max() < 3e-3


def test_erf_is_accurate_near_zero_where_one_minus_erfc_is_not():
  # erf's whole reason to exist as its own function: 1 - erfc(x) cancels for
  # small x, where erf is small and erfc -> 1. Per-point relative error, on a
  # geometric grid so small x is actually sampled.
  import scipy.special as sp
  x = np.geomspace(1e-3, 1.0, 64).astype(np.float16).reshape(1, 64)
  ref = sp.erf(x.astype(np.float32))
  direct = _run(special.erf, x)
  naive = 1.0 - _run(special.erfc, x)
  direct_err = np.abs((direct - ref) / ref).max()
  naive_err = np.abs((naive - ref) / ref).max()
  assert direct_err < 5e-3, direct_err
  # the naive form is off by more than 100% somewhere on this grid
  assert naive_err > 1.0, naive_err
  assert direct_err < naive_err / 100


def test_erf_is_odd_and_exactly_zero_at_the_origin():
  x = np.linspace(-0.5, 0.5, 65).astype(np.float16).reshape(1, 65)  # odd count -> hits 0
  out = _run(special.erf, x)
  assert (np.abs(x[0]) < 1e-6).any(), "grid must contain 0"
  assert np.abs(out[0][np.abs(x[0]) < 1e-6]).max() == 0.0
  assert np.abs(out[0] + out[0][::-1]).max() < 1e-3


def test_sinh_decomposition_matches_numpy():
  x = np.linspace(-10.0, 10.0, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.sinh, x)
  ref = np.sinh(x.astype(np.float32))
  assert np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6) < 5e-3


def test_cosh_decomposition_matches_numpy():
  x = np.linspace(-10.0, 10.0, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.cosh, x)
  ref = np.cosh(x.astype(np.float32))
  assert np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6) < 5e-3


def test_asinh_decomposition_matches_numpy():
  x = np.linspace(-10.0, 10.0, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.asinh, x)
  ref = np.arcsinh(x.astype(np.float32))
  assert np.abs(out - ref).max() < 1e-1


def test_acosh_decomposition_matches_numpy():
  x = np.linspace(1.0001, 10.0, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.acosh, x)
  ref = np.arccosh(x.astype(np.float32))
  assert np.abs(out - ref).max() < 5e-3


def test_atanh_decomposition_matches_numpy():
  x = np.linspace(-0.99, 0.99, 128).astype(np.float16).reshape(1, 128)
  out = _run(special.atanh, x)
  ref = np.arctanh(x.astype(np.float32))
  assert np.abs(out - ref).max() < 3e-3
