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
