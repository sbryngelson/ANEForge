"""Linear and constant detrending (aneforge.dsp.detrend) against scipy.signal.detrend."""
import numpy as np
import pytest

import aneforge.dsp as dsp

scipy_signal = pytest.importorskip("scipy.signal")


@pytest.mark.parametrize("n", [2, 17, 128, 512])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_detrend_constant_1d(n, dtype):
  rng = np.random.default_rng(n)
  x = (rng.standard_normal(n) + 10.0).astype(dtype)
  out = dsp.detrend(x, type="constant")
  ref = scipy_signal.detrend(x, type="constant")
  assert out.shape == x.shape and out.dtype == dtype
  tol = 1e-6 if dtype == np.float32 else 1e-12
  assert np.allclose(out, ref, atol=tol)


@pytest.mark.parametrize("n", [2, 17, 128, 512])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_detrend_linear_1d(n, dtype):
  rng = np.random.default_rng(n + 100)
  t = np.arange(n, dtype=dtype)
  x = (rng.standard_normal(n).astype(dtype) + 2.5 * t - 4.0)
  out = dsp.detrend(x, type="linear")
  ref = scipy_signal.detrend(x, type="linear")
  assert out.shape == x.shape and out.dtype == dtype
  atol = 5e-4 if dtype == np.float32 else 1e-11
  rtol = 1e-5 if dtype == np.float32 else 1e-11
  assert np.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("mode", ["constant", "linear"])
def test_detrend_2d_axes(axis, mode):
  rng = np.random.default_rng(42)
  x = rng.standard_normal((16, 32)).astype(np.float32)
  out = dsp.detrend(x, type=mode, axis=axis)
  ref = scipy_signal.detrend(x, type=mode, axis=axis)
  assert out.shape == x.shape and out.dtype == np.float32
  assert np.allclose(out, ref, atol=1e-5)


def test_detrend_3d_axes():
  rng = np.random.default_rng(123)
  x = rng.standard_normal((4, 7, 11)).astype(np.float64)
  for ax in (0, 1, 2, -1):
    for m in ("constant", "linear"):
      out = dsp.detrend(x, type=m, axis=ax)
      ref = scipy_signal.detrend(x, type=m, axis=ax)
      assert out.shape == x.shape and out.dtype == np.float64
      assert np.allclose(out, ref, atol=1e-12)


def test_detrend_pure_polynomial_residual():
  # Reference-free: fitting a pure line with linear detrend yields near 0.0 residual
  t = np.linspace(-5.0, 5.0, 100, dtype=np.float64)
  line = 3.5 * t + 1.2
  res_linear = dsp.detrend(line, type="linear")
  assert np.max(np.abs(res_linear)) < 1e-12

  # Fitting a constant with constant detrend yields 0.0 residual
  const_sig = np.full(50, 42.0, dtype=np.float64)
  res_const = dsp.detrend(const_sig, type="constant")
  assert np.max(np.abs(res_const)) < 1e-12


def test_detrend_edge_cases():
  # Empty array
  empty = np.array([], dtype=np.float32)
  assert dsp.detrend(empty, type="linear").shape == (0,)
  assert dsp.detrend(empty, type="constant").shape == (0,)

  # Single element
  single = np.array([42.0], dtype=np.float32)
  assert np.allclose(dsp.detrend(single, type="linear"), 0.0)
  assert np.allclose(dsp.detrend(single, type="constant"), 0.0)

  # Integer input converts to float64
  int_arr = [1, 2, 3, 4]
  out_int = dsp.detrend(int_arr, type="linear")
  assert out_int.dtype == np.float64
  assert np.allclose(out_int, 0.0)


def test_detrend_type_aliases():
  x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
  assert np.allclose(dsp.detrend(x, type="l"), dsp.detrend(x, type="linear"))
  assert np.allclose(dsp.detrend(x, type="c"), dsp.detrend(x, type="constant"))


def test_detrend_invalid_type_raises():
  x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  with pytest.raises(ValueError, match="detrend: type must be 'linear' or 'constant'"):
    dsp.detrend(x, type="quadratic")
