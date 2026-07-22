"""2-D FFT on the ANE: aneforge.fft.fft2/ifft2 as one fused program per transform."""
from __future__ import annotations
import os
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pytest
import aneforge.fft as agfft


def relerr(a, b):
  a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


@pytest.mark.parametrize("shape", [(32, 32), (16, 48), (64, 64)])  # square + rectangular
def test_fft2_matches_numpy(shape):
  M, N = shape
  rng = np.random.default_rng(M * 100 + N)
  xr = rng.standard_normal((M, N)).astype(np.float32)
  xi = rng.standard_normal((M, N)).astype(np.float32)
  Xr, Xi = agfft.fft2(xr, xi)
  ref = np.fft.fft2(xr.astype(np.float64) + 1j * xi.astype(np.float64))
  assert relerr(Xr + 1j * Xi, ref) <= 2e-2


def test_fft2_real_field():
  # real input (imag fed as zeros): the PDE/Poisson use case
  rng = np.random.default_rng(3)
  f = rng.standard_normal((32, 32)).astype(np.float32)
  Xr, Xi = agfft.fft2(f, np.zeros_like(f))
  ref = np.fft.fft2(f.astype(np.float64))
  assert relerr(Xr + 1j * Xi, ref) <= 2e-2


def test_ifft2_roundtrip():
  # ifft2(fft2(x)) ~ x; also pins the 1/(M*N) normalization
  rng = np.random.default_rng(7)
  xr = rng.standard_normal((32, 32)).astype(np.float32)
  xi = rng.standard_normal((32, 32)).astype(np.float32)
  Xr, Xi = agfft.fft2(xr, xi)
  br, bi = agfft.ifft2(Xr, Xi)
  assert relerr(br + 1j * bi, xr + 1j * xi) <= 5e-2


def test_ifft2_matches_numpy():
  rng = np.random.default_rng(11)
  Xr = rng.standard_normal((16, 16)).astype(np.float32)
  Xi = rng.standard_normal((16, 16)).astype(np.float32)
  br, bi = agfft.ifft2(Xr, Xi)
  ref = np.fft.ifft2(Xr.astype(np.float64) + 1j * Xi.astype(np.float64))
  assert relerr(br + 1j * bi, ref) <= 2e-2


def test_ifft2_large_magnitude_spectrum():
  # large PDE spectrum: 1/(M*N) scaling must fold into the per-axis twiddles or the
  # intermediate after the first axis overflows fp16 (max 65504).
  N = 64
  x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
  X, Y = np.meshgrid(x, x, indexing="ij")
  u = np.sin(X) * np.cos(2 * Y) + 0.3 * np.cos(2 * X) * np.cos(2 * Y)
  spec = np.fft.fft2(u) * 8.0          # |values| up to ~1.6e4, inside fp16 range
  br, bi = agfft.ifft2(spec.real.astype(np.float32), spec.imag.astype(np.float32))
  ref = np.fft.ifft2(spec)
  assert np.all(np.isfinite(br)) and np.all(np.isfinite(bi))
  assert relerr(br + 1j * bi, ref) <= 2e-2


def test_fft2_plan_is_one_program_and_cached():
  p1 = agfft.fft2_plan(32, 32)
  p2 = agfft.fft2_plan(32, 32)
  assert p1 is p2                       # plan cache, like fft_plan
  assert p1.model.n_ops > 0             # one compiled e5rt program, not a host loop
