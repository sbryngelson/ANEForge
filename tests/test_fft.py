"""1-D FFT family on the ANE: aneforge.fft fft/ifft/rfft/irfft, checked against numpy."""
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


@pytest.mark.parametrize("N", [64, 256, 1024])
def test_fft_matches_numpy(N):
  rng = np.random.default_rng(N)
  xr = rng.standard_normal(N).astype(np.float32)
  xi = rng.standard_normal(N).astype(np.float32)
  Xr, Xi = agfft.fft(xr, xi, N)
  ref = np.fft.fft(xr.astype(np.float64) + 1j * xi.astype(np.float64))
  assert relerr(Xr + 1j * Xi, ref) <= 2e-2


def test_ifft_roundtrip():
  rng = np.random.default_rng(7)
  xr = rng.standard_normal(256).astype(np.float32)
  xi = rng.standard_normal(256).astype(np.float32)
  Xr, Xi = agfft.fft(xr, xi, 256)
  br, bi = agfft.ifft(Xr, Xi, 256)
  assert relerr(br + 1j * bi, xr + 1j * xi) <= 5e-2


def test_rfft_matches_numpy():
  rng = np.random.default_rng(5)
  x = rng.standard_normal(1024).astype(np.float32)
  Rr, Ri = agfft.rfft(x, 1024)
  ref = np.fft.fft(x.astype(np.float64))
  assert relerr(Rr + 1j * Ri, ref) <= 2e-2


@pytest.mark.parametrize("N", [128, 512])
def test_irfft_roundtrip(N):
  # irfft(rfft(x)) reconstructs the real signal; numpy oracle via the half spectrum
  rng = np.random.default_rng(N)
  x = rng.standard_normal(N).astype(np.float32)
  Xr, Xi = agfft.rfft(x, N)
  back = agfft.irfft(Xr, Xi, N)
  assert relerr(back, x) <= 5e-2
  ref = np.fft.irfft(np.asarray(Xr, np.float32)[: N // 2 + 1] + 1j * np.asarray(Xi, np.float32)[: N // 2 + 1], n=N)
  assert relerr(back, ref) <= 5e-2


def test_irfft_returns_real_only():
  # a Hermitian-symmetric spectrum reconstructs a real signal; the returned array is real
  rng = np.random.default_rng(13)
  x = rng.standard_normal(256).astype(np.float32)
  Xr, Xi = agfft.rfft(x, 256)
  back = agfft.irfft(Xr, Xi, 256)
  assert np.iscomplexobj(np.asarray(back)) is False
