"""Spectral analysis tools (aneforge.dsp periodogram, csd, coherence) against scipy.signal."""
import numpy as np
import pytest

import aneforge.dsp as dsp
from _helpers import requires_ane

pytestmark = requires_ane  # transforms dispatch to the ANE
scipy_signal = pytest.importorskip("scipy.signal")


def _sig(n=2048, fs=1000.0):
  t = np.arange(n) / fs
  rng = np.random.default_rng(0)
  return (np.sin(2 * np.pi * 50.0 * t) + 0.5 * np.sin(2 * np.pi * 120.0 * t)
          + 0.1 * rng.standard_normal(n)).astype(np.float32), fs


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize("window", ["boxcar", "hann", "hamming"])
def test_periodogram_matches_scipy(scaling, window):
  x, fs = _sig(1024, fs=500.0)
  f, P = dsp.periodogram(x, fs=fs, window=window, scaling=scaling)
  fr, Pr = scipy_signal.periodogram(x, fs=fs, window=window, detrend=False, scaling=scaling)
  assert np.allclose(f, fr)
  err = np.abs(P - Pr).max() / (np.abs(Pr).max() + 1e-12)
  assert err <= 2e-2, f"scaling={scaling} window={window}: relerr {err:.2e}"


def test_periodogram_finds_peaks():
  x, fs = _sig(2048, fs=1000.0)
  f, P = dsp.periodogram(x, fs=fs, window="hann")
  for tone in (50.0, 120.0):
    k = np.argmin(np.abs(f - tone))
    assert P[k] > 10 * np.median(P)


def test_periodogram_rejects_non_pow2():
  x, fs = _sig(500)
  with pytest.raises(ValueError, match="N must be a power of two"):
    dsp.periodogram(x, fs=fs)


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
def test_csd_matches_scipy(scaling):
  rng = np.random.default_rng(42)
  x = rng.standard_normal(2048).astype(np.float32)
  y = rng.standard_normal(2048).astype(np.float32)
  fs = 200.0
  f, Pxy = dsp.csd(x, y, fs=fs, nperseg=256, scaling=scaling)
  fr, Pxy_ref = scipy_signal.csd(x, y, fs=fs, nperseg=256, detrend=False, scaling=scaling)
  assert np.allclose(f, fr)
  err = np.abs(Pxy - Pxy_ref).max() / (np.abs(Pxy_ref).max() + 1e-12)
  assert err <= 2e-2, f"csd scaling={scaling}: relerr {err:.2e}"


def test_csd_rejects_non_pow2():
  x, fs = _sig(1024)
  with pytest.raises(ValueError, match="nperseg must be a power of two"):
    dsp.csd(x, x, fs=fs, nperseg=150)


def test_coherence_matches_scipy():
  rng = np.random.default_rng(99)
  x = rng.standard_normal(2048).astype(np.float32)
  y = rng.standard_normal(2048).astype(np.float32)
  fs = 100.0
  f, Cxy = dsp.coherence(x, y, fs=fs, nperseg=256)
  fr, Cxy_ref = scipy_signal.coherence(x, y, fs=fs, nperseg=256, detrend=False)
  assert np.allclose(f, fr)
  assert np.all(Cxy >= 0.0) and np.all(Cxy <= 1.0)
  err = np.abs(Cxy - Cxy_ref).max()
  assert err <= 2e-2, f"coherence max abs diff {err:.2e}"


def test_coherence_self_is_unity():
  x, fs = _sig(2048)
  f, Cxx = dsp.coherence(x, x, fs=fs, nperseg=256)
  assert np.allclose(Cxx, 1.0, atol=1e-3)
