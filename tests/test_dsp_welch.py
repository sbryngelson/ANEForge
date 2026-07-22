"""Welch PSD (aneforge.dsp.welch) against scipy.signal.welch."""
import numpy as np
import pytest

import aneforge.dsp as dsp
from _helpers import requires_ane

pytestmark = requires_ane  # the underlying stft dispatches its FFTs to the ANE
scipy_signal = pytest.importorskip("scipy.signal")


def _sig(n=4096, fs=1000.0):
  t = np.arange(n) / fs
  rng = np.random.default_rng(0)
  return (np.sin(2 * np.pi * 50.0 * t) + 0.5 * np.sin(2 * np.pi * 120.0 * t)
          + 0.1 * rng.standard_normal(n)).astype(np.float32), fs


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
def test_welch_matches_scipy(scaling):
  x, fs = _sig()
  f, P = dsp.welch(x, fs=fs, nperseg=256, scaling=scaling)
  fr, Pr = scipy_signal.welch(x, fs=fs, nperseg=256, detrend=False, scaling=scaling)
  assert np.allclose(f, fr)
  err = np.abs(P - Pr).max() / (np.abs(Pr).max() + 1e-12)
  assert err <= 2e-2, f"scaling={scaling}: relerr {err:.2e}"


def test_welch_finds_the_tones():
  x, fs = _sig()
  f, P = dsp.welch(x, fs=fs, nperseg=256)
  for tone in (50.0, 120.0):
    k = np.argmin(np.abs(f - tone))
    assert P[k] > 10 * np.median(P)                     # the tones stand well out of the noise floor

def test_welch_rejects_non_pow2_nperseg():
  x, fs = _sig(1024)
  with pytest.raises(ValueError):
    dsp.welch(x, fs=fs, nperseg=200)
