"""Inverse Short-Time Fourier Transform (aneforge.dsp.istft) with overlap-add."""
import numpy as np
import pytest

import aneforge.dsp as dsp
from _helpers import requires_ane

pytestmark = requires_ane  # underlying transforms dispatch to the ANE
scipy_signal = pytest.importorskip("scipy.signal")


def _sig(n=2048, fs=1000.0):
  t = np.arange(n) / fs
  rng = np.random.default_rng(0)
  return (np.sin(2 * np.pi * 50.0 * t) + 0.5 * np.sin(2 * np.pi * 120.0 * t)
          + 0.1 * rng.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("win_len", [128, 256])
@pytest.mark.parametrize("window", ["hann", "hamming"])
def test_istft_roundtrip(win_len, window):
  x = _sig(2048)
  hop = win_len // 2
  Zr, Zi = dsp.stft(x, win=win_len, hop=hop, window=window)
  rec = dsp.istft(Zr, Zi, win=win_len, hop=hop, window=window)

  # Check interior samples where the overlap-add envelope is fully built
  interior = slice(win_len, min(len(x), len(rec)) - win_len)
  diff = np.max(np.abs(rec[interior] - x[interior]))
  relerr = diff / np.max(np.abs(x[interior]))
  assert relerr < 1e-2, f"win_len={win_len} window={window}: relerr {relerr:.2e}"


def test_istft_custom_array_window():
  x = _sig(1024)
  win = np.hanning(128).astype(np.float32)
  Zr, Zi = dsp.stft(x, win=win, hop=64)
  rec = dsp.istft(Zr, Zi, win=win, hop=64)
  interior = slice(128, min(len(x), len(rec)) - 128)
  err = np.max(np.abs(rec[interior] - x[interior])) / np.max(np.abs(x[interior]))
  assert err < 1e-2


def test_istft_rejects_shape_mismatch():
  zr = np.zeros((129, 10), dtype=np.float32)
  zi = np.zeros((129, 12), dtype=np.float32)
  with pytest.raises(ValueError, match="must be 2D with identical shapes"):
    dsp.istft(zr, zi)


def test_istft_rejects_non_2d():
  zr = np.zeros(129, dtype=np.float32)
  zi = np.zeros(129, dtype=np.float32)
  with pytest.raises(ValueError, match="must be 2D with identical shapes"):
    dsp.istft(zr, zi)
