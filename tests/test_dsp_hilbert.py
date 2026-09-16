"""Analytic signal (aneforge.dsp.hilbert) against scipy.signal.hilbert."""
import numpy as np
import pytest

import aneforge.dsp as dsp
from _helpers import requires_ane

pytestmark = requires_ane  # hilbert dispatches its FFTs to the ANE
scipy_signal = pytest.importorskip("scipy.signal")


# powers of two, even non-powers of two, and ODD lengths (which need scipy's other filter branch)
@pytest.mark.parametrize("N", [256, 512, 1024, 300, 384, 255, 101])
def test_hilbert_matches_scipy(N):
  rng = np.random.default_rng(20260530)
  x = rng.standard_normal(N).astype(np.float32)
  z_re, z_im = dsp.hilbert(x)
  ref = scipy_signal.hilbert(x.astype(np.float64))
  assert z_re.shape == (N,), f"re shape {z_re.shape} != ({N},)"
  assert z_im.shape == (N,), f"im shape {z_im.shape} != ({N},)"
  # real part should match the input (analytic signal keeps the signal as its real part)
  re_err = float(np.linalg.norm(z_re - x.astype(np.float64)) / (np.linalg.norm(x.astype(np.float64)) + 1e-30))
  assert re_err < 5e-3, f"real part relerr {re_err:.3e}"
  # imaginary part (Hilbert transform) against scipy
  im_err = float(np.linalg.norm(z_im - ref.imag) / (np.linalg.norm(ref.imag) + 1e-30))
  assert im_err < 5e-3, f"imag part relerr {im_err:.3e}"


def test_hilbert_sine_gives_minus_cosine():
  """Analytic signal of sin(t): the Hilbert transform is -cos(t) (phase shift -90 deg)."""
  N = 256
  t = np.arange(N, dtype=np.float32) * (2 * np.pi / N)
  x = np.sin(t).astype(np.float32)
  _, z_im = dsp.hilbert(x)
  ref = -np.cos(t.astype(np.float64))
  err = float(np.linalg.norm(z_im - ref) / (np.linalg.norm(ref) + 1e-30))
  assert err < 5e-3, f"sin->-cos analytic signal relerr {err:.3e}"

