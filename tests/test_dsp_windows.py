"""Window functions (aneforge.dsp) against scipy.signal.windows. Host-side, no ANE."""
import numpy as np
import pytest

import aneforge.dsp as dsp

sw = pytest.importorskip("scipy.signal.windows")


@pytest.mark.parametrize("M", [16, 65])
@pytest.mark.parametrize("sym", [False, True])
def test_bartlett_matches_scipy(M, sym):
  assert np.allclose(dsp.bartlett(M, sym=sym), sw.bartlett(M, sym=sym), atol=1e-6)


@pytest.mark.parametrize("M", [16, 65])
@pytest.mark.parametrize("sym", [False, True])
def test_tukey_matches_scipy(M, sym):
  assert np.allclose(dsp.tukey(M, sym=sym), sw.tukey(M, sym=sym), atol=1e-6)


@pytest.mark.parametrize("beta", [6.0, 14.0])
@pytest.mark.parametrize("M", [16, 65])
@pytest.mark.parametrize("sym", [False, True])
def test_kaiser_matches_scipy(M, sym, beta):
  assert np.allclose(dsp.kaiser(M, beta=beta, sym=sym), sw.kaiser(M, beta=beta, sym=sym), atol=1e-6)


@pytest.mark.parametrize("M", [1, 16, 65])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("name", ["flattop", "blackmanharris", "nuttall", "cosine"])
def test_cosine_family_matches_scipy(M, sym, name):
  assert np.allclose(getattr(dsp, name)(M, sym=sym), getattr(sw, name)(M, sym=sym), atol=1e-6)


@pytest.mark.parametrize("std", [2.0, 7.0])
@pytest.mark.parametrize("M", [1, 16, 65])
@pytest.mark.parametrize("sym", [False, True])
def test_gaussian_matches_scipy(M, sym, std):
  assert np.allclose(dsp.gaussian(M, std=std, sym=sym), sw.gaussian(M, std=std, sym=sym), atol=1e-6)


def test_get_window_resolves_new_names():
  for name in ("kaiser", "bartlett", "tukey", "flattop", "blackmanharris",
               "nuttall", "cosine", "gaussian"):
    w = dsp.get_window(name, 32)
    assert w.shape == (32,) and w.dtype == np.float32


def test_get_window_array_path_carries_beta():
  w = dsp.get_window(dsp.kaiser(48, beta=6.0), 48)
  assert np.allclose(w, sw.kaiser(48, beta=6.0, sym=False), atol=1e-6)
