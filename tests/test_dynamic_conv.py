"""af.dynamic_conv: native conv with a runtime-tensor weight (batch-1 only)."""
from __future__ import annotations
import numpy as np
import pytest
from _helpers import requires_ane
import aneforge as af

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE


rng = np.random.default_rng(0)


def _npconv(x, w):
  Cout, Cin, kH, kW = w.shape
  Hout, Wout = x.shape[2] - kH + 1, x.shape[3] - kW + 1
  r = np.zeros((1, Cout, Hout, Wout), np.float32)
  for o in range(Cout):
    for i in range(Hout):
      for j in range(Wout):
        r[0, o, i, j] = np.sum(x[0, :, i:i+kH, j:j+kW] * w[o])
  return r


def _cos(a, b):
  a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_dynamic_conv_b2_guard_and_dtype():
  # batch>=2 rejected at build time
  with pytest.raises(ValueError, match="batch N=1"):
    af.dynamic_conv(af.input((2, 1, 8, 8)), af.input((8, 1, 3, 3)))
  # constant numpy weight is the wrong call (use af.conv)
  with pytest.raises(TypeError, match="must be a Tensor"):
    af.dynamic_conv(af.input((1, 1, 8, 8)), np.ones((8, 1, 3, 3), np.float32))
  assert af.dynamic_conv(af.input((1, 2, 8, 8)), af.input((4, 2, 3, 3))).shape == (1, 4, 6, 6)


@requires_ane
def test_dynamic_conv_fed_weight():
  xv = rng.standard_normal((1, 2, 8, 8)).astype(np.float16)
  wv = rng.standard_normal((4, 2, 3, 3)).astype(np.float16)
  net = af.compile(af.dynamic_conv(af.input((1, 2, 8, 8)), af.input((4, 2, 3, 3))))
  out = np.asarray(net(xv, wv)).reshape(1, 4, 6, 6)
  assert _cos(out, _npconv(xv.astype(np.float32), wv.astype(np.float32))) > 0.99


@requires_ane
def test_dynamic_conv_hypernetwork():
  # code vector generates the conv kernel on-engine (linear -> reshape -> conv)
  xv = rng.standard_normal((1, 2, 8, 8)).astype(np.float16)
  cv = rng.standard_normal((1, 8)).astype(np.float16)
  Wgen = (rng.standard_normal((8, 4 * 2 * 3 * 3)) * 0.1).astype(np.float32)
  x, code = af.input((1, 2, 8, 8)), af.input((1, 8))
  net = af.compile(af.dynamic_conv(x, (code @ Wgen).reshape(4, 2, 3, 3)))
  out = np.asarray(net(xv, cv)).reshape(1, 4, 6, 6)
  ref = _npconv(xv.astype(np.float32), (cv.astype(np.float32) @ Wgen).reshape(4, 2, 3, 3))
  assert _cos(out, ref) > 0.99
