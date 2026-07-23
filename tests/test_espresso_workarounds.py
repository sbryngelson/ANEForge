"""Regression tests for Espresso mis-lowering workarounds (#112, #113): correct MIL in, wrong
program out - so the emitters route around the fusion. Measured on M5 / macOS 26.5."""
import numpy as np
import pytest

import aneforge as af
from aneforge.graph import _const
from _helpers import requires_ane

pytestmark = requires_ane


def _run(out, xv):
  net = af.compile(out, opt=0, _check_precision=False)
  try: return np.asarray(net(xv), np.float64)
  finally: net.release()


def _round_half_away(x):                          # ANE round semantics (np.round is half-to-even)
  return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))

@pytest.mark.parametrize("uop,npop", [("floor", np.floor), ("ceil", np.ceil), ("round", _round_half_away)])
def test_rounding_then_scalar_mul_is_not_dropped(uop, npop):
  # Espresso fuses the scalar into the rounding op's scale slot and drops it (#112)
  xv = np.array([[11.2265625, -3.7, 2.5, -11.226]], np.float16)
  x = af.input((1, 4))
  got = _run(getattr(x, uop)() * -1.206, xv)
  ref = npop(xv.astype(np.float64)) * -1.206
  assert np.abs(got - ref).max() < 0.05, f"{uop}: {got.ravel()} != {ref.ravel()}"


def test_rounding_then_scalar_const_array_mul():
  # same fusion through the binary-mul path with a size-1 const_array operand
  xv = np.array([[11.2265625, -3.7, 2.5, -11.226]], np.float16)
  x = af.input((1, 4))
  got = _run(x.floor() * _const(np.asarray(-1.206, np.float16)), xv)
  ref = np.floor(xv.astype(np.float64)) * -1.206
  assert np.abs(got - ref).max() < 0.05


def test_rounding_then_scalar_add_still_plain():
  # adds after rounding was never affected; the workaround must not disturb it
  xv = np.array([[11.2265625, -3.7]], np.float16)
  x = af.input((1, 2))
  got = _run(x.floor() + 5.0, xv)
  assert np.array_equal(got, np.floor(xv.astype(np.float64)) + 5.0)


def test_softplus_large_inputs_do_not_collapse():
  # native ANE softplus returns 0 for x >= ln(65504) ~= 11.09 (#113); the stable split fixes it
  vals = np.array([[-30000.0, -12.0, -1.0, 0.0, 5.0, 10.0, 11.0, 12.0, 88.0, 30000.0]], np.float16)
  x = af.input((1, 10))
  got = _run(x.softplus(), vals)
  ref = np.logaddexp(0.0, vals.astype(np.float64))
  assert np.abs(got - ref).max() / max(1.0, np.abs(ref).max()) < 1e-3
  assert got.ravel()[-1] == 30000.0                # the old lowering returned exactly 0 here


def test_softplus_normal_range_unchanged():
  rng = np.random.default_rng(0)
  xv = (rng.standard_normal((1, 64)) * 3).astype(np.float16)
  x = af.input((1, 64))
  got = _run(x.softplus(), xv)
  ref = np.logaddexp(0.0, xv.astype(np.float64))
  assert np.abs(got - ref).max() < 1e-2
