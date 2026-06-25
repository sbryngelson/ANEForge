"""Zero-copy I/O views (input_view/output_view) produce identical results to the copy path (__call__/eval)."""
import warnings

import numpy as np
import pytest

import aneforge as af


def test_zero_copy_matches_call():
  warnings.simplefilter("ignore")
  rng = np.random.default_rng(0)
  W = (rng.standard_normal((16, 16, 3, 3)) * 0.05).astype(np.float32)
  x = af.input((1, 16, 8, 8))
  net = af.compile(af.conv(x, W, pad=1).relu())
  img = (rng.standard_normal((1, 16, 8, 8)) * 0.5).astype(np.float16)

  ref = np.asarray(net(img.astype(np.float32))).astype(np.float64)

  v = net.input_view()
  assert v.dtype == np.float16 and v.shape == (1, 16, 8, 8)
  np.copyto(v, img)                       # write directly into ANE-visible memory
  net.execute()
  got = np.asarray(net.output_view()).astype(np.float64)

  assert np.abs(ref - got).max() < 1e-2, f"zero-copy diverged: {np.abs(ref-got).max()}"


def test_input_view_is_persistent_and_writable():
  """The view aliases the persistent buffer: a second write + execute reflects the new input."""
  warnings.simplefilter("ignore")
  rng = np.random.default_rng(1)
  W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
  x = af.input((1, 8, 8, 8))
  net = af.compile(af.conv(x, W, pad=1).relu().mean((2, 3)).reshape(1, 8))
  v = net.input_view()

  a = (rng.standard_normal((1, 8, 8, 8)) * 0.5).astype(np.float16)
  b = (rng.standard_normal((1, 8, 8, 8)) * 0.5).astype(np.float16)
  np.copyto(v, a); net.execute(); out_a = np.array(net.output_view())
  np.copyto(v, b); net.execute(); out_b = np.array(net.output_view())
  assert not np.allclose(out_a, out_b), "different inputs gave identical outputs via the view"


def test_unknown_port_raises():
  x = af.input((1, 4))
  net = af.compile(x.relu())
  with pytest.raises(KeyError):
    net._prog.input_view("nope")
