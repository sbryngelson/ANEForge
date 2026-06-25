"""Build-time guards on the graph builders (rms_norm rank, conv/conv_transpose kW<=15)."""
import numpy as np
import pytest

import aneforge as af


def test_rms_norm_rank3_raises():
  x = af.input((2, 3, 8))
  with pytest.raises(ValueError) as e:
    x.rms_norm(np.ones(8, np.float32))
  assert "rms_norm expects 2D [M,D]" in str(e.value)
  assert "(2, 3, 8)" in str(e.value)


def test_rms_norm_rank2_builds():
  x = af.input((2, 8))
  t = x.rms_norm(np.ones(8, np.float32))
  assert t.op == "rms_norm"
  assert t.shape == (2, 8)


def test_conv_transpose_kw16_raises():
  x = af.input((1, 4, 8, 8))
  W = np.zeros((4, 3, 2, 16), np.float16)  # [Cin, Cout, kH, kW]
  with pytest.raises(ValueError) as e:
    af.conv_transpose(x, W, stride=2)
  assert "kW=16" in str(e.value)
  assert "<=15" in str(e.value)


def test_conv_transpose_kw15_builds():
  x = af.input((1, 4, 8, 20))
  W = np.zeros((4, 3, 2, 15), np.float16)  # [Cin, Cout, kH, kW]
  t = af.conv_transpose(x, W, stride=1)
  assert t.op == "conv_transpose"
  assert t.shape == (1, 3, 9, 34)


def test_conv_kw16_still_raises():
  x = af.input((1, 4, 8, 20))
  W = np.zeros((3, 4, 1, 16), np.float16)  # [Cout, Cin, kH, kW]
  with pytest.raises(ValueError) as e:
    af.conv(x, W)
  assert "kW=16" in str(e.value)
