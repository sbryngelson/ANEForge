"""compile(target=...) - gate lowering on the target ANE family (decompose / reject / saturation / preflight)."""
import numpy as np
import pytest

import aneforge as af
from _helpers import requires_ane


@requires_ane
def test_compile_decomposes_sin_for_h13_and_runs():
  # h13 lacks native sin -> compile substitutes the special.py polynomial; runs on M5
  xv = np.linspace(-1.0, 1.0, 64).astype(np.float16).reshape(1, 64)
  net = af.compile(af.input((1, 64)).sin(), target="h13")
  out = np.asarray(net(xv)).astype(np.float32)
  assert np.abs(out - np.sin(xv.astype(np.float32))).max() < 3e-3


@requires_ane
def test_compile_decomposes_cos_for_h13_and_runs():
  xv = np.linspace(-1.0, 1.0, 64).astype(np.float16).reshape(1, 64)
  net = af.compile(af.input((1, 64)).cos(), target="h13")
  out = np.asarray(net(xv)).astype(np.float32)
  assert np.abs(out - np.cos(xv.astype(np.float32))).max() < 3e-3


def test_compile_rejects_unreachable_op_with_clear_error():
  from aneforge.graph import Tensor
  node = Tensor((1, 8, 16, 16), "crop_resize", [af.input((1, 8, 16, 16))])
  with pytest.raises(NotImplementedError, match="crop_resize"):
    af.compile(node, target="h13")


def test_compile_rejects_below_mil_floor():
  # families below H13 cannot run the MIL path at all
  with pytest.raises(NotImplementedError, match="H13"):
    af.compile(af.input((1, 64)).relu(), target="h11")


@requires_ane
def test_compile_default_target_is_native_noop_on_m5():
  # no target -> host M5: sin native, no rewrite
  xv = np.linspace(-1.0, 1.0, 64).astype(np.float16).reshape(1, 64)
  net = af.compile(af.input((1, 64)).sin())
  out = np.asarray(net(xv)).astype(np.float32)
  assert np.abs(out - np.sin(xv.astype(np.float32))).max() < 3e-3


# --- the M1/H13 slice-saturation guard ---
def _saturation_warnings(graph, target):
  import warnings
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    af.compile(graph, target=target)
  return [w for w in rec if "4094" in str(w.message) or "saturat" in str(w.message).lower()]


def test_h13_last_axis_offset_slice_warns():
  # last-axis-offset slice routes through the H13 Q.4 DMA that saturates at 4094; must warn
  g = af.input((1, 8)).slice_by_size([0, 2], [1, 4])   # last-axis begin=2 > 0
  assert _saturation_warnings(g, "h13")


def test_h13_zero_offset_slice_does_not_warn():
  # begin=0 on the last axis avoids the offset-DMA path entirely (exact on H13)
  g = af.input((1, 8)).slice_by_size([0, 0], [1, 4])
  assert not _saturation_warnings(g, "h13")


def test_m5_offset_slice_does_not_warn():
  # the saturation is H13-specific; M5 (family 5) stays in plain fp16
  g = af.input((1, 8)).slice_by_size([0, 2], [1, 4])
  assert not _saturation_warnings(g, "h16s")


# --- cross_compile_check: static preflight pre-gate (family-cap CI keystone) ---
def test_cross_compile_check_rejects_family_cap_violation_statically(monkeypatch):
  # conv kW=14 fits A16 (<=15) but exceeds A13 cap (<=13); preflight must reject without compiling
  from aneforge import _compile, _runtime
  g = af.conv(af.input((1, 3, 32, 32)), np.zeros((8, 3, 3, 14), np.float32), pad=0)

  def _must_not_compile(*a, **k):
    raise AssertionError("compiler reached despite a static family-cap violation")
  monkeypatch.setattr(_runtime, "compile_check", _must_not_compile)
  assert _compile.cross_compile_check(g, "h13") is False


def test_cross_compile_check_passes_valid_graph_to_compiler(monkeypatch):
  # no static violation -> flows through to the (stubbed) e5rt compiler verdict
  from aneforge import _compile, _runtime
  g = af.conv(af.input((1, 3, 32, 32)), np.zeros((8, 3, 3, 11), np.float32), pad=0)
  monkeypatch.setattr(_runtime, "compile_check", lambda *a, **k: 0)
  assert _compile.cross_compile_check(g, "h13") is True
