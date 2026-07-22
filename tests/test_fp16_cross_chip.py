"""Cross-chip fp16 divergence predictor wired into cross_compile_check + Trainer (warn-only)."""
import warnings

import numpy as np

import aneforge as af
from aneforge import autograd as ag
from aneforge import _targets as TG
from aneforge._compile import cross_compile_check
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE


def _crosschip_warnings(rec):
  return [str(r.message) for r in rec
          if issubclass(r.category, af.CrossChipFP16Warning)]


def test_cross_compile_check_warns_reduction_route(monkeypatch):
  # host A16, cross-compiling a reduction for A13 -> <=1-ULP route warning; pin host family
  monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
  TG._cpu_brand.cache_clear()
  g = af.input((1, 256)).sum((1,))
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    ok = cross_compile_check(g, "h13")
  assert ok is True                       # still compiles, warn-only
  msgs = _crosschip_warnings(rec)
  assert len(msgs) == 1
  assert "ULP" in msgs[0]


def test_cross_compile_check_same_family_no_warning(monkeypatch):
  # cross-compiling for the host's own family (A16) must not warn; pin host=A16
  monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
  TG._cpu_brand.cache_clear()
  g = af.input((1, 256)).sum((1,))
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    assert cross_compile_check(g, "h16s") is True
  assert _crosschip_warnings(rec) == []


def test_cross_compile_check_no_fp16_axis_no_warning(monkeypatch):
  # all-elementwise graph has no reduction/slice route axis -> no fp16 warning
  monkeypatch.delenv("ANEFORGE_TARGET", raising=False)
  g = af.input((1, 64)).relu()
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    assert cross_compile_check(g, "h13") is True
  assert _crosschip_warnings(rec) == []


# --- A13 conv-training loss_scale saturation warning (Trainer; warn-only) ----------
def _conv_trainer(loss_scale):
  w = ag.conv_param(np.random.randn(3, 2, 3, 3))     # kW=3 -> width-offset im2col
  x = af.input((2, 2, 6, 6))
  y = ag.conv2d(x, w)
  loss = ag.mse(y, af.input((2, 3, 4, 4)))
  return ag.Trainer(loss, [w], lr=0.01, loss_scale=loss_scale,
                    data_inputs={x: np.zeros((2, 2, 6, 6), np.float32)})


def test_trainer_warns_but_keeps_loss_scale_on_a13_conv(monkeypatch):
  monkeypatch.setenv("ANEFORGE_TARGET", "h13")        # target the A13 family
  TG._cpu_brand.cache_clear()
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    t = _conv_trainer(1024.0)
  assert t.scale == 1024.0                             # warn-only, never mutated
  assert any("saturat" in str(r.message) for r in rec)


def test_trainer_no_warning_below_onset_on_a13(monkeypatch):
  monkeypatch.setenv("ANEFORGE_TARGET", "h13")
  TG._cpu_brand.cache_clear()
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    t = _conv_trainer(256.0)                         # below onset, already clean
  assert t.scale == 256.0
  assert not any("saturat" in str(r.message) for r in rec)


def test_trainer_no_warning_on_a16(monkeypatch):
  # A16 takes the clean slice route -> silent even at high loss_scale; pin host=A16
  monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
  TG._cpu_brand.cache_clear()
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    t = _conv_trainer(1024.0)
  assert t.scale == 1024.0
  assert not any("saturat" in str(r.message) for r in rec)


def test_trainer_a13_no_conv_unaffected(monkeypatch):
  # A13 target but plain (non-conv) graph: no width-offset slice -> silent
  monkeypatch.setenv("ANEFORGE_TARGET", "h13")
  TG._cpu_brand.cache_clear()
  p = ag.parameter(np.random.randn(4, 4))
  x = af.input((1, 4))
  loss = ag.mse(x @ p, af.input((1, 4)))
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    t = ag.Trainer(loss, [p], lr=0.01, loss_scale=1024.0,
                   data_inputs={x: np.zeros((1, 4), np.float32)})
  assert t.scale == 1024.0
  assert not any("saturat" in str(r.message) for r in rec)
