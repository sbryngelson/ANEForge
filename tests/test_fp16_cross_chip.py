"""Direction B wiring: the cross-chip fp16 divergence predictor surfaced as WARNINGS in
``cross_compile_check`` (preflight) and the A13/M1 conv-training loss_scale guard in
``Trainer``. The predictor itself is unit-tested in test_targets.py; here we check it is
correctly wired into the compile/train paths and never REJECTS or mutates (warn-only:
the auto-cap was dropped after the M1 end-to-end run refuted it - a real CNN trains
identically at loss_scale 128/1024/65536).

cross_compile_check compiles on this M5 host for another family's TargetArchitecture, so
these touch the dylib. The Trainer guard is host-static but the construction does
compile tiny programs.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -m pytest tests/test_fp16_cross_chip.py -q
"""
import warnings

import numpy as np

import aneforge as af
from aneforge import autograd as ag
from aneforge import _targets as TG
from aneforge._compile import cross_compile_check


def _crosschip_warnings(rec):
    return [str(r.message) for r in rec
            if issubclass(r.category, af.CrossChipFP16Warning)]


def test_cross_compile_check_warns_reduction_route(monkeypatch):
    # host = M5 (A16); cross-compiling a reduction for h13 (A13) -> <=1-ULP route warning.
    # Pin the simulated host family so the test is host-independent (passes on M1 too).
    monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
    TG._cpu_brand.cache_clear()
    g = af.input((1, 256)).sum((1,))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ok = cross_compile_check(g, "h13")
    assert ok is True                       # still compiles - warn-only
    msgs = _crosschip_warnings(rec)
    assert len(msgs) == 1
    assert "ULP" in msgs[0]


def test_cross_compile_check_same_family_no_warning(monkeypatch):
    # cross-compiling for the host's own family (A16) must not warn.
    # Pin host=A16 so this is host-independent (the host's own family == the target).
    monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
    TG._cpu_brand.cache_clear()
    g = af.input((1, 256)).sum((1,))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert cross_compile_check(g, "h16s") is True
    assert _crosschip_warnings(rec) == []


def test_cross_compile_check_no_fp16_axis_no_warning(monkeypatch):
    # an all-elementwise graph has no reduction/slice route axis -> no fp16 warning.
    monkeypatch.delenv("ANEFORGE_TARGET", raising=False)
    g = af.input((1, 64)).relu()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert cross_compile_check(g, "h13") is True
    assert _crosschip_warnings(rec) == []


# --- the A13 conv-training loss_scale saturation WARNING (Trainer; warn-only) ----------
def _conv_trainer(loss_scale):
    w = ag.conv_param(np.random.randn(3, 2, 3, 3))     # kW=3 -> width-offset im2col
    x = af.input((2, 2, 6, 6))
    y = ag.conv2d(x, w)
    loss = ag.mse(y, af.input((2, 3, 4, 4)))
    return ag.Trainer(loss, [w], lr=0.01, loss_scale=loss_scale,
                      data_inputs={x: np.zeros((2, 2, 6, 6), np.float32)})


def test_trainer_warns_but_keeps_loss_scale_on_a13_conv(monkeypatch):
    monkeypatch.setenv("ANEFORGE_TARGET", "h13")        # target the M1/A13 family
    TG._cpu_brand.cache_clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t = _conv_trainer(1024.0)
    assert t.scale == 1024.0                             # warn-only: never mutated
    assert any("saturat" in str(r.message) for r in rec)


def test_trainer_no_warning_below_onset_on_a13(monkeypatch):
    monkeypatch.setenv("ANEFORGE_TARGET", "h13")
    TG._cpu_brand.cache_clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t = _conv_trainer(256.0)                         # already clean in the repro
    assert t.scale == 256.0
    assert not any("saturat" in str(r.message) for r in rec)


def test_trainer_no_warning_on_a16(monkeypatch):
    # M5/A16 takes the clean slice route -> silent even at a high loss_scale.
    # Pin host=A16 so this is host-independent (passes on M1 too, which is otherwise A13).
    monkeypatch.setenv("ANEFORGE_TARGET", "h16s")
    TG._cpu_brand.cache_clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        t = _conv_trainer(1024.0)
    assert t.scale == 1024.0
    assert not any("saturat" in str(r.message) for r in rec)


def test_trainer_a13_no_conv_unaffected(monkeypatch):
    # A13 target but a plain (non-conv) graph: no width-offset slice -> silent.
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
