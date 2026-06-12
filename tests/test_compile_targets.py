"""compile(target=...) — gate lowering on the target ANE family.

The gate runs once at the top of compile(): resolve the family, substitute below-floor
ops that have an in-graph decomposition (sin/cos -> special.py), and raise a clear
compile-time error on an op the family cannot run (instead of a dispatch-time crash).

On the M5 dev host (family 5) everything is native, so target=None is a no-op fast path
and existing compiles are byte-identical. The decompose path IS verifiable here, because
the rewritten graph (sin -> mul/exp polynomial) is all-native and runs on the M5.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -m pytest tests/test_compile_targets.py -q
"""
import numpy as np
import pytest

import aneforge as af


def test_compile_decomposes_sin_for_h13_and_runs():
    # target='h13' (family 2) -> native sin is unavailable -> compile substitutes the
    # special.py polynomial. The rewritten graph runs on this M5 and matches numpy.
    xv = np.linspace(-1.0, 1.0, 64).astype(np.float16).reshape(1, 64)
    net = af.compile(af.input((1, 64)).sin(), target="h13")
    out = np.asarray(net(xv)).astype(np.float32)
    assert np.abs(out - np.sin(xv.astype(np.float32))).max() < 3e-3


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


def test_compile_default_target_is_native_noop_on_m5():
    # no target -> detect host (M5/family 5) -> sin is native, no rewrite, runs as before
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
    # a last-axis-offset slice routes through the H13 Q.4 fixed-point DMA that saturates
    # at 4094 (=65504/16); on M1 this must warn loudly (it silently corrupts otherwise).
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
    # A conv with kW=14 fits A16 (<=15) but exceeds the A13 cap (<=13). cross_compile_check
    # for h13 must reject it from preflight ALONE — not lean on the host cross-compiler,
    # which may not enforce a different family's dim caps. The e5rt compiler must not even
    # be reached for a statically-rejectable graph.
    from aneforge import _compile, _runtime
    g = af.conv(af.input((1, 3, 32, 32)), np.zeros((8, 3, 3, 14), np.float32), pad=0)

    def _must_not_compile(*a, **k):
        raise AssertionError("compiler reached despite a static family-cap violation")
    monkeypatch.setattr(_runtime, "compile_check", _must_not_compile)
    assert _compile.cross_compile_check(g, "h13") is False


def test_cross_compile_check_passes_valid_graph_to_compiler(monkeypatch):
    # A graph with no static violation must still flow through to the e5rt compiler and
    # return its verdict (here stubbed to success) — the pre-gate is a filter, not a wall.
    from aneforge import _compile, _runtime
    g = af.conv(af.input((1, 3, 32, 32)), np.zeros((8, 3, 3, 11), np.float32), pad=0)
    monkeypatch.setattr(_runtime, "compile_check", lambda *a, **k: 0)
    assert _compile.cross_compile_check(g, "h13") is True
