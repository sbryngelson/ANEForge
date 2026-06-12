"""Cross-target compile validation: does a graph compile for another ANE family,
checked from THIS host? (aneforge._compile.cross_compile_check)

The e5rt compiler honors a ``TargetArchitecture`` option, and the compile step (produces
a library) is separable from the device-load step. So on this M5 we can compile-check a
graph for h13 (M1) etc. without an actual M1 — the keystone CI lever: validate that the
op corpus compiles for every chip family on one box. (Numeric correctness still needs the
real silicon; this is compile-level validation.)

These tests reproduce the measured op-floor ladder through aneforge's own dylib:
relu needs H13+ (the MIL hard floor); native sin/cos need A15+.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -m pytest tests/test_cross_compile.py -q
"""
import pytest
import aneforge as af
from aneforge._compile import cross_compile_check
from aneforge._targets import detect_family

# cross_compile_check uses the HOST compiler, which can only emit ops the host's family
# supports. A host below A15 (family 4) has no native-`sin` codegen, so cross-compiling a
# native-sin graph fails for EVERY target (measured on M1/family-2: h13/h15/h17s all False).
# Tests that cross-compile native A15+ ops therefore require an A15+/M5 host.
_NEEDS_A15_HOST = pytest.mark.skipif(
    detect_family() < 4,
    reason="cross_compile_check can only emit native A15+ ops (e.g. sin) from an A15+/M5 host")


def test_relu_compiles_for_h13():
    assert cross_compile_check(af.input((1, 64)).relu(), "h13") is True
    assert cross_compile_check(af.input((1, 64)).relu(), 2) is True   # family int also accepted


def test_relu_rejected_below_the_mil_floor():
    # "MIL is only supported for H13+ ANE architectures" — even relu fails on h11/h12
    assert cross_compile_check(af.input((1, 64)).relu(), "h11") is False
    assert cross_compile_check(af.input((1, 64)).relu(), "h12") is False


@_NEEDS_A15_HOST
def test_native_sin_follows_the_a15_floor():
    g = af.input((1, 64)).sin()           # RAW native MIL sin (A15+), not the decomposition
    assert cross_compile_check(g, "h13") is False    # family 2 — rejected
    assert cross_compile_check(g, "h14") is False    # family 3 — still below A15
    assert cross_compile_check(g, "h15") is True     # family 4 — native
    assert cross_compile_check(g, 5) is True          # M5


def test_fused_graph_cross_compiles_for_m1():
    x = af.input((1, 8, 16, 16))
    g = x.relu().mean((2, 3))             # all-native fused graph
    assert cross_compile_check(g, "h13") is True


@_NEEDS_A15_HOST
def test_a17_a18_and_m11_targets_compile():
    # The compiler's 28-target table includes A17 (h17*), A18 (h18) and the M11
    # efficiency ANE; all accept TargetArchitecture. h17s is the measured M5 target.
    g = af.input((1, 64)).sin()           # A15+ op: native on every one of these
    for arch in ("h17s", "h17d", "h18", "m11"):
        assert cross_compile_check(g, arch) is True


def test_sd15_group_norm_tiling_cross_compiles():
    # The rank-4 tiled group_norm makes SD-1.5's big maps fit the per-axis cap. 640ch@64/G32
    # (flat per-group 81920) tiles to D=20/H*W=4096 — under even the A13 16384 cap, so it
    # must compile on every family including h13 (M1). 512ch@128/G32 -> H*W=16384 fits too.
    import numpy as np
    for C, H, W, G in [(640, 64, 64, 32), (512, 128, 128, 32)]:
        g = af.input((1, C, H, W)).group_norm(np.ones(C, np.float16), np.zeros(C, np.float16), G)
        assert cross_compile_check(g, "h13") is True   # A13 / M1
        assert cross_compile_check(g, 5) is True         # A16 / M5 (host)


def test_unknown_arch_raises_not_silently_passes():
    # e5rt silently falls back to the HOST target on an unknown TargetArchitecture
    # string (measured: 'zzz' compiles), so cross_compile_check must gate the name —
    # otherwise a typo'd CI matrix would validate nothing.
    import pytest
    with pytest.raises(ValueError, match="unknown ANE target arch"):
        cross_compile_check(af.input((1, 64)).relu(), "h19")
