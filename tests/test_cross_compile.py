"""Cross-target compile validation: does a graph compile for another ANE family from this host?"""
import pytest
import aneforge as af
from aneforge._compile import cross_compile_check
from aneforge._targets import detect_family
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE

# cross_compile_check uses the HOST compiler, which can only emit ops the host's family
# supports; native A15+ ops (e.g. sin) require an A15+/M5 host.
_NEEDS_A15_HOST = pytest.mark.skipif(
  detect_family() < 4,
  reason="cross_compile_check can only emit native A15+ ops (e.g. sin) from an A15+/M5 host")


def test_relu_compiles_for_h13():
  assert cross_compile_check(af.input((1, 64)).relu(), "h13") is True
  assert cross_compile_check(af.input((1, 64)).relu(), 2) is True   # family int also accepted


def test_relu_rejected_below_the_mil_floor():
  # MIL is only supported for H13+; even relu fails on h11/h12
  assert cross_compile_check(af.input((1, 64)).relu(), "h11") is False
  assert cross_compile_check(af.input((1, 64)).relu(), "h12") is False


@_NEEDS_A15_HOST
def test_native_sin_follows_the_a15_floor():
  g = af.input((1, 64)).sin()           # RAW native MIL sin (A15+), not the decomposition
  assert cross_compile_check(g, "h13") is False    # family 2 - rejected
  assert cross_compile_check(g, "h14") is False    # family 3 - still below A15
  assert cross_compile_check(g, "h15") is True     # family 4 - native
  assert cross_compile_check(g, 5) is True          # M5


def test_fused_graph_cross_compiles_for_m1():
  x = af.input((1, 8, 16, 16))
  g = x.relu().mean((2, 3))             # all-native fused graph
  assert cross_compile_check(g, "h13") is True


@_NEEDS_A15_HOST
def test_a17_a18_and_m11_targets_compile():
  # A17 (h17*), A18 (h18), M11 all accept TargetArchitecture
  g = af.input((1, 64)).sin()           # A15+ op: native on every one of these
  for arch in ("h17s", "h17d", "h18", "m11"):
    assert cross_compile_check(g, arch) is True


def test_sd15_group_norm_tiling_cross_compiles():
  # rank-4 tiled group_norm makes SD-1.5's big maps fit the per-axis cap (must compile on h13)
  import numpy as np
  for C, H, W, G in [(640, 64, 64, 32), (512, 128, 128, 32)]:
    g = af.input((1, C, H, W)).group_norm(np.ones(C, np.float16), np.zeros(C, np.float16), G)
    assert cross_compile_check(g, "h13") is True   # A13 / M1
    assert cross_compile_check(g, 5) is True         # A16 / M5 (host)


def test_unknown_arch_raises_not_silently_passes():
  # e5rt silently falls back to the HOST target on unknown arch, so the name must be gated
  import pytest
  with pytest.raises(ValueError, match="unknown ANE target arch"):
    cross_compile_check(af.input((1, 64)).relu(), "h19")
