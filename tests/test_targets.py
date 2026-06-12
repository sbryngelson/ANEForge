"""Per-chip ANE target capability core (aneforge._targets).

Host-independent: every fact here is the measured per-family capability data from the
M1/M5 reverse-engineering (MinimumFamily op floors + the ZinIrHalParameters numeric limits +
the "MIL is only supported for H13+ ANE architectures" hard floor). No hardware needed
to run these — family is passed explicitly. Detection + cross-compile are separate.
"""
import pytest

from aneforge import _targets as T


# --- the hard architectural floor: aneforge's e5rt/MIL path needs H13+ (A13, family 2) ---
def test_mil_hard_floor_is_family_2():
    assert T.MIN_FAMILY == 2
    assert T.supports_mil(2) is True
    assert T.supports_mil(1) is False  # A12 and older cannot run the MIL path at all


# --- chip <-> compiler-family mapping (OS arch string runs +1 above the compiler family) ---
@pytest.mark.parametrize("arch,fam", [
    ("h13", 2), ("H13", 2),          # M1
    ("h17s", 5), ("H16s", 5),        # M5: OS "h17s" == compiler-internal H16s
    ("h14", 3), ("h15", 4), ("h16", 5),
])
def test_family_of_arch(arch, fam):
    assert T.family_of_arch(arch) == fam


def test_m1_and_m5_anchors():
    # the two physically-measured anchors
    assert T.Family.A13 == 2
    assert T.Family.A16 == 5


# --- op gating: native / decompose / reject, per family ---
def test_core_ops_native_on_every_supported_chip():
    # F0/F2 ops are native on M1 (family 2) and up — aneforge core nets have no gap here
    for op in ("conv", "matmul", "softmax", "layer_norm", "sdpa", "erf", "rsqrt", "atan"):
        assert T.op_status(op, 2) == "native"
        assert T.op_status(op, 5) == "native"


def test_sincos_floor_with_decomposition():
    # sin/cos are A15+ (family 4): native on M5, but aneforge HAS a Horner decomposition
    # for below-floor chips, so the status is "decompose", never a hard reject.
    assert T.op_status("sin", 5) == "native"
    assert T.op_status("cos", 4) == "native"
    assert T.op_status("sin", 2) == "decompose"   # M1: fall back to special.py Horner
    assert T.op_status("cos", 3) == "decompose"    # A14


def test_texture_engine_ops_reject_below_a14():
    # resize-HW/crop/resample/affine/gather need the texture engine (family >= 3); with no
    # decomposition wired, they hard-reject on M1 (family 2) rather than crash at dispatch.
    assert T.op_status("crop_resize", 2) == "reject"
    assert T.op_status("crop_resize", 3) == "native"
    assert T.has_texture_engine(2) is False
    assert T.has_texture_engine(3) is True


def test_bridge_ops_codegen_reject_on_m1():
    # topk/sort/dynamic_slice PASS validation but REJECT at codegen on M1 (family 2);
    # native at higher families. No decomposition -> reject, not decompose.
    for op in ("topk", "sort", "dynamic_slice"):
        assert T.op_status(op, 2) == "reject"
        assert T.op_status(op, 5) == "native"


def test_dropout_random_is_a15():
    # 0x4a9: A15+ -> unsupported on M1 AND A14; host-side RNG is the decomposition.
    assert T.op_status("dropout", 3) == "decompose"
    assert T.op_status("dropout", 4) == "native"


# --- numeric HAL limits per family (measured from the live compiler) ---
def test_max_tensor_dim_quadruples_at_a16():
    assert T.limit("max_tensor_dim", 2) == 16384   # M1..A15
    assert T.limit("max_tensor_dim", 4) == 16384
    assert T.limit("max_tensor_dim", 5) == 65536   # A16/M5


def test_reduction_transpose_threshold():
    assert T.limit("reduction_transpose_extent", 2) == 192   # A13/A14
    assert T.limit("reduction_transpose_extent", 4) == 384   # A15+


# --- preflight: "will this graph run on chip X?" (static graph walk, no hardware) ---
def test_preflight_flags_sin_as_decompose_on_m1():
    import aneforge as af
    g = af.input((1, 64)).sin()
    r = T.preflight(g, family=2)
    assert any(o.op == "sin" for o in r.decompose)
    assert r.ok  # decompose is recoverable (use special.sin), not a hard failure


def test_preflight_all_native_on_m5():
    import aneforge as af
    g = af.input((1, 64)).sin().relu()
    r = T.preflight(g, family=5)
    assert r.ok and not r.decompose and not r.reject


def test_preflight_oversize_dim_needs_tiling_below_a16():
    import aneforge as af
    g = af.input((1, 20000)).relu()           # 20000 > 16384 (A13..A15), <= 65536 (A16)
    assert T.preflight(g, family=2).oversize and not T.preflight(g, family=2).ok
    assert not T.preflight(g, family=5).oversize and T.preflight(g, family=5).ok


def test_preflight_catches_group_norm_internal_oversize():
    # The rank-4 tiled lowering reshapes to [1,G,C/groups,H*W]; the largest INTERNAL axis
    # is max(C/groups, H*W). C32@130x130/G1 -> H*W=16900 exceeds the 16384 cap below A16 but
    # fits the A16 65536 max. preflight must see this internal extent (the compiler does).
    import numpy as np

    import aneforge as af
    C, H, W, G = 32, 130, 130, 1
    g = af.input((1, C, H, W)).group_norm(np.ones(C, np.float16), np.zeros(C, np.float16), G)
    assert not T.preflight(g, 2).ok       # M1 / A13 — rejects (16384 cap, H*W=16900)
    assert not T.preflight(g, 4).ok       # A15 — still 16384
    assert T.preflight(g, 5).ok           # M5 / A16 — fits 65536


def test_preflight_group_norm_tiled_fits_small_axes():
    # SD-1.5 640ch@64/G32: flat (C/G)*H*W=81920 overflowed, but tiled axes are
    # D=20, H*W=4096 — both under even the A13 16384 cap, so it fits everywhere.
    import numpy as np

    import aneforge as af
    g = af.input((1, 640, 64, 64)).group_norm(np.ones(640, np.float16), np.zeros(640, np.float16), 32)
    assert T.preflight(g, 2).ok and T.preflight(g, 5).ok


def test_preflight_reject_op_is_not_ok_below_floor():
    import aneforge as af
    from aneforge.graph import Tensor
    x = af.input((1, 8, 16, 16))
    node = Tensor(x.shape, "crop_resize", [x])   # texture-engine op, no decomposition
    assert T.preflight(node, family=2).reject and not T.preflight(node, family=2).ok
    assert T.preflight(node, family=5).ok          # native at A14+


# --- runtime family detection (read the host chip) ---
def test_family_from_brand_full_mseries_ladder():
    # M1/M5 are measured anchors; M2/M3/M4 resolve by the VERIFIED M(n)=H(n+12) ladder
    # M2=H14/A14, M3=H15/A15, M4=H16/A16. The Pro/Max variant
    # changes core count, not capability family.
    assert T._family_from_brand("Apple M1") == 2          # A13
    assert T._family_from_brand("Apple M1 Max") == 2
    assert T._family_from_brand("Apple M2") == 3          # A14
    assert T._family_from_brand("Apple M2 Pro") == 3
    assert T._family_from_brand("Apple M3 Max") == 4      # A15
    assert T._family_from_brand("Apple M4") == 5          # A16
    assert T._family_from_brand("Apple M5 Pro") == 5      # A16 (H17s)


def test_family_from_brand_beyond_map_is_conservative_floor():
    # a chip past the verified ladder (a future M6+) defaults to the safe minimum: a
    # family-2 program runs on every H13+ chip (higher families are supersets).
    assert T._family_from_brand("Apple M6 Max") == T.MIN_FAMILY
    assert T._family_from_brand("Apple Silicon Unknown") == T.MIN_FAMILY


def test_detect_family_env_override(monkeypatch):
    monkeypatch.setenv("ANEFORGE_TARGET", "h13")
    assert T.detect_family() == 2
    monkeypatch.setenv("ANEFORGE_TARGET", "h17s")
    assert T.detect_family() == 5


@pytest.mark.skipif(T.detect_family() != 5,
                    reason="host-intrinsic: asserts THIS box detects as family 5 (M5) — "
                           "skip on other hosts (e.g. the M1/family-2 dev box)")
def test_detect_family_on_this_m5_host(monkeypatch):
    monkeypatch.delenv("ANEFORGE_TARGET", raising=False)
    # this dev box is an M5 Pro -> compiler family 5 (H16s)
    assert T.detect_family() == 5


# --- cross-chip fp16 divergence predictor (Direction B) --------------------------------
# Static, table-driven: the per-family HAL fields (slice x16 saturation, 0x494 reduce->
# square fuse, 0x3f0 reduction route) predict whether an op's fp16 VALUE can diverge
# between two chip families.
A13, A14, A15, A16 = T.Family.A13, T.Family.A14, T.Family.A15, T.Family.A16


@pytest.mark.parametrize("kind,shape,a,b,kw,expect", [
    # slice with nonzero LAST-axis (width) begin-offset + an A13 target + possibly-large
    # values -> the x16 crop-DMA finite->inf saturation (the one dramatic case).
    ("slice", (2, 2, 4, 4), A13, A16, {"begin": [0, 0, 0, 2]}, "saturation"),
    ("slice_by_size", (2, 2, 4, 4), A16, A13, {"begin": [0, 0, 0, 1]}, "saturation"),
    # ... but a finite magnitude bound under 4094 downgrades it (magnitude-gated).
    ("slice", (2, 2, 4, 4), A13, A16, {"begin": [0, 0, 0, 2], "max_abs": 100.0}, "none"),
    # ... offset on a NON-last axis is clean (only width-offset routes through the DMA).
    ("slice", (2, 2, 4, 4), A13, A16, {"begin": [0, 0, 2, 0]}, "none"),
    # ... A14 is ALSO affected (M2 silicon: a single width slice saturates 4094->inf), so
    # A14 vs a clean A16 diverges; both-affected (A13 vs A14) would match.
    ("slice", (2, 2, 4, 4), A14, A16, {"begin": [0, 0, 0, 2]}, "saturation"),
    # reduce->square 0x494 fuse is a MEASURED no-op (A13/A14/A16 all compute fp16(sum)^2),
    # so it never drives "round1"; only the 0x3f0 route threshold (192 A13/A14 vs 384 A15+)
    # diverges a reduce-square pair.
    ("reduce_square", (4, 1), A13, A16, {}, "ulp1"),     # route thresh 192 vs 384 differ
    ("reduce_square", (4, 1), A13, A14, {}, "none"),     # both 192, fuse uniform -> none
    ("reduce_square", (4, 1), A14, A16, {}, "ulp1"),     # route thresh differs
    # plain reduction/softmax/norm: 0x3f0 route threshold 192(A13/A14) vs 384(A15+).
    ("reduce", (4, 1), A13, A16, {}, "ulp1"),
    ("softmax", (1, 256), A14, A15, {}, "ulp1"),
    # ... A13 vs A14 share the 192 threshold -> no divergence.
    ("reduce", (4, 1), A13, A14, {}, "none"),
    # same family -> never diverges.
    ("reduce", (4, 1), A16, A16, {}, "none"),
    ("slice", (2, 2, 4, 4), A13, A13, {"begin": [0, 0, 0, 2]}, "none"),
    # an op with no fp16-route axis -> none.
    ("add", (4, 4), A13, A16, {}, "none"),
])
def test_predict_fp16_divergence(kind, shape, a, b, kw, expect):
    assert T.predict_fp16_divergence(kind, shape, a, b, **kw) == expect


def test_predict_fp16_unknown_magnitude_is_conservative():
    # max_abs=None (unknown) means the value COULD exceed 4094 -> flag saturation.
    assert T.predict_fp16_divergence(
        "slice", (1, 8), A13, A16, begin=[0, 4]) == "saturation"


def test_fp16_slice_sat_threshold():
    assert T.FP16_SLICE_SAT == 65504.0 / 16    # fp16max / 16 = 4094.0


def test_per_class_extents():
    # A14 measured exact (A14_MAXDIM_CAPS.md): spatial 16384 / channel 65536 / transpose 2^23-1;
    # A16 (M5 probe): spatial 65536 / channel 65536 / transpose >=2^24-1.
    assert T.limit("max_tensor_dim", 3) == 16384 and T.limit("max_tensor_dim", 5) == 65536
    assert T.limit("channel_extent", 3) == 65536 and T.limit("channel_extent", 5) == 65536
    assert T.limit("transpose_extent", 3) == 8388607 and T.limit("transpose_extent", 5) == 16777215


def test_preflight_channel_axis_not_spatial_capped():
    import aneforge as af
    g = af.input((1, 65536, 1, 1)).relu()   # C=65536 legal on A14 (old single cap over-blocked 4x)
    assert T.preflight(g, family=3).ok
    assert not T.preflight(af.input((1, 20000)).relu(), family=3).ok   # flat W stays spatial-capped


def test_preflight_transpose_wide_extent():
    import aneforge as af
    g = af.input((1 << 20, 2)).transpose([1, 0])   # 2^20 << 2^23-1; old cap over-blocked 512x
    assert T.preflight(g, family=3).ok


def test_a13_maxdim_measured_equals_a14_monotone():
    # A13 row measured on live M1: W/H 16384, C 65536 = A14.
    assert T.limit("max_tensor_dim", 2) == T.limit("max_tensor_dim", 3) == 16384
    assert T.limit("channel_extent", 2) == 65536
    # monotone: every cap non-decreasing A13 -> A16 (no inversion).
    for name in ("max_tensor_dim", "channel_extent", "transpose_extent", "conv_kw_max"):
        assert T.limit(name, 2) <= T.limit(name, 5), name


def test_conv_kw_per_family_preflight():
    import numpy as np
    import aneforge as af
    # kW=14: rejected on A13 (<=13), fits A16 (<=15). Monotone, enforced family-aware.
    g = af.conv(af.input((1, 3, 32, 32)), np.zeros((8, 3, 3, 14), np.float32), pad=0)
    assert not T.preflight(g, family=2).ok          # M1: kW 14 > 13
    assert T.preflight(g, family=5).ok              # M5: kW 14 <= 15
    assert T.limit("conv_kw_max", 2) == 13 and T.limit("conv_kw_max", 5) == 15
