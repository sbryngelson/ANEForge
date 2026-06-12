"""group_norm rank-4 tiling (Direction C): the SD-1.5 large-feature-map wall.

The native group_norm now lowers to [1,G,C/groups,H*W] and reduces the trailing two
axes (instead of flattening to [1,G,(C/groups)*H*W]), so every internal axis stays under
the ANE per-axis cap. This lets SD-1.5's 640ch@64 and 512ch@128 maps compile + run where
the flattened extent (81920 / 262144) overflowed. These tests pin small-shape correctness,
big-shape correctness vs fp32, and the relaxed construction guard.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -m pytest tests/test_group_norm_tiling.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import aneforge as af


def _ane_available():
    try:
        from aneforge._runtime import _find_dylib
        _find_dylib()
        return True
    except Exception:
        return False


requires_ane = pytest.mark.skipif(not _ane_available(), reason="ANE/e5rt dylib unavailable")


def _np_gn(x, gamma, beta, groups, eps=1e-5):
    n, c, h, w = x.shape
    xr = x.reshape(n, groups, c // groups, h, w)
    mu = xr.mean((2, 3, 4), keepdims=True)
    var = xr.var((2, 3, 4), keepdims=True)
    xn = ((xr - mu) / np.sqrt(var + eps)).reshape(n, c, h, w)
    g = gamma.astype(np.float32).reshape(1, c, 1, 1)
    b = beta.astype(np.float32).reshape(1, c, 1, 1)
    return xn * g + b


def _run(C, H, W, G, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, C, H, W)).astype(np.float32)
    gamma = rng.standard_normal(C).astype(np.float16)
    beta = (rng.standard_normal(C) * 0.1).astype(np.float16)
    t = af.input((1, C, H, W)).group_norm(gamma, beta, G)
    m = af.compile(t)
    out = m(x)
    m.release()
    return out, _np_gn(x, gamma, beta, G)


# --- small-shape correctness (rank-4 tiled form matches numpy in fp16) ----------------
@requires_ane
@pytest.mark.parametrize("C,H,W,G", [(8, 4, 4, 2), (32, 16, 16, 8), (64, 8, 8, 16)])
def test_group_norm_small_shapes(C, H, W, G):
    out, ref = _run(C, H, W, G)
    assert np.abs(out - ref).max() < 0.03


# --- big-shape correctness: the formerly-rejected SD-1.5 maps now run -----------------
@requires_ane
@pytest.mark.parametrize("C,H,W,G", [(640, 64, 64, 32), (512, 128, 128, 32)])
def test_group_norm_sd15_large_maps_run(C, H, W, G):
    # flat per-group (C/G)*H*W = 81920 / 262144 used to overflow the per-axis cap; the
    # rank-4 tiling keeps max(C/G, H*W) under it so these compile + run in fp16.
    got, ref = _run(C, H, W, G)
    assert np.abs(got - ref).max() < 0.03


# --- construction guard now keyed on the largest tiled axis, not the product ----------
def test_group_norm_guard_allows_large_product_small_axes():
    # 640ch@64/G32: product 81920 > 65536 (old guard rejected) but max(C/G,H*W)=4096 fits.
    af.input((1, 640, 64, 64)).group_norm(np.ones(640, np.float16), np.zeros(640, np.float16), 32)
    # 512ch@128/G32: product 262144, tiled axes D=16 / H*W=16384 both fit.
    af.input((1, 512, 128, 128)).group_norm(np.ones(512, np.float16), np.zeros(512, np.float16), 32)


def test_group_norm_guard_rejects_oversize_single_axis():
    # A single tiled axis above 65536 still raises: H*W = 257*257 = 66049 > 65536.
    with pytest.raises(ValueError, match="largest tiled axis"):
        af.input((1, 4, 257, 257)).group_norm(np.ones(4, np.float16), np.zeros(4, np.float16), 1)
