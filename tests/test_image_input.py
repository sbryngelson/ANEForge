"""Tests for af.image_input - uint8 image input dequantised to fp16 ON the engine.

Camera/decoded-video bytes feed the model directly; the uint8->fp16 cast + the
``scale*x + bias`` normalisation run as in-graph ANE ops. Each test compiles+runs a
real graph fed a uint8 array and checks it against a host-fp16 reference.

    PYTHONPATH=. python3 -m pytest tests/test_image_input.py -q
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
rng = np.random.default_rng(0)


def _cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


@requires_ane
def test_scalar_dequant_identity():
    """uint8 input -> scale 1/255 dequant matches the host float-convert exactly."""
    img = rng.integers(0, 256, size=(1, 8), dtype=np.uint8)
    x = af.image_input((1, 8), scale=1 / 255.0, bias=0.0)
    net = af.compile(x)
    got = net(img); net.release()
    ref = (img.astype(np.float32) / 255.0).astype(np.float16).astype(np.float32)
    assert np.abs(got - ref).max() == 0.0


@requires_ane
def test_scalar_dequant_with_bias():
    """scale*x + bias dequant (e.g. [-1, 1] normalisation)."""
    img = rng.integers(0, 256, size=(1, 3, 8, 8), dtype=np.uint8)
    x = af.image_input((1, 3, 8, 8), scale=2 / 255.0, bias=-1.0)
    net = af.compile(x.mean((2, 3)))
    got = net(img); net.release()
    deq = (img.astype(np.float32) * np.float16(2 / 255.0).astype(np.float32) - 1.0)
    ref = deq.mean((2, 3), keepdims=True).astype(np.float16).astype(np.float32)
    assert _cos(got, ref) > 0.999
    assert np.abs(got - ref).max() < 5e-3


@requires_ane
def test_odd_numel_uint8_input():
    """An odd-element-count uint8 image (75 bytes, padded to a uint16 feed) compiles and runs."""
    N, C, H, W = 1, 3, 5, 5
    img = rng.integers(0, 256, size=(N, C, H, W), dtype=np.uint8)

    x = af.image_input((N, C, H, W), scale=1 / 255.0)
    net = af.compile(x.relu())
    got = net(img); net.release()

    xf = af.input((N, C, H, W))
    ref = af.compile(xf.relu())
    exp = ref((img.astype(np.float32) / 255.0).astype(np.float16))
    ref.release()
    assert np.abs(got - exp).max() == 0.0


@requires_ane
def test_uint8_conv_stack_matches_host_fp16():
    """End-to-end vision graph on a uint8 image == the same graph on a host-fp16 feed."""
    N, C, H, W = 1, 3, 32, 32
    img = rng.integers(0, 256, size=(N, C, H, W), dtype=np.uint8)
    W1 = (rng.standard_normal((8, 3, 3, 3)) * 0.1).astype(np.float16)
    W2 = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float16)

    x = af.image_input((N, C, H, W), scale=1 / 255.0)
    out_u8 = af.compile(af.conv(af.conv(x, W1, pad=1).relu(), W2, pad=1).relu().mean((2, 3)))

    xf = af.input((N, C, H, W))
    ref = af.compile(af.conv(af.conv(xf, W1, pad=1).relu(), W2, pad=1).relu().mean((2, 3)))

    got = out_u8(img)
    exp = ref((img.astype(np.float32) / 255.0).astype(np.float16))
    out_u8.release(); ref.release()
    assert _cos(got, exp) > 0.9999
    assert np.abs(got - exp).max() < 5e-3


@requires_ane
def test_per_channel_scale_bias():
    """ImageNet-style per-channel (scale, bias) broadcast over NCHW channels."""
    N, C, H, W = 1, 3, 16, 16
    img = rng.integers(0, 256, size=(N, C, H, W), dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    sc, bi = (1.0 / 255.0) / std, -mean / std

    x = af.image_input((N, C, H, W), scale=sc, bias=bi)
    net = af.compile(x.mean((2, 3)))
    got = net(img); net.release()

    host = ((img.astype(np.float32) / 255.0) - mean[None, :, None, None]) / std[None, :, None, None]
    ref = host.mean((2, 3), keepdims=True).astype(np.float16).astype(np.float32)
    assert _cos(got, ref) > 0.999
    assert np.abs(got - ref).max() < 1e-2


@requires_ane
def test_uint8_input_accepts_only_integers():
    """A float array fed to a uint8 image port is rejected (clear error, not silent)."""
    x = af.image_input((1, 8))
    net = af.compile(x)
    with pytest.raises(TypeError):
        net(np.zeros((1, 8), np.float32))
    net.release()


def test_image_input_per_channel_shape_validation():
    """Per-channel scale/bias must be length-C on an NCHW image (build-time check)."""
    with pytest.raises(ValueError):
        af.image_input((1, 3, 8, 8), scale=np.ones(5, np.float32))
    with pytest.raises(ValueError):
        af.image_input((1, 8), scale=np.ones(3, np.float32))   # per-channel needs rank-4


def test_input_dtype_validation():
    """af.input rejects an unknown wire dtype."""
    with pytest.raises(ValueError):
        af.input((1, 8), dtype="int32")
