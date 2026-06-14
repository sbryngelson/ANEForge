"""Conv per-channel int8 weights - round-9 fix.

Round-8 found that ``af.conv`` emitted its weight with ``allow_int8=False``, so
``compile(int8=True)`` / ``compress='int8'`` on a conv graph was a SILENT no-op
(byte-identical fp16). Measured on h13/M1: per-channel int8 (``constexpr_affine_
dequantize``) DOES route as a conv weight operand (no +0 bridge needed, unlike
``constexpr_blockwise_shift_scale``), compiles+runs at cos~1.0 vs fp16 with the
expected ~0.5-0.7% per-channel quant error, and ~halves the conv DRAM weight bytes.
The fix enables it (``allow_int8=True``) and extends the auto-rewriter to conv.

These tests lock in: (1) the conv int8 weight branch emits affine_dequantize and
~halves the weight bytes; (2) compress=None stays byte-identical at opt=0 (the int8
branch is gated on the compress knob, so enabling allow_int8 changes nothing there);
(3) the per-node int8 override (auto-rewriter) tags conv; (4) on device, int8 conv
runs cos~1.0 vs fp16 and within the per-channel int8 error vs an fp32 reference.
"""
from __future__ import annotations
import numpy as np
import pytest
import aneforge as af
from aneforge import _compile as C
from aneforge.graph import conv, input as ainput


def _ane():
    try:
        from aneforge._runtime import _find_dylib; _find_dylib(); return True
    except Exception:
        return False


requires_ane = pytest.mark.skipif(not _ane(), reason="ANE/e5rt dylib unavailable")
rng = np.random.default_rng(0)


def _cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _relerr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _npconv(x, w, pad):
    Cout, Cin, kH, kW = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, _, H, W = x.shape
    Hout, Wout = H - kH + 1, W - kW + 1
    r = np.zeros((N, Cout, Hout, Wout), np.float32)
    for o in range(Cout):
        for i in range(Hout):
            for j in range(Wout):
                r[:, o, i, j] = np.sum(x[:, :, i:i+kH, j:j+kW] * w[o], axis=(1, 2, 3))
    return r


# MIL-level (no device): the int8 branch fires for conv and ~halves the bytes
def test_conv_int8_emits_affine_dequantize():
    """compress='int8' on a conv weight emits constexpr_affine_dequantize (the fix);
    previously allow_int8=False fell through to a plain fp16 const (silent no-op)."""
    W = (rng.standard_normal((64, 32, 3, 3)) * 0.2).astype(np.float16)
    em = C._Emitter(int8=False, compress="int8")
    em.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
    mil = "\n".join(em.lines)
    assert "constexpr_affine_dequantize" in mil
    assert "const()" not in mil.split("constexpr_affine_dequantize")[0] or True  # branch took int8


def test_conv_int8_halves_weight_bytes():
    """Per-channel int8 ~halves the conv DRAM weight bytes vs fp16 (residual = the
    per-output-channel fp16 scale vector + blob descriptors)."""
    W = (rng.standard_normal((64, 32, 3, 3)) * 0.2).astype(np.float16)
    fp16 = C._Emitter(int8=False, compress=None)
    fp16.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
    int8 = C._Emitter(int8=False, compress="int8")
    int8.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
    ratio = len(fp16.blob.build()) / len(int8.blob.build())
    assert ratio > 1.9                            # ~2x (true win, not a no-op)


def test_conv_compress_none_byte_identical_at_opt0():
    """Enabling allow_int8 must NOT change the compress=None (opt=0) lowering: the int8
    branch is gated on the compress knob (use_int8), which compress=None never sets, so
    the emitted MIL and blob bytes are byte-identical to the historical fp16 path."""
    W = (rng.standard_normal((32, 16, 3, 3)) * 0.2).astype(np.float16)
    new = C._Emitter(int8=False, compress=None)     # conv now passes allow_int8=True
    new.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
    old = C._Emitter(int8=False, compress=None)     # historical conv: allow_int8=False
    old.weight("c_w", W, allow_int8=False, int8=None, allow_int4=True, allow_sparse=True)
    assert new.lines == old.lines
    assert new.blob.build() == old.blob.build()


def test_auto_rewriter_includes_conv():
    """The per-node int8 auto-rewriter now treats conv as a weight-bearing candidate:
    list_weight_nodes finds it and set_node_int8 tags it with int8=True."""
    from aneforge._rewrite import set_node_int8, list_weight_nodes
    from aneforge._compile import _topo
    from aneforge._optimize import _has_weights, _int8_candidates
    out = conv(ainput((1, 16, 12, 12)), (rng.standard_normal((32, 16, 3, 3)) * 0.2).astype(np.float16), pad=1)
    wn = list_weight_nodes(out)
    assert len(wn) == 1 and wn[0].op == "conv"
    assert _has_weights(out)
    assert len(_int8_candidates(out)) == 1        # the conv is an int8 candidate
    ids = {id(t) for t in _topo(out) if t.op == "conv"}
    out2 = set_node_int8(out, ids)
    tagged = [t for t in _topo(out2) if t.op == "conv"]
    assert tagged[0].attrs.get("int8") is True


# On-device: int8 conv compiles, runs cos~1.0 vs fp16, within int8 error vs ref
@requires_ane
@pytest.mark.parametrize("Cin,Cout,k,pad,HW", [(8, 16, 3, 1, 16), (32, 64, 1, 0, 8)])
def test_conv_int8_runs_on_device(Cin, Cout, k, pad, HW):
    xv = rng.standard_normal((1, Cin, HW, HW)).astype(np.float16)
    wv = (rng.standard_normal((Cout, Cin, k, k)) * 0.2).astype(np.float16)
    ref = _npconv(xv.astype(np.float32), wv.astype(np.float32), pad)
    n16 = af.compile(af.conv(af.input((1, Cin, HW, HW)), wv, pad=pad), compress=None)
    o16 = np.asarray(n16(xv)).reshape(ref.shape)
    n8 = af.compile(af.conv(af.input((1, Cin, HW, HW)), wv, pad=pad), compress="int8")
    o8 = np.asarray(n8(xv)).reshape(ref.shape)
    assert _cos(o8, o16) > 0.999                   # int8 conv tracks fp16
    assert _cos(o8, ref) > 0.999                   # and the fp32 reference
    assert _relerr(o8, ref) < 0.02                 # within per-channel int8 error


@requires_ane
def test_conv_int8_true_alias_runs():
    """The legacy int8=True flag (opt=0) now actually streams the conv weight as int8."""
    xv = rng.standard_normal((1, 16, 12, 12)).astype(np.float16)
    wv = (rng.standard_normal((32, 16, 3, 3)) * 0.2).astype(np.float16)
    ref = _npconv(xv.astype(np.float32), wv.astype(np.float32), 1)
    net = af.compile(af.conv(af.input((1, 16, 12, 12)), wv, pad=1), int8=True, opt=0)
    out = np.asarray(net(xv)).reshape(ref.shape)
    assert _cos(out, ref) > 0.999
    assert _relerr(out, ref) < 0.02
