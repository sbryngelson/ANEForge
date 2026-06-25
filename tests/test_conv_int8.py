"""Conv per-channel int8 weights - round-9 fix (allow_int8 on conv + auto-rewriter)."""
from __future__ import annotations
import numpy as np
import pytest
from _helpers import requires_ane
import aneforge as af
from aneforge import _compile as C
from aneforge.graph import conv, input as ainput


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
  """compress='int8' on a conv weight emits constexpr_affine_dequantize (the fix)."""
  W = (rng.standard_normal((64, 32, 3, 3)) * 0.2).astype(np.float16)
  em = C._Emitter(int8=False, compress="int8")
  em.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
  mil = "\n".join(em.lines)
  assert "constexpr_affine_dequantize" in mil


def test_conv_int8_halves_weight_bytes():
  """Per-channel int8 ~halves the conv DRAM weight bytes vs fp16."""
  W = (rng.standard_normal((64, 32, 3, 3)) * 0.2).astype(np.float16)
  fp16 = C._Emitter(int8=False, compress=None)
  fp16.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
  int8 = C._Emitter(int8=False, compress="int8")
  int8.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
  ratio = len(fp16.blob.build()) / len(int8.blob.build())
  assert ratio > 1.9                            # ~2x (true win, not a no-op)


def test_conv_compress_none_byte_identical_at_opt0():
  """Enabling allow_int8 must NOT change the compress=None (opt=0) lowering (byte-identical)."""
  W = (rng.standard_normal((32, 16, 3, 3)) * 0.2).astype(np.float16)
  new = C._Emitter(int8=False, compress=None)     # conv now passes allow_int8=True
  new.weight("c_w", W, allow_int8=True, int8=None, allow_int4=True, allow_sparse=True)
  old = C._Emitter(int8=False, compress=None)     # historical conv: allow_int8=False
  old.weight("c_w", W, allow_int8=False, int8=None, allow_int4=True, allow_sparse=True)
  assert new.lines == old.lines
  assert new.blob.build() == old.blob.build()


def test_auto_rewriter_includes_conv():
  """The per-node int8 auto-rewriter treats conv as a weight-bearing candidate."""
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
