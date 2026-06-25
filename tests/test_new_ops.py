"""New forward ops (floor/ceil/round/sign/pow/comparisons/reverse/tile/reduce_log_sum_exp) + their vjps."""
from __future__ import annotations
import numpy as np
import pytest
import aneforge as af
from aneforge import autograd as agrad


def _ane_available():
  try:
    from aneforge._runtime import _find_dylib; _find_dylib(); return True
  except Exception:
    return False


requires_ane = pytest.mark.skipif(not _ane_available(), reason="ANE/e5rt dylib unavailable")
rng = np.random.default_rng(0)


def _run(net, *f):
  r = np.asarray(net(*[a.astype(np.float16) for a in f]), np.float32); net.release(); return r


@requires_ane
@pytest.mark.parametrize("name,fn,ref", [
    ("floor", lambda x: x.floor(), lambda v: np.floor(v)),
    ("ceil", lambda x: x.ceil(), lambda v: np.ceil(v)),
    ("round", lambda x: x.round(), lambda v: np.round(v)),
    ("sign", lambda x: x.sign(), lambda v: np.sign(v)),
    ("reverse", lambda x: x.reverse([1]), lambda v: v[:, ::-1]),
    ("tile", lambda x: x.tile([1, 2]), lambda v: np.tile(v, (1, 2))),
])
def test_unary_structural_forward(name, fn, ref):
  xv = (rng.standard_normal((1, 8)) * 3).astype(np.float32)
  net = af.compile(fn(af.input((1, 8))))
  out = _run(net, xv).reshape(ref(xv).shape)
  assert np.allclose(out, ref(xv), atol=3e-2), name


@requires_ane
@pytest.mark.parametrize("cmp,np_op", [
    ("less", lambda a, b: a < b), ("greater_equal", lambda a, b: a >= b),
    ("less_equal", lambda a, b: a <= b), ("not_equal", lambda a, b: a != b),
])
def test_comparisons(cmp, np_op):
  xv = rng.standard_normal((1, 8)).astype(np.float32); yv = rng.standard_normal((1, 8)).astype(np.float32)
  x = af.input((1, 8)); y = af.input((1, 8))
  net = af.compile(af.select(getattr(x, cmp)(y), x, y))
  ref = np.where(np_op(xv, yv), xv, yv)
  assert np.allclose(_run(net, xv, yv).reshape(ref.shape), ref, atol=3e-2)


@requires_ane
def test_reduce_log_sum_exp_forward():
  xv = rng.standard_normal((1, 8)).astype(np.float32)
  net = af.compile(af.input((1, 8)).reduce_log_sum_exp((1,)))
  ref = np.log(np.exp(xv).sum(1, keepdims=True))
  assert np.allclose(_run(net, xv).reshape(ref.shape), ref, atol=1e-1)


# -- vjps for the new differentiable ops --
def _g(loss, x, ls=256.0):
  from tests.test_autograd import _eval_grad_tensor
  gd = agrad.backward(loss, [x], loss_scale=ls)
  return _eval_grad_tensor(gd[x], [x]).reshape(x.shape) / ls


def _cos(a, b):
  return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@requires_ane
def test_reverse_grad():
  xv = rng.standard_normal((2, 6)).astype(np.float32); wv = rng.standard_normal((2, 6)).astype(np.float32)
  x = agrad.parameter(xv)
  loss = (x.reverse([1]) * agrad.parameter(wv)).sum((0, 1))
  assert _cos(_g(loss, x), wv[:, ::-1]) > 0.99


@requires_ane
def test_reduce_log_sum_exp_grad():
  xv = rng.standard_normal((2, 6)).astype(np.float32)
  x = agrad.parameter(xv)
  loss = x.reduce_log_sum_exp((1,)).sum((0, 1))
  sm = np.exp(xv - xv.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
  assert _cos(_g(loss, x), sm) > 0.99


@requires_ane
def test_pow_grad():
  xv = (np.abs(rng.standard_normal((2, 6))) + 0.5).astype(np.float32); wv = rng.standard_normal((2, 6)).astype(np.float32)
  x = agrad.parameter(xv); p = agrad.parameter(np.full((2, 6), 2.0, np.float32))
  loss = (x.pow(p) * agrad.parameter(wv)).sum((0, 1))
  assert _cos(_g(loss, x), 2 * xv * wv) > 0.99


@requires_ane
def test_prelu_forward_and_grad():
  N, C, H, W = 1, 4, 2, 2
  xv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  wv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  al = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
  net = af.compile(af.input((N, C, H, W)).prelu(al))
  out = _run(net, xv).reshape(N, C, H, W)
  ref = np.where(xv > 0, xv, al.reshape(1, C, 1, 1) * xv)
  assert np.allclose(out, ref, atol=3e-2)
  x = agrad.parameter(xv)
  loss = (x.prelu(al) * agrad.parameter(wv)).sum((0, 1, 2, 3))
  dref = np.where(xv > 0, 1.0, al.reshape(1, C, 1, 1)) * wv
  assert _cos(_g(loss, x), dref) > 0.99


@requires_ane
def test_equal_comparison():
  xv = rng.standard_normal((1, 8)).astype(np.float32)
  x = af.input((1, 8))
  net = af.compile(af.select(x.equal(x), x, x))   # x==x everywhere -> returns x
  assert np.allclose(_run(net, xv).reshape(1, 8), xv, atol=3e-2)
