"""Smoke test for aneforge's op surface: one-op graph per op, compiled/run vs numpy."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import aneforge
sys.path.insert(0, str(Path(__file__).resolve().parent))      # tests/ -> import _helpers
import aneforge as af
from _helpers import requires_ane


def check(name, build, ref, *ins, tol=0.02):
  net = af.compile(build())
  out = net(*[i.astype(np.float16) for i in ins])
  r = ref(*[i.astype(np.float32) for i in ins])
  relerr = float(np.abs(out - r).max() / (np.abs(r).max() + 1e-6))
  status = "OK" if relerr < tol else "FAIL"
  print(f"  {name:16s} {status}  relerr {relerr:.5f}")
  return relerr < tol


def _run_all():
  rng = np.random.default_rng(0)
  x = rng.standard_normal((4, 8)).astype(np.float16)
  xp = (np.abs(rng.standard_normal((4, 8))) + 0.5).astype(np.float16)  # positive domain
  img = rng.standard_normal((1, 4, 8, 8)).astype(np.float16)
  ok = []

  print("activations:")
  ok += [
      check("sigmoid", lambda: af.input((4, 8)).sigmoid(), lambda a: 1/(1+np.exp(-a)), x),
      check("tanh", lambda: af.input((4, 8)).tanh(), np.tanh, x),
      check("exp", lambda: af.input((4, 8)).exp(), np.exp, x),
      check("log", lambda: af.input((4, 8)).log(), np.log, xp),
      check("sqrt", lambda: af.input((4, 8)).sqrt(), np.sqrt, xp),
      check("rsqrt", lambda: af.input((4, 8)).rsqrt(), lambda a: 1/np.sqrt(a), xp),
      check("softplus", lambda: af.input((4, 8)).softplus(), lambda a: np.log1p(np.exp(a)), x),
      check("elu", lambda: af.input((4, 8)).elu(), lambda a: np.where(a > 0, a, np.exp(a)-1), x),
      check("leaky_relu", lambda: af.input((4, 8)).leaky_relu(), lambda a: np.where(a > 0, a, 0.01*a), x),
      check("clip", lambda: af.input((4, 8)).clip(-1, 1), lambda a: np.clip(a, -1, 1), x),
      check("relu6", lambda: af.input((4, 8)).relu6(), lambda a: np.clip(a, 0, 6), x),
      check("square", lambda: af.input((4, 8)).square(), lambda a: a*a, x),
      check("erf", lambda: af.input((4, 8)).erf(), lambda a: np.vectorize(math.erf)(a), x),
  ]

  print("binary elementwise:")
  ok += [
      check("div", lambda: af.input((4, 8)) / af.input((4, 8)), lambda a, b: a/b, x, xp),
      check("maximum", lambda: af.maximum(af.input((4, 8)), af.input((4, 8))), np.maximum, x, xp),
      check("minimum", lambda: af.minimum(af.input((4, 8)), af.input((4, 8))), np.minimum, x, xp),
  ]

  print("reductions:")
  ok += [
      check("sum", lambda: af.input((4, 8)).sum(1), lambda a: a.sum(1, keepdims=True), x),
      check("amax", lambda: af.input((4, 8)).amax(1), lambda a: a.max(1, keepdims=True), x),
      check("amin", lambda: af.input((4, 8)).amin(1), lambda a: a.min(1, keepdims=True), x),
  ]

  print("spatial:")
  ok.append(check("avg_pool", lambda: af.input((1, 4, 8, 8)).avg_pool(2),
                  lambda a: a.reshape(1, 4, 4, 2, 4, 2).mean(axis=(3, 5)), img))

  W = rng.standard_normal((4, 3, 2, 2)).astype(np.float16)  # [Cin, Cout, kH, kW]

  def ct_ref(a):
    o = np.zeros((1, 3, 16, 16), np.float32); Wf = W.astype(np.float32)
    for co in range(3):
      for ci in range(4):
        for i in range(8):
          for j in range(8):
            o[0, co, i*2:i*2+2, j*2:j*2+2] += a[0, ci, i, j] * Wf[ci, co]
    return o
  ok.append(check("conv_transpose", lambda: af.conv_transpose(af.input((1, 4, 8, 8)), W, stride=2), ct_ref, img))

  C = 4
  g = (np.abs(rng.standard_normal(C)) + 0.5).astype(np.float16)
  b = rng.standard_normal(C).astype(np.float16)
  m = rng.standard_normal(C).astype(np.float16)
  v = (np.abs(rng.standard_normal(C)) + 0.5).astype(np.float16)

  def bn_ref(a):
    r = lambda t: t.astype(np.float32).reshape(1, C, 1, 1)
    return (a - r(m)) / np.sqrt(r(v) + 1e-3) * r(g) + r(b)
  ok.append(check("batch_norm", lambda: af.batch_norm(af.input((1, 4, 8, 8)), g, b, m, v, eps=1e-3), bn_ref, img))

  print(f"\n{sum(ok)}/{len(ok)} ops correct on the ANE")
  return ok


@requires_ane
def test_op_smoketest():   # pytest entry: every one-op graph compiles + matches numpy on the ANE
  ok = _run_all()
  assert all(ok), f"only {sum(ok)}/{len(ok)} ops correct on the ANE"


if __name__ == "__main__":
  sys.exit(0 if all(_run_all()) else 1)
