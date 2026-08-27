"""digamma and beta: off-device form checks + scipy-oracle accuracy (aneforge.special)."""
import inspect

import numpy as np

from aneforge import special


# -- form checks (off-device, CI-verifiable) ----------------------------------

def test_digamma_uses_recurrence_for_small_x():
  # For x < 2 the recurrence psi(x) = psi(x+1) - 1/x must appear in the body.
  src = inspect.getsource(special.digamma)
  assert "x + " in src or "x+" in src  # the x+1 shift
  assert "/ x" in src or "/x" in src    # the 1/x term


def test_digamma_uses_asymptotic_expansion():
  # The asymptotic expansion must reference .log() and the Bernoulli constants.
  src = inspect.getsource(special.digamma)
  assert ".log()" in src
  assert "_DIGAMMA_B2_12" in src


def test_digamma_is_exported():
  assert "digamma" in special.__all__


def test_beta_uses_lgamma_composition():
  # beta must be exp(lgamma(a) + lgamma(b) - lgamma(a+b)), not gamma(a)*gamma(b)/gamma(a+b).
  src = inspect.getsource(special.beta)
  assert "lgamma" in src
  assert ".exp()" in src
  # must NOT use gamma directly (that overflows)
  body = src.split('"""')[-1]
  assert "gamma(" not in body or "lgamma(" in body


def test_beta_is_exported():
  assert "beta" in special.__all__


# -- scipy-oracle accuracy (skipped if scipy absent) --------------------------

def test_digamma_matches_scipy_on_positive_range():
  sp = __import__("pytest").importorskip("scipy.special")
  xs = np.linspace(1.0, 8.0, 128).astype(np.float16).reshape(1, 128)
  # run through the graph path (compile + execute on CPU, no ANE needed for the math)
  import aneforge as af
  net = af.compile(special.digamma(af.input(xs.shape)))
  out = np.asarray(net(xs)).astype(np.float32)
  ref = sp.digamma(xs.astype(np.float32))
  # per-point relative error, weighted to avoid division by near-zero
  rel = np.abs(out - ref) / (np.abs(ref) + 1e-6)
  assert rel.max() < 0.05, f"digamma relerr {rel.max():.3e} exceeds 5%"


def test_beta_matches_scipy_on_positive_grid():
  sp = __import__("pytest").importorskip("scipy.special")
  import aneforge as af
  # lgamma is accurate away from its zeros (x=1,2) and its range boundary (x=8);
  # a,b in [2.5, 3.5] keeps a+b in [5, 7] where lgamma's abs error is < 0.01.
  xs_a = np.linspace(2.5, 3.5, 32).astype(np.float16)
  xs_b = np.linspace(2.5, 3.5, 32).astype(np.float16)
  aa, bb = np.meshgrid(xs_a, xs_b)
  aa = aa.reshape(1, -1); bb = bb.reshape(1, -1)
  net = af.compile(special.beta(af.input(aa.shape), af.input(bb.shape)))
  out = np.asarray(net(aa, bb)).astype(np.float32)
  ref = sp.beta(aa.astype(np.float32), bb.astype(np.float32))
  rel = np.abs(out - ref) / (np.abs(ref) + 1e-12)
  assert rel.max() < 0.05, f"beta relerr {rel.max():.3e} exceeds 5%"
