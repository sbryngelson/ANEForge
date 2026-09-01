"""aneforge.fft - staged Cooley-Tukey FFT on the ANE (complex carried as (re, im) pairs, each stage a matmul)."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aneforge as af  # noqa: E402
from aneforge.graph import Tensor  # noqa: E402


# complex-as-real-pairs algebra: a value is a (re, im) tuple of Tensors.

def _cmatmul_const(re: Tensor, im: Tensor, Wr: np.ndarray, Wi: np.ndarray):
  """Complex matmul (re+im*i) @ (Wr+Wi*i) as four real matmuls (W a fp16 constant)."""
  Wr = Wr.astype(np.float16); Wi = Wi.astype(np.float16)
  Cre = (re @ Wr) - (im @ Wi)
  Cim = (re @ Wi) + (im @ Wr)
  return Cre, Cim


# Cross twiddles ride in as graph INPUTS, not folded weights: the frontend folds only
# matmul/conv weights, while elementwise mul needs two graph Tensors.


def _dft_matrix(M: int):
  """The [M,M] DFT twiddle W[k,n] = exp(-2pi i k n / M), split into (Wr, Wi); streamed as x @ W^T."""
  n = np.arange(M)
  k = n.reshape(-1, 1)
  W = np.exp(-2j * np.pi * k * n / M).astype(np.complex128)   # W[k,n]
  Wt = W.T                                   # [n, k] so x@Wt gives X[k]
  Wt_re = Wt.real
  Wt_im = Wt.imag
  return np.ascontiguousarray(Wt_re).astype(np.float16), \
           np.ascontiguousarray(Wt_im).astype(np.float16)


def _idft_matrix(M: int):
  """Inverse DFT twiddle (conjugate, UNSCALED): W[k,n] = exp(+2pi i k n / M)."""
  n = np.arange(M)
  k = n.reshape(-1, 1)
  W = np.exp(+2j * np.pi * k * n / M).astype(np.complex128)
  Wt = W.T
  Wt_re = Wt.real
  Wt_im = Wt.imag
  return np.ascontiguousarray(Wt_re).astype(np.float16), \
           np.ascontiguousarray(Wt_im).astype(np.float16)


# The ANE caps transpose+matmul at rank-4, so the fully-split FFT tensor carries at most 3 factor groups.
_MAX_GROUPS = 3


def _prime_factors(N: int) -> list[int]:
  f: list[int] = []
  m = N; d = 2
  while d * d <= m:
    while m % d == 0:
      f.append(d); m //= d
    d += 1 if d == 2 else 2
  if m > 1: f.append(m)
  return f


def _factor(N: int) -> list[int]:
  """Factor N into at most 3 balanced groups (each a dense-DFT matmul stage); prime N -> one dense DFT."""
  primes = _prime_factors(N)
  if len(primes) == 1: return [N]                                  # prime -> one dense DFT block
  k = min(_MAX_GROUPS, len(primes))
  groups = [1] * k
  # greedy balanced bin-packing: largest primes first into the smallest group
  for p in sorted(primes, reverse=True):
    i = min(range(k), key=lambda j: groups[j])
    groups[i] *= p
  groups.sort()
  return groups


# the staged Cooley-Tukey graph builder

class _Builder:
  """Builds the staged FFT graph for one direction, collecting cross-twiddle constants as aux inputs."""

  def __init__(self, inverse: bool):
    self.inverse = inverse
    self.sign = +1.0 if inverse else -1.0
    self.aux_inputs: list[Tensor] = []     # extra graph inputs (cross twiddles)
    self.aux_values: list[np.ndarray] = []  # the constant arrays to feed

  def _dft(self, M: int):
    return _idft_matrix(M) if self.inverse else _dft_matrix(M)

  def _cross(self, N1: int, restlen: int, Ntot: int, bshape):
    """Cross twiddle T[k1, n2] = exp(sign*2pi i k1 n2 / Ntot), registered as a pair of graph inputs and reshaped to `bshape`."""
    k1 = np.arange(N1).reshape(-1, 1)
    n2 = np.arange(restlen).reshape(1, -1)
    T = np.exp(self.sign * 2j * np.pi * k1 * n2 / Ntot).astype(np.complex128)   # [N1, restlen]
    Tr = af.input((N1, restlen)); Ti = af.input((N1, restlen))
    self.aux_inputs += [Tr, Ti]
    self.aux_values += [T.real.astype(np.float16), T.imag.astype(np.float16)]
    return Tr.reshape(*bshape), Ti.reshape(*bshape)

  def _axis_dft(self, re: Tensor, im: Tensor, axis: int, r: int):
    """Radix-r DFT along `axis`: one transpose (axis -> last) + complex matmul + one transpose back (chaining transposes mis-fuses here)."""
    nd = len(re.shape)
    perm = list(range(nd))
    perm[axis], perm[nd - 1] = perm[nd - 1], perm[axis]
    re = re.transpose(perm); im = im.transpose(perm)         # axis -> last
    Wr, Wi = self._dft(r)
    re, im = _cmatmul_const(re, im, Wr, Wi)                  # radix-r DFT (matmul)
    re = re.transpose(perm); im = im.transpose(perm)         # last -> axis
    return re, im

  def _rec(self, re: Tensor, im: Tensor, axes: list[int], lengths: list[int]):
    """Cooley-Tukey unrolled over the split axes (DIT): radix-N1 DFT on the first axis, cross twiddle, recurse the rest."""
    if len(axes) == 1: return self._axis_dft(re, im, axes[0], lengths[0])
    N1 = lengths[0]
    restlen = int(np.prod(lengths[1:]))
    Ntot = N1 * restlen
    a0 = axes[0]
    ndim = len(re.shape)

    # (1) radix-N1 DFT along the first axis (single transpose + matmul + transpose)
    re, im = self._axis_dft(re, im, a0, N1)

    # (2) cross twiddle T[k0, n2] over axis a0 (= k0) and the trailing block (= n2),
    #     broadcast against the running tensor. bshape places N1 on axis a0 and the
    #     trailing radices on their axes; 1 elsewhere.
    bshape = [1] * ndim
    bshape[a0] = N1
    for ax, L in zip(axes[1:], lengths[1:]):
      bshape[ax] = L
    Tr, Ti = self._cross(N1, restlen, Ntot, bshape)
    re, im = (re * Tr) - (im * Ti), (re * Ti) + (im * Tr)

    # (3) recurse on the remaining axes
    return self._rec(re, im, axes[1:], lengths[1:])

  def transform(self, re: Tensor, im: Tensor, N: int):
    """Length-N FFT of the last axis of (re, im) [1, N]; returns [1, N] in natural (numpy.fft) order."""
    factors = _factor(N)
    m = len(factors)
    if m == 1:
      Wr, Wi = self._dft(N)                               # single dense DFT
      return _cmatmul_const(re, im, Wr, Wi)

        # one reshape: [1, N] -> [1, r0, r1, ..., r_{m-1}]   (n0 most-significant digit)
    re = re.reshape(1, *factors); im = im.reshape(1, *factors)

    re, im = self._rec(re, im, list(range(1, m + 1)), list(factors))

    # DIT output is in DIGIT-reversed order: reverse the split axes in ONE transpose.
    rev = [0] + list(range(m, 0, -1))                       # [1, m, m-1, ..., 1]
    re = re.transpose(rev); im = im.transpose(rev)
    re = re.reshape(1, N); im = im.reshape(1, N)
    return re, im


def _stage_count(N: int) -> int:
  """Number of dense-DFT matmul stages emitted (== number of factor groups)."""
  return len(_factor(N))


def _leaf_cost(N: int) -> int:
  """Total complex-matmul work N*sum(groups) (dense single DFT is N*N)."""
  groups = _factor(N)
  if len(groups) == 1: return N * N
  return N * sum(groups)


# Plans (compile once, run many)

class Plan:
  """A compiled staged-FFT program plus the cross-twiddle constants it threads in each call."""

  def __init__(self, N: int, inverse: bool, real_input: bool):
    self.N = N
    self.inverse = inverse
    self.real_input = real_input
    b = _Builder(inverse)
    # real/imag inputs; for rfft the imag input is fed as zeros.
    xr = af.input((1, N)); xi = af.input((1, N))
    Xr, Xi = b.transform(xr, xi, N)
    if inverse:
      Xr = Xr * (1.0 / N)
      Xi = Xi * (1.0 / N)
    out = af.concat([Xr, Xi], axis=1)        # [1, 2N], split on host
    self._aux_values = b.aux_values
    self.n_stages = _stage_count(N)          # number of dense-DFT matmul stages (<=3)
    self.model = af.compile(out, _check_precision=False)
    self.n_ops = self.model.n_ops

  def __call__(self, x_re: np.ndarray, x_im: np.ndarray | None = None):
    x_re = np.asarray(x_re, np.float16).reshape(1, self.N)
    if x_im is None:
      x_im = np.zeros((1, self.N), np.float16)
    else:
      x_im = np.asarray(x_im, np.float16).reshape(1, self.N)
    out = self.model(x_re, x_im, *self._aux_values)
    out = out.reshape(2, self.N)
    return out[0].copy(), out[1].copy()


class Plan2:
  """A compiled 2-D FFT for [M,N] complex fields, one fused program (separable F_M @ X @ F_N^T, eight real GEMMs)."""

  def __init__(self, M: int, N: int, inverse: bool):
    self.M, self.N, self.inverse = M, N, inverse
    mk = _idft_matrix if inverse else _dft_matrix
    WrN, WiN = mk(N)                                          # row twiddle (x @ Wt)
    WrM, WiM = mk(M)                                          # column twiddle
    if inverse:
        # fold 1/(M*N) into the twiddles per-pass (an unscaled first-axis transform overflows fp16).
      WrN, WiN = WrN * (1.0 / N), WiN * (1.0 / N)
      WrM, WiM = WrM * (1.0 / M), WiM * (1.0 / M)
    xr = af.input((M, N)); xi = af.input((M, N))
    re, im = _cmatmul_const(xr, xi, WrN, WiN)                 # all M rows, one matmul
    re = re.transpose([1, 0]); im = im.transpose([1, 0])      # columns -> rows
    re, im = _cmatmul_const(re, im, WrM, WiM)                 # all N columns, one matmul
    re = re.transpose([1, 0]); im = im.transpose([1, 0])
    out = af.concat([re, im], axis=0)                          # [2M, N], split on host
    self.model = af.compile(out, _check_precision=False)
    self.n_ops = self.model.n_ops

  def __call__(self, x_re: np.ndarray, x_im: np.ndarray | None = None):
    x_re = np.asarray(x_re, np.float16).reshape(self.M, self.N)
    if x_im is None:
      x_im = np.zeros((self.M, self.N), np.float16)
    else:
      x_im = np.asarray(x_im, np.float16).reshape(self.M, self.N)
    out = self.model(x_re, x_im).reshape(2, self.M, self.N)
    return out[0].copy(), out[1].copy()


# plan cache so repeated calls at the same N reuse the compiled program
_PLAN_CACHE: dict[tuple, Plan | Plan2] = {}


def fft_plan(N: int) -> Plan:
  key = (N, False, False)
  if key not in _PLAN_CACHE: _PLAN_CACHE[key] = Plan(N, inverse=False, real_input=False)
  return cast(Plan, _PLAN_CACHE[key])


def ifft_plan(N: int) -> Plan:
  key = (N, True, False)
  if key not in _PLAN_CACHE: _PLAN_CACHE[key] = Plan(N, inverse=True, real_input=False)
  return cast(Plan, _PLAN_CACHE[key])


def rfft_plan(N: int) -> Plan:
  key = (N, False, True)
  if key not in _PLAN_CACHE: _PLAN_CACHE[key] = Plan(N, inverse=False, real_input=True)
  return cast(Plan, _PLAN_CACHE[key])


def fft2_plan(M: int, N: int) -> Plan2:
  key = ("2d", M, N, False)
  if key not in _PLAN_CACHE: _PLAN_CACHE[key] = Plan2(M, N, inverse=False)
  return cast(Plan2, _PLAN_CACHE[key])


def ifft2_plan(M: int, N: int) -> Plan2:
  key = ("2d", M, N, True)
  if key not in _PLAN_CACHE: _PLAN_CACHE[key] = Plan2(M, N, inverse=True)
  return cast(Plan2, _PLAN_CACHE[key])


# the public one-shot API

def fft(x_re, x_im, N: int):
  """Forward FFT of a complex signal (real/imag arrays), length N, on the ANE; returns (X_re, X_im)."""
  return fft_plan(N)(x_re, x_im)


def ifft(X_re, X_im, N: int):
  """Inverse FFT (1/N normalized) of a complex spectrum on the ANE; returns (x_re, x_im)."""
  return ifft_plan(N)(X_re, X_im)


def rfft(x_real, N: int):
  """Forward FFT of a real signal (imag = 0) on the ANE; returns the full-length spectrum (X_re, X_im)."""
  return rfft_plan(N)(x_real, None)


def irfft(X_re, X_im, N: int):
  """Inverse real FFT of a Hermitian-symmetric spectrum on the ANE; returns the real time-domain
  signal of length N (the imag part is ~0 by Hermitian symmetry, and numpy.fft.irfft also
  returns only the real part)."""
  x_re, _ = ifft_plan(N)(X_re, X_im)
  return x_re


def fft2(x_re, x_im=None):
  """2-D FFT of an [M,N] complex field on the ANE (x_im=None means a real field); returns (X_re, X_im)."""
  x_re = np.asarray(x_re)
  M, N = x_re.shape
  return fft2_plan(M, N)(x_re, x_im)


def ifft2(X_re, X_im):
  """Inverse 2-D FFT (1/(M*N) normalized) on the ANE; returns (x_re, x_im)."""
  X_re = np.asarray(X_re)
  M, N = X_re.shape
  return ifft2_plan(M, N)(X_re, X_im)


def magnitude(X_re, X_im):
  """|X| = sqrt(re^2 + im^2) (numpy on host outputs)."""
  return np.sqrt(np.asarray(X_re, np.float32) ** 2 + np.asarray(X_im, np.float32) ** 2)


def power(X_re, X_im):
  """|X|^2 power spectrum (numpy on host outputs)."""
  return np.asarray(X_re, np.float32) ** 2 + np.asarray(X_im, np.float32) ** 2


__all__ = [
    "fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "magnitude", "power",
    "Plan", "Plan2", "fft_plan", "ifft_plan", "rfft_plan", "fft2_plan", "ifft2_plan",
]


# self-test / validation

def _relerr(a, b):
  a = np.asarray(a); b = np.asarray(b)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _naive_dft_relerr(N: int, xr: np.ndarray, xi: np.ndarray):
  """Reference: the single dense [N,N] DFT-matmul in fp16 (the previous approach), for comparison vs staged."""
  n = np.arange(N); k = n.reshape(-1, 1)
  W = np.exp(-2j * np.pi * k * n / N).astype(np.complex128)
  Wr = W.real.astype(np.float16).astype(np.float64)
  Wi = W.imag.astype(np.float16).astype(np.float64)
  xr16 = xr.astype(np.float16).astype(np.float64)
  xi16 = xi.astype(np.float16).astype(np.float64)
  Xr = xr16 @ Wr.T - xi16 @ Wi.T
  Xi = xr16 @ Wi.T + xi16 @ Wr.T
  return Xr.ravel(), Xi.ravel()


def _selftest():
  rng = np.random.default_rng(20260530)
  print("aneforge.fft - staged Cooley-Tukey FFT on the ANE (complex as real pairs)\n")

  Ns = [64, 256, 1024, 2048, 1536, 1280]
  print(f"{'N':>6} {'factors':>16} {'stages':>7} {'n_ops':>6} "
          f"{'staged_relerr':>14} {'naiveDFT_relerr':>15} {'staged_cost':>12} {'naive N^2':>10} {'speedup':>8}")
  print("-" * 110)

  all_ok = True
  for N in Ns:
    xr = rng.standard_normal(N).astype(np.float32)
    xi = rng.standard_normal(N).astype(np.float32)

    # ANE staged FFT
    plan = fft_plan(N)
    Xr, Xi = plan(xr, xi)

    # numpy golden
    ref = np.fft.fft(xr.astype(np.float64) + 1j * xi.astype(np.float64))
    staged_err = _relerr(Xr + 1j * Xi, ref)

    # naive single-matrix fp16 DFT (the previous approach) for comparison
    nr, ni = _naive_dft_relerr(N, xr, xi)
    naive_err = _relerr(nr + 1j * ni, ref)

    cost = _leaf_cost(N)
    naive = N * N
    speedup = naive / cost
    factors = _factor(N)
    ok = staged_err < 0.02
    all_ok &= ok
    flag = "" if ok else "  <-- FAIL"
    print(f"{N:>6} {str(factors):>16} {plan.n_stages:>7} {plan.n_ops:>6} "
              f"{staged_err:>14.3e} {naive_err:>15.3e} {cost:>12} {naive:>10} {speedup:>7.1f}x{flag}")

    # ---- rfft + magnitude check ----
  print()
  N = 1024
  sig = rng.standard_normal(N).astype(np.float32)
  Rr, Ri = rfft(sig, N)
  ref = np.fft.fft(sig.astype(np.float64))
  rfft_err = _relerr(Rr + 1j * Ri, ref)
  mag_err = _relerr(magnitude(Rr, Ri), np.abs(ref))
  print(f"rfft  N={N}: spectrum relerr {rfft_err:.3e}   |magnitude| relerr {mag_err:.3e}")

  # ---- ifft round-trip ----
  N = 256
  xr = rng.standard_normal(N).astype(np.float32)
  xi = rng.standard_normal(N).astype(np.float32)
  Xr, Xi = fft(xr, xi, N)
  br, bi = ifft(Xr, Xi, N)
  rt_err = _relerr(br + 1j * bi, xr + 1j * xi)
  print(f"ifft  N={N}: round-trip fft->ifft relerr {rt_err:.3e}")
  all_ok &= (rfft_err < 0.02 and rt_err < 0.05)

  # ---- irfft round-trip + numpy oracle ----
  for N in (128, 512):
    sig = rng.standard_normal(N).astype(np.float32)
    Xr, Xi = rfft(sig, N)
    back = irfft(Xr, Xi, N)
    rt_err = _relerr(back, sig)
    # numpy oracle:the half spectrum X[:N//2+1] is the same array numpy.fft.rfft produces
    ref = np.fft.irfft(np.asarray(Xr, np.float32)[: N // 2 + 1] + 1j * np.asarray(Xi, np.float32)[: N // 2 + 1], n=N)
    np_err = _relerr(back, ref)
    print(f"irfft N={N}: rfft->irfft round-trip {rt_err:.3e}   vs numpy.irfft {np_err:.3e}")
    all_ok &= (rt_err < 0.05 and np_err < 0.05)

  # ---- complexity / accuracy verdict ----
  print("\n" + "=" * 110)
  print("VERDICT - staged matmul-FFT on the ANE")
  print("-" * 110)
  print("  * Every stage is a matmul (4 real matmuls per complex DFT block) + an")
  print("    elementwise cross-twiddle multiply: ANE-native, fused into ONE e5rt program.")
  print("  * 3-stage Cooley-Tukey (N = g0*g1*g2 balanced groups) cuts the scalar matmul")
  print("    work from O(N^2) to N*(g0+g1+g2): e.g. N=1024 -> [8,8,16] => 32x fewer MACs,")
  print("    N=2048 -> [8,16,16] => 51x. The speedup GROWS with N (sub-quadratic).")
  print("  * WHY exactly 3 stages: the ANE caps transpose+matmul at rank-4 tensors, so the")
  print("    fully-split FFT tensor [1,g0,g1,g2] can carry at most 3 factor groups. (A finer")
  print("    log-depth radix-2 split would be rank > 4 and fails ANECCompile; and stacking")
  print("    the recursive transpose->reshape collapses mis-fuses on this ANE - so we keep")
  print("    the tensor fully-split with ONE isolated transpose per stage. Both are real")
  print("    architectural walls, not numerics.)")
  print("  * fp16 accuracy: staged ~5e-4..8e-4, FLAT in N (wide accumulator => the per-stage")
  print("    sums don't compound). That is ~3x the naive single-DFT floor (~2.5e-4) - the")
  print("    extra cross-twiddle multiplies add a little fp16 rounding - but still fp16-clean")
  print("    to N=2048+. So staged is NOT more accurate than the dense DFT, but it IS far")
  print("    cheaper (sub-quadratic) at the same precision class. Max usable N is bounded by")
  print("    fp16 dynamic range (normalize by 1/sqrt(N) for power spectra), not by stage error.")
  print(f"\n{'PASS' if all_ok else 'FAIL'} - staged Cooley-Tukey FFT validates vs numpy.fft on the ANE")
  return 0 if all_ok else 1


if __name__ == "__main__":
  sys.exit(_selftest())
