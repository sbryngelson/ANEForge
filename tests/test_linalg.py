"""Pytest coverage for the on-ANE linear-algebra kernels (aneforge.linalg)."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pytest
import aneforge.linalg as L
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE

f16 = np.float16


def relerr(a, b):
  a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _general(m, n, cond, seed):
  r = np.random.default_rng(seed)
  U = np.linalg.qr(r.standard_normal((m, max(m, n))))[0][:, :min(m, n)]
  V = np.linalg.qr(r.standard_normal((n, n)))[0]
  s = np.geomspace(1.0, cond, min(m, n))
  return (U * s) @ V[:min(m, n)].T if m >= n else ((U * s) @ V[:, :min(m, n)].T)


def _square(n, cond, seed):
  r = np.random.default_rng(seed)
  U = np.linalg.qr(r.standard_normal((n, n)))[0]; V = np.linalg.qr(r.standard_normal((n, n)))[0]
  return (U * np.geomspace(1.0, cond, n)) @ V.T


def _spd(n, cond, seed):
  r = np.random.default_rng(seed); Q = np.linalg.qr(r.standard_normal((n, n)))[0]
  return (Q * np.geomspace(1.0, cond, n)) @ Q.T


def _sym_gapped(n, cond, seed):
  r = np.random.default_rng(seed); Q = np.linalg.qr(r.standard_normal((n, n)))[0]
  ev = np.concatenate([[float(cond)], np.geomspace(0.25 * cond, 1.0, n - 1)])  # clear leading gap
  return (Q * ev) @ Q.T


# ----------------------------- solvers ----------------------------- #

@pytest.mark.parametrize("cond,tol", [(1e1, 5e-3), (1e2, 5e-2)])
def test_conjugate_gradient(cond, tol):
  A = _spd(48, cond, int(cond) + 1); x = np.random.default_rng(0).standard_normal(48)
  assert relerr(L.conjugate_gradient(f16(A), f16(A @ x), iters=40), x) <= tol


def test_jacobi():
  r = np.random.default_rng(3); n = 32
  A = r.standard_normal((n, n)) * 0.1 + np.diag(4.0 + r.random(n) * np.arange(1, n + 1))
  x = r.standard_normal(n)
  assert relerr(L.jacobi(f16(A), f16(A @ x), iters=80), x) <= 1e-2


@pytest.mark.parametrize("cond,tol", [(1e1, 1e-2), (1e2, 3e-2)])
def test_gmres_general_solve(cond, tol):
  A = _square(10, cond, int(cond) + 2); x = np.random.default_rng(1).standard_normal(10)
  assert relerr(L.gmres(f16(A), f16(A @ x)), x) <= tol


@pytest.mark.parametrize("shape", [(80, 24), (40, 40)])  # overdetermined + square
def test_lsqr(shape):
  m, n = shape; r = np.random.default_rng(5)
  A = _general(m, n, 1e1, 7); xt = r.standard_normal(n); b = A @ xt + 0.01 * r.standard_normal(m)
  ref = np.linalg.lstsq(A, b, rcond=None)[0]
  assert relerr(L.lsqr(f16(A), f16(b), iters=80), ref) <= 5e-2


def test_iterative_refine():
  A = _spd(48, 1e1, 5); x = np.random.default_rng(2).standard_normal(48); b = A @ x
  x0 = L.conjugate_gradient(f16(A), f16(b), iters=6)            # under-iterated
  xr = L.iterative_refine(f16(A), f16(b), x0, iters=3, inner=80)
  assert relerr(xr, x) < relerr(x0, x)                          # refinement improves it


def test_least_squares():
  r = np.random.default_rng(9); A = _general(80, 24, 1e1, 9)
  xt = r.standard_normal(24); b = A @ xt + 0.01 * r.standard_normal(80)
  ref = np.linalg.lstsq(A, b, rcond=None)[0]
  assert relerr(L.least_squares(f16(A), f16(b), iters=40, refine=2), ref) <= 5e-2


# --------------------- eigen / singular values --------------------- #

@pytest.mark.parametrize("cond", [1e1, 1e2])
def test_dominant_eig(cond):
  A = _sym_gapped(40, cond, 7); ev = np.linalg.eigvalsh(A); ref = ev[np.argmax(np.abs(ev))]
  lam, _ = L.dominant_eig(f16(A), iters=80)
  assert abs(lam - ref) / abs(ref) <= 1e-2


@pytest.mark.parametrize("shape", [(50, 30), (30, 50)])  # tall + wide
def test_dominant_svd(shape):
  m, n = shape; r = np.random.default_rng(sum(shape))
  U = np.linalg.qr(r.standard_normal((m, min(m, n))))[0]; V = np.linalg.qr(r.standard_normal((n, min(m, n))))[0]
  s = np.concatenate([[10.0], np.geomspace(2.5, 1.0, min(m, n) - 1)])   # gap
  A = (U * s) @ V.T; ref = np.linalg.svd(A, compute_uv=False)[0]
  sig, _, _ = L.dominant_svd(f16(A), iters=80)
  assert abs(sig - ref) / ref <= 2e-2


def test_generalized_eigh():            # A x = lambda B x (sygv), composed on-engine
  n = 8; r = np.random.default_rng(9)
  Q = np.linalg.qr(r.standard_normal((n, n)))[0]
  A = (Q * (np.geomspace(1, 1e1, n) * r.choice([-1, 1], n))) @ Q.T
  Qb = np.linalg.qr(r.standard_normal((n, n)))[0]; B = (Qb * np.geomspace(1, 1e1, n)) @ Qb.T
  Lc = np.linalg.cholesky(B); Li = np.linalg.inv(Lc)
  ref = np.sort(np.linalg.eigvalsh(Li @ A @ Li.T))
  assert relerr(L.generalized_eigh(f16(A), f16(B)), ref) <= 2e-2


def test_eigvals_general_real():        # nonsymmetric A, real spectrum, unshifted QR on-engine
  r = np.random.default_rng(0); V = r.standard_normal((6, 6))
  A = V @ np.diag([8., 5., 3., 2., 1.5, 1.]) @ np.linalg.inv(V)
  got = np.sort_complex(L.eigvals(f16(A), iters=60)); ref = np.sort_complex(np.linalg.eigvals(A))
  assert np.linalg.norm(got - ref) / np.linalg.norm(ref) <= 1e-2


def test_eigvals_general_complex():     # nonsymmetric A with complex-conjugate eigenpairs
  r = np.random.default_rng(0); V = r.standard_normal((6, 6)); B = np.zeros((6, 6))
  for i, (a, b) in enumerate([(3, 2), (1.5, 1), (0.8, 0.5)]):
    B[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[a, b], [-b, a]]
  A = V @ B @ np.linalg.inv(V)
  got = np.sort_complex(L.eigvals(f16(A), iters=60)); ref = np.sort_complex(np.linalg.eigvals(A))
  assert np.linalg.norm(got - ref) / np.linalg.norm(ref) <= 4e-2


@pytest.mark.parametrize("cond", [1e1, 1e2])
def test_eigh_full_spectrum(cond):
  A = _sym_gapped(8, cond, int(cond) + 9)
  assert relerr(L.eigh(f16(A), sweeps=8), np.sort(np.linalg.eigvalsh(A))) <= 1.5e-2


def test_eigh_iterate_matches_unrolled():
  # iterate path and unrolled path use the same rotations -> fp16-exact at small n
  A = _sym_gapped(8, 1e1, 19)
  un = L.eigh(f16(A), sweeps=8)
  it = L.eigh(f16(A), sweeps=8, iterate=True)
  assert relerr(it, un) <= 1e-6
  assert relerr(it, np.sort(np.linalg.eigvalsh(A))) <= 1.5e-2


def test_eigh_iterate_beyond_unrolled_ceiling():
  # n=32 exceeds the ~n=20 unrolled cap; only the iterate path reaches it
  A = _sym_gapped(32, 1e1, 32)
  ref = np.sort(np.linalg.eigvalsh(A))
  assert relerr(L.eigh(f16(A), sweeps=10, iterate=True), ref) <= 3e-2


@pytest.mark.parametrize("shape", [(8, 8), (10, 8), (8, 10)])  # square + tall + WIDE (the bug)
def test_svd_full_shapes(shape):
  m, n = shape; A = _general(m, n, 1e1, int(m * 10 + n))
  assert relerr(L.svd(f16(A), sweeps=8), np.linalg.svd(A, compute_uv=False)) <= 3e-2


def test_svdvals_topk():
  r = np.random.default_rng(7); Ul = np.linalg.qr(r.standard_normal((128, 8)))[0]
  Vr = np.linalg.qr(r.standard_normal((96, 8)))[0]
  A = (Ul * np.geomspace(10, 1, 8)) @ Vr.T + 0.01 * r.standard_normal((128, 96))
  S = L.svdvals_topk(f16(A), k=8, oversample=2, power_iters=1)
  assert relerr(S, np.linalg.svd(A, compute_uv=False)[:8]) <= 3e-2


def test_randomized_svd_values():
  r = np.random.default_rng(7); Ul = np.linalg.qr(r.standard_normal((128, 8)))[0]
  Vr = np.linalg.qr(r.standard_normal((96, 8)))[0]
  A = (Ul * np.geomspace(10, 1, 8)) @ Vr.T + 0.01 * r.standard_normal((128, 96))
  _, Sv, _ = L.randomized_svd(f16(A), k=8, oversample=5, power_iters=2)
  assert relerr(Sv, np.linalg.svd(A, compute_uv=False)[:8]) <= 1e-2


# ------------------- direct factorizations (recon) ------------------- #

@pytest.mark.parametrize("shape", [(8, 8), (10, 6)])  # square + tall
def test_qr_reconstruction(shape):
  m, n = shape; A = _general(m, n, 1e1, int(m + n))
  Q, R = L.qr(f16(A))
  assert relerr(Q @ R, A) <= 5e-3
  assert relerr(Q.T @ Q, np.eye(n)) <= 5e-2          # orthonormal columns (fp16-loose)


@pytest.mark.parametrize("cond", [1e1, 1e2])
def test_cholesky_reconstruction(cond):
  A = _spd(8, cond, int(cond) + 22); Lc = L.cholesky(f16(A))
  assert relerr(Lc @ Lc.T, A) <= 5e-3


@pytest.mark.parametrize("cond", [1e1, 1e2])
def test_lu_reconstruction(cond):
  A = _square(8, cond, int(cond) + 23); Lm, Um = L.lu(f16(A))
  assert relerr(Lm @ Um, A) <= 2e-2


def test_factorizations_large_entries():
  # entries above the slice-x16 saturation threshold (4094) but inside fp16 range: the
  # routed accessor keeps chol/lu finite where a direct width-offset slice returns inf
  for mx in (4500.0, 8000.0):
    A = _spd(6, 1e1, 31); A = A / np.abs(A).max() * mx
    Lc = L.cholesky(f16(A))
    assert np.isfinite(Lc).all() and relerr(Lc @ Lc.T, A) <= 5e-3
    B = np.random.default_rng(32).standard_normal((6, 6)) + np.eye(6) * mx
    Lm, Um = L.lu(f16(B))
    assert np.isfinite(Lm @ Um).all() and relerr(Lm @ Um, B) <= 2e-2


def test_lu_pivoted():                  # P A = L U with on-engine argmax pivoting (segmented)
  A = _square(8, 1e1, 99)
  A[0, 0] = 1e-3                       # tiny leading pivot: unpivoted would be unstable
  P, Lp, Up = L.lu_pivoted(f16(A))
  assert relerr(P @ A, Lp @ Up) <= 3e-3
  assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)   # P is a permutation
  assert np.allclose(np.tril(Up, -1), 0)                          # U upper-triangular


# ------------- det/slogdet, pinv, solve_triangular, bicgstab ------------- #

@pytest.mark.parametrize("cond", [1e1, 1e2])
def test_det_matches_numpy(cond):
  A = _square(8, cond, int(cond) + 41)
  ref = np.linalg.det(A)
  assert abs(L.det(f16(A)) - ref) / (abs(ref) + 1e-30) <= 5e-2

def test_slogdet_matches_numpy():
  A = _square(8, 1e2, 43) * 3.0                       # scaled so the raw product is fp16-hostile
  sr, lr = np.linalg.slogdet(A)
  sg, lg = L.slogdet(f16(A))
  assert sg == sr and abs(lg - lr) <= 5e-2 * max(1.0, abs(lr))

def test_pinv_rank_k_matches_numpy():
  r = np.random.default_rng(7)
  Ul = np.linalg.qr(r.standard_normal((40, 6)))[0]; Vr = np.linalg.qr(r.standard_normal((30, 6)))[0]
  A = (Ul * np.geomspace(8, 1, 6)) @ Vr.T             # exactly rank 6
  P = L.pinv(f16(A), k=6)
  assert relerr(P, np.linalg.pinv(A)) <= 5e-2
  assert relerr(A @ P @ A, A) <= 5e-2                 # Moore-Penrose: A P A = A

@pytest.mark.parametrize("lower", [True, False])
def test_solve_triangular(lower):
  r = np.random.default_rng(11); n = 8
  T = np.tril(r.standard_normal((n, n))) + np.eye(n) * 4.0
  if not lower: T = T.T
  x = r.standard_normal(n); b = T @ x
  assert relerr(L.solve_triangular(f16(T), f16(b), lower=lower), x) <= 5e-3

def test_solve_triangular_large_entries():
  # entries above the slice-x16 saturation threshold: the routed accessor keeps the solve finite
  r = np.random.default_rng(12); n = 6
  T = np.tril(r.standard_normal((n, n))) + np.eye(n) * 6.0
  T = T / np.abs(T).max() * 5000.0
  x = r.standard_normal(n); b = T @ x
  y = L.solve_triangular(f16(T), f16(b))
  assert np.isfinite(y).all() and relerr(y, x) <= 1e-2

@pytest.mark.parametrize("shape", [(6, 5), (16, 16), (32, 8)])
@pytest.mark.parametrize("order", ["fro", 1, np.inf])
def test_norm_matches_numpy(shape, order):
  m, n = shape
  A = f16(np.random.default_rng(m * 100 + n).standard_normal((m, n)))
  got, ref = L.norm(A, order), np.linalg.norm(A.astype(np.float64), order)
  assert abs(got - ref) / abs(ref) <= 5e-3, f"norm(order={order!r}) {shape}: {got} vs {ref}"


def test_norm_rejects_bad_input():
  with pytest.raises(ValueError): L.norm(np.zeros((2, 2, 2), f16))       # not 2-D
  with pytest.raises(ValueError): L.norm(np.zeros((2, 2), f16), 2)       # spectral norm: use svdvals


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5, 8])
def test_matrix_power_matches_numpy(k):
  # well-conditioned (cond 2) so the fp16 error from repeated squaring stays bounded
  r = np.random.default_rng(7); n = 8
  Q = np.linalg.qr(r.standard_normal((n, n)))[0]
  A = f16((Q * np.geomspace(1.0, 2.0, n)) @ Q.T)
  got = L.matrix_power(A, k)
  ref = np.linalg.matrix_power(A.astype(np.float64), k)
  assert relerr(got, ref) <= 5e-3, f"matrix_power(A, {k}): relerr {relerr(got, ref)}"


def test_matrix_power_identity_and_rejects():
  A = f16(np.random.default_rng(8).standard_normal((5, 5)))
  assert np.array_equal(L.matrix_power(A, 0), np.eye(5, dtype=np.float32))   # n=0 -> I
  assert relerr(L.matrix_power(A, 1), A) == 0.0                              # n=1 -> A itself
  with pytest.raises(ValueError): L.matrix_power(np.zeros((2, 3), f16), 2)   # not square
  with pytest.raises(ValueError): L.matrix_power(A, -1)                      # needs an inverse


@pytest.mark.parametrize("lower", [True, False])
def test_solve_triangular_matrix_rhs(lower):
  # the m columns solve in one fused graph; each must match the single-vector answer exactly, since
  # the substitution is the same recurrence widened from a [1,1] row slice to [1,m]
  r = np.random.default_rng(21); n, m = 7, 3
  T = np.tril(r.standard_normal((n, n))) + np.eye(n) * 4.0
  if not lower: T = T.T
  X = r.standard_normal((n, m)); B = T @ X
  got = L.solve_triangular(f16(T), f16(B), lower=lower)
  assert got.shape == (n, m), got.shape
  assert relerr(got, X) <= 5e-3, f"matrix rhs: relerr {relerr(got, X)}"
  for j in range(m):                                    # column j == solving that column alone
    solo = L.solve_triangular(f16(T), f16(B[:, j]), lower=lower)
    assert relerr(got[:, j], solo) <= 5e-3, f"column {j} disagrees with the vector solve"


def test_solve_triangular_vector_shape_unchanged():
  # a 1-D b must still come back 1-D, not [n, 1]
  r = np.random.default_rng(22); n = 5
  T = np.tril(r.standard_normal((n, n))) + np.eye(n) * 4.0
  assert L.solve_triangular(f16(T), f16(r.standard_normal(n))).shape == (n,)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_solve_matches_numpy(n):
  # general (non-symmetric) A: this is the path conjugate_gradient cannot take
  r = np.random.default_rng(30 + n)
  A = f16(r.standard_normal((n, n)) + np.eye(n) * n)
  b = f16(r.standard_normal(n))
  got = L.solve(A, b)
  ref = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
  assert relerr(got, ref) <= 5e-3, f"solve n={n}: relerr {relerr(got, ref)}"


def test_solve_matrix_rhs():
  r = np.random.default_rng(31); n, m = 6, 3
  A = f16(r.standard_normal((n, n)) + np.eye(n) * n)
  B = f16(r.standard_normal((n, m)))
  got = L.solve(A, B)
  ref = np.linalg.solve(A.astype(np.float64), B.astype(np.float64))
  assert got.shape == (n, m) and relerr(got, ref) <= 5e-3, f"solve matrix rhs: {got.shape}"


def test_solve_residual_is_small():
  # the property that matters to a caller: A x ~ b, checked without a reference solver
  r = np.random.default_rng(32); n = 8
  A = f16(r.standard_normal((n, n)) + np.eye(n) * n)
  b = f16(r.standard_normal(n))
  x = L.solve(A, b)
  res = A.astype(np.float64) @ x.astype(np.float64) - b.astype(np.float64)
  assert np.abs(res).max() / np.abs(b.astype(np.float64)).max() <= 5e-3


def test_solve_rejects():
  A = f16(np.random.default_rng(33).standard_normal((4, 4)) + np.eye(4) * 4)
  with pytest.raises(ValueError): L.solve(np.zeros((3, 4), f16), np.zeros(3, f16))   # not square
  with pytest.raises(ValueError): L.solve(A, np.zeros(3, f16))                       # b rows != n
  with pytest.raises(np.linalg.LinAlgError):
    L.solve(f16([[1.0, 2.0], [2.0, 4.0]]), f16([1.0, 2.0]))                          # singular
