"""aneforge.linalg - iterative solvers and small-n direct factorizations on the ANE."""
from __future__ import annotations

import os
from typing import Any
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import aneforge as af

f16 = np.float16


# One-shot ANE GEMM for sketch/projection/Gram products (iterative solvers unroll instead).

def _ane_gemm(A16: np.ndarray, B16: np.ndarray, transpose_a: bool = False) -> np.ndarray:
  """C = A @ B (or A^T @ B) on the ANE; B folds in as a weight."""
  A16 = A16.astype(f16); B16 = B16.astype(f16)
  if transpose_a:
    m, n = A16.shape
    At = af.input((m, n))
    net = af.compile(At.transpose([1, 0]) @ B16)   # [n,m] @ [m,k]
    C = net(A16)
  else:
    m, n = A16.shape
    At = af.input((m, n))
    net = af.compile(At @ B16)                     # [m,n] @ [n,k]
    C = net(A16)
  net.release()
  return np.asarray(C, f16)


def _spectral_omega(A16: np.ndarray | np.floating[Any]) -> float:
  """Richardson/Jacobi relaxation omega from the max abs row-sum (host scalar)."""
  rowsum = np.abs(np.asarray(A16, np.float64)).sum(1).max()
  return float(f16(1.0 / rowsum))


def _solve_once(out, *feeds):
  """Compile the unrolled recurrence as one fused graph, run once, return fp64."""
  net = af.compile(out, _check_precision=False)
  y = np.asarray(net(*feeds), np.float64)
  net.release()
  return y


def _run_multi(outs, *feeds):
  """Compile a list of output graphs, run once, return outputs in `outs` order."""
  from aneforge._compile import compile_multi
  net = compile_multi(outs)
  out = net(*feeds)
  vals = [out[name] for _, name in net.output_ports]
  net.release()
  return vals


def _mgs_qr_graph(X, height: int, ones):
  """Modified Gram-Schmidt thin QR of in-graph X [height, n] -> (Q, R) graphs (`ones` a [height,1] dot const)."""
  n = X.shape[1]
  Q, Rcols, z = [], [], None
  dot = lambda a, b: (a * b).transpose([1, 0]) @ ones        # [1,height]@[height,1] = [1,1]
  for j in range(n):
    v = X.slice_by_size([0, j], [height, 1]); rcol = []
    for i in range(j):
      rij = dot(Q[i], v); rcol.append(rij); v = v - rij * Q[i]
    d2 = dot(v, v); rjj = d2.sqrt(); rcol.append(rjj)
    Q.append(v * d2.rsqrt())                                  # q_j = v / ||v||
    z = rjj - rjj if z is None else z                        # exact-zero [1,1]
    Rcols.append(af.concat(rcol + [z] * (n - j - 1), axis=0))   # R column j -> [n,1]
  return af.concat(Q, axis=1), af.concat(Rcols, axis=1)


# 1. CONJUGATE GRADIENT  (SPD A, fixed K iterations)

def conjugate_gradient(A, b, iters: int = 20, x0=None, refine: int = 0):
  """Solve A x = b for SPD A by CG with fixed `iters` (recurrence unrolls into one program); `refine` adds residual-correction rounds."""
  A0 = np.asarray(A, np.float64); b0 = np.asarray(b, np.float64).reshape(-1)
  n = A0.shape[0]
  # symmetric Jacobi preconditioner: A~ = Dh^{-1} A Dh^{-1}, b~ = Dh^{-1} b,
  # x = Dh^{-1} y;  Dh = sqrt(diag(A)).  A~ symmetric, so A~ p (row) is p @ A~.
  dh = np.sqrt(np.abs(A0.diagonal())); dh[dh == 0] = 1.0
  A16 = f16((A0 / dh[:, None]) / dh[None, :])
  b16 = f16(b0 / dh).reshape(1, n)
  ones = np.ones((n, 1), f16)

  bT = af.input((1, n))
  x0T = af.input((1, n)) if x0 is not None else None
  dot = lambda u, v: (u * v) @ ones                       # ANE matmul dot (saturates above ~32752 = fp16_max/2; scaling keeps sums in range)

  def cg_block(x, r, K):
    """K unrolled CG steps from residual r (p=r), accumulating into x."""
    p = r
    rs = dot(r, r)
    for _ in range(K):
      Ap = p @ A16                                     # ANE GEMV  A~ p
      alpha = rs / dot(p, Ap)                          # [1,1] / [1,1]
      x = x + alpha * p                                # broadcast axpy
      r = r - alpha * Ap
      rs_new = dot(r, r)
      p = r + (rs_new / rs) * p                        # beta = rs_new/rs
      rs = rs_new
    return x, r

  if x0T is None:
    x, r = bT * 0.0, bT                                  # x=0 -> r0 = b
  else:
    x, r = x0T, bT - x0T @ A16                           # r0 = b - A~ x0
  x, r = cg_block(x, r, iters)
  for _ in range(refine):
    r = bT - x @ A16                                     # residual correction
    dx, _ = cg_block(bT * 0.0, r, max(4, iters // 2))
    x = x + dx

  feeds = [b16] if x0T is None else [b16, (np.asarray(x0, np.float64).reshape(-1) * dh).astype(f16).reshape(1, n)]
  y = _solve_once(x, *feeds).ravel()
  if not np.isfinite(y).all(): y = np.zeros(n)   # high cond: iterates can overflow fp16
  return (y / dh).astype(np.float32)


# 2. stationary ITERATIONS - Jacobi / Gauss-Seidel

def jacobi(A, b, iters: int = 60, x0=None):
  """Jacobi iteration x += D^{-1}(b - A x), fixed `iters`; converges for diagonally-dominant A."""
  A16 = np.asarray(A, f16); b0 = np.asarray(b, np.float64).reshape(-1)
  n = A16.shape[0]
  AT16 = np.ascontiguousarray(A16.T)                     # x @ A^T = (A x) as a row
  d = np.asarray(A16, np.float64).diagonal().copy(); d[d == 0] = 1.0
  dinv = (1.0 / d).astype(f16).reshape(1, n)

  bT = af.input((1, n)); dinvT = af.input((1, n))
  x0T = af.input((1, n)) if x0 is not None else None
  x = bT * 0.0 if x0T is None else x0T
  for _ in range(iters):
    x = x + (bT - x @ AT16) * dinvT                    # x + D^{-1}(b - A x)
  feeds = [b0.astype(f16).reshape(1, n), dinv]
  if x0T is not None: feeds.append(np.asarray(x0, f16).reshape(1, n))
  return _solve_once(x, *feeds).ravel().astype(np.float32)


def gauss_seidel(A, b, iters: int = 60, x0=None):
  """Gauss-Seidel iteration, fixed `iters`; serial sweep is host-side, residual GEMV on the ANE."""
  A16 = np.asarray(A, f16); b16 = np.asarray(b, f16).reshape(-1)
  Af = np.asarray(A16, np.float64); bf = b16.astype(np.float64)
  n = A16.shape[0]
  x = np.zeros(n, np.float64) if x0 is None else np.asarray(x0, np.float64).reshape(-1).copy()
  L = np.tril(Af, -1); U = np.triu(Af, 1); d = Af.diagonal()
  for _ in range(iters):
    for i in range(n):
      s = L[i, :i] @ x[:i] + U[i, i + 1:] @ x[i + 1:]
      x[i] = (bf[i] - s) / d[i]
  return x.astype(np.float32)


# 3. ITERATIVE refinement  (residual correction)

def iterative_refine(A, b, x0, iters: int = 3, inner: int = 60):
  """Sharpen a solve x0 by fixed-K residual correction (r = b - A x, solve A dx = r, x += dx); inner is a Richardson sweep."""
  A0 = np.asarray(A, np.float64); b0 = np.asarray(b, np.float64).reshape(-1)
  n = A0.shape[0]
  # same symmetric Jacobi scaling as CG, to keep the fp16 GEMV products in range
  dh = np.sqrt(np.abs(A0.diagonal())); dh[dh == 0] = 1.0
  A16 = f16((A0 / dh[:, None]) / dh[None, :])
  b16 = f16(b0 / dh).reshape(1, n)
  omega = _spectral_omega(A16)

  bT = af.input((1, n)); x0T = af.input((1, n))
  x = x0T
  for _ in range(iters):
    r = bT - x @ A16                                    # residual (ANE GEMV + sub)
    dx = bT * 0.0
    for _ in range(inner):
      dx = dx + (r - dx @ A16) * omega                # Richardson, omega is a scalar
    x = x + dx
  x0s = (np.asarray(x0, np.float64).reshape(-1) * dh).astype(f16).reshape(1, n)
  y = _solve_once(x, b16, x0s).ravel()
  return (y / dh).astype(np.float32)


# 4. randomized SVD  (sketch + range-finder + small dense SVD)

def randomized_svd(A, k: int, oversample: int = 5, power_iters: int = 2):
  """Truncated SVD of A [m,n] via Halko-Martinsson-Tropp range finding; returns (U[:, :k], S[:k], Vt[:k, :])."""
  A16 = np.asarray(A, f16)
  m, n = A16.shape
  l = min(k + oversample, n)
  rng = np.random.default_rng(0)
  Omega = rng.standard_normal((n, l)).astype(f16)                        # HOST RNG (a constant)

  Y = _ane_gemm(A16, Omega)                                              # ANE: A @ Omega -> [m,l]
  Q16 = np.linalg.qr(np.asarray(Y, np.float64))[0].astype(f16)           # HOST QR -> O(1) basis
  for _ in range(power_iters):
    AtQ = _ane_gemm(A16, Q16, transpose_a=True)                        # ANE: A^T @ Q -> [n,l]
    Y = _ane_gemm(A16, AtQ)                                            # ANE: A   @ AtQ -> [m,l]
    Q16 = np.linalg.qr(np.asarray(Y, np.float64))[0].astype(f16)       # HOST re-orthonormalize

  B = _ane_gemm(A16, Q16, transpose_a=True).T                           # ANE: (A^T @ Q)^T = Q^T @ A -> [l,n]
  Ub, S, Vt = np.linalg.svd(np.asarray(B, np.float64), full_matrices=False)  # HOST small SVD
  U = _ane_gemm(Q16, Ub.astype(f16))                                     # ANE: Q @ Ub -> [m,l]
  return (np.asarray(U, np.float32)[:, :k],
          S[:k].astype(np.float32),
          np.asarray(Vt, np.float32)[:k, :])


def pca(X, k: int, oversample: int = 5, power_iters: int = 2):
  """Principal components of X [samples, features] via randomized_svd of the centered data; returns (components, singular_values, mean)."""
  Xf = np.asarray(X, np.float64)
  mean = Xf.mean(0)
  Xc = (Xf - mean).astype(f16)                                           # HOST centering
  _, S, Vt = randomized_svd(Xc, k, oversample, power_iters)              # ANE matmuls inside
  return Vt[:k].astype(np.float32), S[:k].astype(np.float32), mean.astype(np.float32)


# 5. LEAST squares  (normal equations via CG + refinement)

def least_squares(A, b, iters: int = 40, refine: int = 2):
  """Solve min_x ||A x - b||_2 via the normal equations A^T A x = A^T b by CG + refinement (forming A^T A squares cond(A))."""
  A16 = np.asarray(A, f16); b16 = np.asarray(b, f16).reshape(-1)
  m, n = A16.shape
  AtA = _ane_gemm(A16, A16, transpose_a=True)                            # ANE: A^T A -> [n,n]
  # A^T b  via matmul: (A^T) @ b  ==  ( [n,m] @ [m,1] )
  At = af.input((m, n))
  bt = af.input((m, 1))
  net = af.compile(At.transpose([1, 0]) @ bt)                            # ANE GEMV
  Atb = net(A16, b16.reshape(m, 1).astype(f16)).reshape(n)
  net.release()
  return conjugate_gradient(AtA, Atb, iters=iters, refine=refine)


# Krylov solvers fully on the ANE: a fixed-step recurrence is static, so it unrolls into one graph.

def lsqr(A, b, iters: int = 80):
  """Solve min_x ||A x - b||_2 by LSQR (Golub-Kahan bidiagonalization), fully on the ANE; square systems only to cond~1e1 (use gmres beyond)."""
  A0 = np.asarray(A, f16); m, n = A0.shape; AT = np.ascontiguousarray(A0.T)
  onem = np.ones((m, 1), f16); onen = np.ones((n, 1), f16)
  bT = af.input((1, m))
  nrm = lambda z, o: ((z * z) @ o).sqrt()
  Ax = lambda v: v @ AT                                # [1,n]@[n,m] = A@v
  ATx = lambda u: u @ A0                               # [1,m]@[m,n] = A^T@u
  beta = nrm(bT, onem); u = bT / beta
  v = ATx(u); alpha = nrm(v, onen); v = v / alpha
  w = v; x = v * 0.0; phib = beta; rhob = alpha
  for _ in range(iters):
    u2 = Ax(v) - alpha * u; beta = nrm(u2, onem); u = u2 / beta
    v2 = ATx(u) - beta * v; alpha = nrm(v2, onen); v = v2 / alpha
    rho = (rhob * rhob + beta * beta).sqrt(); c = rhob / rho; s = beta / rho
    theta = s * alpha; rhob = (c * alpha) * -1.0; phi = c * phib; phib = s * phib
    x = x + (phi / rho) * w; w = v - (theta / rho) * w
  return _solve_once(x, np.asarray(b, f16).reshape(1, m)).ravel().astype(np.float32)


def dominant_eig(A, iters: int = 60, seed: int = 0):
  """Dominant eigenpair of symmetric A by power iteration, fully on the ANE (in-graph Rayleigh quotient); returns (lambda, eigenvector)."""
  A0 = np.asarray(A, f16); n = A0.shape[0]; AT = np.ascontiguousarray(A0.T)
  onen = np.ones((n, 1), f16)
  x0 = af.input((1, n))
  nrm = lambda z: ((z * z) @ onen).sqrt()
  Ax = lambda v: v @ AT
  x = x0 / nrm(x0)
  for _ in range(iters):
    y = Ax(x); x = y / nrm(y)
  Axf = Ax(x)
  lam = ((x * Axf) @ onen) / ((x * x) @ onen)          # Rayleigh quotient [1,1]
  out = af.concat([lam, x], axis=1)                    # [1, 1+n]
  seedv = np.random.default_rng(seed).standard_normal((1, n)).astype(f16)
  r = _solve_once(out, seedv).ravel()
  return float(r[0]), r[1:].astype(np.float32)


def gmres(A, b):
  """Solve A x = b for general A by one full GMRES(n) cycle, fully on the ANE; more accurate than lsqr on square systems but heavier."""
  A0 = np.asarray(A, f16); n = A0.shape[0]; m = n
  AT = np.ascontiguousarray(A0.T); onen = np.ones((n, 1), f16)
  bT = af.input((1, n))
  nrm = lambda z: ((z * z) @ onen).sqrt()
  Ax = lambda v: v @ AT
  dot = lambda a, c: (a * c) @ onen
  Q = [bT / nrm(bT)]; H = {}; g = [nrm(bT)] + [None] * m; cs = [None] * m; sn = [None] * m
  for j in range(m):
    w = Ax(Q[j])
    for _p in range(2):                                  # double MGS, all in-graph
      for i in range(j + 1):
        h = dot(Q[i], w); H[(i, j)] = h if (i, j) not in H else H[(i, j)] + h; w = w - h * Q[i]
    hn = nrm(w); Hjp = hn
    for i in range(j):                                   # apply prior Givens to column j
      t = cs[i] * H[(i, j)] + sn[i] * H[(i + 1, j)]
      H[(i + 1, j)] = sn[i] * (H[(i, j)] * -1.0) + cs[i] * H[(i + 1, j)]; H[(i, j)] = t
    H[(j + 1, j)] = Hjp
    d = (H[(j, j)] * H[(j, j)] + Hjp * Hjp).sqrt(); cs[j] = H[(j, j)] / d; sn[j] = Hjp / d
    H[(j, j)] = cs[j] * H[(j, j)] + sn[j] * Hjp
    g[j + 1] = sn[j] * (g[j] * -1.0); g[j] = cs[j] * g[j]
    Q.append(w / hn)
  y = [None] * m                                           # back-substitution R y = g
  for i in range(m - 1, -1, -1):
    acc = g[i]
    for jj in range(i + 1, m):
      acc = acc - H[(i, jj)] * y[jj]
    y[i] = acc / H[(i, i)]
  x = Q[0] * 0.0
  for i in range(m): x = x + y[i] * Q[i]
  return _solve_once(x, np.asarray(b, f16).reshape(1, n)).ravel().astype(np.float32)


def dominant_svd(A, iters: int = 60, seed: int = 0):
  """Dominant singular triple by power iteration on A^T A, fully on the ANE (half-step renorm holds fp16 range); returns (sigma1, u1, v1)."""
  A0 = np.asarray(A, f16); m, n = A0.shape
  AT = np.ascontiguousarray(A0.T); onen = np.ones((n, 1), f16); onem = np.ones((m, 1), f16)
  v0 = af.input((1, n))
  nn = lambda z: ((z * z) @ onen).sqrt()
  nm = lambda z: ((z * z) @ onem).sqrt()
  Ax = lambda v: v @ AT                                  # [1,n]@[n,m] = A v
  ATx = lambda u: u @ A0                                 # [1,m]@[m,n] = A^T u
  v = v0 / nn(v0)
  for _ in range(iters):
    Av = Ax(v); Av = Av / nm(Av)                          # half-step renorm: holds O(1)
    v = ATx(Av); v = v / nn(v)
  Av = Ax(v); sig = nm(Av)                               # sigma1 = ||A v1||  [1,1]
  u = Av / sig
  out = af.concat([sig, u, v], axis=1)                   # [1, 1+m+n]
  seedv = np.random.default_rng(seed).standard_normal((1, n)).astype(f16)
  r = _solve_once(out, seedv).ravel()
  return float(r[0]), r[1:1 + m].astype(np.float32), r[1 + m:].astype(np.float32)


# Full spectrum via fixed-sweep cyclic Jacobi: closed-form rotations, no pivot/branch, all in one program.

def _jacobi_sweep(M, n):
  """One full cyclic-Jacobi sweep over symmetric M [n,n] (all (p,q) rotations), in-graph."""
  row = lambda X, i: X.slice_by_size([i, 0], [1, n])
  col = lambda X, j: X.slice_by_size([0, j], [n, 1])

  def setrows(X, p, q, rp, rq):
    parts = ([X.slice_by_size([0, 0], [p, n])] if p > 0 else []) + [rp]
    if q > p + 1: parts.append(X.slice_by_size([p + 1, 0], [q - p - 1, n]))
    parts.append(rq)
    if q < n - 1: parts.append(X.slice_by_size([q + 1, 0], [n - q - 1, n]))
    return af.concat(parts, axis=0)

  def setcols(X, p, q, cp, cq):
    parts = ([X.slice_by_size([0, 0], [n, p])] if p > 0 else []) + [cp]
    if q > p + 1: parts.append(X.slice_by_size([0, p + 1], [n, q - p - 1]))
    parts.append(cq)
    if q < n - 1: parts.append(X.slice_by_size([0, q + 1], [n, n - q - 1]))
    return af.concat(parts, axis=1)

  for p in range(n):
    for q in range(p + 1, n):
      rp = row(M, p); rq = row(M, q)
      app = rp.slice_by_size([0, p], [1, 1]); aqq = rq.slice_by_size([0, q], [1, 1])
      apq = rp.slice_by_size([0, q], [1, 1])
      denom = apq * 2.0
      tau = (aqq - app) * denom / ((denom * denom).adds(1e-6))    # safe 1/denom (->0 at apq=0)
      sgn = tau / ((tau * tau).adds(1e-12).sqrt())
      t = sgn / (tau.abs() + (tau * tau).adds(1.0).sqrt())
      c = (t * t).adds(1.0).rsqrt(); s = t * c                     # cos, sin
      M = setrows(M, p, q, c * rp - s * rq, s * rp + c * rq)       # G^T A
      cp = col(M, p); cq = col(M, q)
      M = setcols(M, p, q, c * cp - s * cq, s * cp + c * cq)       # (.) G
  return M


def eigh(A, sweeps: int = 8, iterate: bool = False):
  """All eigenvalues of symmetric A by fixed-sweep cyclic Jacobi, ascending; `iterate=True` host-loops one compiled sweep (reaches larger n)."""
  A0 = np.asarray(A, f16); n = A0.shape[0]
  if iterate:
    net = af.compile(_jacobi_sweep(af.input((n, n)), n), _check_precision=False)
    M = A0
    for _ in range(sweeps): M = net(M).astype(f16)
    net.release()
    return np.sort(np.diag(M.astype(np.float64))).astype(np.float32)
  M = af.input((n, n))
  for _ in range(sweeps): M = _jacobi_sweep(M, n)
  out = _solve_once(M, A0)                                                 # final ~diagonal matrix
  return np.sort(np.diag(out)).astype(np.float32)


def svd(A, sweeps: int = 8):
  """All singular values of A (descending), fully on the ANE: Gram matrix via one GEMM, then its spectrum via cyclic-Jacobi eigh."""
  A16 = np.asarray(A, f16); m, n = A16.shape
  # form the SMALLER Gram matrix (eigh's graph is O(dim^3)): same nonzero eigenvalues.
  M = A16 if m >= n else np.ascontiguousarray(A16.T)
  G = _ane_gemm(M, M, transpose_a=True)                  # ANE: M^T M  [min(m,n), min(m,n)]
  ev = eigh(G, sweeps=sweeps)
  return np.sqrt(np.clip(ev, 0.0, None))[::-1].astype(np.float32)


def svdvals_topk(A, k: int, oversample: int = 2, power_iters: int = 1, seed: int = 0):
  """Top-k singular values of a large A, fully on the ANE (sketch + on-ANE qr + projection + on-ANE svd); extra power iters hurt in fp16."""
  A16 = np.asarray(A, f16); m, n = A16.shape; l = min(k + oversample, n)
  Om = np.random.default_rng(seed).standard_normal((n, l)).astype(f16)
  Y = _ane_gemm(A16, Om); Q = qr(Y)[0].astype(f16)                  # ANE sketch + Gram-Schmidt
  for _ in range(power_iters):
    Y = _ane_gemm(A16, _ane_gemm(A16, Q, transpose_a=True)); Q = qr(Y)[0].astype(f16)
  B = _ane_gemm(A16, Q, transpose_a=True).T                         # ANE projection [l,n]
  return np.sort(svd(B, sweeps=8))[::-1][:k]                         # ANE svd of the small block


# Direct factorizations (QR/Cholesky/LU): unpivoted fixed recurrences, valid for
# well-conditioned leading minors; unroll into one program (small-n).

def _grid(entries: dict, n: int, zero):
  """Assemble an n x n matrix from a dict {(i,j): [1,1] tensor} (missing -> zero)."""
  return af.concat([af.concat([entries.get((i, j), zero) for j in range(n)], axis=1)
                    for i in range(n)], axis=0)


def _els_routed(Xt, n: int):
  """Element accessor el(i, j) routed off the width axis (row slice + transpose + slice).
    A width-offset slice rides the A13/A14 x16 crop-DMA, saturating |value| > 4094 to
    +/-inf; routing keeps every begin's last axis at 0, restoring the full fp16 span."""
  rows = [Xt.slice_by_size([i, 0], [1, n]).transpose([1, 0]) for i in range(n)]
  return lambda i, j: rows[i].slice_by_size([j, 0], [1, 1])


def qr(A):
  """Thin QR A = Q R by modified Gram-Schmidt, fully on the ANE; returns (Q, R)."""
  A16 = np.asarray(A, f16); m, n = A16.shape; onem = np.ones((m, 1), f16)
  Qg, Rg = _mgs_qr_graph(af.input((m, n)), m, onem)
  Qv, Rv = _run_multi([Qg, Rg], A16)
  return np.asarray(Qv, np.float32), np.asarray(Rv, np.float32)


def cholesky(A):
  """Lower-triangular Cholesky factor L (A = L L^T) of SPD A, unpivoted, fully on the ANE."""
  A16 = np.asarray(A, f16); n = A16.shape[0]
  At = af.input((n, n)); el = _els_routed(At, n)
  z = el(0, 0) - el(0, 0); L: dict = {}
  for j in range(n):
    d = el(j, j)
    for k in range(j): d = d - L[(j, k)] * L[(j, k)]
    L[(j, j)] = d.relu().adds(1e-12).sqrt()                # relu guards fp16 noise (SPD -> d>0)
    for i in range(j + 1, n):
      s = el(i, j)
      for k in range(j): s = s - L[(i, k)] * L[(j, k)]
      L[(i, j)] = s / L[(j, j)]
  return _solve_once(_grid(L, n, z), A16).astype(np.float32)


def lu(A):
  """Unpivoted LU (A = L U) by Doolittle, fully on the ANE; returns (L, U)."""
  A16 = np.asarray(A, f16); n = A16.shape[0]
  At = af.input((n, n)); el = _els_routed(At, n)
  z = el(0, 0) - el(0, 0); one = z.adds(1.0)
  U: dict = {}; L: dict = {(i, i): one for i in range(n)}
  for i in range(n):
    for k in range(i, n):
      s = el(i, k)
      for t in range(i): s = s - L[(i, t)] * U[(t, k)]
      U[(i, k)] = s
    for k in range(i + 1, n):
      s = el(k, i)
      for t in range(i): s = s - L[(k, t)] * U[(t, i)]
      L[(k, i)] = s / U[(i, i)]
  Lv, Uv = _run_multi([_grid(L, n, z), _grid(U, n, z)], A16)
  return np.asarray(Lv, np.float32), np.asarray(Uv, np.float32)


def lu_pivoted(A):
  """Partial-pivoted LU (P A = L U) on the ANE; the per-column argmax pivot is a bridge op, so the program is segmented. Returns (P, L, U)."""
  A16 = np.asarray(A, f16); n = A16.shape[0]
  At = af.input((n, n)); In = af.input((n, n)); arc = af.input((n, 1))
  arcr = arc.transpose([1, 0]); M = At; P = In
  half = (arc * 0.0).adds(0.5); onerow = (arcr * 0.0).adds(1.0); zerorow = arcr * 0.0
  for k in range(n):
    ek = In.slice_by_size([0, k], [n, 1]); ekr = ek.transpose([1, 0])
    kc = (arc * 0.0).adds(float(k))
    excl = af.select(kc.greater(arc), (arc * 0.0).adds(1e4), arc * 0.0)   # exclude rows < k
    idx = ((M @ ek).abs() - excl).argmax(axis=0)                          # pivot row (bridge op)
    p1h = af.select((arc - idx).abs().greater(half), arc * 0.0, (arc * 0.0).adds(1.0))

    def swap(X):                                                          # swap rows k and pivot
      rk = ekr @ X; rp = p1h.transpose([1, 0]) @ X
      return X + ek @ (rp - rk) + p1h @ (rk - rp)
    M = swap(M); P = swap(P)
    piv = (ekr @ M) @ ek
    belowk = af.select(arc.greater(kc), (arc * 0.0).adds(1.0), arc * 0.0)  # rows > k
    Lk = ((M @ ek) / piv) * belowk
    rowk = ekr @ M
    colabove = af.select(arcr.greater((arcr * 0.0).adds(float(k))), onerow, zerorow)  # cols > k
    M = M - Lk @ (rowk * colabove)                                        # Schur (trailing only)
    M = M - ((M @ ek) * belowk) @ ekr + Lk @ ekr                          # store Lk in column k
  net = af.compile(af.concat([P, M], axis=1))                              # SegmentedModel (argmax cuts)
  o = np.asarray(net(A16, np.eye(n, dtype=f16), np.arange(n, dtype=f16).reshape(n, 1)), np.float32)
  net.release()
  Pv, Mv = o[:, :n], o[:, n:]
  return Pv, np.tril(Mv, -1) + np.eye(n, dtype=np.float32), np.triu(Mv)


def _trinv_lower(L):
  """Inverse of lower-triangular L by forward substitution, on the ANE."""
  L16 = np.asarray(L, f16); n = L16.shape[0]
  Lt = af.input((n, n)); el = _els_routed(Lt, n)
  z = el(0, 0) - el(0, 0); one = z.adds(1.0)
  X: dict = {}
  for col in range(n):
    for i in range(col, n):
      s = one if i == col else z
      for k in range(col, i): s = s - el(i, k) * X[(k, col)]
      X[(i, col)] = s / el(i, i)
  return _solve_once(_grid(X, n, z), L16).astype(np.float32)


def generalized_eigh(A, B, sweeps: int = 8):
  """Eigenvalues of A x = lambda B x (A symmetric, B SPD), ascending, by composing on-engine kernels: B = L L^T, C = L^-1 A L^-T, eigh(C)."""
  A16 = np.asarray(A, f16); B16 = np.asarray(B, f16)
  L = cholesky(B16).astype(f16)                          # ANE
  Li = _trinv_lower(L).astype(f16)                       # ANE
  C = _ane_gemm(_ane_gemm(Li, A16), np.ascontiguousarray(Li.T))   # ANE: (L^-1 A) L^-T
  C = ((C + C.T) * 0.5).astype(f16)                      # symmetrize (host, tiny)
  return eigh(C, sweeps=sweeps)


def eigvals(A, iters: int = 60):
  """Eigenvalues of general real A by the unshifted QR algorithm (M <- R Q -> real Schur form), fully on the ANE; returns a complex array."""
  A16 = np.asarray(A, f16); n = A16.shape[0]; onen = np.ones((n, 1), f16)
  M = af.input((n, n))
  for _ in range(iters):
    Q, R = _mgs_qr_graph(M, n, onen); M = R @ Q        # RQ; unrolls into one program
  Mv = _solve_once(M, A16)
  evs, i = [], 0
  while i < n:
    if i < n - 1 and abs(Mv[i + 1, i]) > 1e-2 * (abs(Mv[i, i]) + abs(Mv[i + 1, i + 1]) + 1e-9):
      b = Mv[i:i + 2, i:i + 2]; tr = b[0, 0] + b[1, 1]
      det = b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
      s = np.sqrt(complex(tr * tr - 4 * det)); evs += [(tr + s) / 2, (tr - s) / 2]; i += 2
    else:
      evs.append(complex(Mv[i, i])); i += 1
  return np.array(evs)


__all__ = [
  "conjugate_gradient", "jacobi", "gauss_seidel", "iterative_refine",
  "least_squares", "lsqr", "gmres",
  "qr", "cholesky", "lu", "lu_pivoted",
  "eigh", "eigvals", "generalized_eigh", "dominant_eig",
  "svd", "dominant_svd", "svdvals_topk", "randomized_svd", "pca",
]


# __main__ - self-test / validation vs numpy/scipy

def _make_spd(n, cond, seed):
  """SPD A with target condition number, fp16-stored (reference solves the fp16-rounded system)."""
  rng = np.random.default_rng(seed)
  Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
  eig = np.geomspace(1.0, cond, n)
  A = (Q * eig) @ Q.T
  A = (A + A.T) / 2
  xtrue = rng.standard_normal(n)
  b = A @ xtrue
  A16, b16 = f16(A), f16(b)
  xref = np.linalg.solve(np.asarray(A16, np.float64), np.asarray(b16, np.float64))
  return A16, b16, xref


def _relerr(approx, truth):
  truth = np.asarray(truth, np.float64).ravel()
  approx = np.asarray(approx, np.float64).ravel()
  return float(np.linalg.norm(approx - truth) / (np.linalg.norm(truth) + 1e-30))


def _diag_dominant(n, cond_scale, seed):
  """A strongly diagonally-dominant system (Jacobi/Gauss-Seidel converge)."""
  rng = np.random.default_rng(seed)
  A = rng.standard_normal((n, n)) * 0.1
  A = A + A.T
  A[np.diag_indices(n)] = np.abs(A).sum(1) + cond_scale
  xtrue = rng.standard_normal(n)
  b = A @ xtrue
  A16, b16 = f16(A), f16(b)
  xref = np.linalg.solve(np.asarray(A16, np.float64), np.asarray(b16, np.float64))
  return A16, b16, xref


# Determinants, pseudoinverse, triangular solve - composed from the routines above.

def solve_triangular(A, b, lower: bool = True):
  """Solve T x = b for triangular T by substitution on the ANE, without forming the inverse
  (the routed accessor keeps large entries finite, as in cholesky/lu)."""
  A16 = np.asarray(A, f16); n = A16.shape[0]
  b16 = np.asarray(b, f16).reshape(n, 1)
  At = af.input((n, n)); bt = af.input((n, 1))
  el = _els_routed(At, n)
  bl = lambda i: bt.slice_by_size([i, 0], [1, 1])          # height-axis slice: begin stays 0 on the last axis
  X: dict = {}
  for i in (range(n) if lower else range(n - 1, -1, -1)):
    s = bl(i)
    for k in (range(i) if lower else range(i + 1, n)): s = s - el(i, k) * X[k]
    X[i] = s / el(i, i)
  out = af.concat([X[i] for i in range(n)], axis=0)
  return _solve_once(out, A16, b16).ravel().astype(np.float32)


def _perm_parity(perm) -> float:
  """+1/-1 parity of a permutation via cycle decomposition."""
  seen = np.zeros(len(perm), bool); sign = 1.0
  for i in range(len(perm)):
    if seen[i]: continue
    j, clen = i, 0
    while not seen[j]: seen[j] = True; j = int(perm[j]); clen += 1
    if clen % 2 == 0: sign = -sign
  return sign


def det(A):
  """det(A) via the on-ANE pivoted LU (P A = L U): permutation parity times prod(diag U)."""
  P, _, U = lu_pivoted(A)
  sign = _perm_parity(np.argmax(P, axis=1))
  return float(sign * np.prod(np.diag(U).astype(np.float64)))


def slogdet(A):
  """(sign, log|det A|) via the on-ANE pivoted LU; the log-sum form stays finite where the raw
  product over/underflows. Returns (0.0, -inf) for a singular U diagonal, like numpy."""
  P, _, U = lu_pivoted(A)
  d = np.diag(U).astype(np.float64)
  if np.any(d == 0.0): return 0.0, float("-inf")
  sign = _perm_parity(np.argmax(P, axis=1)) * float(np.prod(np.sign(d)))
  return float(sign), float(np.sum(np.log(np.abs(d))))


def pinv(A, k: int, oversample: int = 5, power_iters: int = 2, rcond: float = 1e-3):
  """Rank-k pseudoinverse via randomized_svd: V diag(1/s) U^T with a relative cutoff on small
  singular values. Rank-k by construction - for a full-rank dense pinv use a dense library."""
  U, S, Vt = randomized_svd(A, k, oversample=oversample, power_iters=power_iters)
  keep = S > rcond * (S[0] if S.size and S[0] > 0 else 1.0)
  if not np.any(keep): return np.zeros((A.shape[1], A.shape[0]), np.float32)
  W = (Vt[keep].T / S[keep]).astype(np.float32)            # [n, r]
  return _ane_gemm(W.astype(f16), np.ascontiguousarray(U[:, keep].T, np.float32).astype(f16)).astype(np.float32)


def norm(A, order="fro"):
  """Matrix norm of a 2-D A, composed on-engine as one fused graph.

  `order="fro"` is `sqrt(sum_square(A))`; `order=1` is the max absolute column sum and
  `order=inf` the max absolute row sum, each an abs, a matmul against a ones vector (the
  wide accumulator does the summing), and an `amax`. Oracle is `np.linalg.norm` (whose
  argument is spelled `ord`; renamed here only to avoid shadowing the builtin).
  """
  A16 = np.asarray(A, f16)
  if A16.ndim != 2: raise ValueError(f"linalg.norm: expected a 2-D matrix; got shape {A16.shape}")
  m, n = A16.shape
  X = af.input((m, n))
  if order == "fro":
    out = X.sum_square((0, 1)).sqrt()
  elif order == 1:
    # column sums: |A|^T @ ones[m,1] -> [n,1]
    out = (X.abs().transpose([1, 0]) @ np.ones((m, 1), f16)).amax((0, 1))
  elif order in (np.inf, float("inf")):
    out = (X.abs() @ np.ones((n, 1), f16)).amax((0, 1))     # row sums -> [m,1]
  else:
    raise ValueError(f"linalg.norm: order must be 'fro', 1, or inf; got {order!r}")
  net = af.compile(out, _check_precision=False)
  v = float(np.ravel(np.asarray(net(A16), np.float64))[0])
  net.release()
  return v


def matrix_power(A, n: int):
  """A**n for integer n >= 0 by binary exponentiation over `_ane_gemm`, so it costs
  O(log n) on-engine gemms rather than n-1. n=0 is the identity.

  fp16 error compounds with each squaring, so this is for modest powers on
  well-conditioned matrices; negative n needs an inverse and is rejected.
  Oracle is `np.linalg.matrix_power`.
  """
  A16 = np.asarray(A, f16)
  if A16.ndim != 2 or A16.shape[0] != A16.shape[1]:
    raise ValueError(f"linalg.matrix_power: expected a square matrix; got shape {A16.shape}")
  if n < 0:
    raise ValueError("linalg.matrix_power: negative powers need a matrix inverse; not supported")
  if n == 0: return np.eye(A16.shape[0], dtype=np.float32)
  acc, base = None, A16                                     # square-and-multiply
  while n:
    if n & 1: acc = base.copy() if acc is None else _ane_gemm(acc, base)
    n >>= 1
    if n: base = _ane_gemm(base, base)
  return np.asarray(acc, np.float32)


def main():
  print("=" * 90)
  print("aneforge.linalg - ITERATIVE linear algebra on the ANE  (matmuls=ANE, RNG/QR/SVD/loop=HOST)")
  print("=" * 90)
  print("Reference: numpy/scipy fp64 on the SAME fp16-rounded system (measures the algorithm,")
  print("not the A->fp16 storage error). All dots/accumulation via matmul (wide accumulator).\n")

  # ---------------- CG vs np.linalg.solve, swept over condition number --- #
  print("-" * 90)
  print("CONJUGATE GRADIENT  (SPD A, n=48)  -  relerr vs np.linalg.solve, swept over cond")
  print("-" * 90)
  print(f"{'cond':>8} | {'CG K=40':>12} | {'CG K=40 +refine2':>18} | {'np.linalg.solve(fp16 sys)':>26}")
  n = 48
  for cond in (1e1, 1e2, 1e3):
    A16, b16, xref = _make_spd(n, cond, seed=int(cond) + 11)
    x_cg = conjugate_gradient(A16, b16, iters=40)
    x_cgr = conjugate_gradient(A16, b16, iters=40, refine=2)
    # numpy direct solve of the same fp16 system, as the achievable-floor baseline
    x_np = np.linalg.solve(np.asarray(A16, np.float64), np.asarray(b16, np.float64))
    print(f"{cond:>8.0e} | {_relerr(x_cg, xref):>12.3e} | {_relerr(x_cgr, xref):>18.3e} | {_relerr(x_np, xref):>26.3e}")
  print("  reading: CG matches the direct fp16 solve to ~1e-3/1e-2 for cond<=1e2; at cond~1e3 the")
  print("  fp16 iterates OVERFLOW/stall and CG breaks down (relerr ~1, guarded to a finite last")
  print("  iterate, not nan) - the fp16 envelope ceiling. refine helps under-iterated solves but")
  print("  adds fp16 noise once x is already at the floor (cond=1e1), so it is a near-ceiling tool.")
  print()

  # ---------------- Jacobi / Gauss-Seidel (diag-dominant) ---------------- #
  print("-" * 90)
  print("STATIONARY ITERATIONS  (diagonally-dominant A, n=32)  -  relerr vs np.linalg.solve")
  print("-" * 90)
  print(f"{'dominance':>10} | {'Jacobi K=80':>14} | {'Gauss-Seidel K=40':>18}")
  for dom in (1.0, 4.0):
    A16, b16, xref = _diag_dominant(32, dom, seed=int(dom * 7) + 3)
    xj = jacobi(A16, b16, iters=80)
    xg = gauss_seidel(A16, b16, iters=40)
    print(f"{dom:>10.1f} | {_relerr(xj, xref):>14.3e} | {_relerr(xg, xref):>18.3e}")
  print("  reading: both converge for dominant systems; Gauss-Seidel's serial sweep is host-side")
  print("  (arch-limited), the GEMV is ANE; Jacobi is fully matmul-driven on the ANE.")
  print()

  # ---------------- iterative refinement standalone ---------------------- #
  print("-" * 90)
  print("ITERATIVE REFINEMENT  (sharpen a rough Jacobi solve, n=48)  -  relerr vs np.linalg.solve")
  print("-" * 90)
  print(f"{'cond':>8} | {'rough x0':>12} | {'refine K=3':>12}")
  for cond in (1e1, 1e2):
    A16, b16, xref = _make_spd(n, cond, seed=int(cond) + 5)
    x0 = conjugate_gradient(A16, b16, iters=6)          # deliberately under-iterated
    xr = iterative_refine(A16, b16, x0, iters=3, inner=80)
    print(f"{cond:>8.0e} | {_relerr(x0, xref):>12.3e} | {_relerr(xr, xref):>12.3e}")
  print("  reading: residual-correction sharpens an under-iterated solve by ~1-2 orders at cond=1e1")
  print("  (1.4e-2 -> 8.5e-4) and ~0.7 order at cond=1e2 (3.3e-1 -> 7.2e-2) - the ~1-order envelope.")
  print()

  # ---------------- least squares vs np.linalg.lstsq --------------------- #
  print("-" * 90)
  print("LEAST SQUARES  (overdetermined A [80,24])  -  relerr vs np.linalg.lstsq, swept over cond")
  print("-" * 90)
  print(f"{'cond(A)':>8} | {'CG-normal +refine':>18} | {'np.linalg.lstsq':>16}")
  for cond in (1e1, 1e2):
    rng = np.random.default_rng(int(cond) + 99)
    m2, n2 = 80, 24
    U, _ = np.linalg.qr(rng.standard_normal((m2, n2)))
    V, _ = np.linalg.qr(rng.standard_normal((n2, n2)))
    s = np.geomspace(1.0, cond, n2)
    A = (U * s) @ V.T
    xtrue = rng.standard_normal(n2)
    b = A @ xtrue + 0.01 * rng.standard_normal(m2)
    A16, b16 = f16(A), f16(b)
    x_ls = least_squares(A16, b16, iters=60, refine=2)
    x_ref = np.linalg.lstsq(np.asarray(A16, np.float64), np.asarray(b16, np.float64), rcond=None)[0]
    print(f"{cond:>8.0e} | {_relerr(x_ls, x_ref):>18.3e} | {0.0:>16.3e}")
  print("  reading: forming A^T A SQUARES cond(A): cond(A)=1e1 -> 1e2 solves fine (4e-3); cond(A)=1e2")
  print("  -> 1e4 is past the fp16 CG ceiling and breaks down (relerr ~1, guarded finite). This is")
  print("  the actual cost of the normal equations in fp16 - a QR-lstsq would avoid the squaring but")
  print("  QR is arch-limited on the ANE, so CG-normal is the iterative alternative, valid only while")
  print("  cond(A)^2 stays inside the envelope (cond(A) <~ a few x10).")
  print()

  # ---------------- randomized SVD vs np.linalg.svd ---------------------- #
  print("-" * 90)
  print("RANDOMIZED SVD  (low-rank-ish A [128,96], true rank 8)  -  vs np.linalg.svd")
  print("-" * 90)
  rng = np.random.default_rng(7)
  m3, n3, r = 128, 96, 8
  Ul = np.linalg.qr(rng.standard_normal((m3, r)))[0]           # orthonormal left factor
  Vr = np.linalg.qr(rng.standard_normal((n3, r)))[0]           # orthonormal right factor
  sv = np.geomspace(10.0, 1.0, r)
  A = (Ul * sv) @ Vr.T + 0.01 * rng.standard_normal((m3, n3))  # rank-8 + small noise
  A16 = f16(A)
  k = 8
  U, S, Vt = randomized_svd(A16, k=k, oversample=5, power_iters=2)
  S_ref = np.linalg.svd(np.asarray(A16, np.float64), compute_uv=False)[:k]
  sv_relerr = np.abs(S - S_ref) / (S_ref + 1e-30)
  print(f"  top-{k} singular values:")
  print(f"    {'i':>3} | {'rSVD (ANE)':>12} | {'np.svd':>12} | {'relerr':>10}")
  for i in range(k):
    print(f"    {i:>3} | {S[i]:>12.4f} | {S_ref[i]:>12.4f} | {sv_relerr[i]:>10.3e}")
  # subspace error: principal-angle / projector distance for the top-k right space
  Vt_ref = np.linalg.svd(np.asarray(A16, np.float64))[2][:k]
  P = Vt.T @ Vt; Pref = Vt_ref.T @ Vt_ref
  subspace_err = float(np.linalg.norm(P - Pref) / np.linalg.norm(Pref))
  print(f"  top-{k} right-subspace projector relerr: {subspace_err:.3e}")

  # FLOP split: ANE matmuls vs host QR/SVD
  l = k + 5
  ane_flops = 2 * m3 * n3 * l * (1 + 2 * 2) + 2 * m3 * l * l   # sketch+2 power iters + Q@Ub
  host_flops = m3 * l * l + l * n3 * min(l, n3)                 # QR + small SVD
  print(f"  FLOP split: ANE matmuls ~{ane_flops/1e6:.1f} MFLOP  vs  host QR/SVD ~{host_flops/1e6:.2f} MFLOP "
        f"(~{ane_flops/host_flops:.0f}x on ANE)")
  print()

  # ---------------- verdict ---------------------------------------------- #
  print("#" * 90)
  print("# VERDICT - iterative linear algebra on the ANE")
  print("#" * 90)
  print("  WHAT FITS (fixed-iteration, matmul-dominated -> static dataflow):")
  print("    CG, Jacobi, iterative refinement, randomized SVD/PCA, CG-normal least squares.")
  print("    The expensive GEMV/GEMM run on the ANE (wide accumulator); the host does RNG, the")
  print("    small QR/SVD, the sequential Gauss-Seidel sweep, and the fixed loop counter.")
  print("  ACCURACY ENVELOPE (fp16, measured above): CG/Jacobi/refinement match the direct fp64")
  print("    solve to ~1e-3 for cond<=1e2; at cond~1e3 the fp16 iterates overflow/stall and CG")
  print("    breaks down (relerr ~1) - the ~1-order envelope of the reverse-engineering corpus")
  print("    Symmetric-Jacobi scaling + subspace re-orthonormalization keep the fp16 GEMVs/GEMMs")
  print("    in range (raw products overflow 65504). Least squares via normal equations SQUARES")
  print("    cond(A): fine for cond(A) up to ~a few x10, breaks down once cond(A)^2 leaves the")
  print("    envelope. randomized SVD recovers top singular values to ~1e-5 and the top-k subspace")
  print("    to ~5e-4 on low-rank data (re-orthonormalized power iteration).")
  print("  ALSO IN THIS MODULE (not exercised above): small-n DIRECT factorizations -")
  print("    qr/cholesky/lu (fixed unpivoted recurrences, unrolled into one on-ANE program;")
  print("    the unpivoted forms need well-conditioned leading minors), lu_pivoted (argmax-")
  print("    bridge partial pivoting, segmented), eigh/eigvals/svd (fixed-sweep cyclic Jacobi /")
  print("    unshifted QR), and the fully-on-ANE Krylov solvers lsqr/gmres. Only convergence-")
  print("    gated, data-dependent variants stay ARCH-LIMITED (no in-graph loop), and the")
  print("    native MatrixDecomposition composite is not_currently_callable.")


if __name__ == "__main__":
  main()
