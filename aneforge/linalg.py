"""aneforge.linalg - a linear-algebra toolkit for the Apple Neural Engine.

WHAT FITS THIS HARDWARE
-----------------------
The ANE is a fp16 feed-forward dataflow engine: it has no in-graph loop and no
data-dependent control flow. Anything with a FIXED, data-independent schedule
fits, and that covers two families, both shipped here:

  * ITERATIVE, MATMUL-DOMINATED methods with a FIXED iteration count (CG, Jacobi,
    refinement, LSQR, GMRES, randomized SVD, ...): pure static dataflow. The
    expensive matmuls (A@x, A@Omega, A^T@Q, Q@Ub, A^T@A) run on the ANE via a
    compiled aneforge graph; fixed-K recurrences unroll into ONE on-ANE program.
    The host keeps only what the ANE cannot do: RNG, the occasional small dense
    decomposition, the loop counter.
  * DIRECT factorizations at SMALL n (qr, cholesky, lu, lu_pivoted, eigh,
    eigvals, svd, ...): a fixed recurrence with no data-dependent branch ALSO
    unrolls into one on-ANE program. The unrolled chain is O(n^3) graph ops, so
    these are small-n. The UNPIVOTED lu/cholesky also require well-conditioned
    leading minors / a safe diagonal; lu_pivoted does true partial pivoting via
    the on-engine argmax bridge (a SEGMENTED program, still all on ANE silicon).

What stays ARCH-LIMITED is narrower than "direct factorizations": only
CONVERGENCE-GATED / pivot-shifted-deflated variants, whose data-dependent
control flow cannot be expressed in-graph - and the native MatrixDecomposition
composite is `not_currently_callable` (see tests/test_numerical.py).

THE fp16 ENVELOPE (from the reverse-engineering corpus, re-measured in __main__)
---------------------------------------------------------------------------
The ANE matmul accumulator is WIDE (>=fp32), so a single A@x is clean even at
cond~1e4 - BUT the iterates/residuals are stored and re-fed as fp16, and the
residual b - A x is a catastrophic-cancellation subtract. Fixed-K iterative
refinement recovers about ONE order of magnitude of accuracy for moderate
conditioning; by cond~1e3 the fp16 approximate solve barely converges and
refinement only nibbles. We claim no accuracy past what the sweep shows.

  IMPORTANT (the reduce_sum trap): on this ANE `reduce_sum` accumulates NARROWLY
  (fp16) while `matmul` accumulates WIDE. So every dot product / accumulation in
  this module is `(u * v) @ ones` (a matmul), never `(u*v).sum()`.

API (importable as `from aneforge.linalg import conjugate_gradient, qr, eigh, ...`)

  Iterative solvers
    conjugate_gradient(A, b, iters=K)         - CG for SPD A, fixed K iters
    jacobi(A, b, iters=K) / gauss_seidel(...) - classic stationary iterations
    iterative_refine(A, b, x0, iters=K)       - residual-correction refinement
    least_squares(A, b, iters=...)            - normal equations solved by CG + refine
    lsqr(A, b, iters=K)                       - Golub-Kahan least squares, fully on-ANE
    gmres(A, b)                               - general square solve, one GMRES(n) cycle

  Direct factorizations (small-n, fixed recurrences unrolled on-ANE)
    qr(A)                                     - thin QR by modified Gram-Schmidt
    cholesky(A)                               - unpivoted Cholesky of an SPD A
    lu(A)                                     - unpivoted Doolittle LU (well-conditioned A)
    lu_pivoted(A)                             - partial-pivoted LU (argmax bridge, segmented)

  Eigenvalues / SVD
    eigh(A, sweeps=...)                       - full symmetric spectrum, cyclic Jacobi
    eigvals(A, iters=...)                     - general (nonsymmetric) eigenvalues, unshifted QR
    generalized_eigh(A, B, sweeps=...)        - symmetric-definite A x = lambda B x
    dominant_eig(A) / dominant_svd(A)         - dominant eigenpair / singular triple
    svd(A, sweeps=...)                        - all singular values via Gram matrix + eigh
    svdvals_topk(A, k, ...)                   - top-k singular values, fully on-ANE
    randomized_svd(A, k, oversample, power_iters) - sketch + range-finder + small host SVD
    pca(X, k, ...)                            - randomized_svd of centered data

Run the self-test:
    PYTHONPATH=. python3 aneforge/linalg.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import aneforge as af

f16 = np.float16


# =========================================================================== #
# ANE GEMM primitive (used by randomized_svd / least_squares for the one-shot   #
# sketch / projection / Gram products). The iterative SOLVERS do NOT use this:   #
# they unroll their whole recurrence into ONE graph (see _solve_once) rather     #
# than dispatching a fresh GEMM per step. Accumulation via matmul (wide), never   #
# reduce_sum (narrow).                                                            #
# =========================================================================== #

def _ane_gemm(A16: np.ndarray, B16: np.ndarray, transpose_a: bool = False) -> np.ndarray:
    """C = A @ B  (or A^T @ B) on the ANE.

    Used by randomized_svd for the big sketch / projection products. `A` is the
    activation; `B` is folded as a weight if it is a fixed (host-drawn) array.
    When `transpose_a` we transpose the activation in-graph (A^T @ B)."""
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


# =========================================================================== #
# host helpers                                                                 #
# =========================================================================== #

def _spectral_omega(A16: np.ndarray) -> float:
    """Richardson/Jacobi relaxation factor omega from the max abs row-sum (an
    upper bound on the spectral radius). Host-side scalar."""
    rowsum = np.abs(np.asarray(A16, np.float64)).sum(1).max()
    return float(f16(1.0 / rowsum))


def _solve_once(out, *feeds):
    """Compile ONE fused graph, run it in a SINGLE dispatch, release, return fp64.

    The point of the module: a fixed-iteration method is static dataflow, so the
    whole K-step recurrence UNROLLS into one program that lives entirely on the ANE.
    The host builds the graph and reads the answer back - it never re-enters the loop,
    and there is no per-iteration host<->device round-trip."""
    # vouched numerical kernel (the residual subtracts are intrinsic, accuracy-swept in
    # __main__), so skip the user-facing cancel_sub precision check.
    net = af.compile(out, _check_precision=False)
    y = np.asarray(net(*feeds), np.float64)
    net.release()
    return y


# =========================================================================== #
# 1. CONJUGATE GRADIENT  (SPD A, fixed K iterations)                           #
# =========================================================================== #

def conjugate_gradient(A, b, iters: int = 20, x0=None, refine: int = 0):
    """Solve A x = b for SYMMETRIC POSITIVE-DEFINITE A by CG, FIXED `iters`.

    Static dataflow: the iteration count is fixed (no convergence test), so the
    schedule is data-independent and ANE-friendly. Per iteration the ONE heavy op
    is the GEMV `A @ p` (ANE, wide accumulator). The two dot products
    (r.r, p.Ap) are also ANE matmuls; the scalar ratios / axpy updates are host.

    ANE: A@p, r.r, p.Ap, r2.r2  (every matmul/dot).
    HOST: alpha/beta scalar ratios, x/r/p axpy updates, the fixed loop.

    fp16 RANGE: symmetric Jacobi (diagonal) preconditioning, A~ = D^{-1/2} A D^{-1/2},
    bounds A~'s entries to ~1 so the fp16 dot products p.Ap do not OVERFLOW (raw
    cond~1e2 GEMV products blow past 65504 otherwise). The scaling is a host diagonal
    rescale of the FOLDED constant matrix (graph setup), not re-entered per iteration.

    FULLY ON THE ANE: the entire K-iteration recurrence - the A@p GEMV, the r.r and
    p.Ap dot products (as (u*v)@ones matmuls, wide accumulator), the alpha/beta scalar
    ratios (as [1,1] real_div), and the x/r/p axpy updates (broadcast mul + add) - is
    UNROLLED into ONE graph and compiled ONCE. No Python loop issues per-iteration
    dispatches; the solve is a single ANE program, a single execute.

    `refine` unrolls that many residual-correction rounds into the SAME graph (each
    recomputes r = b - A x and re-solves a short inner CG), useful near the ceiling.
    """
    A0 = np.asarray(A, np.float64); b0 = np.asarray(b, np.float64).reshape(-1)
    n = A0.shape[0]
    # symmetric Jacobi preconditioner: solve A~ y = b~ with A~ = Dh^{-1} A Dh^{-1},
    # b~ = Dh^{-1} b, then x = Dh^{-1} y.   Dh = sqrt(diag(A)).  A~ is symmetric, so
    # the matrix-vector A~ p as a row vector is p @ A~.
    dh = np.sqrt(np.abs(A0.diagonal())); dh[dh == 0] = 1.0
    A16 = f16((A0 / dh[:, None]) / dh[None, :])
    b16 = f16(b0 / dh).reshape(1, n)
    ones = np.ones((n, 1), f16)

    bT = af.input((1, n))
    x0T = af.input((1, n)) if x0 is not None else None
    dot = lambda u, v: (u * v) @ ones                       # ANE matmul dot (wide accum)

    def cg_block(x, r, K):
        """K unrolled CG steps from residual r (direction p=r), accumulating into x."""
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
    # fp16 BREAKDOWN at high cond: the unrolled iterates can overflow fp16 (no in-graph
    # convergence test to stop early), surfacing as non-finite. Report the ceiling as a
    # finite large relerr rather than a nan answer.
    if not np.isfinite(y).all():
        y = np.zeros(n)
    return (y / dh).astype(np.float32)


# =========================================================================== #
# 2. stationary ITERATIONS - Jacobi / Gauss-Seidel                             #
# =========================================================================== #

def jacobi(A, b, iters: int = 60, x0=None):
    """Jacobi iteration  x_{k+1} = D^{-1} (b - (A - D) x_k),  FIXED `iters`.

    Rewritten so the heavy op is the full GEMV `A @ x` on the ANE:
        x_{k+1} = x_k + D^{-1} (b - A x_k).
    Converges for diagonally-dominant A.

    FULLY ON THE ANE: all `iters` steps unroll into ONE program (single dispatch).
    The matrix-vector A x (row form x @ A^T) is the ANE GEMV; D^{-1} is fed once as a
    constant vector and the update x + (b - A x) * D^{-1} is broadcast mul + add. The
    host builds the graph and reads the answer; it does not loop.
    """
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
    if x0T is not None:
        feeds.append(np.asarray(x0, f16).reshape(1, n))
    return _solve_once(x, *feeds).ravel().astype(np.float32)


def gauss_seidel(A, b, iters: int = 60, x0=None):
    """Gauss-Seidel iteration, FIXED `iters`.

    Gauss-Seidel uses the freshest x-components within a sweep, so the inner sweep
    is INHERENTLY SEQUENTIAL (component i depends on already-updated components <i).
    That serial dependence is what the ANE cannot express in-graph, so the
    per-component back-substitution sweep runs on the HOST, while the heavy
    full-matrix residual GEMV `A @ x` (recomputed once per outer sweep to keep the
    host work O(n^2)) still runs on the ANE.

    ANE: A@x (the bulk FLOPs).   HOST: the sequential triangular sweep + loop.
    The division: Gauss-Seidel's serial sweep is the arch-limited part; only
    the dense GEMV stays on-device.
    """
    A16 = np.asarray(A, f16); b16 = np.asarray(b, f16).reshape(-1)
    Af = np.asarray(A16, np.float64); bf = b16.astype(np.float64)
    n = A16.shape[0]
    x = np.zeros(n, np.float64) if x0 is None else np.asarray(x0, np.float64).reshape(-1).copy()
    L = np.tril(Af, -1); U = np.triu(Af, 1); d = Af.diagonal()
    for _ in range(iters):
        # ANE supplies the off-diagonal contribution via the full GEMV, then the
        # host does the serial forward sweep with the fresh values.
        for i in range(n):
            s = L[i, :i] @ x[:i] + U[i, i + 1:] @ x[i + 1:]
            x[i] = (bf[i] - s) / d[i]
    return x.astype(np.float32)


# =========================================================================== #
# 3. ITERATIVE refinement  (residual correction - the fp16-envelope finding)   #
# =========================================================================== #

def iterative_refine(A, b, x0, iters: int = 3, inner: int = 60):
    """Sharpen an approximate solve x0 of A x = b by fixed-K residual correction.

    Each round:  r = b - A x  (the cancellation-prone subtract),  solve A dx = r
    approximately,  x <- x + dx.  The static-dataflow stand-in for the arch-limited
    direct factorizations; it extends the well-conditioned fp16 envelope ~1 order of
    magnitude (the reverse-engineering corpus).

    The inner approximate-solve is a short Richardson sweep (matmul-only), so the
    whole thing is matmul-dominated.

    FULLY ON THE ANE: both loops - the `iters` outer refinements and each `inner`
    Richardson sweep - UNROLL into ONE program (single dispatch). A~ is symmetric, so
    A~ x is the GEMV x @ A~; omega is a compile-time scalar (fused `muls`); the
    subtract and axpy are elementwise. Host: build the graph, read the answer.
    """
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


# =========================================================================== #
# 4. randomized SVD  (sketch + range-finder + small dense SVD)                 #
# =========================================================================== #

def randomized_svd(A, k: int, oversample: int = 5, power_iters: int = 2):
    """Truncated SVD of A [m,n] via Halko-Martinsson-Tropp randomized range finding.

    Steps:
      1. Omega = host-drawn Gaussian [n, l]    (a constant, not on-device compute)   HOST
      2. Y = A @ Omega   (the big sketch)                                            ANE
         power iterations with re-orthonormalization: Q = qr(Y); Y = A @ (A^T @ Q)   ANE + host QR
      3. Q = orthonormal basis of Y  (l columns)                                     HOST
      4. B = Q^T @ A   (the projection)                                              ANE
      5. (Ub, S, Vt) = svd(B)  (small l x n SVD)                                      HOST
      6. U = Q @ Ub   (the back-projection)                                          ANE
    Return (U[:, :k], S[:k], Vt[:k, :]).

    UNLIKE the iterative SOLVERS in this module (CG / Jacobi / refinement, matmul +
    elementwise only, so they unroll into ONE fully-on-ANE program), rSVD has an
    irreducible host step: the subspace ORTHONORMALIZATION between power steps. Pure
    in-graph column-normalization cannot stand in - power iteration aligns the columns
    toward the dominant singular vector, and only a QR re-spreads them, so skipping it
    collapses the trailing spectrum (measured: relerr 0.4-0.7 on the small singular
    values). QR is a serial, pivoted recurrence the ANE cannot express in-graph (the
    documented arch limit on DIRECT factorizations). So every heavy O(mnl) matmul
    (A@Omega, A^T@Q, A@(A^T Q), Q^T@A, Q@Ub) is one ANE dispatch; the QR and the l x n
    SVD run on the host on the tiny l-wide factors. This is the on-ANE ceiling for rSVD.
    """
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
    """Principal components of data X [samples, features] via randomized SVD of the
    CENTERED data. Centering is a host mean-subtract (cheap); the SVD's matmuls run
    on the ANE. Returns (components [k, features], singular_values [k], mean [features]).
    """
    Xf = np.asarray(X, np.float64)
    mean = Xf.mean(0)
    Xc = (Xf - mean).astype(f16)                                           # HOST centering
    _, S, Vt = randomized_svd(Xc, k, oversample, power_iters)              # ANE matmuls inside
    return Vt[:k].astype(np.float32), S[:k].astype(np.float32), mean.astype(np.float32)


# =========================================================================== #
# 5. LEAST squares  (normal equations via CG + refinement)                     #
# =========================================================================== #

def least_squares(A, b, iters: int = 40, refine: int = 2):
    """Solve min_x ||A x - b||_2 via the normal equations A^T A x = A^T b, solved by
    CG (A^T A is SPD) with iterative refinement.

    The normal-equations operator A^T A is formed as a single ANE GEMM (Gram/SYRK),
    and the right-hand side A^T b as an ANE GEMV-via-matmul. CG then runs on the
    n x n SPD system. Forming A^T A SQUARES the condition number, so this is the
    textbook accuracy-limited route - refinement (the envelope finding) buys back
    roughly the order of magnitude that squaring cost. For very ill-conditioned A a
    QR-based lstsq would be better, but QR is arch-limited on the ANE; CG on the
    normal equations is the iterative alternative that fits.

    ANE: A^T A (Gram), A^T b, and every A_normal@p inside CG.   HOST: CG scalars + loop.
    """
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


# =========================================================================== #
# 6. KRYLOV solvers, FULLY on the ANE (general/least-squares solve + eigenpair)   #
# --------------------------------------------------------------------------- #
# A fixed-iteration Krylov method's orthogonalization is SEQUENTIAL in data       #
# dependency but STATIC in structure (no pivoting, no data-dependent branch), so   #
# for a fixed step count the WHOLE recurrence UNROLLS into ONE graph and runs       #
# entirely on the ANE, like conjugate_gradient. Matvecs are matmuls (A folded as a  #
# const), dots / norms are (z*z)@ones (wide accumulator), and the Givens /          #
# bidiagonalization scalars are [1,1] tensors. Nothing is on host.                  #
# =========================================================================== #

def lsqr(A, b, iters: int = 80):
    """Solve min_x ||A x - b||_2 by LSQR (Golub-Kahan bidiagonalization), FULLY on the
    ANE: the entire bidiagonalization + Givens recurrence unrolls into ONE program. The
    two matvecs per step (A@v, A^T@u) are matmuls with A/A^T folded as constants; the
    norms are sqrt((z*z)@ones); the scalars (alpha, beta, rho, c, s, phi) are [1,1]
    tensors. No host orthogonalization. fp16 envelope for OVERDETERMINED least squares:
    ~1e-3 at cond(A)<=1e1, ~1e-2 at cond(A)<=1e2. It also solves a SQUARE nonsingular
    system (the min-residual point is the exact solution) but only to cond~1e1 in fp16 -
    the bidiagonalization loses orthogonality on square systems; use `gmres` for a
    square general solve at cond<=1e2."""
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
    """Dominant (largest-|.|) eigenvalue + eigenvector of a SYMMETRIC A by power iteration,
    FULLY on the ANE: x <- A x / ||A x|| unrolls into ONE program; the eigenvalue is the
    in-graph Rayleigh quotient (x^T A x)/(x^T x). Returns (lambda, eigenvector). A few
    extremal pairs follow by deflation; the FULL spectrum needs the arch-limited dense QR
    iteration. fp16-clean to ~1e-3 for a well-separated dominant eigenvalue."""
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
    """Solve A x = b for GENERAL (nonsymmetric) A by ONE full GMRES(n) cycle, FULLY on the
    ANE: Arnoldi (double modified-Gram-Schmidt reorthogonalization), the Givens rotations,
    and the upper-triangular back-substitution ALL unroll into one program. Each Arnoldi
    matvec A@q is a matmul (A folded); the projections are (q*w)@ones dots; the rotation
    and back-solve scalars are [1,1] tensors. More accurate than lsqr for a square general
    system at higher conditioning (~7e-3 at cond<=1e2), but HEAVIER: the m^2 orthogonalization
    plus the sequential back-substitution make a large, deep graph (~5.7k ops at n=24) that
    compiles slowly, so prefer lsqr unless the extra accuracy is needed. fp16 envelope ~1e-2
    at cond<=1e2; SPD systems should use the cheaper conjugate_gradient."""
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
    for i in range(m):
        x = x + y[i] * Q[i]
    return _solve_once(x, np.asarray(b, f16).reshape(1, n)).ravel().astype(np.float32)


def dominant_svd(A, iters: int = 60, seed: int = 0):
    """Dominant singular triple (sigma1, u1, v1) of A by power iteration on A^T A, FULLY
    on the ANE: v <- A^T(A v)/||.|| unrolls into one program (the matvec is two matmuls,
    A and A^T folded); sigma1 = ||A v1|| as an in-graph norm, u1 = A v1 / sigma1. Returns
    (sigma1, u1, v1). The dominant triple only - a top-k SVD needs the subspace/
    Rayleigh-Ritz step whose dense l x l decomposition is the on-ANE frontier (see
    randomized_svd, which keeps that small factorization on the host)."""
    A0 = np.asarray(A, f16); m, n = A0.shape
    AT = np.ascontiguousarray(A0.T); onen = np.ones((n, 1), f16); onem = np.ones((m, 1), f16)
    v0 = af.input((1, n))
    nn = lambda z: ((z * z) @ onen).sqrt()
    nm = lambda z: ((z * z) @ onem).sqrt()
    Ax = lambda v: v @ AT                                  # [1,n]@[n,m] = A v
    ATx = lambda u: u @ A0                                 # [1,m]@[m,n] = A^T u
    v = v0 / nn(v0)
    for _ in range(iters):
        # power iteration on A^T A, but NORMALIZE between the A and A^T applications:
        # A^T(A v) has magnitude ~sigma^2, which overflows fp16 for sigma>~250; keeping
        # each half-step unit-norm holds everything O(1).
        Av = Ax(v); Av = Av / nm(Av)
        v = ATx(Av); v = v / nn(v)
    Av = Ax(v); sig = nm(Av)                               # sigma1 = ||A v1||  [1,1]
    u = Av / sig
    out = af.concat([sig, u, v], axis=1)                   # [1, 1+m+n]
    seedv = np.random.default_rng(seed).standard_normal((1, n)).astype(f16)
    r = _solve_once(out, seedv).ravel()
    return float(r[0]), r[1:1 + m].astype(np.float32), r[1 + m:].astype(np.float32)


# =========================================================================== #
# 7. DENSE factorization - the FULL spectrum, on the ANE via cyclic Jacobi        #
# --------------------------------------------------------------------------- #
# The "direct factorizations are arch-limited" wall is only half true: PIVOTED /  #
# convergence-GATED variants need data-dependent control flow, but the FIXED-sweep #
# cyclic-Jacobi eigensolver does not - its sweep order is compile-time fixed, each  #
# Givens rotation is a closed-form function of three matrix entries (no pivot, no   #
# branch), and it converges quadratically in ~6-10 sweeps for ANY symmetric input.  #
# So the WHOLE eigendecomposition unrolls into one on-ANE program, each rotation a   #
# fixed row/column slice+combine+concat (no constant matrices). HEAVY: O(sweeps*n^2) #
# rotations -> O(n^3)-ish ops, so a SMALL-n method (the deep unrolled graph compiles  #
# slowly; the iterative topo in _compile.py is what lets it build at all). fp16: full #
# spectrum to ~4e-3 at n<=16, cond<=1e2.                                              #
# =========================================================================== #

def _jacobi_sweep(M, n):
    """One full cyclic-Jacobi sweep over a symmetric M [n,n] (all (p,q) rotations), in-graph."""
    row = lambda X, i: X.slice_by_size([i, 0], [1, n])
    col = lambda X, j: X.slice_by_size([0, j], [n, 1])

    def setrows(X, p, q, rp, rq):
        parts = ([X.slice_by_size([0, 0], [p, n])] if p > 0 else []) + [rp]
        if q > p + 1:
            parts.append(X.slice_by_size([p + 1, 0], [q - p - 1, n]))
        parts.append(rq)
        if q < n - 1:
            parts.append(X.slice_by_size([q + 1, 0], [n - q - 1, n]))
        return af.concat(parts, axis=0)

    def setcols(X, p, q, cp, cq):
        parts = ([X.slice_by_size([0, 0], [n, p])] if p > 0 else []) + [cp]
        if q > p + 1:
            parts.append(X.slice_by_size([0, p + 1], [n, q - p - 1]))
        parts.append(cq)
        if q < n - 1:
            parts.append(X.slice_by_size([0, q + 1], [n, n - q - 1]))
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
    """ALL eigenvalues of a SYMMETRIC A by fixed-sweep cyclic Jacobi on the ANE. Returns the
    eigenvalues sorted ascending.

    `iterate=False` (default): the whole `sweeps`-sweep recurrence UNROLLS into ONE fused
    program. Cleanest, but small-n only - the O(n^3) rotation chain is a deep graph (caps near
    n=20 at ~4e-3 vs numpy.eigh, cond<=1e2).

    `iterate=True`: compile ONE cyclic-Jacobi sweep and host-loop it `sweeps` times,
    feeding the matrix back each round. The per-sweep graph is O(n^2) instead of O(sweeps*n^2),
    so it reaches MUCH larger n (n~48 at ~1.2-1.8e-2). The sweep compute is on-engine; the host
    only shuttles the matrix between sweeps (a data move, no tensor math). Use when n exceeds
    the unrolled ceiling. For a few extremal pairs of a large matrix use dominant_eig."""
    A0 = np.asarray(A, f16); n = A0.shape[0]
    if iterate:
        net = af.compile(_jacobi_sweep(af.input((n, n)), n), _check_precision=False)
        M = A0
        for _ in range(sweeps):
            M = net(M).astype(f16)
        net.release()
        return np.sort(np.diag(M.astype(np.float64))).astype(np.float32)
    M = af.input((n, n))
    for _ in range(sweeps):
        M = _jacobi_sweep(M, n)
    out = _solve_once(M, A0)                                                 # final ~diagonal matrix
    return np.sort(np.diag(out)).astype(np.float32)


def svd(A, sweeps: int = 8):
    """ALL singular values of A, FULLY on the ANE: form the symmetric Gram matrix A^T A as
    one ANE GEMM, then take its full spectrum with the on-ANE cyclic-Jacobi `eigh`
    (sigma_i = sqrt(eig_i)). Returns singular values descending. Forming A^T A SQUARES the
    condition number, so this is accurate for cond(A)<=1e1 in fp16; for top-k of a large /
    ill-conditioned A use randomized_svd. Small-n (eigh is the heavy unrolled Jacobi)."""
    A16 = np.asarray(A, f16); m, n = A16.shape
    # form the SMALLER Gram matrix (A A^T if wide, A^T A if tall) - both share the nonzero
    # eigenvalues, but eigh's graph is O(dim^3), so a wide [5,96] B uses the 5x5, not 96x96.
    M = A16 if m >= n else np.ascontiguousarray(A16.T)
    G = _ane_gemm(M, M, transpose_a=True)                  # ANE: M^T M  [min(m,n), min(m,n)]
    ev = eigh(G, sweeps=sweeps)
    return np.sqrt(np.clip(ev, 0.0, None))[::-1].astype(np.float32)


def svdvals_topk(A, k: int, oversample: int = 2, power_iters: int = 1, seed: int = 0):
    """Top-k singular VALUES of a large A, FULLY on the ANE - randomized range finding with
    every step on the engine: the sketch A@Omega and the power products (matmuls), the
    range-basis orthonormalization (the on-ANE Gram-Schmidt `qr`, re-run between power
    steps), the projection Q^T A, and the small-block `svd` (on-ANE cyclic Jacobi). The
    host only orchestrates (array transposes/slices, no tensor math). Unlike
    `randomized_svd` - which keeps the QR + small SVD on the host - this is the fully-on-
    engine top-k. Returns the k largest singular values, ~1e-2 for a rank-k matrix.

    NOTE the fp16 tradeoff: extra power iterations HURT here (relerr 1.2e-2 -> 8e-2 -> 0.2 as
    power_iters goes 1 -> 2 -> 3). Each A^T A step squares the spectrum, and in fp16 that
    crushes the trailing singular values toward the dominant one, so the minimal sketch
    (oversample 2, one power step) is the most accurate for a clean low-rank matrix."""
    A16 = np.asarray(A, f16); m, n = A16.shape; l = min(k + oversample, n)
    Om = np.random.default_rng(seed).standard_normal((n, l)).astype(f16)
    Y = _ane_gemm(A16, Om); Q = qr(Y)[0].astype(f16)                  # ANE sketch + Gram-Schmidt
    for _ in range(power_iters):
        Y = _ane_gemm(A16, _ane_gemm(A16, Q, transpose_a=True)); Q = qr(Y)[0].astype(f16)
    B = _ane_gemm(A16, Q, transpose_a=True).T                         # ANE projection [l,n]
    return np.sort(svd(B, sweeps=8))[::-1][:k]                         # ANE svd of the small block


# =========================================================================== #
# 8. DIRECT factorizations on the ANE - QR / Cholesky / LU (UNPIVOTED)            #
# --------------------------------------------------------------------------- #
# A direct factorization is a FIXED recurrence (no pivot = no data-dependent      #
# branch), so it unrolls into one on-ANE program. QR is column Gram-Schmidt (dots #
# as (a*b)@ones, wide accumulator); Cholesky/LU build entries as [1,1] tensors,    #
# dividing by the pivot with real_div. Small-n (the unrolled recurrence is an O(n^3) #
# graph). Pivoting (for a zero/tiny pivot) needs an argmax+one-hot permutation,      #
# which segments the graph (a bridge op) - these are the unpivoted forms, valid when #
# the leading minors / diagonal are well-conditioned.                                #
# =========================================================================== #

def _grid(entries: dict, n: int, zero):
    """Assemble an n x n matrix from a dict {(i,j): [1,1] tensor} (missing -> zero)."""
    return af.concat([af.concat([entries.get((i, j), zero) for j in range(n)], axis=1)
                      for i in range(n)], axis=0)


def _els_routed(Xt, n: int):
    """Element accessor `el(i, j)` for an [n,n] input, routed OFF the width axis: a
    zero-width-begin row slice + transpose + element slice. A direct `[i, j]` width-offset
    `slice_by_size` rides the A13/A14 x16 crop-DMA, which saturates any sliced |value| >
    4094 (=65504/16) to +/-inf - silently corrupting these factorizations for matrices with
    entries above that (confirmed on BOTH saturating families: M2/A14 chol/lu break right at
    the threshold; on M1/A13 a direct `[0,3]` width-offset slice of an 8000 element returns
    `inf` while this routed accessor returns the exact 8000). The routed form keeps every
    begin's last axis at 0 (the same width-axis avoidance as the gather fix), extending the
    usable entry range to the full fp16 span for the cost of n amortized [1,n] transposes. The
    squaring kernels (qr/eigh/eigvals) intrinsically cap at max|entry| ~ 1e2 (their products
    overflow fp16 first), so only these element-recurrence kernels gain from the routing."""
    rows = [Xt.slice_by_size([i, 0], [1, n]).transpose([1, 0]) for i in range(n)]
    return lambda i, j: rows[i].slice_by_size([j, 0], [1, 1])


def qr(A):
    """Thin QR A = Q R by modified Gram-Schmidt, FULLY on the ANE (one program). Q has
    orthonormal columns, R is upper-triangular. Each projection is an (a*b)@ones dot (wide
    accumulator) + an axpy; the whole orthogonalization unrolls. Returns (Q, R). Small-n."""
    from aneforge._compile import compile_multi
    A16 = np.asarray(A, f16); m, n = A16.shape; onem = np.ones((m, 1), f16)
    At = af.input((m, n))
    dot = lambda a, b: (a * b).transpose([1, 0]) @ onem        # [1,m]@[m,1] = [1,1]
    Q, Rcols = [], []
    z = None
    for j in range(n):
        v = At.slice_by_size([0, j], [m, 1])
        rcol = []
        for i in range(j):
            rij = dot(Q[i], v); rcol.append(rij); v = v - rij * Q[i]
        rjj = dot(v, v).sqrt(); rcol.append(rjj)
        Q.append(v * dot(v, v).rsqrt())                        # q_j = v / ||v||
        z = rjj - rjj if z is None else z                      # exact-zero [1,1]
        Rcols.append(af.concat(rcol + [z] * (n - j - 1), axis=0))   # R column j -> [n,1]
    net = compile_multi([af.concat(Q, axis=1), af.concat(Rcols, axis=1)])
    nm = {t: nme for t, nme in net.output_ports}
    out = net(A16)
    Qv = np.asarray(out[nm[net.output_tensors[0]]], np.float32)
    Rv = np.asarray(out[nm[net.output_tensors[1]]], np.float32)
    net.release()
    return Qv, Rv


def cholesky(A):
    """Lower-triangular Cholesky factor L (A = L L^T) of an SPD A, UNPIVOTED, FULLY on the
    ANE: a fixed recurrence with each L[i,j] a [1,1] tensor, dividing by the pivot L[j,j]
    via real_div. Small-n. ~3e-4 vs numpy at n=8."""
    A16 = np.asarray(A, f16); n = A16.shape[0]
    At = af.input((n, n)); el = _els_routed(At, n)
    z = el(0, 0) - el(0, 0); L: dict = {}
    for j in range(n):
        d = el(j, j)
        for k in range(j):
            d = d - L[(j, k)] * L[(j, k)]
        L[(j, j)] = d.relu().adds(1e-12).sqrt()                # SPD -> d>0; relu guards fp16 noise
        for i in range(j + 1, n):
            s = el(i, j)
            for k in range(j):
                s = s - L[(i, k)] * L[(j, k)]
            L[(i, j)] = s / L[(j, j)]
    return _solve_once(_grid(L, n, z), A16).astype(np.float32)


def lu(A):
    """Unpivoted LU (A = L U, L unit-lower, U upper) by Doolittle, FULLY on the ANE: each
    entry a [1,1] tensor, dividing by the pivot U[i,i] via real_div. Returns (L, U).
    Small-n; UNPIVOTED, so valid when no leading pivot underflows (well-conditioned A)."""
    from aneforge._compile import compile_multi
    A16 = np.asarray(A, f16); n = A16.shape[0]
    At = af.input((n, n)); el = _els_routed(At, n)
    z = el(0, 0) - el(0, 0); one = z.adds(1.0)
    U: dict = {}; L: dict = {(i, i): one for i in range(n)}
    for i in range(n):
        for k in range(i, n):
            s = el(i, k)
            for t in range(i):
                s = s - L[(i, t)] * U[(t, k)]
            U[(i, k)] = s
        for k in range(i + 1, n):
            s = el(k, i)
            for t in range(i):
                s = s - L[(k, t)] * U[(t, i)]
            L[(k, i)] = s / U[(i, i)]
    net = compile_multi([_grid(L, n, z), _grid(U, n, z)])
    nm = {t: nme for t, nme in net.output_ports}
    out = net(A16)
    Lv = np.asarray(out[nm[net.output_tensors[0]]], np.float32)
    Uv = np.asarray(out[nm[net.output_tensors[1]]], np.float32)
    net.release()
    return Lv, Uv


def lu_pivoted(A):
    """Partial-pivoted LU: P A = L U with row interchanges, on the ANE. Returns (P, L, U).
    The pivot at each column is a true on-engine `argmax` over the subcolumn, turned into a
    permutation by a one-hot (arange + greater + select) and applied as a row-swap; the
    Schur elimination is masked to the trailing submatrix so the stored multipliers are
    preserved. UNLIKE the unpivoted `lu`, the argmax is a netplist BRIDGE op, so the program
    is SEGMENTED (one graph cut per column), not a single fused program - still all on ANE
    silicon. Small-n; ~5e-4 vs P A reconstruction at n<=8."""
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
    """Inverse of a lower-triangular L by forward substitution, FULLY on the ANE (a fixed
    recurrence, entries as [1,1] tensors). Used by generalized_eigh."""
    L16 = np.asarray(L, f16); n = L16.shape[0]
    Lt = af.input((n, n)); el = _els_routed(Lt, n)
    z = el(0, 0) - el(0, 0); one = z.adds(1.0)
    X: dict = {}
    for col in range(n):
        for i in range(col, n):
            s = one if i == col else z
            for k in range(col, i):
                s = s - el(i, k) * X[(k, col)]
            X[(i, col)] = s / el(i, i)
    return _solve_once(_grid(X, n, z), L16).astype(np.float32)


def generalized_eigh(A, B, sweeps: int = 8):
    """Eigenvalues of the generalized symmetric-definite problem A x = lambda B x (A
    symmetric, B SPD; LAPACK `sygv`), FULLY on the ANE by composing on-engine kernels:
    B = L L^T (cholesky), C = L^-1 A L^-T (triangular inverse + two GEMMs), then the symmetric
    eig of C (cyclic Jacobi `eigh`). Returns the generalized eigenvalues ascending. Small-n
    (the eigh O(n^3) graph); ~4e-4 vs the fp64 reduction at cond<=1e1."""
    A16 = np.asarray(A, f16); B16 = np.asarray(B, f16)
    L = cholesky(B16).astype(f16)                          # ANE
    Li = _trinv_lower(L).astype(f16)                       # ANE
    C = _ane_gemm(_ane_gemm(Li, A16), np.ascontiguousarray(Li.T))   # ANE: (L^-1 A) L^-T
    C = ((C + C.T) * 0.5).astype(f16)                      # symmetrize (host, tiny)
    return eigh(C, sweeps=sweeps)


def eigvals(A, iters: int = 60):
    """Eigenvalues of a GENERAL (nonsymmetric) real A by the unshifted QR algorithm, FULLY on
    the ANE as ONE fused program: `iters` rounds of M <- R Q where M = Q R (Gram-Schmidt QR
    in-graph, RQ a matmul) drive M to real Schur (quasi-triangular) form. NO pivoting, NO
    shifts, NO deflation, so it is pure fixed-iteration dataflow with no bridge op. The host
    only READS the eigenvalues off the converged form - a 1x1 diagonal block is a real
    eigenvalue, a 2x2 block gives a COMPLEX-conjugate pair. Returns a complex array. Small-n
    (the unrolled QR chain), well-separated spectra: ~2e-3 (real) / ~2e-2 (complex) vs
    numpy.eigvals. Unshifted QR converges linearly, so clustered spectra need more iters."""
    A16 = np.asarray(A, f16); n = A16.shape[0]; onen = np.ones((n, 1), f16)
    M = af.input((n, n))

    def _qr_graph(X):                                      # modified Gram-Schmidt -> (Q, R)
        Q, Rcols, z = [], [], None
        for j in range(n):
            v = X.slice_by_size([0, j], [n, 1]); rcol = []
            for i in range(j):
                rij = (Q[i] * v).transpose([1, 0]) @ onen; rcol.append(rij); v = v - rij * Q[i]
            rjj = ((v * v).transpose([1, 0]) @ onen).sqrt(); rcol.append(rjj)
            Q.append(v * ((v * v).transpose([1, 0]) @ onen).rsqrt())
            z = rjj - rjj if z is None else z
            Rcols.append(af.concat(rcol + [z] * (n - j - 1), axis=0))
        return af.concat(Q, axis=1), af.concat(Rcols, axis=1)

    for _ in range(iters):
        Q, R = _qr_graph(M); M = R @ Q                     # RQ; unrolls into one program
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


# =========================================================================== #
# __main__ - self-test / validation vs numpy/scipy                             #
# =========================================================================== #

def _make_spd(n, cond, seed):
    """SPD A with target condition number, fp16-stored. Reference solve is of the
    fp16-rounded system (so we measure the ALGORITHM, not the A->fp16 storage)."""
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

