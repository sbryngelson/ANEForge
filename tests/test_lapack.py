"""LAPACK-on-the-ANE characterization corpus.

LAPACK is built on three things the ANE cannot do: PIVOTING (data-dependent argmax +
row swap), a SEQUENTIAL data-sized inner recurrence, and fp64. So LAPACK's *routines*
do not port. But each LAPACK *problem* has a fixed-iteration / matmul-dominated method
that IS static dataflow, and those UNROLL into a single on-ANE program (no host compute).

This file is the map: for each LAPACK problem family it runs the ANE method fully on the
engine, validates against the numpy fp64 reference on the SAME fp16-rounded system, and
records the conditioning envelope. The families whose only method is a pivoted/recursive
DIRECT factorization (full spectrum, pivoted LU/QR) are the documented walls - listed,
not run. Accumulation is via matmul (the ANE's wide >=fp32 accumulator).

Run: PYTHONPATH=. python3 tests/test_lapack.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import aneforge.linalg as L

f16 = np.float16
rng0 = np.random.default_rng(0)


def relerr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _general(n, cond, seed):
    r = np.random.default_rng(seed)
    U = np.linalg.qr(r.standard_normal((n, n)))[0]; V = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (U * np.geomspace(1, cond, n)) @ V.T


def _spd(n, cond, seed):
    r = np.random.default_rng(seed); Q = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (Q * np.geomspace(1, cond, n)) @ Q.T


def _sym(n, cond, seed):
    r = np.random.default_rng(seed); Q = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (Q * (np.geomspace(1, cond, n) * r.choice([-1, 1], n))) @ Q.T


# (family, LAPACK driver, ANE method, runs at cond -> (relerr, envelope_tol))
def check_spd_solve(cond):
    A = _spd(48, cond, int(cond) + 1); x = rng0.standard_normal(48); b = A @ x
    return relerr(L.conjugate_gradient(f16(A), f16(b), iters=40), x), 5e-2


def check_general_solve(cond):
    n = 12; A = _general(n, cond, int(cond) + 2); x = np.random.default_rng(1).standard_normal(n)
    return relerr(L.gmres(f16(A), f16(A @ x)), x), 3e-2


def check_least_squares(cond):
    r = np.random.default_rng(int(cond) + 3)
    U = np.linalg.qr(r.standard_normal((80, 24)))[0]; V = np.linalg.qr(r.standard_normal((24, 24)))[0]
    A = (U * np.geomspace(1, cond, 24)) @ V.T; xt = r.standard_normal(24); b = A @ xt + 0.01 * r.standard_normal(80)
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    return relerr(L.lsqr(f16(A), f16(b), iters=80), ref), 1.5e-1


def check_dominant_eig(cond):
    A = _sym(40, cond, 7); ev = np.linalg.eigvalsh(A); ref = ev[np.argmax(np.abs(ev))]
    lam, _ = L.dominant_eig(f16(A), iters=80)
    return abs(lam - ref) / abs(ref), 1e-2


def check_dominant_svd(cond):
    # power iteration recovers the dominant triple when there is a spectral GAP (its
    # envelope is the gap, not conditioning): sigma1 = cond, sigma2 <= 0.25*cond.
    r = np.random.default_rng(int(cond) + 5)
    U = np.linalg.qr(r.standard_normal((50, 30)))[0]; V = np.linalg.qr(r.standard_normal((30, 30)))[0]
    s = np.concatenate([[float(cond)], np.geomspace(0.25 * cond, 1.0, 29)])   # clear leading gap
    A = (U * s) @ V.T; ref = np.linalg.svd(A, compute_uv=False)[0]
    sig, _, _ = L.dominant_svd(f16(A), iters=80)
    return abs(sig - ref) / ref, 2e-2


def check_qr(cond):        # A = Q R reconstruction (QR is not unique; check the product)
    A = _general(8, cond, int(cond) + 21); Q, R = L.qr(f16(A))
    return relerr(Q @ R, A), 5e-3


def check_cholesky(cond):  # A = L L^T reconstruction, SPD
    A = _spd(8, cond, int(cond) + 22); Lc = L.cholesky(f16(A))
    return relerr(Lc @ Lc.T, A), 5e-3


def check_lu(cond):        # A = L U reconstruction, unpivoted
    A = _general(8, cond, int(cond) + 23); Lm, Um = L.lu(f16(A))
    return relerr(Lm @ Um, A), 2e-2


def check_nonsym_eig(cond):  # nonsymmetric eigenvalues by unshifted QR, fully on-engine
    r = np.random.default_rng(int(cond)); V = r.standard_normal((6, 6))
    A = V @ np.diag(np.geomspace(8, 1, 6)) @ np.linalg.inv(V)   # real, well-separated spectrum
    got = np.sort_complex(L.eigvals(f16(A), iters=60)); ref = np.sort_complex(np.linalg.eigvals(A))
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref)), 2e-2


def check_pivoted_lu(cond):  # P A = L U with on-engine argmax pivoting (segmented)
    A = _general(8, cond, int(cond) + 41); A[0, 0] = 1e-3   # tiny leading pivot
    P, Lp, Up = L.lu_pivoted(f16(A))
    return relerr(P @ A, Lp @ Up), 3e-3


def check_full_eig(cond):  # ALL eigenvalues via on-ANE cyclic Jacobi (small n)
    A = _sym(8, cond, int(cond) + 9)
    return relerr(L.eigh(f16(A), sweeps=8), np.sort(np.linalg.eigvalsh(A))), 1.5e-2


def check_generalized_eig(cond):  # A x = lambda B x (sygv): chol + trinv + eigh, on-ANE
    A = _sym(8, cond, int(cond) + 31); B = _spd(8, cond, int(cond) + 32)
    Li = np.linalg.inv(np.linalg.cholesky(B))
    ref = np.sort(np.linalg.eigvalsh(Li @ A @ Li.T))
    return relerr(L.generalized_eigh(f16(A), f16(B)), ref), 3e-2


def check_full_svd(cond):  # ALL singular values via eigh(A^T A) on ANE; cond^2 -> cond(A)<=1e1
    r = np.random.default_rng(int(cond) + 13)
    U = np.linalg.qr(r.standard_normal((10, 8)))[0]; V = np.linalg.qr(r.standard_normal((8, 8)))[0]
    A = (U * np.geomspace(1, cond, 8)) @ V.T
    return relerr(L.svd(f16(A), sweeps=8), np.linalg.svd(A, compute_uv=False)), 3e-2


def check_topk_svd(cond):  # cond unused (low-rank); fully-on-ANE randomized range finding
    r = np.random.default_rng(7); Ul = np.linalg.qr(r.standard_normal((128, 8)))[0]
    Vr = np.linalg.qr(r.standard_normal((96, 8)))[0]
    A = (Ul * np.geomspace(10, 1, 8)) @ Vr.T + 0.01 * r.standard_normal((128, 96))
    S = L.svdvals_topk(f16(A), k=8, oversample=2, power_iters=1)
    return relerr(S, np.linalg.svd(A, compute_uv=False)[:8]), 3e-2


CASES = [
    ("SPD solve (posv)",        "conjugate_gradient",  "fully on ANE",          check_spd_solve,     (1e1, 1e2)),
    ("general solve (gesv)",    "gmres",               "fully on ANE",          check_general_solve, (1e1, 1e2)),
    ("least squares (gels)",    "lsqr",                "fully on ANE",          check_least_squares, (1e1, 1e2)),
    ("dominant eig (syev,k=1)", "dominant_eig",        "fully on ANE",          check_dominant_eig,  (1e1, 1e2)),
    ("dominant SVD (gesvd,k=1)","dominant_svd",        "fully on ANE",          check_dominant_svd,  (1e1, 1e2)),
    ("QR (geqrf)",              "qr (Gram-Schmidt)",   "fully on ANE (small n)", check_qr,            (1e1, 1e2)),
    ("Cholesky (potrf)",        "cholesky (unpivoted)","fully on ANE (small n)", check_cholesky,      (1e1, 1e2)),
    ("LU (getrf, unpivoted)",   "lu (Doolittle)",      "fully on ANE (small n)", check_lu,            (1e1, 1e2)),
    ("pivoted LU (gesv)",       "lu_pivoted (argmax)", "on ANE, SEGMENTED",     check_pivoted_lu,    (1e1,)),
    ("FULL eig sym (syev)",     "eigh (cyclic Jacobi)","fully on ANE (small n)", check_full_eig,      (1e1, 1e2)),
    ("generalized eig (sygv)",  "chol+trinv+eigh",     "fully on ANE (small n)", check_generalized_eig, (1e1,)),
    ("nonsymmetric eig (geev)", "eigvals (unshifted QR)","fully on ANE (small n)", check_nonsym_eig,    (1e1,)),
    ("FULL SVD (gesvd)",        "svd = eigh(A^TA)",     "fully on ANE (small n)", check_full_svd,      (1e1,)),
    ("top-k SVD (gesvd)",       "svdvals_topk",        "fully on ANE",          check_topk_svd,      (1e1,)),
]

REACHABLE_UNBUILT: list = []   # QR / Cholesky / LU now built (above); nothing left in this tier

# The genuine walls - they need data-dependent CONTROL FLOW or OUTPUT SIZE, which static
# dataflow cannot express even with fixed iteration:
WALLED = [
    ("rank-revealing / adaptive-tolerance", "data-dependent OUTPUT SIZE (how many values > tol) - "
        "cannot emit a runtime-sized result; only all-n + a mask. The one irreducible wall."),
]


def main():
    print("=" * 86)
    print("LAPACK on the ANE - problem families, fitting method, fp16 conditioning envelope")
    print("=" * 86)
    print(f"{'family':>26} | {'ANE method':>18} | {'where':>20} | {'cond':>5} | {'relerr':>9} | st")
    print("-" * 86)
    red = 0
    for fam, method, where, fn, conds in CASES:
        for cond in conds:
            err, tol = fn(cond)
            ok = np.isfinite(err) and err <= tol
            red += 0 if ok else 1
            print(f"{fam:>26} | {method:>18} | {where:>20} | {cond:>5.0e} | {err:>9.2e} | "
                  f"{'ok' if ok else 'RED'}")
    if REACHABLE_UNBUILT:
        print("\nREACHABLE, not yet built:")
        for fam, why in REACHABLE_UNBUILT:
            print(f"  - {fam}: {why}")
    print("\nWALLED - needs data-dependent CONTROL FLOW or OUTPUT SIZE (static dataflow can't):")
    for fam, why in WALLED:
        print(f"  - {fam}: {why}")
    print("\nreading: solvers, least squares, dominant pairs, the FULL dense factorizations")
    print("  (QR / Cholesky / LU) AND the FULL spectral decompositions (symmetric eig + SVD via")
    print("  cyclic Jacobi) ALL run ENTIRELY on the ANE as single unrolled programs. 'Direct")
    print("  factorization is walled' is FALSE: an UNPIVOTED / FIXED-sweep factorization is a")
    print("  static recurrence, so it unrolls (small-n: O(n^3) deep graph; the iterative topo in")
    print("  _compile.py is what lets it build). fp16: solvers clean ~cond1e1 usable ~1e2; FULL")
    print("  SVD is cond^2 (cond(A)<=1e1); dominant eig/SVD need a spectral GAP. The remaining")
    print("  pivoted LU runs via a true argmax pivot (SEGMENTED, one bridge cut/column); the")
    print("  nonsymmetric eig runs as unrolled unshifted QR (one program, complex pairs from 2x2")
    print("  blocks). The ONLY irreducible wall left is rank-revealing - a data-dependent OUTPUT")
    print("  SIZE a static program cannot emit (it can only return all-n values plus a mask).")
    print("\n" + "=" * 86)
    print(f"GATE: {'GREEN' if red == 0 else 'RED'}  "
          f"({sum(len(c[4]) for c in CASES) - red}/{sum(len(c[4]) for c in CASES)} within envelope)")
    return 1 if red else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
