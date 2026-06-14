"""Eigenvalues and singular values on the ANE - the full spectral decompositions.

The full symmetric eigendecomposition runs as fixed-sweep cyclic Jacobi, every
rotation a fixed slice+combine - one unrolled program (caps near n=20), or with
``eigh(iterate=True)`` one compiled sweep host-looped to reach n~48. On top of it:
the full SVD (Jacobi on the Gram matrix), the generalized symmetric problem
A x = lambda B x (LAPACK sygv: chol + trinv + eigh, composed on-engine), and the
NONSYMMETRIC eigenvalues (LAPACK geev) as unrolled unshifted QR iteration. Top-k
singular values of a large matrix use a randomized sketch - every step on-engine.

    python3 examples/eigenvalues_svd.py
"""
import sys

from _common import head, f16, relerr, spd, general, sym   # sets env + repo-root path
import numpy as np
from aneforge.linalg import eigh, eigvals, generalized_eigh, svd, svdvals_topk


def main():
    head("SPECTRUM - full symmetric eig + SVD (cyclic Jacobi) + geev + sygv, on the ANE")
    r = np.random.default_rng(7)

    # full symmetric spectrum: one unrolled Jacobi program (n=8)
    As = sym(8, 1e2, 7)
    ev = eigh(f16(As))
    print(f"\n  eigh (all 8 eigenvalues)  relerr vs np.eigh  = {relerr(ev, np.sort(np.linalg.eigvalsh(As))):.2e}")

    # iterate=True: ONE compiled sweep host-looped - same rotations, same result,
    # but the per-sweep graph is O(n^2), so it scales past the unrolled cap (n~48).
    evi = eigh(f16(As), iterate=True)
    print(f"  eigh(iterate=True)        relerr vs unrolled  = {relerr(evi, ev):.2e}   (reaches n~48)")

    # full SVD = Jacobi on the Gram matrix (cond^2: accurate for cond(A) <= 1e1)
    Ag = general(8, 1e1, 5)
    sv = svd(f16(Ag))
    print(f"  svd  (all 8 sing. values) relerr vs np.svd   = {relerr(sv, np.linalg.svd(Ag, compute_uv=False)):.2e}")

    # generalized symmetric problem A x = lambda B x (sygv): chol+trinv+eigh on-engine
    Asym = sym(8, 1e1, 31); B = spd(8, 1e1, 32)
    Li = np.linalg.inv(np.linalg.cholesky(B))
    gref = np.sort(np.linalg.eigvalsh(Li @ Asym @ Li.T))
    print(f"  generalized_eigh (sygv)   relerr vs np ref    = {relerr(generalized_eigh(f16(Asym), f16(B)), gref):.2e}")

    # NONSYMMETRIC eigenvalues (geev): unrolled unshifted QR, complex pairs from 2x2 blocks
    V = np.random.default_rng(10).standard_normal((6, 6))
    An = V @ np.diag(np.geomspace(8, 1, 6)) @ np.linalg.inv(V)
    got = np.sort_complex(eigvals(f16(An), iters=60)); nref = np.sort_complex(np.linalg.eigvals(An))
    print(f"  eigvals (geev, nonsym)    relerr vs np.eig    = {float(np.linalg.norm(got - nref) / np.linalg.norm(nref)):.2e}")

    # top-k singular values of a LARGE matrix: randomized sketch, every step on-engine
    Ul = np.linalg.qr(r.standard_normal((128, 8)))[0]; Vr = np.linalg.qr(r.standard_normal((96, 8)))[0]
    Abig = (Ul * np.geomspace(10, 1, 8)) @ Vr.T + 0.01 * r.standard_normal((128, 96))
    St = svdvals_topk(f16(Abig), k=8)
    print(f"  svdvals_topk (128x96, k=8) relerr vs np.svd   = {relerr(St, np.linalg.svd(Abig, compute_uv=False)[:8]):.2e}")

    print("  reading: the full eig/SVD are fixed-sweep cyclic Jacobi, every rotation a fixed")
    print("  slice+combine (unrolled caps near n=20; iterate=True host-loops one compiled sweep")
    print("  to n~48). geev runs as unrolled unshifted QR; sygv composes chol+trinv+eigh; top-k")
    print("  uses a randomized sketch with the orthonormalization and projected svd on-engine.")


if __name__ == "__main__":
    sys.exit(main())
