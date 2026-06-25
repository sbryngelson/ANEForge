"""Direct factorizations on the ANE: QR / Cholesky / LU as unrolled recurrences, plus pivoted LU. Run: python3 examples/factorize.py"""
import sys

from _common import head, f16, relerr, spd, general   # sets env + repo-root path
import numpy as np
from aneforge.linalg import qr, cholesky, lu, lu_pivoted


def main():
    head("FACTORIZE - QR / Cholesky / LU as fixed recurrences, fully on the ANE (n=8)")
    A = general(8, 1e1, 3); Asp = spd(8, 1e1, 4)
    Q, R = qr(f16(A)); L_ = cholesky(f16(Asp)); Ll, Uu = lu(f16(A))
    print(f"\n  QR        |A - QR|       = {relerr(Q @ R, A):.2e}   (Q^T Q - I: {relerr(Q.T @ Q, np.eye(8)):.2e})")
    print(f"  Cholesky  |A - L L^T|     = {relerr(L_ @ L_.T, Asp):.2e}")
    print(f"  LU        |A - L U|      = {relerr(Ll @ Uu, A):.2e}")

    Ap = general(8, 1e1, 41); Ap[0, 0] = 1e-3      # tiny leading pivot: unpivoted would be unstable
    P, Lp, Up = lu_pivoted(f16(Ap))
    print(f"  pivoted LU |P A - L U|   = {relerr(P @ Ap, Lp @ Up):.2e}   (on-engine argmax pivot, segmented)")

    print("  reading: a direct factorization is a fixed recurrence (no pivot = no data-dependent")
    print("  branch), so it unrolls into one program. Small-n: the O(n^3) graph caps QR/Cholesky/")
    print("  LU near n=32. Pivoting needs a data-dependent permutation: the argmax is a bridge op,")
    print("  so pivoted LU runs SEGMENTED (one cut per column) instead of fully fused.")


if __name__ == "__main__":
    sys.exit(main())
