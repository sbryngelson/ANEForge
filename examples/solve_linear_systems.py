"""Solve A x = b on the ANE - conjugate gradient, K iterations as ONE fused program.

The ANE has no in-graph loop, but a FIXED-iteration Krylov solver is static dataflow:
every GEMV and dot product of all K CG iterations unrolls into a single on-engine
program (the matrix folds in as a constant, so the program size is iteration-bound,
not n-bound - the same solve runs unchanged at n=512). GMRES (general systems) and
LSQR (least squares) follow the same pattern; see aneforge.linalg.

Honest envelope (docs/api/math.md): fp16, clean to ~1e-3 at cond 1e2, then the
fp16 iterates stall.

    python3 examples/solve_linear_systems.py
"""
import sys

from _common import head, f16, relerr, spd   # sets env + repo-root path; import before aneforge
import numpy as np
from aneforge.linalg import conjugate_gradient


def main():
    head("SOLVE - SPD system by conjugate gradient, K iters UNROLLED into ONE ANE program")
    print(f"\n{'cond(A)':>8} | {'CG relerr (ANE)':>16} | {'np.linalg.solve':>16}")
    for cond in (1e1, 1e2, 1e3):
        A = spd(48, cond, int(cond) + 11); x = np.random.default_rng(0).standard_normal(48)
        b = A @ x
        xcg = conjugate_gradient(f16(A), f16(b), iters=40)
        xnp = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))
        print(f"{cond:>8.0e} | {relerr(xcg, x):>16.3e} | {relerr(xnp, x):>16.3e}")
    print("  reading: clean to ~1e-3 at cond 1e2, then the fp16 iterates stall. The matrix")
    print("  folds as a constant, so the program is size-independent: this same solve runs")
    print("  unchanged at n=512 (CG/GMRES/LSQR scale; the explicit factorizations in")
    print("  examples/factorize.py are the small-n ones).")


if __name__ == "__main__":
    sys.exit(main())
