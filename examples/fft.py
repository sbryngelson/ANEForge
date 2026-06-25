"""FFT on the ANE: staged Cooley-Tukey as dense-DFT matmuls (sub-quadratic MACs). Run: python3 examples/fft.py"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
from aneforge.fft import fft as ane_fft, ifft as ane_ifft, _factor, _leaf_cost


def main():
    _common.head("FFT - staged Cooley-Tukey as dense-DFT matmuls on the ANE (sub-quadratic MACs)")
    print(f"\n{'N':>6} | {'factors':>12} | {'roundtrip err':>14} | {'spectrum err':>14} | {'MAC saving':>10}")
    rng = np.random.default_rng(3)
    for N in (256, 1024, 2048):
        x_re = rng.standard_normal(N).astype(np.float32); x_im = np.zeros(N, np.float32)
        X_re, X_im = ane_fft(x_re, x_im, N); xr_re, xr_im = ane_ifft(X_re, X_im, N)
        Xref = np.fft.fft(x_re.astype(np.float64))
        spec = np.linalg.norm((X_re + 1j * X_im) - Xref) / (np.linalg.norm(Xref) + 1e-30)
        rnd = np.linalg.norm((xr_re + 1j * xr_im) - x_re) / (np.linalg.norm(x_re) + 1e-30)
        fac = "x".join(str(g) for g in _factor(N))
        print(f"{N:>6} | {fac:>12} | {rnd:>14.3e} | {spec:>14.3e} | {(N*N)/_leaf_cost(N):>9.1f}x")
    print("\n  reading: N*sum(factors) complex MACs vs the dense DFT's N^2; matches np.fft to fp16.")


if __name__ == "__main__":
    sys.exit(main())
