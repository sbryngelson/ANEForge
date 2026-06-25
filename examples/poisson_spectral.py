"""Spectral Poisson solver on the ANE: lap(u)=f by the Fourier method, each 2-D FFT one fused ANE program. Run: python3 examples/poisson_spectral.py"""
import sys
import time

from _common import head, relerr   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge.fft as agfft


def spectral_poisson(N=64, L=2.0 * np.pi):
    """Solve lap(u)=f on a periodic [0,L)^2 grid by the Fourier method, the 2-D FFTs
    on the ANE (one fused program each). Manufactured solution -> reports relerr +
    the lap(u)~f spectral-residual check."""
    # grid + manufactured smooth solution (periodic, zero-mean sinusoids)
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u_true = (np.sin(X) * np.cos(2 * Y)
              + 0.5 * np.sin(3 * X) * np.sin(Y)
              + 0.3 * np.cos(2 * X) * np.cos(2 * Y))
    u_true -= u_true.mean()

    # analytic forcing f = lap(u_true)
    f = (-(1 + 4) * np.sin(X) * np.cos(2 * Y)
         - (9 + 1) * 0.5 * np.sin(3 * X) * np.sin(Y)
         - (4 + 4) * 0.3 * np.cos(2 * X) * np.cos(2 * Y))

    # spectral wavenumbers (periodic, fundamental period L)
    k = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi          # angular wavenumbers
    KX, KY = np.meshgrid(k, k, indexing="ij")
    denom = -(KX**2 + KY**2)
    denom[0, 0] = 1.0                                     # avoid /0; DC set below

    # the solve: ANE transform -> host divide -> ANE inverse
    f_re, f_im = agfft.fft2(f)                            # one fused ANE program
    u_re = f_re / denom
    u_im = f_im / denom
    u_re[0, 0] = 0.0; u_im[0, 0] = 0.0                    # gauge: zero-mean (DC=0)
    u, _ = agfft.ifft2(u_re, u_im)                        # one fused ANE program
    u = u - u.mean()                                      # zero-mean gauge

    err = relerr(u, u_true)

    # verify the spectral Laplacian round-trip lap(u) ~ f
    cu_re, cu_im = agfft.fft2(u)
    lap_re = cu_re * denom; lap_im = cu_im * denom
    lap_re[0, 0] = 0.0; lap_im[0, 0] = 0.0
    fc_re, _ = agfft.ifft2(lap_re, lap_im)
    lap_err = relerr(fc_re - fc_re.mean(), f - f.mean())

    return err, lap_err, u, u_true


def main():
    N = 64
    head(f"SPECTRAL POISSON SOLVER  lap(u)=f  on a {N}x{N} periodic grid, FFTs on the ANE")
    print("    method: u = real(iFFT2( FFT2(f) / -(kx^2+ky^2) ))")
    print("    each 2-D FFT = ONE fused ANE program (8 GEMMs; was 128 host-looped dispatches)")
    t0 = time.perf_counter()
    err, lap_err, u, u_true = spectral_poisson(N)
    dt = time.perf_counter() - t0
    print(f"    recovered u vs manufactured u_true (zero-mean):  relerr = {err:.3e}")
    print(f"    spectral Laplacian round-trip   lap(u) ~ f:      relerr = {lap_err:.3e}")
    print(f"    solve + verification wall time (4 transforms + compiles): {dt*1e3:.0f} ms")
    # lap(u)~f chains two extra transforms on u, so its fp16 floor is looser (<1e-1).
    ok = err < 5e-2 and lap_err < 1e-1
    print(f"    -> {'PASS' if ok else 'FAIL'}  (fp16 transform/divide floor; the method is "
          f"exact in exact arithmetic)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
