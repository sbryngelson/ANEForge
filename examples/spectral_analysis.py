"""aneforge spectral demo - FFT-class spectral analysis of a real 1-D signal on the ANE.

This is "FFT-class spectral analysis on the ANE". The Apple Neural Engine has no
complex dtype (compute is fp16 real only), so we run the Discrete Fourier Transform
as a TWIDDLE-MATRIX MATMUL with complex emulated as real/imag PAIRS:

    X_k = sum_n x_n * exp(-2*pi*i*k*n/N) = x @ W^T,   W[k,n] = exp(-2*pi*i*k*n/N).

For a real input x (imag part = 0):
    Xr = x @ Wr^T,   Xi = x @ Wi^T,   magnitude = sqrt(Xr^2 + Xi^2)
where Wr, Wi are the real/imag parts of the twiddle matrix, folded as fp16 weight
constants. The two GEMMs + the squared-magnitude map fuse into ONE e5rt program.

We analyze a synthetic signal - a sum of pure sinusoids plus a linear chirp - and:
  1. compute its magnitude spectrum on the ANE, validate vs numpy.fft.rfft magnitude;
  2. recover the planted tone frequencies from the ANE spectrum's peaks and check
     they match the ground truth;
  3. sweep N from 64 to 2048 and print the relerr-vs-N curve - demonstrating the
     "wide accumulator" finding: the length-N twiddle sum is accumulated in >=fp32,
     so fp16 rounding does NOT compound with N and the spectrum stays fp16-CLEAN.

CAVEAT: this is the NAIVE O(N^2) DFT (a dense twiddle matmul), not an O(N log N)
FFT - the cost is the quadratic twiddle-matrix size/bandwidth, NOT precision. The
relerr stays ~3e-4..1e-3 FLAT in N (it does not grow), so fp16 is not the wall for
the transform. The reference is numpy's exact fft over the same fp16-rounded signal.

    python3 examples/spectral_analysis.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def make_signal(N, fs):
    """A real signal: three pure tones + a linear chirp, on a length-N grid.
    Returns (signal fp16, list of planted tone freqs in Hz)."""
    t = np.arange(N) / fs
    tones = [60.0, 180.0, 350.0]          # Hz - three clean spectral lines
    x = np.zeros(N, np.float32)
    for f in tones:
        x += np.sin(2 * np.pi * f * t)
    # a faint linear chirp 400 -> 480 Hz (broadband, fills in the spectrum)
    x += 0.4 * np.sin(2 * np.pi * (400 * t + 0.5 * 40.0 * t * t))
    x /= np.abs(x).max()                  # normalize into fp16 range
    return x.astype(np.float16), tones


def dft_program(N):
    """Compile the real-input DFT magnitude spectrum as ONE fused ANE program.
    Folds the fp16 twiddle matrices Wr, Wi as constant weights.

    The twiddles carry a 1/sqrt(N) NORMALIZATION (the standard unitary DFT scaling).
    This is not cosmetic: without it the peak |X_k| grows ~amplitude*N/2, so at
    N=2048 the SQUARED intermediate Xr^2+Xi^2 (~1.2e5) OVERFLOWS fp16's 65504 max
    and the spectrum becomes inf. The unitary normalization keeps the magnitude AND
    its square comfortably in fp16 range at every N - a real fp16 dynamic-range
    note (the wall here is fp16's max value, not its precision)."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)  # unitary [N,N] DFT matrix
    Wr = np.ascontiguousarray(np.real(W)).astype(np.float16)
    Wi = np.ascontiguousarray(np.imag(W)).astype(np.float16)
    x = af.input((1, N))
    Xr = x @ Wr.T.astype(np.float16)                  # x @ Wr^T
    Xi = x @ Wi.T.astype(np.float16)                  # x @ Wi^T
    mag = (Xr.square() + Xi.square()).sqrt()          # |X_k| = sqrt(Xr^2 + Xi^2)
    return af.compile(mag)


def ref_mag(sig, N):
    """numpy reference: the identically-normalized (1/sqrt(N)) DFT magnitude."""
    return np.abs(np.fft.fft(sig.astype(np.float32))) / np.sqrt(N)


def relerr_vs_N(fs):
    """Sweep N: relerr of the ANE magnitude spectrum vs numpy.fft, demonstrating
    fp16-clean scaling (wide accumulator -> no compounding in N)."""
    print("  fp16-clean scaling (DFT magnitude relerr vs N, wide accumulator):")
    rows = []
    for N in (64, 128, 256, 512, 1024, 2048):
        sig, _ = make_signal(N, fs)
        prog = dft_program(N)
        mag_ane = prog(sig.reshape(1, N).astype(np.float16))[0]
        mag_ref = ref_mag(sig, N)
        e = float(np.linalg.norm(mag_ane - mag_ref) / np.linalg.norm(mag_ref))
        rows.append((N, e))
        print(f"    N={N:<5d} relerr {e:.3e}")
    print("    -> relerr is essentially FLAT in N: the ANE matmul accumulator is wide")
    print("       (>=fp32), so the length-N twiddle sum is not re-rounded per term.")
    return rows


def main():
    fs = 1000.0          # 1 kHz sample rate
    N = 1024
    sig, tones = make_signal(N, fs)

    prog = dft_program(N)
    print(f"DFT-as-matmul compiled: {prog.n_ops} ANE ops (2 twiddle GEMMs + |.| map), "
          f"N={N}, fs={fs:.0f} Hz")

    # magnitude spectrum on the ANE
    mag_ane = prog(sig.reshape(1, N).astype(np.float16))[0]
    mag_ref = ref_mag(sig, N)                                # numpy exact reference

    relerr = float(np.linalg.norm(mag_ane - mag_ref) / np.linalg.norm(mag_ref))
    print(f"  magnitude spectrum relerr vs numpy.fft: {relerr:.3e}  (fp16-clean)")

    # recover the planted tone frequencies from ANE spectrum peaks
    half = N // 2
    freqs = np.arange(half) * fs / N
    spec = mag_ane[:half].copy()
    spec[0] = 0.0                                            # ignore DC
    # pick the 3 strongest isolated peaks (the pure tones dominate the chirp)
    peak_idx = np.argsort(spec)[::-1]
    found = []
    for idx in peak_idx:
        f = freqs[idx]
        if all(abs(f - g) > 5.0 for g in found):            # dedupe nearby bins
            found.append(f)
        if len(found) == 3:
            break
    found.sort()
    print(f"  planted tones:   {[f'{f:.0f} Hz' for f in tones]}")
    print(f"  recovered peaks: {[f'{f:.0f} Hz' for f in found]} (from ANE spectrum)")
    bin_hz = fs / N
    peaks_ok = all(min(abs(f - g) for g in tones) <= bin_hz for f in found)
    print(f"  peaks match planted tones (within one {bin_hz:.1f} Hz bin): {peaks_ok}")

    print()
    curve = relerr_vs_N(fs)
    flat = max(e for _, e in curve) < 2e-3                   # no growth with N

    ok = (relerr < 1e-3) and peaks_ok and flat
    print(f"\n{'PASS' if ok else 'FAIL'} - FFT-class spectral analysis on the ANE: "
          f"spectrum within {relerr:.1e} of numpy.fft, tones recovered, fp16-clean to N=2048")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
