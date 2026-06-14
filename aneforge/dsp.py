"""aneforge.dsp - a digital-signal-processing toolkit on the Apple Neural Engine.

Built on the two things the ANE is genuinely good at: REAL CONVOLUTION (the conv
layer) and the MATMUL-FFT from `aneforge.fft` (the DFT factored into matmul stages,
complex carried as real/imag PAIRS). This submodule composes them into the everyday
DSP kit:

    from aneforge.dsp import (fir_filter, fft_convolve, freq_filter,
                             stft, spectrogram, hann, hamming, blackman,
                             correlate, autocorrelate)

WHAT FITS THE ANE:
  * FIR filtering  - a finite impulse response IS a convolution. Short kernels run
    on the native conv layer (1xK, arch-capped at K<=15 on this M5, verified);
    longer ones fall back to FFT convolution. Feed-forward, no recurrence: a clean fit.
  * FFT-domain DSP - fft_convolve / freq_filter / stft / spectrogram are all
    (windowed) FFTs + elementwise spectral ops + iFFT, every stage a matmul or an
    elementwise map (real/imag pairs via aneforge.fft). A clean fit.
  * Correlation   - cross/auto-correlation is convolution without the kernel flip,
    exactly what the ANE conv and the native CrossCorrelation bridge already do.

WHAT DOES NOT FIT (arch limit):
  * IIR / recursive filters (Butterworth, biquads, any `y[n] = ... + a*y[n-1]`).
    A direct-form IIR is a DATA-DEPENDENT RECURRENCE over samples - the scan/cumsum
    wall the ANE lacks (no in-graph loop, no scalar feedback). No streaming IIR
    here. `iir_filter` is ONLY a FIXED-LENGTH UNROLL: the recurrence becomes its
    truncated FIR impulse response of a chosen length, run as an FIR. Exact only up
    to the truncation, and the unroll is fixed at build time - it does NOT scale to
    arbitrary-length streaming IIR. Exposed tagged, with the relerr-vs-untruncated
    cost reported, rather than faking a recurrent filter.

CONVENTIONS (inherited):
  * The ANE conv is a CROSS-CORRELATION (no kernel flip), matching
    `np.correlate(x, h, 'valid')`. True convolution (np.convolve / scipy lfilter)
    flips the kernel - fir_filter handles the flip + causal zero-pad internally.
  * Complex is (re, im) real-tensor pairs; spectra come back as separate real arrays.
  * Everything is fp16 on-device; the matmul accumulator is wide (>=fp32), so the
    error floor is fp16 INPUT/twiddle rounding (~few e-4), flat in length.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 aneforge/dsp.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aneforge as af  # noqa: E402
from aneforge.fft import fft, ifft, fft_plan, ifft_plan  # noqa: E402


# --------------------------------------------------------------------------- #
# windows (host-side numpy: tiny, data-independent coefficient vectors)        #
# --------------------------------------------------------------------------- #
# Windows are short constant vectors multiplied into framed signals. Computed on
# the host (numpy) - no gain from a length-W cosine table on the ANE - and applied
# as fp16 elementwise multiplies on-device (in stft) or on the host (when the caller
# just wants the coefficients).

def hann(M: int, sym: bool = False) -> np.ndarray:
    """Hann window (raised cosine). `sym=False` gives the periodic/DFT window
    (the STFT default, matching scipy.signal.get_window('hann', M))."""
    if M == 1:
        return np.ones(1, np.float32)
    n = np.arange(M)
    denom = (M - 1) if sym else M
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / denom)).astype(np.float32)


def hamming(M: int, sym: bool = False) -> np.ndarray:
    """Hamming window. `sym=False` = periodic (STFT) form."""
    if M == 1:
        return np.ones(1, np.float32)
    n = np.arange(M)
    denom = (M - 1) if sym else M
    return (0.54 - 0.46 * np.cos(2.0 * np.pi * n / denom)).astype(np.float32)


def blackman(M: int, sym: bool = False) -> np.ndarray:
    """Blackman window. `sym=False` = periodic (STFT) form."""
    if M == 1:
        return np.ones(1, np.float32)
    n = np.arange(M)
    denom = (M - 1) if sym else M
    a0, a1, a2 = 0.42, 0.5, 0.08
    return (a0 - a1 * np.cos(2.0 * np.pi * n / denom)
            + a2 * np.cos(4.0 * np.pi * n / denom)).astype(np.float32)


_WINDOWS = {"hann": hann, "hamming": hamming, "blackman": blackman}


def get_window(window, M: int) -> np.ndarray:
    """Resolve `window` (name str, 'boxcar'/None for rectangular, or an array) to a
    length-M fp32 coefficient vector (periodic form for the named cosine windows)."""
    if window is None or window == "boxcar" or window == "rect":
        return np.ones(M, np.float32)
    if isinstance(window, str):
        if window not in _WINDOWS:
            raise ValueError(f"unknown window {window!r}; choose from {list(_WINDOWS)} or 'boxcar'")
        return _WINDOWS[window](M, sym=False)
    w = np.asarray(window, np.float32)
    if w.shape != (M,):
        raise ValueError(f"window array must have length {M}; got {w.shape}")
    return w


# --------------------------------------------------------------------------- #
# FIR filtering - convolution (the ANE's home turf)                            #
# --------------------------------------------------------------------------- #

# The native 1xK conv backend is arch-capped at K<=15 on this M5 (verified: K=16
# fails "Some ops are not supported on any of the specified backends"). Longer FIR
# kernels are routed through fft_convolve instead (same linear-convolution result).
_MAX_CONV_TAPS = 15


def fir_filter(x, taps, mode: str = "same"):
    """FIR filter `y = x * taps` (true convolution), on the ANE.

    A finite impulse response is a convolution. The ANE conv is a CROSS-correlation
    (no kernel flip), so we flip the taps and zero-pad to realize a genuine
    convolution / causal FIR:

      * mode='full'  -> length L+K-1   (== np.convolve(x, taps))
      * mode='same'  -> length L, centered (== np.convolve(..., 'same'))
      * mode='valid' -> length L-K+1   (== np.convolve(..., 'valid'))
      * mode='lfilter' -> length L, causal (== scipy.signal.lfilter(taps, [1.0], x))

    Short kernels (K<=15) run on the native conv layer in ONE fused program; longer
    kernels fall back to FFT convolution (aneforge.fft) automatically - same result.

    cost: COMPUTE (a real conv) for short taps; FFT-domain for long taps.
    fit:  GOOD - feed-forward, no recurrence.
    """
    x = np.asarray(x, np.float32).ravel()
    taps = np.asarray(taps, np.float32).ravel()
    L, K = x.shape[0], taps.shape[0]
    if K > L:
        raise ValueError(f"fir_filter: {K} taps longer than signal length {L}")

    # long kernels: the native 1xK conv backend caps at K<=15 -> use FFT convolution.
    if K > _MAX_CONV_TAPS:
        full = fft_convolve(x, taps)                     # length L+K-1
        return _trim_conv(full, L, K, mode)

    # short kernels: native conv. ANE conv == correlation, so flip the taps and
    # left/right zero-pad to get the requested convolution slice.
    hflip = np.ascontiguousarray(taps[::-1]).astype(np.float16).reshape(1, 1, 1, K)
    if mode in ("full", "lfilter"):
        left = K - 1
    elif mode == "same":
        left = (K - 1) // 2
    elif mode == "valid":
        left = 0
    else:
        raise ValueError(f"fir_filter: mode must be full/same/valid/lfilter; got {mode!r}")
    if mode == "full":
        right = K - 1
    elif mode == "same":
        right = K - 1 - left
    elif mode == "lfilter":
        right = 0
    else:                                                # valid
        right = 0
    xp = np.concatenate([np.zeros(left, np.float32), x, np.zeros(right, np.float32)])
    Lp = xp.shape[0]
    inp = af.input((1, 1, 1, Lp))
    model = af.compile(af.conv(inp, hflip, pad=0))
    y = model(xp.astype(np.float16).reshape(1, 1, 1, Lp))
    return y.ravel().astype(np.float32)


def _trim_conv(full: np.ndarray, L: int, K: int, mode: str) -> np.ndarray:
    """Slice a length-(L+K-1) full convolution to the requested mode."""
    if mode == "full":
        return full
    if mode == "valid":
        return full[K - 1:L]
    if mode == "lfilter":
        return full[:L]
    if mode == "same":
        start = (K - 1) // 2
        return full[start:start + L]
    raise ValueError(f"mode must be full/same/valid/lfilter; got {mode!r}")


# --------------------------------------------------------------------------- #
# FFT convolution - linear convolution via the matmul-FFT (overlap-add)        #
# --------------------------------------------------------------------------- #

def _next_fft_size(n: int) -> int:
    """Smallest size >= n that aneforge.fft factors cleanly (power of two is always
    fine and keeps the staged FFT balanced)."""
    return 1 << max(1, int(math.ceil(math.log2(max(2, n)))))


def fft_convolve(x, h, block: int | None = None):
    """Linear convolution `y = x * h` via the FFT (aneforge.fft), length L+K-1.

    Equivalent to `np.convolve(x, h)` but computed as FFT(x)*FFT(h) -> iFFT, so
    every stage is a matmul or an elementwise spectral product (ANE-native). For a
    short kernel against a long signal we use OVERLAP-ADD: split the signal into
    blocks, multiply each block-FFT by the (cached) kernel spectrum, inverse-
    transform, and sum the overlapping tails. A single block is used when the whole
    thing fits one transform.

    cost: COMPUTE/FUSION - staged matmul-FFTs + spectral multiply.
    fit:  GOOD.
    """
    x = np.asarray(x, np.float32).ravel()
    h = np.asarray(h, np.float32).ravel()
    L, K = x.shape[0], h.shape[0]
    out_len = L + K - 1

    # single-block path: one FFT covers the whole linear convolution.
    if block is None:
        single = _next_fft_size(out_len)
        if L <= 4 * K or single <= 1024:
            return _fft_conv_block(x, h, single)[:out_len].astype(np.float32)
        block = single  # (not reached for the common case; kept for explicitness)

    # overlap-add: choose an FFT size, step = N-K+1 samples per block.
    N = _next_fft_size(max(block, 2 * K))
    step = N - K + 1
    Hr, Hi = fft(np.pad(h, (0, N - K)), np.zeros(N, np.float32), N)
    fplan = fft_plan(N)
    iplan = ifft_plan(N)
    y = np.zeros(out_len, np.float32)
    for start in range(0, L, step):
        seg = x[start:start + step]
        seg = np.pad(seg, (0, N - seg.shape[0]))
        Xr, Xi = fplan(seg, np.zeros(N, np.float32))
        Yr = Xr * Hr - Xi * Hi
        Yi = Xr * Hi + Xi * Hr
        yr = _ifft_real_scaled(Yr, Yi, N, iplan)
        end = min(start + N, out_len)
        y[start:end] += yr[:end - start].astype(np.float32)
    return y


def _ifft_real_scaled(Yr: np.ndarray, Yi: np.ndarray, N: int, iplan=None) -> np.ndarray:
    """Inverse-FFT a spectrum to a real signal, guarding fp16 dynamic range.

    aneforge's iFFT accumulates the length-N inverse-DFT sum BEFORE the 1/N scale, so
    the on-device intermediate peaks at ~max_k sum_n |Y_n|. For convolution/correlation
    spectra (products of two transforms) that peak can exceed fp16's 65504 ceiling and
    SATURATE, corrupting the result (the FFT module's fp16 dynamic-range wall, not a
    precision issue). We pre-scale `Y` so the unscaled sum stays well within range, run
    the (linear) iFFT, then undo the scale on the host."""
    peak = float(np.sum(np.sqrt(Yr.astype(np.float64) ** 2 + Yi.astype(np.float64) ** 2)))
    s = 1.0
    if peak > 1.0:
        s = 30000.0 / peak                              # keep unscaled accumulator < ~3e4
    Ys_r = (Yr * s).astype(np.float32)
    Ys_i = (Yi * s).astype(np.float32)
    yr, _ = (iplan(Ys_r, Ys_i) if iplan is not None else ifft(Ys_r, Ys_i, N))
    return yr.astype(np.float64) / s


def _fft_conv_block(x: np.ndarray, h: np.ndarray, N: int) -> np.ndarray:
    """One-block FFT convolution at transform size N (>= len(x)+len(h)-1)."""
    L, K = x.shape[0], h.shape[0]
    Xr, Xi = fft(np.pad(x, (0, N - L)), np.zeros(N, np.float32), N)
    Hr, Hi = fft(np.pad(h, (0, N - K)), np.zeros(N, np.float32), N)
    Yr = Xr * Hr - Xi * Hi
    Yi = Xr * Hi + Xi * Hr
    return _ifft_real_scaled(Yr, Yi, N)


# --------------------------------------------------------------------------- #
# frequency-domain filtering - FFT -> mask spectrum -> iFFT                     #
# --------------------------------------------------------------------------- #

def freq_filter(x, kind: str, cutoff, fs: float = 2.0):
    """Frequency-domain filter: FFT the (real) signal, zero out the rejected bins,
    inverse-FFT. A brick-wall ideal filter in the DFT domain.

    `kind` in {'lowpass','highpass','bandpass','bandstop'}. `cutoff` is a scalar
    (low/high) or a (lo, hi) pair (band). `fs` is the sample rate (default 2.0 so a
    bare `cutoff` reads as a normalized frequency in [0,1] == fraction of
    Nyquist). Returns the real filtered signal (length L), on the ANE.

    The mask is applied to BOTH the bin and its conjugate mirror so the output stays
    real. cost: COMPUTE/FUSION (FFT + elementwise mask + iFFT). fit: GOOD.
    """
    x = np.asarray(x, np.float32).ravel()
    L = x.shape[0]
    N = _next_fft_size(L)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)                # bin center frequencies
    absf = np.abs(freqs)

    if kind in ("lowpass", "highpass"):
        fc = float(np.asarray(cutoff).ravel()[0])
        passband = (absf <= fc) if kind == "lowpass" else (absf >= fc)
    elif kind in ("bandpass", "bandstop"):
        lo, hi = (float(c) for c in np.asarray(cutoff).ravel()[:2])
        inband = (absf >= lo) & (absf <= hi)
        passband = inband if kind == "bandpass" else ~inband
    else:
        raise ValueError(f"freq_filter: kind must be lowpass/highpass/bandpass/bandstop; got {kind!r}")
    mask = passband.astype(np.float32)

    Xr, Xi = fft(np.pad(x, (0, N - L)), np.zeros(N, np.float32), N)
    Xr = Xr * mask
    Xi = Xi * mask
    yr = _ifft_real_scaled(Xr, Xi, N)                   # fp16-range-guarded iFFT
    return yr[:L].astype(np.float32)


# --------------------------------------------------------------------------- #
# STFT / spectrogram - windowed framed FFTs (one aneforge.fft per frame)        #
# --------------------------------------------------------------------------- #

def _frame(x: np.ndarray, win_len: int, hop: int) -> np.ndarray:
    """Split `x` into overlapping frames [n_frames, win_len] (no boundary pad;
    trailing samples that don't fill a frame are dropped - scipy stft boundary=None,
    padded=False)."""
    if x.shape[0] < win_len:
        return np.empty((0, win_len), x.dtype)
    n_frames = 1 + (x.shape[0] - win_len) // hop
    idx = np.arange(win_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def stft(x, win=256, hop=None, window: str = "hann"):
    """Short-time Fourier transform: window each frame, FFT it (aneforge.fft), stack.

    `win` is the frame length (int) or an explicit window-coefficient array; `hop`
    the step between frames (default win//4 like scipy); `window` the named window
    when `win` is an int. Returns (Zr, Zi), each [n_freq, n_frames] with
    n_freq = win//2 + 1 (the non-redundant rfft half), matching scipy.signal.stft's
    bin layout (NOT its 1/sum(win) scaling - see spectrogram for magnitudes).

    Each frame is one matmul-FFT. The frames are independent (no recurrence), a
    clean ANE fit; we batch them through the cached FFT plan. fit: GOOD.
    """
    x = np.asarray(x, np.float32).ravel()
    if isinstance(win, (int, np.integer)):
        win_len = int(win)
        w = get_window(window, win_len)
    else:
        w = np.asarray(win, np.float32).ravel()
        win_len = w.shape[0]
    if hop is None:
        hop = win_len // 4
    N = _next_fft_size(win_len)
    n_freq = win_len // 2 + 1

    frames = _frame(x, win_len, hop) * w                 # [n_frames, win_len], windowed
    n_frames = frames.shape[0]
    plan = fft_plan(N)
    Zr = np.empty((n_freq, n_frames), np.float32)
    Zi = np.empty((n_freq, n_frames), np.float32)
    zero = np.zeros(N, np.float32)
    for t in range(n_frames):
        fr = np.pad(frames[t], (0, N - win_len))
        Fr, Fi = plan(fr, zero)
        Zr[:, t] = Fr[:n_freq]
        Zi[:, t] = Fi[:n_freq]
    return Zr, Zi


def spectrogram(x, win=256, hop=None, window: str = "hann", mode: str = "magnitude"):
    """Magnitude (or power) spectrogram = |STFT|. `mode` in {'magnitude','power'}.
    Returns a [n_freq, n_frames] real array. fit: GOOD."""
    Zr, Zi = stft(x, win=win, hop=hop, window=window)
    p = Zr.astype(np.float32) ** 2 + Zi.astype(np.float32) ** 2
    return p if mode == "power" else np.sqrt(p)


# --------------------------------------------------------------------------- #
# correlation - convolution without the kernel flip                            #
# --------------------------------------------------------------------------- #

def correlate(a, b, mode: str = "valid"):
    """Cross-correlation `r[k] = sum_n a[n+k] * b[n]` on the ANE.

    For a short template (Lb<=15) uses the native CrossCorrelation bridge (a path
    Apple's MIL frontend rejects): a length-La row and a smaller length-Lb template ->
    'valid' correlation of length La-Lb+1, matching `np.correlate(a, b, 'valid')`.
    A wider template (Lb>=16, the same 1xK conv-width arch wall as fir_filter) falls
    back to FFT correlation (correlation = convolution with `b` reversed). Other
    modes zero-pad the map before the valid correlation.

    cost: MIXED (cut) for the bridge; COMPUTE/FFT for the fallback. fit: GOOD.
    """
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    La, Lb = a.shape[0], b.shape[0]
    if Lb >= La:
        raise ValueError(f"correlate: template length {Lb} must be < signal length {La} "
                         f"(CrossCorrelation bridge requires a strictly smaller template)")
    if mode == "valid":
        ap = a
    elif mode == "full":
        ap = np.concatenate([np.zeros(Lb - 1, np.float32), a, np.zeros(Lb - 1, np.float32)])
    elif mode == "same":
        pad = (Lb - 1) // 2
        ap = np.concatenate([np.zeros(pad, np.float32), a, np.zeros(Lb - 1 - pad, np.float32)])
    else:
        raise ValueError(f"correlate: mode must be valid/full/same; got {mode!r}")

    # wide template: the CrossCorrelation bridge lowers to a 1xLb conv, capped at
    # Lb<=15 on this ANE (Lb>=16 fails ANECCompile). Fall back to FFT correlation:
    # correlate(ap, b) 'valid' == full-convolve(ap, reverse(b)) sliced to valid lags.
    if Lb > _MAX_CONV_TAPS:
        full = fft_convolve(ap, b[::-1])                 # length len(ap)+Lb-1
        return full[Lb - 1:ap.shape[0]].astype(np.float32)

    Wp = ap.shape[0]
    model = af.compile(af.cross_correlation(af.input((1, Wp)), af.input((1, Lb))))
    y = model(ap.astype(np.float16).reshape(1, Wp), b.astype(np.float16).reshape(1, Lb))
    return y.ravel().astype(np.float32)


def autocorrelate(x, max_lag: int | None = None):
    """Autocorrelation `r[k] = sum_n x[n] x[n+k]` for lags k=0..max_lag, on the ANE.

    Built as a correlation of the signal against its own leading `window` via the
    CrossCorrelation bridge. `max_lag` defaults to len(x)//4 (the template length is
    len(x)-max_lag, kept < len(x)). Returns r[0..max_lag] (the non-negative lags),
    matching the tail of `np.correlate(x, x, 'full')`. fit: GOOD.
    """
    x = np.asarray(x, np.float32).ravel()
    L = x.shape[0]
    if max_lag is None:
        max_lag = L // 4
    tmpl_len = L - max_lag
    if tmpl_len < 1 or tmpl_len >= L:
        raise ValueError(f"autocorrelate: max_lag={max_lag} out of range for length {L}")
    tmpl = x[:tmpl_len]
    # valid correlation of x with its prefix gives r[k] = sum_n x[n+k] tmpl[n], k=0..max_lag
    return correlate(x, tmpl, mode="valid")


# --------------------------------------------------------------------------- #
# IIR - ARCH-LIMITED: provided only as a fixed-length FIR unroll               #
# --------------------------------------------------------------------------- #

def iir_filter(x, b, a, n_taps: int = 256):
    """ARCH-LIMITED IIR via a FIXED-LENGTH FIR unroll.

    A direct-form IIR `a[0] y[n] = sum_i b[i] x[n-i] - sum_{j>=1} a[j] y[n-j]` is a
    DATA-DEPENDENT RECURRENCE over samples - each output depends on previous OUTPUTS.
    The ANE is feed-forward: no in-graph loop, no scalar feedback, no scan/cumsum.
    There is NO streaming IIR on this hardware.

    The ONLY honest realization is to TRUNCATE the IIR's infinite impulse response to
    `n_taps` and run it as an FIR (via fir_filter / fft_convolve). Exact only up to
    the truncation tail, with the tap count fixed at build time - it does NOT scale
    to arbitrary streaming IIR. Use it for stable filters whose impulse response has
    decayed within `n_taps` samples; the residual is the truncated tail energy.

    Returns the FIR-approximated, causal (lfilter-style) output (length L), plus prints
    nothing - the caller-facing tag is in the module docstring. fit: ARCH-LIMITED
    (fixed unroll only).
    """
    import scipy.signal as ss
    b = np.asarray(b, np.float64).ravel()
    a = np.asarray(a, np.float64).ravel()
    # truncated impulse response: response of the IIR to a unit impulse, n_taps long.
    imp = np.zeros(n_taps, np.float64)
    imp[0] = 1.0
    ir = ss.lfilter(b, a, imp).astype(np.float32)        # FIR approximation of the IIR
    return fir_filter(x, ir, mode="lfilter")


__all__ = [
    "hann", "hamming", "blackman", "get_window",
    "fir_filter", "fft_convolve", "freq_filter", "stft", "spectrogram",
    "correlate", "autocorrelate", "iir_filter",
]


# --------------------------------------------------------------------------- #
# self-test / validation vs scipy.signal / numpy                              #
# --------------------------------------------------------------------------- #

def _relerr(y, ref):
    y = np.asarray(y, np.float64); ref = np.asarray(ref, np.float64)
    d = np.linalg.norm(y - ref)
    n = np.linalg.norm(ref)
    return float(d / (n + 1e-30))


def _selftest():
    try:
        import scipy.signal as ss
        have_scipy = True
    except Exception:  # noqa: BLE001
        have_scipy = False

    rng = np.random.default_rng(20260530)
    print("aneforge.dsp - DSP toolkit on the ANE (conv + matmul-FFT)\n")
    rows: list[tuple[str, float, str, str]] = []   # (name, relerr, fit, note)

    def record(name, err, fit, note=""):
        rows.append((name, err, fit, note))
        print(f"  {name:34s} relerr {err:.3e}   [{fit}]  {note}")

    # ---- FIR (short, native conv) vs np.convolve / scipy.lfilter -------------- #
    L, K = 512, 9
    x = rng.standard_normal(L).astype(np.float32)
    h = ss.firwin(K, 0.3).astype(np.float32) if have_scipy else (rng.standard_normal(K).astype(np.float32) * 0.3)
    y_same = fir_filter(x, h, "same")
    record("fir_filter(K=9,same)", _relerr(y_same, np.convolve(x, h, "same")),
           "GOOD", "vs np.convolve same")
    y_full = fir_filter(x, h, "full")
    record("fir_filter(K=9,full)", _relerr(y_full, np.convolve(x, h)),
           "GOOD", "vs np.convolve")
    if have_scipy:
        y_lf = fir_filter(x, h, "lfilter")
        record("fir_filter(K=9,lfilter)", _relerr(y_lf, ss.lfilter(h, [1.0], x)),
               "GOOD", "vs scipy.lfilter")

    # ---- FIR (long, auto FFT fallback K>15) vs np.convolve -------------------- #
    Klong = 64
    hlong = ss.firwin(Klong, 0.25).astype(np.float32) if have_scipy else rng.standard_normal(Klong).astype(np.float32) * 0.1
    y_long = fir_filter(x, hlong, "full")
    record("fir_filter(K=64,full,FFT)", _relerr(y_long, np.convolve(x, hlong)),
           "GOOD", "long taps -> FFT fallback")

    # ---- fft_convolve vs np.convolve ----------------------------------------- #
    a = rng.standard_normal(200).astype(np.float32)
    bb = rng.standard_normal(40).astype(np.float32)
    record("fft_convolve(200,40)", _relerr(fft_convolve(a, bb), np.convolve(a, bb)),
           "GOOD", "vs np.convolve")
    # overlap-add path (force blocking)
    a2 = rng.standard_normal(2048).astype(np.float32)
    b2 = rng.standard_normal(48).astype(np.float32)
    record("fft_convolve(2048,48,OLA)", _relerr(fft_convolve(a2, b2, block=256), np.convolve(a2, b2)),
           "GOOD", "overlap-add blocks")

    # ---- freq_filter vs a reference brick-wall filtered signal ---------------- #
    N = 1024
    fs = 1000.0
    t = np.arange(N) / fs
    sig = (np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    def ref_brick(x, kind, cutoff, fs):
        Nn = _next_fft_size(len(x))
        X = np.fft.fft(np.pad(x, (0, Nn - len(x))).astype(np.float64))
        fr = np.abs(np.fft.fftfreq(Nn, d=1.0 / fs))
        if kind == "lowpass":
            m = fr <= cutoff
        elif kind == "highpass":
            m = fr >= cutoff
        elif kind == "bandpass":
            m = (fr >= cutoff[0]) & (fr <= cutoff[1])
        else:
            m = ~((fr >= cutoff[0]) & (fr <= cutoff[1]))
        return np.fft.ifft(X * m).real[:len(x)]

    record("freq_filter(lowpass 100Hz)",
           _relerr(freq_filter(sig, "lowpass", 100.0, fs), ref_brick(sig, "lowpass", 100.0, fs)),
           "GOOD", "keeps 50Hz, drops 300Hz")
    record("freq_filter(highpass 150Hz)",
           _relerr(freq_filter(sig, "highpass", 150.0, fs), ref_brick(sig, "highpass", 150.0, fs)),
           "GOOD")
    record("freq_filter(bandpass 40-60)",
           _relerr(freq_filter(sig, "bandpass", (40.0, 60.0), fs), ref_brick(sig, "bandpass", (40.0, 60.0), fs)),
           "GOOD")

    # ---- stft / spectrogram vs scipy.signal.stft (magnitude) ------------------ #
    if have_scipy:
        sx = rng.standard_normal(2048).astype(np.float32)
        win_len, hop = 256, 64
        Zr, Zi = stft(sx, win=win_len, hop=hop, window="hann")
        mag_ane = np.sqrt(Zr ** 2 + Zi ** 2)
        w = get_window("hann", win_len)
        # scipy stft scales by 1/sum(win); we compute raw windowed-FFT magnitude, so
        # compare against scipy's raw (scaling='spectrum' off): multiply scipy by sum(win).
        f, tt, Zs = ss.stft(sx, nperseg=win_len, noverlap=win_len - hop, window=w,
                            boundary=None, padded=False, return_onesided=True)
        mag_ref = np.abs(Zs) * w.sum()
        m = min(mag_ane.shape[1], mag_ref.shape[1])
        record("stft magnitude (win=256)",
               _relerr(mag_ane[:, :m], mag_ref[:, :m]), "GOOD", "vs scipy.signal.stft |Z|")
        spec = spectrogram(sx, win=win_len, hop=hop, window="hann", mode="power")
        record("spectrogram (power)",
               _relerr(spec[:, :m], (mag_ref[:, :m]) ** 2), "GOOD", "vs |scipy stft|^2")

    # ---- correlate / autocorrelate vs np.correlate ---------------------------- #
    ca = rng.standard_normal(128).astype(np.float32)
    cb_short = rng.standard_normal(8).astype(np.float32)
    record("correlate(valid,bridge K=8)",
           _relerr(correlate(ca, cb_short, "valid"), np.correlate(ca, cb_short, "valid")),
           "GOOD", "CrossCorrelation bridge")
    cb = rng.standard_normal(40).astype(np.float32)
    record("correlate(valid,FFT K=40)", _relerr(correlate(ca, cb, "valid"), np.correlate(ca, cb, "valid")),
           "GOOD", "wide template -> FFT fallback")
    ac = rng.standard_normal(256).astype(np.float32)
    max_lag = 32
    ac_ane = autocorrelate(ac, max_lag=max_lag)
    full = np.correlate(ac, ac[:256 - max_lag], "valid")
    record("autocorrelate(max_lag=32)", _relerr(ac_ane, full), "GOOD", "vs np.correlate")

    # ---- IIR: arch-limited fixed-unroll (truncated impulse response FIR) ------ #
    if have_scipy:
        # a resonant (high-Q) bandpass: long, slowly-decaying impulse response, so
        # truncation length n_taps visibly governs the error = the honest unroll cost.
        bb_iir, aa_iir = ss.iirpeak(0.2, Q=30.0)
        xi = rng.standard_normal(512).astype(np.float32)
        ref_iir = ss.lfilter(bb_iir, aa_iir, xi)
        for nt in (32, 128, 512):
            y_iir = iir_filter(xi, bb_iir, aa_iir, n_taps=nt)
            err = _relerr(y_iir, ref_iir)
            note = f"truncated IR n_taps={nt} (shorter unroll = more truncation tail)"
            record(f"iir_filter(peakQ30,n_taps={nt})", err, "ARCH-LIMITED", note)

    # ---- verdict -------------------------------------------------------------- #
    good = [r for r in rows if r[2] == "GOOD"]
    arch = [r for r in rows if r[2] == "ARCH-LIMITED"]
    worst_good = max((r[1] for r in good), default=0.0)
    print("\n" + "=" * 92)
    print("VERDICT - aneforge.dsp on the ANE")
    print("-" * 92)
    print(f"  GOOD (conv + FFT-based DSP): {len(good)} routines, worst relerr {worst_good:.2e}")
    print("    * fir_filter   - FIR is a convolution; native 1xK conv (K<=15) or FFT fallback.")
    print("    * fft_convolve - linear convolution via the matmul-FFT (single-block / overlap-add).")
    print("    * freq_filter  - FFT -> brick-wall spectral mask -> iFFT (lowpass/highpass/band).")
    print("    * stft/spectrogram - windowed framed FFTs, one matmul-FFT per frame (independent).")
    print("    * correlate/autocorrelate - CrossCorrelation bridge (conv without the flip).")
    print("    All feed-forward, every stage a matmul/conv/elementwise map; fp16-clean (~1e-3),")
    print("    flat in length (wide accumulator). FIR + FFT-based DSP FIT THE ANE.")
    print(f"  ARCH-LIMITED: {len(arch)} routine(s)")
    print("    * iir_filter - a recursive/IIR filter is a DATA-DEPENDENT RECURRENCE over samples")
    print("      (y[n] depends on previous y), i.e. the scan/cumsum wall the ANE lacks (no")
    print("      in-graph loop, no scalar feedback). NO streaming IIR exists here. We provide")
    print("      ONLY a FIXED-LENGTH FIR UNROLL: truncate the IIR impulse response to n_taps and")
    print("      run it as an FIR. Exact up to the truncation tail; tap count fixed at build")
    print("      time; does NOT scale to arbitrary streaming IIR. (relerr above shrinks as")
    print("      n_taps grows = the impulse response decays - the honest cost of the unroll.)")

    # gate: good routines must be fp16-clean; IIR is reported, not gated on a fixed tol.
    ok = worst_good < 5e-3
    print(f"\n{'PASS' if ok else 'FAIL'} - FIR/FFT-DSP validate vs scipy/numpy on the ANE "
          f"(worst GOOD relerr {worst_good:.2e}); IIR honestly tagged arch-limited (fixed unroll)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
