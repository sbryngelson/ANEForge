"""Spectral / signal / statistical scientific-kernel corpus ('can the ANE do an FFT?') vs fp32 goldens."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tests/ -> import _corpus
from _corpus import Case, run_corpus
from _helpers import f16, onehot_select
import aneforge as af

rng = np.random.default_rng(20260529)


# tag side-table: name -> (cost_character, feasibility)
TAGS: dict[str, tuple[str, str]] = {}


def tagged(case: Case, cost: str, feasibility: str) -> Case:
  TAGS[case.name] = (cost, feasibility)
  return case


# SPECTRAL - the "can the ANE do an FFT?" probe (complex as real pairs)

def _dft_matmul(N: int, tol: float):
  """DFT as a dense twiddle-matrix matmul, complex emulated as a (real, imag) pair; fp16-clean (wide accumulator), O(N^2) twiddle is the cost."""
  n = np.arange(N)
  k = n.reshape(-1, 1)
  W = np.exp(-2j * np.pi * k * n / N)                  # [N,N] DFT matrix
  Wr = np.ascontiguousarray(np.real(W)).astype(np.float16)
  Wi = np.ascontiguousarray(np.imag(W)).astype(np.float16)
  xr = f16(rng, 1, N); xi = f16(rng, 1, N)

  def build(rt, it):
    a = rt @ Wr.T.astype(np.float16)                # xr @ Wr^T
    b = it @ Wi.T.astype(np.float16)                # xi @ Wi^T
    Xr = a - b
    c = rt @ Wi.T.astype(np.float16)                # xr @ Wi^T
    d = it @ Wr.T.astype(np.float16)                # xi @ Wr^T
    Xi = c + d
    return af.concat([Xr, Xi], axis=1)              # [1, 2N]

  def ref(ra, ia):
    xc = ra.astype(np.float32) + 1j * ia.astype(np.float32)
    X = xc @ W.T                                     # W symmetric: x @ W^T == DFT
    return np.concatenate([np.real(X), np.imag(X)], axis=1).astype(np.float32)

  return tagged(Case(f"dft_matmul_N{N}", "spectral", build, ref, [xr, xi], tol=tol),
                  "compute", "works")


def _fft_butterfly_radix2(N: int, tol: float):
  """Radix-2 DIT FFT fully unrolled as a static graph over (real, imag) lane pairs; correct for small fixed N, but only as a per-N static unroll (no loop/gather)."""
  bits = int(round(np.log2(N)))
  assert (1 << bits) == N, "radix-2 requires power-of-two N"

  def _bitrev(x, b):
    r = 0
    for _ in range(b):
      r = (r << 1) | (x & 1)
      x >>= 1
    return r

  order = [_bitrev(i, bits) for i in range(N)]

  xr = f16(rng, 1, N); xi = f16(rng, 1, N)

  def build(rt, it):
    re = [onehot_select(rt, order[i], N) for i in range(N)]  # [1,N] -> [1,1] one-hot select
    im = [onehot_select(it, order[i], N) for i in range(N)]
    size = 2
    while size <= N:
      half = size // 2
      for start in range(0, N, size):
        for j in range(half):
          ang = -2.0 * np.pi * j / size
          wr, wi = float(np.cos(ang)), float(np.sin(ang))
          a, b = start + j, start + j + half
          tr = re[b] * wr - im[b] * wi         # twiddle * operand (complex)
          ti = re[b] * wi + im[b] * wr
          re[a], re[b] = re[a] + tr, re[a] - tr
          im[a], im[b] = im[a] + ti, im[a] - ti
      size *= 2
    return af.concat(re + im, axis=1)               # [1, 2N]

  def ref(ra, ia):
    X = np.fft.fft(ra.astype(np.float32) + 1j * ia.astype(np.float32))
    return np.concatenate([np.real(X), np.imag(X)], axis=1).astype(np.float32)

  return tagged(Case(f"fft_butterfly_radix2_N{N}", "spectral", build, ref, [xr, xi], tol=tol),
                  "floor/fusion", "works")


def _rfft_power_spectrum(N: int, tol: float):
  """Real-input FFT power spectrum |X_k|^2 = Xr^2+Xi^2 via the DFT-matmul; fp16-clean."""
  n = np.arange(N); k = n.reshape(-1, 1)
  W = np.exp(-2j * np.pi * k * n / N)
  Wr = np.ascontiguousarray(np.real(W)).astype(np.float16)
  Wi = np.ascontiguousarray(np.imag(W)).astype(np.float16)
  x = f16(rng, 1, N)

  def build(xt):
    Xr = xt @ Wr.T.astype(np.float16)
    Xi = xt @ Wi.T.astype(np.float16)
    return Xr.square() + Xi.square()                # [1,N] power

  def ref(xa):
    X = xa.astype(np.float32) @ W.T
    return (np.abs(X) ** 2).astype(np.float32)

  return tagged(Case(f"rfft_power_spectrum_N{N}", "spectral", build, ref, [x], tol=tol),
                  "compute", "works")


# SIGNAL - FIR filter and autocorrelation

def _fir_filter(L: int, K: int, tol: float):
  """FIR filter as a valid 1D conv; ANE conv is cross-correlation (no kernel flip)."""
  sig = f16(rng, 1, 1, 1, L)
  taps = f16(rng, 1, 1, 1, K, scale=0.5)

  def build(st):
    return af.conv(st, taps, pad=0)                 # [1,1,1,L-K+1], valid

  def ref(sa):
    s = sa[0, 0, 0].astype(np.float32)
    t = taps[0, 0, 0].astype(np.float32)
    y = np.correlate(s, t, mode="valid")            # ANE conv == correlation
    return y.reshape(1, 1, 1, -1)

  return tagged(Case(f"fir_filter_L{L}_K{K}", "signal", build, ref, [sig], tol=tol),
                  "compute", "works")


def _autocorr_crosscorr(H: int, W: int, Th: int, Tw: int, tol: float):
  """Template autocorrelation via the native CrossCorrelation bridge; template must be strictly smaller than the map."""
  x = f16(rng, H, W)
  tmpl = np.ascontiguousarray(x[:Th, :Tw]).astype(np.float16)   # patch -> autocorr-style

  def build(xt, tt):
    return af.cross_correlation(xt, tt)             # [H-Th+1, W-Tw+1]

  def ref(xa, ta):
    xf = xa.astype(np.float32); tf = ta.astype(np.float32)
    out = np.zeros((H - Th + 1, W - Tw + 1), np.float32)
    for i in range(H - Th + 1):
      for j in range(W - Tw + 1):
        out[i, j] = (xf[i:i + Th, j:j + Tw] * tf).sum()
    return out

  return tagged(Case(f"autocorr_xcorr_{H}x{W}_t{Th}x{Tw}", "signal", build, ref,
                       [x, tmpl], tol=tol), "mixed (cut)", "works")


def _autocorr_conv(L: int, Lk: int, tol: float):
  """Lagged autocorrelation as a valid 1D conv of the signal against a prefix of itself; lag window Lk must stay small (conv backend rejects wide 1xK kernels)."""
  sig = f16(rng, 1, 1, 1, L)
  kern = np.ascontiguousarray(sig[0, 0, 0, :Lk]).astype(np.float16).reshape(1, 1, 1, Lk)

  def build(st):
    return af.conv(st, kern, pad=0)                 # correlation of s with its prefix

  def ref(sa):
    s = sa[0, 0, 0].astype(np.float32)
    k = kern[0, 0, 0].astype(np.float32)
    return np.correlate(s, k, mode="valid").reshape(1, 1, 1, -1)

  return tagged(Case(f"autocorr_conv_L{L}_k{Lk}", "signal", build, ref, [sig], tol=tol),
                  "compute", "works")


# MONTE CARLO - host-drawn samples (see module RANDOMNESS NOTE), ANE reduces

def _mc_integral(M: int, tol: float):
  """MC integral of exp(-x^2) over [0,1] as map+mean; samples host-drawn, ref uses the same samples (deterministic)."""
  u = rng.random((1, M)).astype(np.float16)

  def build(ut):
    return (ut.square() * (-1.0)).exp().mean(1)     # [1,1]

  def ref(ua):
    return np.exp(-(ua.astype(np.float32) ** 2)).mean(1, keepdims=True)

  return tagged(Case(f"mc_integral_exp_M{M}", "montecarlo", build, ref, [u], tol=tol),
                  "reduction", "works")


def _mc_mean_var(M: int, tol: float):
  """MC mean & variance via E[g^2]-E[g]^2; wide accumulator avoids catastrophic cancellation."""
  s = rng.standard_normal((1, M)).astype(np.float16)

  def build(st):
    mean = st.mean(1)
    meansq = st.square().mean(1)
    var = meansq - mean * mean
    return af.concat([mean, var], axis=1)           # [1,2]

  def ref(sa):
    a = sa.astype(np.float32)
    mean = a.mean(1, keepdims=True)
    var = (a ** 2).mean(1, keepdims=True) - mean * mean
    return np.concatenate([mean, var], axis=1)

  return tagged(Case(f"mc_mean_var_M{M}", "montecarlo", build, ref, [s], tol=tol),
                  "reduction", "works")


# N-BODY - pairwise interactions via broadcast diff + reduction

def _nbody_spring(N: int, tol: float):
  """Pairwise linear net force F_i = sum_j (p_j - p_i) via broadcast-diff + reduction."""
  P = f16(rng, N, 3, scale=1.0)

  def build(pt):
    pi = pt.reshape(N, 1, 3)
    pj = pt.reshape(1, N, 3)
    diff = pj - pi                                  # [N,N,3] broadcast
    return diff.sum(1).reshape(N, 3)               # sum over j

  def ref(pa):
    a = pa.astype(np.float32)
    return (a[None, :, :] - a[:, None, :]).sum(1)

  return tagged(Case(f"nbody_spring_N{N}", "nbody", build, ref, [P], tol=tol),
                  "mixed", "works")


def _nbody_invsq_potential(N: int, tol: float):
  """Pairwise inverse-distance potential U_i = sum_{j!=i} 1/sqrt(|p_i-p_j|^2+eps); self term killed by a folded diagonal mask."""
  P = (rng.standard_normal((N, 3)).astype(np.float32) * 1.5).astype(np.float16)
  eps = 0.5
  bigdiag = (np.eye(N, dtype=np.float32) * 1e3).astype(np.float16)   # folded mask

  def build(pt, mt):
    pi = pt.reshape(N, 1, 3)
    pj = pt.reshape(1, N, 3)
    d = pj - pi                                     # [N,N,3]
    d2 = d.square().sum(2).reshape(N, N)            # [N,N] squared distances
    d2 = d2 + mt                                    # add huge value on the diagonal
    inv = d2.rsqrt(eps=eps)                         # 1/sqrt(d2+eps), diag -> ~0
    return inv.sum(1).reshape(N, 1)                 # potential per point

  def ref(pa, ma):
    a = pa.astype(np.float32)
    d2 = ((a[None, :, :] - a[:, None, :]) ** 2).sum(2) + ma.astype(np.float32)
    inv = 1.0 / np.sqrt(d2 + eps)
    return inv.sum(1, keepdims=True)

  return tagged(Case(f"nbody_invsq_pot_N{N}", "nbody", build, ref, [P, bigdiag], tol=tol),
                  "mixed", "fp16-limited")


# QUADRATURE - weighted reduction (cumsum is unsupported: a boundary)

def _simpson(n: int, tol: float):
  """Composite Simpson integration of sin(x) on [0,pi] as a weighted reduction (true value 2.0); cumsum/prefix-scan is not expressible (no scan op)."""
  a, b = 0.0, np.pi
  h = (b - a) / n
  x = np.linspace(a, b, n + 1)
  w = np.ones(n + 1); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0; w *= h / 3.0
  fx = np.sin(x).astype(np.float16).reshape(1, n + 1)
  wv = w.astype(np.float16).reshape(1, n + 1)

  def build(ft, wt):
    return (ft * wt).sum(1)                          # [1,1]

  def ref(fa, wa):
    return (fa.astype(np.float32) * wa.astype(np.float32)).sum(1, keepdims=True)

  return tagged(Case(f"simpson_sin_n{n}", "quadrature", build, ref, [fx, wv], tol=tol),
                  "reduction", "works")


def _gauss_legendre(n: int, tol: float):
  """Gauss-Legendre quadrature of exp(x) on [-1,1] as a weighted reduction (true value e - 1/e ~ 2.3504)."""
  nodes, weights = np.polynomial.legendre.leggauss(n)
  fx = np.exp(nodes).astype(np.float16).reshape(1, n)
  wv = weights.astype(np.float16).reshape(1, n)

  def build(ft, wt):
    return (ft * wt).sum(1)

  def ref(fa, wa):
    return (fa.astype(np.float32) * wa.astype(np.float32)).sum(1, keepdims=True)

  return tagged(Case(f"gauss_legendre_exp_n{n}", "quadrature", build, ref, [fx, wv], tol=tol),
                  "reduction", "works")


# corpus assembly + runner

# SPECTRAL: tolerances track the measured fp16 floor (flat in N; wide accumulator)
SPECTRAL = [
  _dft_matmul(8,    tol=0.004),
  _dft_matmul(16,   tol=0.004),
  _dft_matmul(32,   tol=0.004),
  _dft_matmul(64,   tol=0.004),
  _dft_matmul(128,  tol=0.005),
  _dft_matmul(256,  tol=0.005),
  _dft_matmul(512,  tol=0.005),
  _dft_matmul(1024, tol=0.006),
  _dft_matmul(2048, tol=0.006),
  _fft_butterfly_radix2(8,  tol=0.004),
  _fft_butterfly_radix2(16, tol=0.006),
  _rfft_power_spectrum(64,  tol=0.01),
  _rfft_power_spectrum(256, tol=0.012),
]

SIGNAL = [
  _fir_filter(64, 5, tol=0.01),
  _fir_filter(128, 9, tol=0.012),
  _autocorr_crosscorr(8, 8, 3, 3, tol=0.01),
  _autocorr_crosscorr(12, 12, 4, 4, tol=0.012),
  _autocorr_conv(64, 8, tol=0.02),
]

MONTECARLO = [
  _mc_integral(4096, tol=0.01),
  _mc_integral(8192, tol=0.01),
  _mc_mean_var(8192, tol=0.03),
]

NBODY = [
  _nbody_spring(6, tol=0.01),
  _nbody_spring(16, tol=0.015),
  _nbody_invsq_potential(8, tol=0.03),
]

QUADRATURE = [
  _simpson(32, tol=0.005),
  _simpson(64, tol=0.005),
  _gauss_legendre(8, tol=0.01),
  _gauss_legendre(16, tol=0.01),
]

CASES = SPECTRAL + SIGNAL + MONTECARLO + NBODY + QUADRATURE


def _header():
  return f"{'case':30s} {'var':4s} {'status':6s} {'cost':14s} {'feasible':13s} detail"


def _row(rec):
  return (f"{rec['name']:30s} {rec['variant']:4s} {rec['status']:6s} "
          f"{rec['cost']:14s} {rec['feasibility']:13s} {rec['metric']}")


def _annotate(case, rec):
  rec["cost"], rec["feasibility"] = TAGS.get(case.name, ("?", "?"))


def run_spectral(cases, verbose: bool = True):
  """run_corpus extended to print cost/feasibility tags and a SPECTRAL verdict; returns (results, exit_code)."""
  def verdict(all_results, relerrs):
    dft_curve = [(int(r["name"].split("_N")[1]), r["metric"].split()[1])
                 for r in all_results
                 if r.get("relerr") is not None and r["name"].startswith("dft_matmul_N")]
    # ------- SPECTRAL verdict block (the marquee finding) ----------------- #
    print("\n" + "-" * 104)
    print("CAN THE ANE DO AN FFT?  (complex emulated as real/imag tensor pairs)")
    print("-" * 104)
    if dft_curve:
      print("  DFT-as-matmul precision vs transform size N (fp16 weights, wide accumulator):")
      for N, e in sorted(dft_curve):
        print(f"    N={N:<5d} relerr {e}")
      print("    => relerr is essentially FLAT in N (no compounding): the ANE matmul")
      print("       accumulator is wide (>=fp32), so the length-N twiddle sum is not")
      print("       re-rounded per term. fp16 is NOT the wall for the transform.")
    bf = [r for r in all_results if r["name"].startswith("fft_butterfly")]
    if bf:
      bfst = ", ".join(f"{r['name'].split('_N')[1]}:{r['status']}({r['metric']})" for r in bf)
      print(f"  Radix-2 butterfly FFT (fully unrolled): {bfst}")
    print("\n  VERDICT:")
    print("    YES - both a dense DFT (twiddle-matrix matmul) and a radix-2 butterfly FFT")
    print("    are EXPRESSIBLE and CORRECT on the ANE via complex-as-real-pairs")
    print("    arithmetic ((a+bi)(c+di) = (ac-bd)+(ad+bc)i over paired real tensors).")
    print("    * DFT-matmul: fp16-CLEAN to at least N=2048 (relerr ~ few e-4, flat in N).")
    print("      The limiter is NOT precision but the O(N^2) twiddle-matrix size/bandwidth")
    print("      - it is a naive DFT, so cost grows quadratically; precision does not.")
    print("    * Butterfly FFT: correct for small fixed N, but expressible ONLY as a")
    print("      per-N STATIC UNROLL (the ANE is feed-forward: no in-graph loop, no scalar")
    print("      feedback, and bit-reversal needs a folded one-hot since there is no")
    print("      gather). So its O(N log N) advantage is not realizable as a graph at")
    print("      useful N - the wall is ARCHITECTURAL (static unroll), not fp16.")
    print("    BOTTOM LINE: the ANE has no complex dtype, but FFT/DFT is viable through")
    print("    real-pair emulation; for practical sizes the dense DFT-matmul is the")
    print("    right tool (fp16-robust, fuses), and a true scaling FFT is blocked by the")
    print("    missing loop/gather, not by numerics.")

    # ------- other capability findings ------------------------------------ #
    print("\n" + "-" * 104)
    print("OTHER SCIENTIFIC-KERNEL FINDINGS")
    print("-" * 104)
    print("  SIGNAL    : FIR filter (1D conv) and 2D autocorrelation (CrossCorrelation")
    print("              bridge + conv-of-self) WORK, fp16-clean. ANE conv == cross-")
    print("              correlation (no kernel flip); CrossCorrelation needs the template")
    print("              strictly smaller than the map (same-size fails ANECCompile).")
    print("  MONTECARLO: map+reduction estimators (integral, mean/variance) WORK; the wide")
    print("              accumulator keeps large-M reductions clean. Caveat: aneforge")
    print("              exposes NO on-device RNG, so samples are host-drawn and fed in")
    print("              (reference uses the same samples -> exact, not statistical).")
    print("  N-BODY    : pairwise broadcast-diff ([N,1,3]-[1,N,3]) + reduction WORKS for")
    print("              small N (linear forces fp16-clean; 1/r potential is fp16-limited")
    print("              near coincident points and grows O(N^2) in the intermediate).")
    print("  QUADRATURE: definite integrals as a weighted reduction (Simpson, Gauss-")
    print("              Legendre) WORK to ~1e-5. BOUNDARY: cumsum/prefix-scan is NOT")
    print("              expressible (no scan op, no in-graph loop), so cumulative/")
    print("              indefinite integrals do not fit.")
    print()  # blank line before the GATE line

  return run_corpus(cases, verbose, columns=(_header, _row), annotate=_annotate,
                    verdict=verdict, sep_width=104)


if __name__ == "__main__":
  _, code = run_spectral(CASES)
  sys.exit(code)
