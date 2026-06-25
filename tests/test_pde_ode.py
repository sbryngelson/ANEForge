"""Iterative-methods corpus: PDE/ODE/root-finding/series fixed-iteration kernels vs numpy goldens."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ -> import _corpus
from _corpus import Case, run_corpus
from _helpers import f16, onehot_select as _col
import aneforge as af

rng = np.random.default_rng(7)


# tag side-table: name -> (cost_character, feasibility)
TAGS: dict[str, tuple[str, str]] = {}


def tagged(case: Case, cost: str, feasibility: str) -> Case:
  TAGS[case.name] = (cost, feasibility)
  return case


# 1. PDE / STENCILS  (conv-based - the ANE's home turf)

def _jacobi_iteration():
  """Jacobi smoother for 2D Poisson (f=0, K=5 fixed sweeps as neighbour-sum convs); tol=0.02 (a solver form would need a data-dependent convergence stop)."""
  H = W = 24; K = 5
  nbr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
  Kw = (0.25 * nbr).reshape(1, 1, 3, 3).astype(np.float16)
  u0 = f16(rng, 1, 1, H, W, scale=1.0)

  def build(ut):
    h = ut
    for _ in range(K):
      h = af.conv(h, Kw, pad=1)        # u_new = (sum of 4 neighbours)/4
    return h

  def ref(ua):
    h = ua.astype(np.float32)
    Kf = (0.25 * nbr)
    for _ in range(K):
      x = np.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)))
      out = np.zeros_like(h)
      for i in range(H):
        for j in range(W):
          out[0, 0, i, j] = (x[0, 0, i:i + 3, j:j + 3] * Kf).sum()
      h = out
    return h

  return tagged(Case("jacobi_poisson_K5", "pde-ode", build, ref, [u0], tol=0.02),
                "compute", "works")


def _heat_step_2d():
  """Explicit (forward-Euler) 2D heat step folded into one conv, r=0.2 (inside the r<=0.25 stability bound); tol=0.02."""
  H = W = 32; r = 0.2
  lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
  K = np.zeros((1, 1, 3, 3), np.float32)
  K[0, 0] = r * lap
  K[0, 0, 1, 1] += 1.0
  Kf = K.astype(np.float16)
  u0 = f16(rng, 1, 1, H, W, scale=1.0)

  def build(ut): return af.conv(ut, Kf, pad=1)

  def ref(ua):
    x = np.pad(ua.astype(np.float32), ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(ua, np.float32)
    Kff = K[0, 0]
    for i in range(H):
      for j in range(W):
        out[0, 0, i, j] = (x[0, 0, i:i + 3, j:j + 3] * Kff).sum()
    return out

  return tagged(Case("heat_explicit_step_2d", "pde-ode", build, ref, [u0], tol=0.02),
                "compute", "works")


def _wave_step_2d():
  """2D wave leapfrog step (two time levels in, combined kernel K=C^2*Lap+2*I in one conv minus u^{n-1}), C=0.4 inside CFL; tol=0.02."""
  H = W = 24; C = 0.4
  lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
  K = np.zeros((1, 1, 3, 3), np.float32)
  K[0, 0] = (C * C) * lap
  K[0, 0, 1, 1] += 2.0                  # + 2*I  -> conv gives 2 u^n + C^2 Lap u^n
  Kf = K.astype(np.float16)
  un = f16(rng, 1, 1, H, W, scale=1.0)       # u^n
  unm1 = f16(rng, 1, 1, H, W, scale=1.0)     # u^{n-1}

  def build(un_t, unm1_t): return af.conv(un_t, Kf, pad=1) - unm1_t

  def ref(un_a, unm1_a):
    x = np.pad(un_a.astype(np.float32), ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros_like(un_a, np.float32)
    Kff = K[0, 0]
    for i in range(H):
      for j in range(W):
        conv[0, 0, i, j] = (x[0, 0, i:i + 3, j:j + 3] * Kff).sum()
    return conv - unm1_a.astype(np.float32)

  return tagged(Case("wave_leapfrog_step_2d", "pde-ode", build, ref, [un, unm1], tol=0.02),
                "compute", "works")


def _multigrid_smooth():
  """One damped-Jacobi multigrid smoothing sweep (omega=2/3) folded into a single conv; tol=0.02 (a full V-cycle's recursion depth + coarse solve are arch-limited)."""
  H = W = 24; omega = 2.0 / 3.0
  nbr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
  K = np.zeros((1, 1, 3, 3), np.float32)
  K[0, 0] = omega * 0.25 * nbr
  K[0, 0, 1, 1] += (1.0 - omega)
  Kf = K.astype(np.float16)
  u0 = f16(rng, 1, 1, H, W, scale=1.0)

  def build(ut): return af.conv(ut, Kf, pad=1)

  def ref(ua):
    x = np.pad(ua.astype(np.float32), ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(ua, np.float32)
    Kff = K[0, 0]
    for i in range(H):
      for j in range(W):
        out[0, 0, i, j] = (x[0, 0, i:i + 3, j:j + 3] * Kff).sum()
    return out

  return tagged(Case("multigrid_jacobi_smooth", "pde-ode", build, ref, [u0], tol=0.02),
                "compute", "works")


# 2. ODE INTEGRATORS & ROOT FINDERS  (fixed-step recurrences)

def _forward_euler():
  """Forward-Euler on a linear ODE y'=Ay, K=8 fixed steps (gemv+axpy); tol=0.04 covers bounded fp16 compounding (an adaptive-step integrator would be arch-limited)."""
  N, K, dt = 8, 8, 0.05
  A = (rng.standard_normal((N, N)).astype(np.float32) * 0.3)
  A = (A - A.T)                       # skew-symmetric -> norm-preserving, stable
  Ah = A.astype(np.float16)
  y0 = f16(rng, 1, N, scale=1.0)

  def build(yt):
    h = yt
    for _ in range(K):
      rhs = h @ Ah.T.astype(np.float16)     # A y  (gemv via [1,N]@[N,N]^T)
      h = h + rhs * dt
    return h

  def ref(ya):
    Af = A.astype(np.float32); h = ya
    for _ in range(K):
      h = h + dt * (h @ Af.T)
    return h

  return tagged(Case("forward_euler_linear_K8", "pde-ode", build, ref, [y0], tol=0.04),
                "mixed", "works")


def _rk4_step():
  """RK4 on a nonlinear scalar ODE y'=-y+0.1y^2, K=4 fixed steps; tol=0.04 (residual is fp16 compounding over 4 steps x 4 RHS evals, not a bug)."""
  N, K, dt = 12, 4, 0.1

  def f_ag(y):                              # f(y) = -y + 0.1 y^2  via aneforge ops
    return (y * -1.0) + (y.square() * 0.1)

  def f_np(y): return -y + 0.1 * y * y

  y0 = f16(rng, 1, N, scale=0.5)

  def build(yt):
    y = yt
    for _ in range(K):
      k1 = f_ag(y)
      k2 = f_ag(y + k1 * (dt / 2))
      k3 = f_ag(y + k2 * (dt / 2))
      k4 = f_ag(y + k3 * dt)
      incr = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6)
      y = y + incr
    return y

  def ref(ya):
    y = ya
    for _ in range(K):
      k1 = f_np(y)
      k2 = f_np(y + dt / 2 * k1)
      k3 = f_np(y + dt / 2 * k2)
      k4 = f_np(y + dt * k3)
      y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

  return tagged(Case("rk4_nonlinear_K4", "pde-ode", build, ref, [y0], tol=0.04),
                "mixed", "works")


def _newton_scalar():
  """Newton sqrt iter x<-0.5(x+a/x), K=4 fixed steps; tol=0.02 (converged, only fp16 rounding of the final iterate; run-to-tol would be arch-limited)."""
  N, K = 16, 4
  a = (rng.uniform(0.5, 4.0, size=(1, N)).astype(np.float32)).astype(np.float16)  # solve x^2=a

  def build(at):
    x = at                                # x0 = a  (positive, fine for sqrt iter)
    for _ in range(K):
      x = (x + at / x) * 0.5            # 0.5*(x + a/x)
    return x

  def ref(aa):
    x = aa.copy()
    for _ in range(K):
      x = 0.5 * (x + aa / x)
    return x

  return tagged(Case("newton_sqrt_K4", "pde-ode", build, ref, [a], tol=0.02),
                "mixed", "works")


def _newton_vector():
  """Vector Newton via the explicit 2x2 Jacobian inverse, K=4 fixed steps; tol=0.03 (a general N-D Newton needs a linear solve each step = arch-limited)."""
  K = 4
  r, s = 2.0, 0.5                          # target constants
  rs = np.array([[r, s]], np.float16)
  x0 = np.array([[1.2, 0.3]], np.float16)  # start

  def build(xt, rst):
    x0c = _col(xt, 0, 2); x1c = _col(xt, 1, 2)
    rc = _col(rst, 0, 2); sc = _col(rst, 1, 2)
    for _ in range(K):
      F0 = x0c * x0c + x1c * x1c - rc
      F1 = x0c - x1c - sc
      # J = [[2 x0, 2 x1],[1,-1]]; det = -2x0 - 2x1
      det = x0c * (-2.0) + x1c * (-2.0)
      # J^{-1} = (1/det) [[-1, -2 x1],[ -1, 2 x0]]
      dx0 = (F0 * (-1.0) + F1 * (x1c * (-2.0))) / det
      dx1 = (F0 * (-1.0) + F1 * (x0c * 2.0)) / det
      x0c = x0c - dx0
      x1c = x1c - dx1
    return af.concat([x0c, x1c], axis=1)

  def ref(xa, rsa):
    x0c = xa[:, 0:1].astype(np.float32); x1c = xa[:, 1:2].astype(np.float32)
    rc = rsa[:, 0:1].astype(np.float32); sc = rsa[:, 1:2].astype(np.float32)
    for _ in range(K):
      F0 = x0c * x0c + x1c * x1c - rc
      F1 = x0c - x1c - sc
      det = -2 * x0c - 2 * x1c
      dx0 = (-F0 - 2 * x1c * F1) / det
      dx1 = (-F0 + 2 * x0c * F1) / det
      x0c = x0c - dx0
      x1c = x1c - dx1
    return np.concatenate([x0c, x1c], axis=1)

  return tagged(Case("newton_vector_2d_K4", "pde-ode", build, ref, [x0, rs], tol=0.03),
                "mixed", "works")


def _fixed_point():
  """Fixed-point iter x=cos(x), K=10 (Banach contraction -> Dottie number); tol=0.02 (converged value's fp16 rounding plus cos's own fp16 transcendental error)."""
  N, K = 16, 10
  x0 = (rng.uniform(0.0, 1.0, size=(1, N)).astype(np.float32)).astype(np.float16)

  def build(xt):
    x = xt
    for _ in range(K):
      x = x.cos()
    return x

  def ref(xa):
    x = xa.copy()
    for _ in range(K):
      x = np.cos(x)
    return x

  return tagged(Case("fixed_point_cos_K10", "pde-ode", build, ref, [x0], tol=0.02),
                "mixed", "works")


# 3. SERIES / SPECIAL-FUNCTION APPROXIMATION  (Horner chains vs exact)

def _exp_taylor():
  """exp(x) via a degree-8 Taylor/Horner chain, x in [-2,2]; tol=0.03 covers truncation plus fp16 rounding of the 8 fused mul/adds (a genuine approx gap, not a bug)."""
  D = 8
  coeffs = np.array([1.0 / math.factorial(k) for k in range(D + 1)], np.float32)
  cvec = coeffs.astype(np.float16).reshape(1, D + 1)             # c[0..D] = 1/k!
  x = (rng.uniform(-2.0, 2.0, size=(1, 24)).astype(np.float32)).astype(np.float16)

  def build(xt, ct):
    acc = None
    for k in range(D, -1, -1):
      ck = _col(ct, k, D + 1)
      acc = ck if acc is None else (acc * xt + ck)
    return acc

  def ref(xa, ca):
    acc = None
    for k in range(D, -1, -1):
      ck = ca[:, k:k + 1]
      acc = ck if acc is None else (acc * xa + ck)
    return acc   # the SAME truncated series in fp32 (isolates fp16 vs fp32, not trunc)

  return tagged(Case("exp_taylor_deg8", "pde-ode", build, ref, [x, cvec], tol=0.03),
                "floor/fusion", "works")


def _exp_vs_exact():
  """deg-10 exp Taylor vs the EXACT np.exp (includes truncation), x in [-1.5,1.5]; tol=0.03 dominated by fp16 rounding of 10 fused mul/adds."""
  D = 10
  coeffs = np.array([1.0 / math.factorial(k) for k in range(D + 1)], np.float32)
  cvec = coeffs.astype(np.float16).reshape(1, D + 1)
  x = (rng.uniform(-1.5, 1.5, size=(1, 24)).astype(np.float32)).astype(np.float16)

  def build(xt, ct):
    acc = None
    for k in range(D, -1, -1):
      ck = _col(ct, k, D + 1)
      acc = ck if acc is None else (acc * xt + ck)
    return acc

  def ref(xa, ca): return np.exp(xa)   # EXACT special function

  return tagged(Case("exp_taylor_vs_exact", "pde-ode", build, ref, [x, cvec], tol=0.03),
                "floor/fusion", "works")


def _erf_series():
  """erf(x) via its M=8 Maclaurin/Horner series vs exact erf, x in [-1.5,1.5]; tol=0.03 (large |x| would need a regime switch the engine lacks)."""
  M = 8
  u_coeffs = np.array([((-1.0) ** k) / (math.factorial(k) * (2 * k + 1))
                        for k in range(M + 1)], np.float32)            # poly in u=x^2
  cvec = u_coeffs.astype(np.float16).reshape(1, M + 1)
  two_over_sqrtpi = np.float32(2.0 / np.sqrt(np.pi))
  x = (rng.uniform(-1.5, 1.5, size=(1, 24)).astype(np.float32)).astype(np.float16)

  try:
    from scipy.special import erf as sp_erf
    _erf = sp_erf
  except Exception:  # noqa: BLE001
    _erf = np.vectorize(lambda v: __import__("math").erf(float(v)))

  def build(xt, ct):
    u = xt.square()                       # u = x^2
    acc = None
    for k in range(M, -1, -1):
      ck = _col(ct, k, M + 1)
      acc = ck if acc is None else (acc * u + ck)
    return (acc * xt) * float(two_over_sqrtpi)

  def ref(xa, ca): return _erf(xa.astype(np.float32)).astype(np.float32)

  return tagged(Case("erf_maclaurin_deg8", "pde-ode", build, ref, [x, cvec], tol=0.03),
                "floor/fusion", "works")


def _log_series():
  """log(1+z) via a degree-10 Horner series vs exact log1p, z in (-0.5,0.5); tol=0.03 (full-axis log would need data-dependent argument reduction)."""
  D = 10
  # coeffs for poly in z, k=0..D ; c[0]=0 (no constant term), c[k]=(-1)^{k+1}/k
  coeffs = np.array([0.0] + [((-1.0) ** (k + 1)) / k for k in range(1, D + 1)], np.float32)
  cvec = coeffs.astype(np.float16).reshape(1, D + 1)
  z = (rng.uniform(-0.5, 0.5, size=(1, 24)).astype(np.float32)).astype(np.float16)

  def build(zt, ct):
    acc = None
    for k in range(D, -1, -1):
      ck = _col(ct, k, D + 1)
      acc = ck if acc is None else (acc * zt + ck)
    return acc

  def ref(za, ca): return np.log1p(za.astype(np.float32)).astype(np.float32)

  return tagged(Case("log1p_series_deg10", "pde-ode", build, ref, [z, cvec], tol=0.03),
                "floor/fusion", "works")


# 4. ARCH-LIMITED PROBES - convergent/adaptive forms, xfail'd: the data-dependent stop is inexpressible.

def _newton_to_convergence_archlimited():
  """Newton run-to-convergence (xfail): a fixed-K unroll cannot match the variable-count numpy reference, demonstrating the data-dependent stop is inexpressible."""
  N = 12
  # convergence count varies across lanes: numpy runs to a tight residual, the ANE
  # graph is fixed K=2 (too few for the hardest lanes) -> genuine mismatch.
  a = (rng.uniform(0.1, 50.0, size=(1, N)).astype(np.float32)).astype(np.float16)
  K_fixed = 2

  def build(at):
    x = at
    for _ in range(K_fixed):
      x = (x + at / x) * 0.5
    return x

  def ref(aa):
    # run EACH lane to convergence (data-dependent iteration count)
    out = np.empty_like(aa)
    for j in range(aa.shape[1]):
      x = float(aa[0, j])
      for _ in range(100):
        xn = 0.5 * (x + aa[0, j] / x)
        if abs(xn - x) < 1e-7 * max(1.0, abs(xn)):
          x = xn
          break
        x = xn
      out[0, j] = x
    return out

  return tagged(Case("newton_to_convergence", "pde-ode-archlimited", build, ref, [a],
                     tol=0.02,
                     xfail="run-to-convergence needs a data-dependent stop "
                           "(variable iteration count); the feed-forward ANE has no "
                           "in-graph loop/branch. Fixed K=2 unroll cannot match the "
                           "adaptive numpy reference for the hard (large-a) lanes."),
                "mixed (no control flow)", "arch-limited")


def _adaptive_timestep_archlimited():
  """Adaptive-timestep ODE (xfail): a fixed coarse Euler graph diverges from the accurate (exp) solution on stiff lanes, since dt/step-count are data-dependent control flow."""
  N = 8
  # y' = -k y (stiff for large k); a spread of k across lanes forces different ideal
  # step sizes per lane -> no single fixed dt works for all.
  k_rates = (rng.uniform(1.0, 20.0, size=(1, N)).astype(np.float32)).astype(np.float16)
  y0 = (np.ones((1, N), np.float32)).astype(np.float16)
  T = 1.0; K_coarse = 4; dt_coarse = T / K_coarse

  def build(yt, kt):
    y = yt
    for _ in range(K_coarse):
      y = y - (kt * y) * dt_coarse        # forward Euler, large fixed dt
    return y

  def ref(ya, ka):
    # exact solution y(T) = y0 * exp(-k T) (what an adaptive integrator targets)
    return ya.astype(np.float32) * np.exp(-ka.astype(np.float32) * T)

  return tagged(Case("adaptive_timestep_ode", "pde-ode-archlimited", build, ref,
                     [y0, k_rates], tol=0.03,
                     xfail="adaptive dt + variable step count is data-dependent "
                           "control flow; the ANE cannot shrink dt on a runtime "
                           "error estimate. A fixed coarse Euler graph diverges from "
                           "the accurate (exp) solution for stiff (large-k) lanes."),
                "mixed (no control flow)", "arch-limited")


# corpus assembly + runner

PDE_STENCILS = [
  _jacobi_iteration(),
  _heat_step_2d(),
  _wave_step_2d(),
  _multigrid_smooth(),
]

ODE_ROOT = [
  _forward_euler(),
  _rk4_step(),
  _newton_scalar(),
  _newton_vector(),
  _fixed_point(),
]

SERIES = [
  _exp_taylor(),
  _exp_vs_exact(),
  _erf_series(),
  _log_series(),
]

ARCH_LIMITED = [
  _newton_to_convergence_archlimited(),
  _adaptive_timestep_archlimited(),
]

CASES = PDE_STENCILS + ODE_ROOT + SERIES + ARCH_LIMITED


def _header():
  return f"{'case':32s} {'var':4s} {'status':6s} {'cost':22s} {'feasible':13s} detail"


def _row(rec):
  return (f"{rec['name']:32s} {rec['variant']:4s} {rec['status']:6s} "
          f"{rec['cost']:22s} {rec['feasibility']:13s} {rec['metric']}")


def _annotate(case, rec):
  rec["cost"], rec["feasibility"] = TAGS.get(case.name, ("?", "?"))


def run_pde_ode(cases, verbose: bool = True):
  """run_corpus plus an iterative-methods capability verdict block; PASS/XFAIL green, FAIL/ERROR/XPASS red. Returns (results, exit_code)."""
  def verdict(all_results, relerrs):
    # ------- iterative-methods capability verdict block ------------------- #
    print("\n" + "-" * 110)
    print("WHICH ITERATIVE NUMERICAL METHODS FIT THE ANE (fixed feed-forward dataflow)")
    print("-" * 110)
    print("  PDE / STENCILS (conv-based) - the marquee fit:")
    for c in PDE_STENCILS:
      recs = [r for r in all_results if r["name"] == c.name]
      st = recs[0]["status"] if recs else "?"
      print(f"    {c.name:30s} {TAGS[c.name][1]:13s} ({st}; {recs[0]['metric'] if recs else ''})")
    print("  ODE INTEGRATORS & ROOT FINDERS (fixed-step recurrences):")
    for c in ODE_ROOT:
      recs = [r for r in all_results if r["name"] == c.name]
      st = recs[0]["status"] if recs else "?"
      print(f"    {c.name:30s} {TAGS[c.name][1]:13s} ({st}; {recs[0]['metric'] if recs else ''})")
    print("  SERIES / SPECIAL FUNCTIONS (Horner chains vs exact):")
    for c in SERIES:
      recs = [r for r in all_results if r["name"] == c.name]
      st = recs[0]["status"] if recs else "?"
      print(f"    {c.name:30s} {TAGS[c.name][1]:13s} ({st}; {recs[0]['metric'] if recs else ''})")
    print("  ARCH-LIMITED (convergent/adaptive forms - control flow the ANE lacks):")
    for c in ARCH_LIMITED:
      recs = [r for r in all_results if r["name"] == c.name]
      st = recs[0]["status"] if recs else "?"
      print(f"    {c.name:30s} {TAGS[c.name][1]:13s} ({st}; xfail = inexpressible stop)")

    print("\n  Summary verdict:")
    print("    WORKS (fixed-iteration, fully unrolled into a static graph):")
    print("      - PDE stencil sweeps: Jacobi (K sweeps), explicit heat step, leapfrog")
    print("        wave step, damped-Jacobi multigrid smoothing. These lower to native")
    print("        conv - the ANE's home turf - and are fp16-clean (wide accumulator).")
    print("      - ODE integrators: forward-Euler & RK4 advanced a FIXED #steps; the RHS")
    print("        is aneforge ops. Error is fp16 COMPOUNDING (~few %, bounded), not a bug.")
    print("      - Root finders: Newton (scalar via sqrt-iter, vector via 2x2 closed-form")
    print("        Jacobian) & fixed-point x=g(x), all FIXED iteration count.")
    print("      - Special functions: exp/erf/log via Horner series, accurate to a few %")
    print("        vs the EXACT function on a bounded argument interval.")
    print("    ARCH-LIMITED (needs control flow the feed-forward engine lacks):")
    print("      - RUN-TO-CONVERGENCE (stop when ||residual|| < tol): variable iteration")
    print("        count = data-dependent loop/branch. Must fix K up front; a fixed graph")
    print("        cannot match an adaptive reference for the hard lanes.")
    print("      - ADAPTIVE TIMESTEP (shrink dt on local error): data-dependent step size")
    print("        and step count. A fixed coarse step diverges on stiff lanes.")
    print("      - REGIME SWITCHES in special functions (erf/log argument reduction, big-x")
    print("        asymptotics): a runtime branch on the argument; one fixed series only")
    print("        covers a bounded interval.")
    print("      - MULTIGRID V-CYCLE RECURSION / coarse solve: the inter-grid transfers")
    print("        are convs/pools (expressible), but the recursion depth + coarse solve")
    print("        are control-flow / data-sized. Only the single smoothing sweep fits.")
    print("    => The ANE fits iterative scientific methods whose ITERATION STRUCTURE is")
    print("       STATIC AND KNOWN AT COMPILE TIME (fixed sweeps/steps, unrolled). It does")
    print("       NOT fit methods whose iteration count or step size is decided AT RUNTIME")
    print("       from the data (convergence tests, adaptive stepping, regime switches),")
    print("       which need the in-graph loop/branch the feed-forward dataflow lacks.")
    print("       fp16 is not the wall here (compounding stays bounded over modest step")
    print("       counts); the wall is the missing data-dependent control flow.")
    print()  # blank line before the GATE line

  return run_corpus(cases, verbose, columns=(_header, _row), annotate=_annotate,
                    verdict=verdict, sep_width=110)


if __name__ == "__main__":
  _, code = run_pde_ode(CASES)
  sys.exit(code)
