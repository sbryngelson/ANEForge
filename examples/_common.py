"""Shared example scaffolding: env setup, error metrics, test matrices, pass/fail report. Import FIRST."""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import aneforge
import aneforge  # noqa: F401  (importing the package sets KMP_DUPLICATE_LIB_OK before numpy loads OpenMP)

import numpy as np

f16 = np.float16

# Green INFO headers / yellow warnings; tty-guarded and NO_COLOR-aware.
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_GREEN, _YELLOW, _BOLD, _DIM, _RESET = "\033[32m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"


def head(msg=""):
    """Print a section header in bold green (use instead of a `'=' * N` rule)."""
    print(f"{_BOLD}{_GREEN}{msg}{_RESET}" if _COLOR and msg else msg)


def aside(msg=""):
    """Print an explanatory aside, dimmed so it recedes behind the results.
    (Re-dims each line so the colour survives line breaks in any terminal.)"""
    if _COLOR and msg:
        print("\n".join(f"{_DIM}{ln}{_RESET}" for ln in str(msg).split("\n")))
    else:
        print(msg)


# Show each distinct warning ONCE per run (collapse identical repeats at render).
_warned: set = set()


def _colour_showwarning(message, category, filename, lineno, file=None, line=None):
    key = (category, str(message))
    if key in _warned:
        return
    _warned.add(key)
    stream = file if file is not None else sys.stderr
    text = warnings.formatwarning(message, category, filename, lineno, line)
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    stream.write(f"{_YELLOW}{text}{_RESET}" if tty and not os.environ.get("NO_COLOR") else text)


warnings.showwarning = _colour_showwarning


def relerr(a, b):
    """L2 relative error of `a` against the reference `b`."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def maxrel(a, b):
    """Max elementwise error relative to the reference's max magnitude."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))


def report(name, out, ref, route="netplist", exact=False, tol=0.02, abserr=None):
    """One pass/fail line for an op check; returns whether it passed.

    exact: bitwise comparison (permutation/index ops). abserr: absolute-error
    metric for near-zero outputs. Otherwise max relative error vs `tol`.
    """
    out = np.asarray(out, np.float32).reshape(-1)
    ref = np.asarray(ref, np.float32).reshape(-1)
    if exact:
        ok = out.shape == ref.shape and np.array_equal(out, ref)
        print(f"  {name:20s} [{route:9s}] {'OK' if ok else 'MISMATCH'}  exact={ok}")
        return ok
    if abserr is not None:
        err = float(np.abs(out - ref).max())
        ok = err < abserr
        print(f"  {name:20s} [{route:9s}] {'OK' if ok else 'MISMATCH'}  abserr {err:.5f} (<{abserr})")
        return ok
    err = maxrel(out, ref)
    ok = err < tol
    print(f"  {name:20s} [{route:9s}] {'OK' if ok else 'MISMATCH'}  relerr {err:.5f}")
    return ok


# Conditioned random test matrices (geometric spectrum 1..cond).
def spd(n, cond, seed):
    """Symmetric positive definite [n,n] with condition number `cond`."""
    r = np.random.default_rng(seed)
    Q = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (Q * np.geomspace(1.0, cond, n)) @ Q.T


def general(n, cond, seed):
    """General square [n,n] with condition number `cond`."""
    r = np.random.default_rng(seed)
    U = np.linalg.qr(r.standard_normal((n, n)))[0]
    V = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (U * np.geomspace(1.0, cond, n)) @ V.T


def sym(n, cond, seed):
    """Symmetric indefinite [n,n] (mixed-sign spectrum, |eig| spanning 1..cond)."""
    r = np.random.default_rng(seed)
    Q = np.linalg.qr(r.standard_normal((n, n)))[0]
    return (Q * (np.geomspace(1.0, cond, n) * r.choice([-1, 1], n))) @ Q.T
