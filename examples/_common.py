"""Shared example scaffolding - environment setup plus the few helpers every demo
repeated verbatim. Import this FIRST in an example (before aneforge): importing it
sets KMP_DUPLICATE_LIB_OK and puts the repo root on sys.path.

Deliberately minimal: error metrics, conditioned random test matrices, and the
one-line pass/fail report. Demo logic stays in the demos so each file remains a
self-contained, copy-pasteable example.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import aneforge

import numpy as np

f16 = np.float16


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
