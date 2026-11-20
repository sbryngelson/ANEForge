#!/usr/bin/env python3
"""Probe for width-offset slices losing their offset or failing to compile on the ANE (issue #310).

On an Apple M5 Pro under macOS 27.2.1 a fused program that takes two different columns of an input
with last-axis-offset slices (`X.slice_by_size([0, k], [n, 1])`) and uses them as matmul operands
either returns the wrong column or fails to compile, while the same MIL is correct on the CPU and GPU
backends. Routing each slice off the last axis (transpose, slice a row, transpose back, so every
last-axis begin is 0) gives the right answer in every cell. This was found through `linalg.lu_pivoted`,
whose unit vectors were such slices of an identity input.

Each cell is two stages on an [n,n] input A with the identity fed as a second input I:

  stage 1 at column k1, one of
    touch       A + 0 * (e_k1 e_k1^T)                  only references the slice
    swap        swap rows k1 and p of A                (p is a fed one-hot)
    swap+elim   the swap, then one LU elimination step at column k1
  stage 2       (M @ e_k2) e_k2^T                      column k2 of the stage-1 result, in place

and is checked against the same arithmetic in numpy. Each cell is built three ways:

  raw       e_k as the last-axis-offset slice, emitted as written (ANEFORGE_ROUTE_WIDTH_SLICES=0)
  default   the same graph through the library's default lowering, which routes such a slice
  routed    e_k routed by hand in the graph (transposed row slice)

`raw` is the hazard; `default` shows whether the lowering removes it. One compile per fresh
process, breaker disabled.

  python bench/slice_offset_probe.py               # the table
  python bench/slice_offset_probe.py --one swap+elim 3 1 raw
"""
from __future__ import annotations

import argparse
import sys

from bench._probe import chip as _chip
from bench._probe import probe_isolated

N = 8
STAGES = ("touch", "swap", "swap+elim")
FORMS = ("raw", "default", "routed")
# (k1, k2): a nonzero k1 followed by a different k2 is what fails; (0, 1) and equal columns are controls
PAIRS = ((0, 1), (1, 0), (3, 1), (2, 6), (3, 3))
TOL = 0.03


def _attempt(stage: str, k1: int, k2: int, form: str) -> str:
  """One compile+dispatch; 'OK', 'C-FAIL <Err>', or 'D-FAIL wrong(<max abs err>)' / 'D-FAIL <Err>'."""
  import os
  os.environ["ANEFORGE_DISABLE_COMPILE_BREAKER"] = "1"
  os.environ["ANEFORGE_ROUTE_WIDTH_SLICES"] = "0" if form == "raw" else "1"
  import warnings

  import numpy as np
  warnings.filterwarnings("ignore")
  import aneforge as af

  n, p = N, (k1 + 2) % N
  rng = np.random.default_rng(5)
  # every |entry| in [1, 2] with a random sign: whichever row the swap brings up, the pivot is not small
  A = (rng.uniform(1.0, 2.0, (n, n)) * rng.choice([-1.0, 1.0], (n, n))).astype(np.float16)
  eye = np.eye(n, dtype=np.float16)

  # numpy reference of the same arithmetic
  M = A.astype(np.float32).copy()
  if stage != "touch": M[[k1, p]] = M[[p, k1]]
  if stage == "swap+elim":
    lk = M[:, k1] / M[k1, k1]; lk[:k1 + 1] = 0.0
    row = M[k1].copy(); row[:k1 + 1] = 0.0
    M = M - np.outer(lk, row)
  ref = np.zeros((n, n), np.float32); ref[:, k2] = M[:, k2]

  At, In = af.input((n, n)), af.input((n, n))
  feeds = [A, eye]

  def e(k):
    if form == "routed": return In.transpose([1, 0]).slice_by_size([k, 0], [1, n]).transpose([1, 0])
    return In.slice_by_size([0, k], [n, 1])
  ek, Mt = e(k1), At
  ekr = ek.transpose([1, 0])
  if stage == "touch":
    Mt = Mt + (ek @ ekr) * 0.0
  else:
    p1h = af.input((n, 1)); feeds.append(eye[:, p:p + 1])
    rk, rp = ekr @ Mt, p1h.transpose([1, 0]) @ Mt
    Mt = Mt + ek @ (rp - rk) + p1h @ (rk - rp)
  if stage == "swap+elim":
    arc = af.input((n, 1)); feeds.append(np.arange(n, dtype=np.float16).reshape(n, 1))
    arcr, kc = arc.transpose([1, 0]), (arc * 0.0).adds(float(k1))
    below = af.select(arc.greater(kc), (arc * 0.0).adds(1.0), arc * 0.0)
    right = af.select(arcr.greater((arcr * 0.0).adds(float(k1))), (arcr * 0.0).adds(1.0), arcr * 0.0)
    Lk = ((Mt @ ek) / ((ekr @ Mt) @ ek)) * below
    Mt = Mt - Lk @ ((ekr @ Mt) * right)
  e2 = e(k2)
  try:
    net = af.compile((Mt @ e2) @ e2.transpose([1, 0]))
  except Exception as ex:                               # noqa: BLE001 - any failure is the datum
    return f"C-FAIL {type(ex).__name__}"
  try:
    out = np.asarray(net(*feeds), np.float32)
  except Exception as ex:                               # noqa: BLE001
    return f"D-FAIL {type(ex).__name__}"
  err = float(np.abs(out - ref).max()) if np.isfinite(out).all() else float("inf")
  return "OK" if err <= TOL else f"D-FAIL wrong({err:.3g})"


def _isolated(stage: str, k1: int, k2: int, form: str) -> str:
  return probe_isolated(["--worker", "--one", stage, str(k1), str(k2), form], __file__)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--one", nargs=4, metavar=("STAGE", "K1", "K2", "FORM"), help="a single cell")
  ap.add_argument("--worker", action="store_true", help="internal: single compile, print result")
  args = ap.parse_args()
  if args.one:
    stage, k1, k2, form = args.one[0], int(args.one[1]), int(args.one[2]), args.one[3]
    if stage not in STAGES or form not in FORMS: ap.error(f"STAGE in {STAGES}, FORM in {FORMS}")
    if args.worker:
      print(_attempt(stage, k1, k2, form))
      return 0
    print(f"{_chip()}  {stage} k1={k1} k2={k2} {form}: {_isolated(stage, k1, k2, form)}")
    return 0

  print(f"{_chip()}  n={N}   (C-FAIL=compile, D-FAIL wrong(e)=max abs error e vs numpy)\n")
  print(f"  {'stage 1':<10} {'k1':>2} {'k2':>2}   " + " ".join(f"{f:<20}" for f in FORMS))
  bad = [0] * len(FORMS)
  for stage in STAGES:
    for k1, k2 in PAIRS:
      res = [_isolated(stage, k1, k2, f) for f in FORMS]
      bad = [b + (r != "OK") for b, r in zip(bad, res)]
      print(f"  {stage:<10} {k1:>2} {k2:>2}   " + " ".join(f"{r:<20}" for r in res), flush=True)
  print("\nnot OK: " + ", ".join(f"{f} {b}" for f, b in zip(FORMS, bad)) + ". See issue #310; please report "
        "the chip, macOS version and commit with the table.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
