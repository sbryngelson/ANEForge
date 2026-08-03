"""Repro for #153: at opt=1 the cost-model picks the lossy int8 variant and the output saturates to inf.

`opt=1` selects on predicted cost with no measurement (`_compile._compile_opt`), so the lossy int8
config wins whenever it is cheaper, with nothing comparing it against a baseline. On a matmul whose
true sums stay inside fp16 range, opt=0 is finite and opt=1 is not.

  python bench/opt1_int8_saturation_repro.py            # default [31,15]@[15,16]
  python bench/opt1_int8_saturation_repro.py --trials 50 # how absolute it is over random draws
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

warnings.filterwarnings("ignore")

import aneforge as af

CARRIER = 30000.0        # the magnitude that trips the int8 variant's ceiling
DANGER_FRAC = 1 / 16     # matches the fuzzer's FLOAT_DANGER density
BOUND = 3e4              # keep only cases whose true product is inside fp16 range


def _case(rng, M: int, K: int, N: int):
  """A bounded matmul carrying ~30000 magnitudes; returns (x, W, true_bound) or None if unbounded."""
  x = rng.standard_normal((M, K)).astype(np.float32)
  x[rng.random((M, K)) < DANGER_FRAC] = CARRIER
  x = x.astype(np.float16)
  W = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float16)
  bound = float((np.abs(x.astype(np.float32)) @ np.abs(W.astype(np.float32))).max())
  return (x, W, bound) if bound < BOUND else None


def _nonfinite(x, W, opt: int) -> tuple[int, int]:
  out = np.asarray(af.compile(af.input(x.shape) @ W, opt=opt)(x), np.float32)
  return int((~np.isfinite(out)).sum()), out.size


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--shape", nargs=3, type=int, default=[31, 15, 16], metavar=("M", "K", "N"))
  ap.add_argument("--trials", type=int, default=1)
  ap.add_argument("--seed", type=int, default=1)
  args = ap.parse_args()
  M, K, N = args.shape
  rng = np.random.default_rng(args.seed)

  if args.trials == 1:
    c = None
    while c is None:
      c = _case(rng, M, K, N)
    x, W, bound = c
    print(f"[{M},{K}] @ [{K},{N}], true |x|@|W| max = {bound:.1f} (fp16 cliff is ~32760)")
    for opt in (0, 1):
      bad, total = _nonfinite(x, W, opt)
      print(f"  opt={opt}: {bad}/{total} non-finite")
    return 0

  tally = {0: [0, 0], 1: [0, 0]}     # opt -> [cases with any inf, cases kept]
  for _ in range(args.trials):
    c = _case(rng, M, K, N)
    if c is None: continue
    x, W, _ = c
    for opt in (0, 1):
      bad, _n = _nonfinite(x, W, opt)
      tally[opt][1] += 1
      tally[opt][0] += int(bad > 0)
  for opt in (0, 1):
    bad, kept = tally[opt]
    print(f"  opt={opt}: {bad}/{kept} bounded cases produced inf")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
