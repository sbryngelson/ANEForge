"""rms_norm compile/execute scan: the single-affine-slot control for issue #149.

`layer_norm` has two affine terms and fails when exactly one of them is live (see
`layer_norm_compile_probe.py`). `rms_norm` has only `gamma`, so it can never be in that asymmetric
state, which is why it is the control: if the asymmetric-affine theory is right, this scan should be
clean everywhere `layer_norm` is not.

A second consumer of `bench/_probe.py`, and roughly the whole point of that helper: the fresh-process
and breaker handling is inherited, so this file is just a graph, a feed, and a table.

  python bench/rms_norm_compile_probe.py            # D-scan at R=512
  python bench/rms_norm_compile_probe.py --scan-d 256 --seed 7
"""
from __future__ import annotations

import argparse
import sys

from bench._probe import OK, chip, probe_cell, probe_isolated, short

SCAN_DS = (512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192, 10240, 12288, 16384)


def _attempt(R: int, D: int, seed: int) -> str:
  import numpy as np

  rng = np.random.default_rng(seed)
  g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
  x_in = rng.standard_normal((R, D)).astype(np.float16)

  def build():
    import aneforge as af
    return af.input((R, D)).rms_norm(g)

  return probe_cell(build, lambda: x_in)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scan-d", type=int, default=512, metavar="R", help="sweep D at this fixed R")
  ap.add_argument("--seed", type=int, default=13)
  ap.add_argument("--one", nargs=2, type=int, metavar=("R", "D"))
  ap.add_argument("--worker", action="store_true", help="internal: single compile, print result")
  args = ap.parse_args()

  if args.worker and args.one:
    print(_attempt(args.one[0], args.one[1], args.seed))
    return 0

  R = args.scan_d
  print(f"{chip()}: rms_norm at R={R} seed={args.seed}, one fresh process per cell, breaker disabled\n")
  fails = []
  for D in SCAN_DS:
    res = probe_isolated(["--worker", "--seed", str(args.seed), "--one", str(R), str(D)], __file__)
    print(f"  [{R:5d},{D:6d}]  {res}")
    if short(res) != OK:
      fails.append(D)
  print(f"\n{len(SCAN_DS) - len(fails)}/{len(SCAN_DS)} pass."
        + (f" Failing D: {fails}" if fails else " No failures, as the single-affine-slot control"
                                                " predicts (#149)."))
  return 0


if __name__ == "__main__":
  sys.exit(main())
