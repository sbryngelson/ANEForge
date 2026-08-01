"""Breaker-controlled probe for the layer_norm ANE compile failure (issue #149).

The naive sweep is confounded: after one compile failure the circuit breaker paces the next compile
(`ANEFORGE_COMPILE_BACKOFF`), so a single early failure makes later shapes look like they fail too.
This runs **one compile per fresh process** with the breaker disabled, which is the only way to get a
per-shape answer.

  python bench/layer_norm_compile_probe.py --one R D      # single attempt, prints OK / FAIL
  python bench/layer_norm_compile_probe.py                # the R x D matrix, one subprocess per cell
  python bench/layer_norm_compile_probe.py --repeat 5 --one 256 3994   # determinism check

Reports the chip so results are comparable across generations, per the scoping note on #115.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

RS = (64, 128, 256, 512)
DS = (1024, 2048, 3072, 4096)


def _attempt(R: int, D: int, seed: int = 13) -> str:
  """One compile+dispatch of layer_norm at [R, D]; returns 'OK' or 'FAIL <ExcType>'."""
  os.environ["ANEFORGE_DISABLE_COMPILE_BREAKER"] = "1"   # one compile per process: nothing to pace
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  import warnings

  import numpy as np
  warnings.filterwarnings("ignore")
  import aneforge as af

  rng = np.random.default_rng(seed)
  g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
  b = (rng.standard_normal(D).astype(np.float32) * 0.1).astype(np.float16)
  try:
    net = af.compile(af.input((R, D)).layer_norm(g, b))
    out = np.asarray(net(rng.standard_normal((R, D)).astype(np.float16)), np.float32)
    return "OK" if np.isfinite(out).all() else "FAIL nonfinite"
  except Exception as e:                                # noqa: BLE001 - any failure is the datum
    return f"FAIL {type(e).__name__}"


def _subprocess_attempt(R: int, D: int) -> str:
  """Run _attempt in a fresh interpreter so no compiler or breaker state carries over."""
  env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
  p = subprocess.run([sys.executable, __file__, "--one", str(R), str(D)],
                     capture_output=True, text=True, env=env)
  for line in reversed(p.stdout.splitlines()):          # stderr carries E5RT noise; parse stdout
    if line.startswith(("OK", "FAIL")):
      return line.strip()
  return "FAIL nolines"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--one", nargs=2, type=int, metavar=("R", "D"))
  ap.add_argument("--repeat", type=int, default=1)
  args = ap.parse_args()

  if args.one and args.repeat == 1:
    print(_attempt(*args.one))
    return 0

  try:
    from bench import _machine
    chip = _machine.fingerprint()["hardware"]["chip"]
  except Exception:                                     # noqa: BLE001 - probe still works without it
    chip = "unknown chip"

  if args.one:
    R, D = args.one
    res = [_subprocess_attempt(R, D) for _ in range(args.repeat)]
    print(f"{chip}  [{R},{D}] x{args.repeat}: " + "  ".join(res))
    return 0

  print(f"{chip}: layer_norm compile, one fresh process per cell, breaker disabled\n")
  print("        " + "".join(f"D={d:<7}" for d in DS))
  grid = {}
  for R in RS:
    cells = []
    for D in DS:
      grid[(R, D)] = res = _subprocess_attempt(R, D)
      cells.append(res.split()[0])
    print(f"  R={R:<5}" + "".join(f"{c:<9}" for c in cells))
  ok = sum(1 for v in grid.values() if v == "OK")
  print(f"\n{ok}/{len(grid)} shapes compile. Non-monotonic in both axes means this is not a "
        f"size cap: see issue #149.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
