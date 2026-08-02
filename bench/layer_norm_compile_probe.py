"""Breaker-controlled probe for the layer_norm ANE compile failure (issue #149).

The naive sweep is confounded: after one compile failure the circuit breaker paces the next compile
(`ANEFORGE_COMPILE_BACKOFF`), so a single early failure makes later shapes look like they fail too.
This runs **one compile per fresh process** with the breaker disabled, which is the only way to get a
per-shape answer, and reports the chip so cells are comparable across generations (per the scoping
note on #115).

  python bench/layer_norm_compile_probe.py                      # the R x D matrix
  python bench/layer_norm_compile_probe.py --scan-d 512         # D-scan at fixed R: finds the
                                                                # scattered cells a matrix misses
  python bench/layer_norm_compile_probe.py --op rms_norm --scan-d 512   # is it layer_norm-specific?
  python bench/layer_norm_compile_probe.py --one 256 3994 --repeat 5    # determinism check

Findings so far, every cell deterministic across repeats: M1 Max and M2 Pro are indistinguishable and
fail a *scattered* set of D at fixed R, non-monotonic in both axes ([512,1024] fails while [512,8192]
passes with 16x the elements). M5 Pro passes all of those and fails only past a clean threshold
between 8192 and 10240. rms_norm and softmax compile where layer_norm does not, so the trigger is
layer_norm's own decomposition rather than reduce-over-last-axis in general.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

RS = (64, 128, 256, 512)
DS = (1024, 2048, 3072, 4096)
SCAN_DS = (512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192, 10240, 12288, 16384)
OPS = ("layer_norm", "rms_norm", "softmax")


def _attempt(R: int, D: int, op: str, seed: int = 13) -> str:
  """One compile+dispatch of `op` at [R, D]; returns 'OK' or 'FAIL <ExcType>'."""
  os.environ["ANEFORGE_DISABLE_COMPILE_BREAKER"] = "1"   # one compile per process: nothing to pace
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  import warnings

  import numpy as np
  warnings.filterwarnings("ignore")
  import aneforge as af

  rng = np.random.default_rng(seed)
  x = af.input((R, D))
  if op == "layer_norm":
    g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
    b = (rng.standard_normal(D).astype(np.float32) * 0.1).astype(np.float16)
    graph = x.layer_norm(g, b)
  elif op == "rms_norm":
    g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
    graph = x.rms_norm(g)
  else:
    graph = x.softmax(-1)
  try:
    net = af.compile(graph)
    out = np.asarray(net(rng.standard_normal((R, D)).astype(np.float16)), np.float32)
    return "OK" if np.isfinite(out).all() else "FAIL nonfinite"
  except Exception as e:                                # noqa: BLE001 - any failure is the datum
    return f"FAIL {type(e).__name__}"


def _isolated(R: int, D: int, op: str) -> str:
  """Run _attempt in a fresh interpreter so no compiler or breaker state carries over."""
  env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
  p = subprocess.run([sys.executable, __file__, "--op", op, "--one", str(R), str(D)],
                     capture_output=True, text=True, env=env)
  for line in reversed(p.stdout.splitlines()):          # stderr carries E5RT noise; parse stdout
    if line.startswith(("OK", "FAIL")):
      return line.strip()
  return "FAIL nolines"


def _chip() -> str:
  try:
    from bench import _machine
    return _machine.fingerprint()["hardware"]["chip"]
  except Exception:                                     # noqa: BLE001 - the probe works without it
    return "unknown chip"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--one", nargs=2, type=int, metavar=("R", "D"))
  ap.add_argument("--scan-d", type=int, metavar="R", help="sweep D at this fixed R")
  ap.add_argument("--op", choices=OPS, default="layer_norm")
  ap.add_argument("--repeat", type=int, default=1)
  args = ap.parse_args()

  if args.one and args.repeat == 1:                     # the subprocess worker
    print(_attempt(args.one[0], args.one[1], args.op))
    return 0

  chip = _chip()

  if args.one:
    R, D = args.one
    res = [_isolated(R, D, args.op) for _ in range(args.repeat)]
    print(f"{chip}  {args.op} [{R},{D}] x{args.repeat}: " + "  ".join(res))
    return 0

  if args.scan_d:
    R = args.scan_d
    print(f"{chip}: {args.op} at R={R}, one fresh process per cell, breaker disabled\n")
    fails = []
    for D in SCAN_DS:
      res = _isolated(R, D, args.op)
      print(f"  [{R:5d},{D:6d}]  {res}")
      if res.startswith("FAIL"):
        fails.append(D)
    print(f"\n{len(SCAN_DS) - len(fails)}/{len(SCAN_DS)} compile."
          + (f" Failing D: {fails}" if fails else " No failures."))
    if len(fails) > 1 and fails != list(SCAN_DS[-len(fails):]):
      print("Failing D is a scattered set rather than a tail, so no threshold in D explains it.")
    return 0

  print(f"{chip}: {args.op} compile, one fresh process per cell, breaker disabled\n")
  print("        " + "".join(f"D={d:<7}" for d in DS))
  grid = {}
  for R in RS:
    cells = []
    for D in DS:
      grid[(R, D)] = res = _isolated(R, D, args.op)
      cells.append(res.split()[0])
    print(f"  R={R:<5}" + "".join(f"{c:<9}" for c in cells))
  ok = sum(1 for v in grid.values() if v == "OK")
  print(f"\n{ok}/{len(grid)} shapes compile. Non-monotonic in both axes means this is not a size "
        f"cap; --scan-d finds the scattered cells a fixed matrix misses. See issue #149.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
