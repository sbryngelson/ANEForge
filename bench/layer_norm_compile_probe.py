"""Probe for the layer_norm ANE compile/execute failure (issue #149).

IMPORTANT (corrects earlier readings in #149): the failure is NOT a shape frontier. It depends on the
gamma/beta AFFINE values, and the "scattered non-monotonic frontier" first reported was the signature
of one specific affine -- `numpy.default_rng(13)` with both gamma and beta non-constant, the config an
earlier version of this probe hardcoded. Verified across three families (M1 Max, M2 Pro, M5 Pro), the
2x2 over affine constant-ness (`--affine-scan`) separates two distinct, family-specific bugs:

  - M1 / M2 (compile stage): layer_norm compile-fails whenever EXACTLY ONE affine term is
    non-constant (one folded, one live). Uniform -- no shape or seed dependence. A constant term
    folds out, so this is a bug in the affine-emission/fold interaction of the lowering.
  - M5 (execute stage): with BOTH terms live and large D (>=~10240) the compiled program fails to
    execute (rc=-1). Seed-dependent at the edge, not a clean threshold.

M1/M2 and M5 are exact opposites on the asymmetric configs (M1/M2 fail, M5 passes). rms_norm and
softmax never fail here: rms_norm has a single affine slot (it can never be asymmetric) and softmax
has none -- so "layer_norm-specific" is really "needs layer_norm's two-term affine".

Method: one compile per FRESH process with the circuit breaker disabled (a compile failure otherwise
paces the next compile via `ANEFORGE_COMPILE_BACKOFF`, making later shapes look failed too). The chip
and seed are reported so cells are comparable across generations (per the #115 scoping note).

  python bench/layer_norm_compile_probe.py --affine-scan 512      # the 2x2 x D table (the useful one)
  python bench/layer_norm_compile_probe.py --scan-d 512           # D-scan, current --gamma/--beta
  python bench/layer_norm_compile_probe.py --one 512 10240 --repeat 5   # determinism at one cell
  python bench/layer_norm_compile_probe.py --scan-d 512 --seed 7  # vary the affine seed
  python bench/layer_norm_compile_probe.py --op rms_norm --scan-d 512   # a single-slot control
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
GAMMAS = ("rand", "ones", "const")     # const = 1.1 (folds); ones = identity scale
BETAS = ("rand", "zeros", "const")     # const = 0.1 (folds); zeros = no shift
# the affine 2x2 (+ rms control) that separates the two bugs; each is (label, gamma, beta)
AFFINE_CONFIGS = (("cc", "const", "const"), ("rz", "rand", "zeros"),
                  ("or", "ones", "rand"), ("rr", "rand", "rand"), ("rms", "rand", "-"))


def _attempt(R: int, D: int, op: str, seed: int, gamma: str, beta: str) -> str:
  """One compile+dispatch at [R,D]; returns 'OK', 'C-FAIL ...' (compile) or 'D-FAIL ...' (dispatch).

  gamma/beta are drawn single-stream (gamma then beta) from `seed` so beta is identical regardless of
  the gamma choice, matching how a real learned affine and the original probe are constructed."""
  os.environ["ANEFORGE_DISABLE_COMPILE_BREAKER"] = "1"   # one compile per process: nothing to pace
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  import warnings

  import numpy as np
  warnings.filterwarnings("ignore")
  import aneforge as af

  rng = np.random.default_rng(seed)
  g_rand = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
  b_rand = (rng.standard_normal(D).astype(np.float32) * 0.1).astype(np.float16)
  g = {"rand": g_rand, "ones": np.ones(D, np.float16), "const": np.full(D, 1.1, np.float16)}[gamma]
  x = af.input((R, D))
  if op == "rms_norm":
    graph = x.rms_norm(g)
  elif op == "softmax":
    graph = x.softmax(-1)
  else:
    b = {"rand": b_rand, "zeros": np.zeros(D, np.float16), "const": np.full(D, 0.1, np.float16)}[beta]
    graph = x.layer_norm(g, b)
  try:
    net = af.compile(graph)
  except Exception as e:                                # noqa: BLE001 - any failure is the datum
    return f"C-FAIL {type(e).__name__}"
  try:
    out = np.asarray(net(rng.standard_normal((R, D)).astype(np.float16)), np.float32)
    return "OK" if np.isfinite(out).all() else "D-FAIL nonfinite"
  except Exception as e:                                # noqa: BLE001
    return f"D-FAIL {type(e).__name__}"


def _isolated(R: int, D: int, op: str, seed: int, gamma: str, beta: str) -> str:
  """Run _attempt in a fresh interpreter so no compiler or breaker state carries over."""
  env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
  p = subprocess.run([sys.executable, __file__, "--worker", "--op", op, "--seed", str(seed),
                      "--gamma", gamma, "--beta", beta, "--one", str(R), str(D)],
                     capture_output=True, text=True, env=env)
  for line in reversed(p.stdout.splitlines()):          # stderr carries E5RT noise; parse stdout
    if line.startswith(("OK", "C-FAIL", "D-FAIL")):
      return line.strip()
  return "C-FAIL nolines"


def _short(res: str) -> str:
  return res.split()[0]


def _chip() -> str:
  try:
    from bench import _machine
    return _machine.fingerprint()["hardware"]["chip"]
  except Exception:                                     # noqa: BLE001 - the probe works without it
    return "unknown chip"


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--one", nargs=2, type=int, metavar=("R", "D"))
  ap.add_argument("--scan-d", type=int, metavar="R", help="sweep D at this fixed R")
  ap.add_argument("--affine-scan", type=int, metavar="R", help="2x2 affine (+ rms) x D table at fixed R")
  ap.add_argument("--op", choices=OPS, default="layer_norm")
  ap.add_argument("--seed", type=int, default=13, help="affine (and input) RNG seed")
  ap.add_argument("--gamma", choices=GAMMAS, default="rand")
  ap.add_argument("--beta", choices=BETAS, default="rand")
  ap.add_argument("--repeat", type=int, default=1)
  ap.add_argument("--worker", action="store_true", help="internal: single compile, print result")
  args = ap.parse_args()

  if args.worker and args.one:                          # the subprocess worker
    print(_attempt(args.one[0], args.one[1], args.op, args.seed, args.gamma, args.beta))
    return 0

  chip = _chip()
  tag = f"{args.op} gamma={args.gamma} beta={args.beta} seed={args.seed}"

  if args.one:
    R, D = args.one
    res = [_isolated(R, D, args.op, args.seed, args.gamma, args.beta) for _ in range(args.repeat)]
    print(f"{chip}  {tag} [{R},{D}] x{args.repeat}: " + "  ".join(res))
    return 0

  if args.affine_scan:
    R = args.affine_scan
    print(f"{chip}  R={R}  seed={args.seed}   (C-FAIL=compile, D-FAIL=dispatch/execute)")
    print("  D       " + " ".join(f"{lbl:<7}" for lbl, _, _ in AFFINE_CONFIGS))
    for D in SCAN_DS:
      row = [_short(_isolated(R, D, "rms_norm" if lbl == "rms" else "layer_norm", args.seed, g, b))
             for lbl, g, b in AFFINE_CONFIGS]
      print(f"  {D:<7} " + " ".join(f"{c:<7}" for c in row))
    print("\ncc=both const (folds -> OK), rz/or=one term live (asymmetric), rr=both live (realistic),"
          " rms=single slot. See issue #149: asymmetric fails on M1/M2 (compile), both-live+large-D"
          " fails on M5 (execute).")
    return 0

  if args.scan_d:
    R = args.scan_d
    print(f"{chip}: {tag} at R={R}, one fresh process per cell, breaker disabled\n")
    fails = []
    for D in SCAN_DS:
      res = _isolated(R, D, args.op, args.seed, args.gamma, args.beta)
      print(f"  [{R:5d},{D:6d}]  {res}")
      if not res.startswith("OK"):
        fails.append(D)
    print(f"\n{len(SCAN_DS) - len(fails)}/{len(SCAN_DS)} pass."
          + (f" Failing D: {fails}" if fails else " No failures.")
          + "  NOTE: this is one affine/seed; the pass/fail set moves with --seed and --gamma/--beta.")
    return 0

  print(f"{chip}: {tag} matrix, one fresh process per cell, breaker disabled\n")
  print("        " + "".join(f"D={d:<7}" for d in DS))
  grid = {}
  for R in RS:
    cells = []
    for D in DS:
      grid[(R, D)] = res = _isolated(R, D, args.op, args.seed, args.gamma, args.beta)
      cells.append(_short(res))
    print(f"  R={R:<5}" + "".join(f"{c:<9}" for c in cells))
  ok = sum(1 for v in grid.values() if v == "OK")
  print(f"\n{ok}/{len(grid)} pass for this affine/seed. The pass/fail pattern is affine-dependent"
        " (see --affine-scan), not a fixed shape frontier. Issue #149.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
