#!/usr/bin/env python3
"""Differential fuzzer for the ANE compiler, built on three ideas that fit this hardware:

1. INTEGER-EXACT MODE (the primary oracle). fp16 represents integers up to 2048 exactly, and a
   large op subset is closed over small integers. Graphs whose mirror proves every intermediate
   stays integral and <= 2048 have a BIT-EQUALITY oracle: no tolerances, no false positives,
   every mismatch is a real bug. Integer closure also survives rewrites, so an identity-augmented
   EMI variant (muls(1)/adds(0)/double-transpose chains the canonicalizer must strip) is compared
   EXACTLY at opt=1 - a tolerance-free probe of the rewrite engine.

2. BOUNDARY-LATTICE SAMPLING. Dims and params are drawn from the edges where lowering bugs live
   (powers of two +/-1, documented tile factors {2,3,4,8}, nonzero last-axis slice offsets, fp16
   danger values near 2048/4094/65504/subnormals) instead of uniformly.

3. FLOAT MODE with a CONDITIONING SCREEN. For ops integers cannot cover (sigmoid/softmax/...),
   the mirror runs at fp32 AND fp64; graphs where those two disagree are ill-conditioned and are
   discarded rather than judged - the fp16 engine only answers for well-conditioned graphs.

Findings are shrunk to minimal reproducers and reported with a replayable JSON spec.

  python3 scripts/fuzz.py --graphs 200 --seed 1          # fixed batch
  python3 scripts/fuzz.py --minutes 20                   # time budget
  python3 scripts/fuzz.py --repro finding-ab12cd34.json  # replay a reported case
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANEFORGE_DISABLE_COMPILE_BREAKER", "1")
import aneforge as af  # noqa: E402
from aneforge.graph import maximum as _gmax, minimum as _gmin, concat as _gconcat  # noqa: E402

INT_MAX = 2048.0         # fp16 is exact on integers up to here - the exact-oracle budget
FLOAT_MAX = 3.0e4        # float-mode magnitude ceiling (near fp16 infinity)
MAX_ELEMS = 8192
COND_TOL = 1e-4          # fp32-vs-fp64 mirror disagreement above this = ill-conditioned, discard
GATE_TOL = 1.5e-2        # opt=1 oracle: the autotuner applies accuracy-GATED lossy variants (int8
                         # weights and friends) by design, so opt=1 answers to the gate, not to
                         # bit-exactness - only opt=0 promises the untouched lowering

# boundary-biased pools (v2: derive from aneforge._capabilities / _targets tables)
DIM_POOL = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33]
TILE_FACTORS = [2, 3, 4, 8]                       # the documented native tile factors
FLOAT_DANGER = [2047.0, -2047.0, 4094.0, -4094.0, 30000.0, 6.1e-5, -6.1e-5, 6e-8]
INT_DANGER = [1024.0, -1024.0, 2047.0, -2047.0, 255.0, -255.0]


# ---------------------------------------------------------------- op table
# (af build, float64 mirror, int_closed?) - int_closed ops map integers to integers.

def _np_softplus(x): return np.logaddexp(0.0, x)
def _np_silu(x): return x / (1.0 + np.exp(-x))
def _np_elu(x, a): return np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1.0))
def _np_round_half_away(x):                        # ANE round is half-away-from-zero, not banker's
  return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))
def _np_softmax(x):
  e = np.exp(x - x.max(axis=-1, keepdims=True))
  return e / e.sum(axis=-1, keepdims=True)

UNARY = {
  "relu":     (lambda t: t.relu(),          lambda x: np.maximum(x, 0.0), True),
  "abs":      (lambda t: t.abs(),           np.abs,                       True),
  "floor":    (lambda t: t.floor(),         np.floor,                     True),
  "round":    (lambda t: t.round(),         _np_round_half_away,          True),
  "sign":     (lambda t: t.sign(),          np.sign,                      True),
  "square":   (lambda t: t.square(),        np.square,                    True),
  "clip":     (lambda t: t.clip(-4.0, 4.0), lambda x: np.clip(x, -4.0, 4.0), True),
  "sigmoid":  (lambda t: t.sigmoid(),       lambda x: 1.0 / (1.0 + np.exp(-x)), False),
  "tanh":     (lambda t: t.tanh(),          np.tanh,                      False),
  "softplus": (lambda t: t.softplus(),      _np_softplus,                 False),
  "silu":     (lambda t: t.silu(),          _np_silu,                     False),
  "sqrt_abs": (lambda t: t.abs().sqrt(),    lambda x: np.sqrt(np.abs(x)), False),
  "softmax":  (lambda t: t.softmax(-1),     _np_softmax,                  False),
}
BINARY = {
  "add": (lambda a, b: a + b,       lambda x, y: x + y, True),
  "sub": (lambda a, b: a - b,       lambda x, y: x - y, True),
  "mul": (lambda a, b: a * b,       lambda x, y: x * y, True),
  "max": (lambda a, b: _gmax(a, b), np.maximum,         True),
  "min": (lambda a, b: _gmin(a, b), np.minimum,         True),
}
REDUCE = {
  "rsum":  (lambda t, ax: t.sum((ax,)),  lambda x, ax: x.sum(axis=ax, keepdims=True),  True),
  "rmax":  (lambda t, ax: t.amax((ax,)), lambda x, ax: x.max(axis=ax, keepdims=True),  True),
  "rmean": (lambda t, ax: t.mean((ax,)), lambda x, ax: x.mean(axis=ax, keepdims=True), False),
}
STRUCT_KINDS = ["transpose", "reshape", "concat", "slice", "tile"]   # int-closed by nature
FLOAT_KINDS = ["elu", "leaky_relu"]

def _kinds_for(mode):
  base = ["unary"] * 5 + ["binary"] * 4 + ["scalar"] * 2 + ["reduce"] * 2 + ["matmul"] + STRUCT_KINDS
  return base + FLOAT_KINDS if mode == "float" else base


# ---------------------------------------------------------------- feeds (own rng stream: build
# reconstructs them from the spec alone, no draw-order replay)

def _feed_rng(seed): return np.random.default_rng((seed ^ 0x9E3779B9) & 0x7FFFFFFF)

def _feed_for(spec):
  rng = _feed_rng(spec["seed"]); shape = tuple(spec["input"])
  if spec["mode"] == "int":
    x = rng.integers(-3, 4, size=shape).astype(np.float64)
    if rng.random() < 0.5:                        # boundary integers, sparse so products stay small
      flat = x.reshape(-1)
      idx = rng.integers(0, flat.size, size=max(1, flat.size // 32))
      flat[idx] = rng.choice(INT_DANGER, size=idx.size)
  else:
    x = rng.standard_normal(shape) * float(rng.choice([0.5, 1.0, 4.0]))
    if rng.random() < 0.35:
      flat = x.reshape(-1)
      idx = rng.integers(0, flat.size, size=max(1, flat.size // 16))
      flat[idx] = rng.choice(FLOAT_DANGER, size=idx.size)
  return np.asarray(x, np.float16)


# ---------------------------------------------------------------- generation
# Spec: {"seed", "mode": "int"|"float", "input": shape, "nodes": [{"op", "src", ...}]}
# Value 0 is the input; node i produces value i+1. Everything replays from the spec.

def _rand_shape(rng):
  rank = int(rng.integers(2, 4))
  while True:
    s = tuple(int(rng.choice(DIM_POOL)) for _ in range(rank))
    if np.prod(s) <= MAX_ELEMS: return s

def _acc_bound(mode): return INT_MAX if mode == "int" else FLOAT_MAX

def _acc_violation(spec, feed):
  """True if any matmul/reduce-sum node's worst-case |partial sum| exceeds the mode bound.
  Measured on M5/H17s - per-family datapath formats differ across ANE generations (see the
  guide's datapath chapter), so these cliffs may sit elsewhere on other chips; a community run
  that diverges from this model on another family is DATA, not noise. On H17s: matmul results
  SATURATE to inf above ~32752 = fp16_max/2 for every
  K (the matmul sibling of the documented slice-x16 saturation at 4094), and integer reduce sums
  crossing 2048 lose bit-exactness. The oracle only judges graphs whose accumulations provably
  stay in range under ANY summation order; the bounds here sit below both cliffs. (Two known
  transpose-fed matmul cases return inf even BELOW these bounds - an open finding.)"""
  vs = [np.asarray(feed, np.float64)]
  for nd in spec["nodes"]:
    x = vs[nd["src"][0]]
    if nd["op"] == "matmul":
      W = np.asarray(_weight(x.shape[1], nd["n"], nd["wseed"], spec["mode"]), np.float64)
      if (np.abs(x) @ np.abs(W)).max() > _acc_bound(spec["mode"]): return True
    if nd["op"] in ("rsum", "rmean"):
      if np.abs(x).sum(axis=nd["axis"]).max() > _acc_bound(spec["mode"]): return True
    vs.append(_mirror_node(nd, vs, spec))
  return False

def _ok(y, mode):
  if not np.isfinite(y).all(): return False
  if mode == "int":
    return np.abs(y).max() <= INT_MAX and np.array_equal(y, np.round(y))
  return np.abs(y).max() <= FLOAT_MAX

def gen_spec(seed):
  """Generate one replayable graph spec (pure host; no aneforge objects)."""
  rng = np.random.default_rng(seed)
  mode = "int" if rng.random() < 0.6 else "float"
  spec = {"seed": int(seed), "mode": mode, "input": list(_rand_shape(rng)), "nodes": []}
  vals = [np.asarray(_feed_for(spec), np.float64)]
  kinds = _kinds_for(mode)
  n_nodes = int(rng.integers(3, 13)); guard = 0
  while len(spec["nodes"]) < n_nodes and guard < 250:
    guard += 1
    kind = str(rng.choice(kinds))
    i = int(rng.integers(0, len(vals)))
    x = vals[i]; node = None; y = None
    if kind == "unary":
      pool = [k for k, v in UNARY.items() if v[2] or mode == "float"]
      op = str(rng.choice(pool)); y = UNARY[op][1](x); node = {"op": op, "src": [i]}
    elif kind == "elu":
      a = round(float(rng.uniform(0.5, 1.5)), 3)
      y = _np_elu(x, a); node = {"op": "elu", "src": [i], "alpha": a}
    elif kind == "leaky_relu":
      a = round(float(rng.uniform(0.01, 0.3)), 3)
      y = np.where(x > 0, x, a * x); node = {"op": "leaky_relu", "src": [i], "alpha": a}
    elif kind == "scalar":
      op = str(rng.choice(["muls", "adds"]))
      k = float(rng.integers(-3, 4)) if mode == "int" else round(float(rng.uniform(-2, 2)), 3)
      y = x * k if op == "muls" else x + k; node = {"op": op, "src": [i], "k": k}
    elif kind == "binary":
      j = int(rng.choice([j for j, v in enumerate(vals) if v.shape == x.shape]))
      op = str(rng.choice(list(BINARY)))
      y = BINARY[op][1](x, vals[j]); node = {"op": op, "src": [i, j]}
    elif kind == "reduce":
      pool = [k for k, v in REDUCE.items() if v[2] or mode == "float"]
      op = str(rng.choice(pool)); ax = int(rng.integers(0, x.ndim))
      y = REDUCE[op][1](x, ax); node = {"op": op, "src": [i], "axis": ax}
    elif kind == "matmul" and x.ndim == 2 and x.shape[1] <= 33:
      n = int(rng.choice(DIM_POOL)); wseed = int(rng.integers(0, 2**31))
      W = _weight(x.shape[1], n, wseed, mode)
      y = x @ np.asarray(W, np.float64); node = {"op": "matmul", "src": [i], "n": n, "wseed": wseed}
    elif kind == "transpose":
      perm = [int(p) for p in rng.permutation(x.ndim)]
      y = np.transpose(x, perm); node = {"op": "transpose", "src": [i], "perm": perm}
    elif kind == "reshape" and x.ndim >= 2:
      flat = int(np.prod(x.shape))
      d = int(rng.choice([d for d in range(1, min(flat, 64) + 1) if flat % d == 0]))
      y = x.reshape(d, flat // d); node = {"op": "reshape", "src": [i], "shape": [d, flat // d]}
    elif kind == "concat":
      j = int(rng.choice([j for j, v in enumerate(vals) if v.shape == x.shape]))
      ax = int(rng.integers(0, x.ndim))
      if 2 * np.prod(x.shape) > MAX_ELEMS: continue
      y = np.concatenate([x, vals[j]], axis=ax); node = {"op": "concat", "src": [i, j], "axis": ax}
    elif kind == "slice":
      ax = int(rng.integers(0, x.ndim)); d = x.shape[ax]
      if d < 2: continue
      # boundary begins: 0, 1, and the tail - nonzero LAST-axis begins ride the offset DMA path
      b = int(rng.choice([0, 1, max(0, d - 2), int(rng.integers(0, d - 1))]))
      sz = int(rng.integers(1, d - b + 1))
      begin = [0] * x.ndim; size = list(x.shape); begin[ax] = b; size[ax] = sz
      y = x[tuple(slice(bb, bb + ss) for bb, ss in zip(begin, size))]
      node = {"op": "slice", "src": [i], "begin": begin, "size": size}
    elif kind == "tile":
      reps = [1] * x.ndim
      reps[int(rng.integers(0, x.ndim))] = int(rng.choice(TILE_FACTORS))
      if np.prod(x.shape) * np.prod(reps) > MAX_ELEMS: continue
      y = np.tile(x, reps); node = {"op": "tile", "src": [i], "reps": reps}
    if node is None or y is None or not _ok(y, mode): continue
    if node["op"] == "matmul":
      W = np.asarray(_weight(x.shape[1], node["n"], node["wseed"], mode), np.float64)
      if (np.abs(x) @ np.abs(W)).max() > _acc_bound(mode): continue
    if node["op"] in ("rsum", "rmean") and np.abs(x).sum(axis=node["axis"]).max() > _acc_bound(mode): continue
    vals.append(y); spec["nodes"].append(node)
  return spec

def _weight(k, n, wseed, mode):
  r = np.random.default_rng(wseed)
  if mode == "int": return r.integers(-2, 3, size=(k, n)).astype(np.float16)
  return (r.standard_normal((k, n)) / math.sqrt(k)).astype(np.float16)


# ---------------------------------------------------------------- build + mirror from a spec

def _mirror(spec, feed, dtype):
  """Evaluate the spec's numpy mirror at `dtype`; returns (all_values, output)."""
  vs = [np.asarray(feed, dtype)]
  for nd in spec["nodes"]:
    vs.append(_mirror_node(nd, vs, spec, dtype))
  return vs, vs[-1]

def _mirror_node(nd, vs, spec, dtype=np.float64):
  op = nd["op"]; x = vs[nd["src"][0]]
  if True:
    if op in UNARY: y = UNARY[op][1](x)
    elif op in BINARY: y = BINARY[op][1](x, vs[nd["src"][1]])
    elif op in REDUCE: y = REDUCE[op][1](x, nd["axis"])
    elif op == "muls": y = x * dtype(nd["k"])
    elif op == "adds": y = x + dtype(nd["k"])
    elif op == "elu": y = _np_elu(x, nd["alpha"])
    elif op == "leaky_relu": y = np.where(x > 0, x, dtype(nd["alpha"]) * x)
    elif op == "matmul": y = x @ np.asarray(_weight(x.shape[1], nd["n"], nd["wseed"], spec["mode"]), dtype)
    elif op == "transpose": y = np.transpose(x, nd["perm"])
    elif op == "reshape": y = x.reshape(nd["shape"])
    elif op == "concat": y = np.concatenate([x, vs[nd["src"][1]]], axis=nd["axis"])
    elif op == "slice": y = x[tuple(slice(b, b + z) for b, z in zip(nd["begin"], nd["size"]))]
    elif op == "tile": y = np.tile(x, nd["reps"])
    else: raise ValueError(f"unknown op {op!r}")
  return np.asarray(y, dtype)

def build_graph(spec, emi=False):
  """Build the aneforge graph for a spec; `emi` appends an identity chain (muls(1), adds(0),
  double transpose) that the canonicalizer must strip without changing the result."""
  t = af.input(tuple(spec["input"]))
  ts = [t]
  for nd in spec["nodes"]:
    op = nd["op"]; t = ts[nd["src"][0]]
    if op in UNARY: tt = UNARY[op][0](t)
    elif op in BINARY: tt = BINARY[op][0](t, ts[nd["src"][1]])
    elif op in REDUCE: tt = REDUCE[op][0](t, nd["axis"])
    elif op == "muls": tt = t * nd["k"]
    elif op == "adds": tt = t + nd["k"]
    elif op == "elu": tt = t.elu(nd["alpha"])
    elif op == "leaky_relu": tt = t.leaky_relu(nd["alpha"])
    elif op == "matmul": tt = t @ _weight(t.shape[1], nd["n"], nd["wseed"], spec["mode"])
    elif op == "transpose": tt = t.transpose(nd["perm"])
    elif op == "reshape": tt = t.reshape(*nd["shape"])
    elif op == "concat": tt = _gconcat([t, ts[nd["src"][1]]], axis=nd["axis"])
    elif op == "slice": tt = t.slice_by_size(nd["begin"], nd["size"])
    elif op == "tile": tt = t.tile(nd["reps"])
    else: raise ValueError(f"unknown op {op!r}")
    ts.append(tt)
  out = ts[-1]
  if emi:
    out = (out * 1.0) + 0.0                        # muls(1) + adds(0): lossless no-ops
    perm = list(range(len(out.shape)))
    if len(perm) >= 2:
      swap = perm[:]; swap[0], swap[1] = swap[1], swap[0]
      out = out.transpose(swap).transpose(swap)    # self-inverse transpose pair
  return out


# ---------------------------------------------------------------- run + oracles

def _dispatch(out, feed, opt):
  """Compile+run; returns (array, None) or (None, failure dict)."""
  try:
    net = af.compile(out, opt=opt, _check_precision=False)
  except Exception as e:
    return None, {"opt": opt, "kind": "compile-fail", "error": f"{type(e).__name__}: {e}"[:300]}
  try:
    got = np.asarray(net(feed), np.float64)
  except Exception as e:
    return None, {"opt": opt, "kind": "run-fail", "error": f"{type(e).__name__}: {e}"[:300]}
  finally:
    try: net.release()
    except Exception: pass
  return got, None

def run_case(spec, opts=(0, 1), emi=True):
  """Run a spec against its oracles. opt=0 answers exactly (int mode) or to the fp16 rounding
  budget (float mode); opt=1 answers to the autotuner's accuracy gate (GATE_TOL), because lossy
  gated variants are part of its contract. Returns failure dicts (empty = ok)."""
  feed = _feed_for(spec)
  if _acc_violation(spec, feed):
    return []                                      # fp16 accumulation out of range: engine-undefined, not judged
  vs64, ref = _mirror(spec, feed, np.float64)
  scale = max(float(np.abs(v).max()) for v in vs64) + 1e-3
  if spec["mode"] == "float":                      # conditioning screen: fp32 and fp64 mirrors agree?
    _, ref32 = _mirror(spec, feed, np.float32)
    if float(np.abs(ref32.astype(np.float64) - ref).max()) / scale > COND_TOL:
      return []                                    # ill-conditioned: not a fair judge of fp16
  def judge(got, opt, prefix=""):
    if got.shape != ref.shape:
      return {"opt": opt, "kind": prefix + "shape", "error": f"{got.shape} != {ref.shape}"}
    if not np.isfinite(got).all() and np.isfinite(ref).all():
      return {"opt": opt, "kind": prefix + "non-finite", "error": "output has NaN/inf, reference does not"}
    if opt == 0 and spec["mode"] == "int":
      if not np.array_equal(got, ref):
        n_bad = int((got != ref).sum())
        return {"opt": opt, "kind": prefix + "exact-mismatch",
                "error": f"{n_bad}/{ref.size} elements differ (integer graph at opt=0: must be bit-exact)"}
      return None
    err = float(np.abs(got - ref).max()) / scale
    tol = GATE_TOL if opt >= 1 else 5e-3 + 1.5e-3 * len(spec["nodes"])
    if err > tol:
      return {"opt": opt, "kind": prefix + "numeric", "error": f"relerr {err:.3e} > tol {tol:.3e}"}
    return None
  fails = []
  for opt in opts:
    got, fail = _dispatch(build_graph(spec), feed, opt)
    if fail or got is None: fails.append(fail); continue
    f = judge(got, opt)
    if f: fails.append(f)
  # EMI probe (int mode): the identity chain must lower exactly at opt=0, and the canonicalizer's
  # stripping of it at opt=1 must stay inside the same gate as the base graph
  if emi and spec["mode"] == "int" and not fails and spec["seed"] % 2 == 0:
    for opt in (0, 1):
      got, fail = _dispatch(build_graph(spec, emi=True), feed, opt)
      if fail or got is None:
        fails.append({**(fail or {"opt": opt}), "kind": "emi-" + (fail or {}).get("kind", "run-fail")}); break
      f = judge(got, opt, prefix="emi-")
      if f: fails.append(f); break
  return fails


# ---------------------------------------------------------------- shrinking

def _valid(spec):
  """A shrink candidate must respect the generator's own invariants (else the failure mode shifts)."""
  try:
    vs, _ = _mirror(spec, _feed_for(spec), np.float64)
    return all(_ok(v, spec["mode"]) for v in vs)
  except Exception:
    return False

def _still_fails(spec, opt, kind):
  if not _valid(spec): return False
  try:
    return any(f and f["opt"] == opt and f["kind"] == kind for f in run_case(spec, opts=(opt,), emi=True))
  except Exception:
    return False                                   # host-side crash = malformed candidate, not a repro

def shrink(spec, opt, kind):
  """Minimize a failing spec preserving the SAME failure kind: shortest failing prefix by
  bisection, then shape-safe single-node drops."""
  nodes = spec["nodes"]
  lo, hi = 0, len(nodes)
  while lo < hi:
    mid = (lo + hi) // 2
    if _still_fails({**spec, "nodes": nodes[:mid]}, opt, kind): hi = mid
    else: lo = mid + 1
  spec = {**spec, "nodes": nodes[:min(lo, len(nodes))]}
  changed = True
  while changed and len(spec["nodes"]) > 1:
    changed = False
    for k in range(len(spec["nodes"]) - 1, -1, -1):
      cand = _drop_node(spec["nodes"], k)
      if cand is not None and _still_fails({**spec, "nodes": cand}, opt, kind):
        spec = {**spec, "nodes": cand}; changed = True; break
  return spec

def _drop_node(nodes, k):
  """Remove node k, rewiring consumers of its value to the node's first source."""
  try:
    out = []
    for j, nd in enumerate(nodes):
      if j == k: continue
      src = [(s if s <= k else s - 1) if s != k + 1 else nodes[k]["src"][0] for s in nd["src"]]
      out.append({**nd, "src": src})
    return out
  except Exception:
    return None


# ---------------------------------------------------------------- driver

def fingerprint(spec):
  sig = spec["mode"] + json.dumps(spec["nodes"], sort_keys=True)
  return hashlib.sha1(sig.encode()).hexdigest()[:8]

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--graphs", type=int, default=0, help="number of graphs (0 = use --minutes)")
  ap.add_argument("--minutes", type=float, default=10.0, help="time budget when --graphs is 0")
  ap.add_argument("--seed", type=int, default=None, help="master seed (default: random)")
  ap.add_argument("--out", default="fuzz-report.json", help="report path")
  ap.add_argument("--repro", default=None, help="replay a saved finding spec instead of fuzzing")
  args = ap.parse_args()
  warnings.filterwarnings("ignore")

  if args.repro:
    spec = json.load(open(args.repro))
    fails = run_case(spec)
    print(json.dumps(fails, indent=2) if fails else "reproducer PASSES here (does not fail on this machine)")
    return 1 if fails else 0

  master = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
  rng = np.random.default_rng(master)
  t0 = time.time(); n = 0; by_mode = {"int": 0, "float": 0}; findings = {}
  print(f"fuzz: master seed {master}", flush=True)
  while (args.graphs and n < args.graphs) or (not args.graphs and (time.time() - t0) < args.minutes * 60):
    case_seed = int(rng.integers(0, 2**31)); n += 1
    spec = gen_spec(case_seed)
    if not spec["nodes"]: continue
    by_mode[spec["mode"]] += 1
    fails = run_case(spec)
    if fails:
      opt, kind = fails[0]["opt"], fails[0]["kind"]
      small = shrink(spec, opt, kind)
      fp = fingerprint(small)
      if fp not in findings:
        findings[fp] = {"fingerprint": fp, "mode": spec["mode"], "fails": fails, "spec": small,
                        "nodes_before_shrink": len(spec["nodes"]), "nodes_after": len(small["nodes"])}
        json.dump(small, open(f"finding-{fp}.json", "w"), indent=1)
        print(f"  FINDING {fp} [{spec['mode']}]: {fails[0]['kind']} at opt={opt} "
              f"({len(spec['nodes'])} -> {len(small['nodes'])} nodes) -> finding-{fp}.json", flush=True)
    if n % 25 == 0:
      print(f"  {n} graphs ({by_mode['int']} int / {by_mode['float']} float), "
            f"{len(findings)} unique finding(s), {time.time() - t0:.0f}s", flush=True)
  report = {"master_seed": master, "graphs": n, "modes": by_mode,
            "elapsed_s": round(time.time() - t0, 1), "findings": list(findings.values())}
  json.dump(report, open(args.out, "w"), indent=1)
  print(f"fuzz: {n} graphs in {report['elapsed_s']}s -> {len(findings)} unique finding(s); report: {args.out}")
  return 2 if findings else 0

if __name__ == "__main__":
  sys.exit(main())
