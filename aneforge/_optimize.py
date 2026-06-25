"""aneforge graph autotuner: a deterministic measurement-cache search over the metamorphic-proven-safe variant space."""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from ._compile import compile as _raw_compile, _topo as _raw_topo
from ._cost import estimate, precision_risk
from .graph import Tensor

# Variant-correctness tolerance vs the opt=0 baseline (fp16-noise: lossless passes, lossy int8 rejected).
_ACCURACY_TOL = 5e-3
# A lossy variant must beat the baseline by this factor to be chosen (no measurement-noise "wins").
_MIN_LOSSY_SPEEDUP = 1.10


# Per-root memo of _topo post-order. The graph is never mutated during a tune (compile only
# writes node._name, which topo ignores), so memoizing only avoids recompute. Identity-checked
# on read so a reused id can never return a stale order; capped to bound it.
_TOPO_MEMO: dict[int, tuple] = {}


def _topo(out):
  hit = _TOPO_MEMO.get(id(out))
  if hit is not None and hit[0] is out: return hit[1]
  order = _raw_topo(out)
  if len(_TOPO_MEMO) >= 256: _TOPO_MEMO.clear()
  _TOPO_MEMO[id(out)] = (out, order)
  return order


# deterministic graph-structure hash
def _graph_key(out, input_shapes) -> str:
  """Deterministic hash of the graph STRUCTURE + input shapes (not weight values)."""
  order = _topo(out)
  pos = {id(t): i for i, t in enumerate(order)}
  h = hashlib.sha256()
  for t in order:
    h.update(t.op.encode())
    h.update(repr(tuple(t.shape)).encode())
    h.update(repr([pos[id(s)] for s in t.srcs]).encode())
    for k in sorted(t.attrs):
      v = t.attrs[k]
      if isinstance(v, np.ndarray):
        h.update(f"{k}:arr{v.shape}:{v.dtype}".encode())
      elif k != "idx":   # idx is an input-ordering detail, already in pos
        h.update(f"{k}:{v}".encode())
  h.update(repr([tuple(s) for s in input_shapes]).encode())
  return h.hexdigest()[:32]


# persistent measurement cache
def _cache_dir() -> Path:
  env = os.environ.get("ANEFORGE_CACHE_DIR")
  if env:
    d = Path(env)
  else:
    # repo-local cache (falls back to ~/.cache/aneforge if not writable).
    repo = Path(__file__).resolve().parents[1]
    d = repo / ".aneforge_cache"
  try:
    d.mkdir(parents=True, exist_ok=True)
    return d
  except Exception:
    d = Path.home() / ".cache" / "aneforge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path() -> Path:
  return _cache_dir() / "autotune.json"


def _load_cache() -> dict:
  p = _cache_path()
  if p.exists():
    try: return json.loads(p.read_text())
    except Exception: return {}
  return {}


def _save_cache(cache: dict) -> None:
  try:
    _cache_path().write_text(json.dumps(cache, indent=2, sort_keys=True))
  except Exception:
    pass


# attention query-tile autotune
# The query-tile count sets how the [H,S,T] score is fissioned. The S-based heuristic is not
# the per-shape optimum, so measure the best count once per (chip,S,T,heads,head_dim) and cache.
_TILE_MEMO: dict = {}
_TILE_CANDIDATES = (1, 2, 3, 4, 5, 6, 8)


def _heuristic_tiles(S: int) -> int:
  return max(1, (S + 256) // 512)


def _time_attention_core(S: int, T: int, n_heads: int, dh: int, n_tiles: int, reps: int = 20) -> float:
  """Median wall time (s) of the query-tiled attention core on the ANE."""
  from . import graph as g
  qh, kh, vh = g.input((n_heads, S, dh)), g.input((n_heads, T, dh)), g.input((n_heads, T, dh))
  kt = kh.transpose([0, 2, 1]); scale = 1.0 / dh ** 0.5
  if n_tiles == 1:
    o = ((qh @ kt) * scale).softmax(-1) @ vh
  else:
    tile = -(-S // n_tiles); parts = []
    for start in range(0, S, tile):
      n = min(tile, S - start)
      qt = qh.slice_by_size([0, start, 0], [n_heads, n, dh])
      parts.append(((qt @ kt) * scale).softmax(-1) @ vh)
    o = g.concat(parts, axis=1)
  prog = _raw_compile(o)
  rng = np.random.default_rng(0)
  args = [rng.standard_normal((n_heads, S, dh)).astype(np.float32) * 0.1,
          rng.standard_normal((n_heads, T, dh)).astype(np.float32) * 0.1,
          rng.standard_normal((n_heads, T, dh)).astype(np.float32) * 0.1]
  try:
    for _ in range(5):
      prog(*args)
    ts = []
    for _ in range(reps):
      t0 = time.perf_counter(); prog(*args); ts.append(time.perf_counter() - t0)
    return sorted(ts)[reps // 2]
  finally:
    try: prog.release()
    except Exception: pass


def attention_tiles(S: int, n_heads: int, dh: int, T: int | None = None,
                    tune: bool = False, candidates=_TILE_CANDIDATES) -> int:
  """Query-tile count for an attention score: a chip+shape-cached tuned value, else the S-based heuristic."""
  T = S if T is None else T
  from ._targets import _cpu_brand
  key = f"attn_tiles:{_cpu_brand() or 'unknown'}:{S}:{T}:{n_heads}:{dh}"
  if key in _TILE_MEMO: return _TILE_MEMO[key]
  cache = _load_cache()
  if key in cache: return _TILE_MEMO.setdefault(key, int(cache[key]))
  want = tune or os.environ.get("ANEFORGE_TUNE_ATTENTION", "").lower() in ("1", "true", "yes")
  if not want: return _heuristic_tiles(S)
  best_n, best_t = _heuristic_tiles(S), float("inf")
  for nt in candidates:
    try: dt = _time_attention_core(S, T, n_heads, dh, nt)
    except Exception: continue
    if dt < best_t: best_t, best_n = dt, nt
  cache[key] = best_n; _save_cache(cache); _TILE_MEMO[key] = best_n
  return best_n


def tune_attention(S: int, n_heads: int, dh: int, T: int | None = None,
                   candidates=_TILE_CANDIDATES) -> int:
  """Measure and cache the best query-tile count for this attention shape on this chip."""
  return attention_tiles(S, n_heads, dh, T=T, tune=True, candidates=candidates)


# variant space (legal == metamorphic-proven-safe)
def _has_weights(out) -> bool:
  """True if the graph carries any int8-eligible weight (matmul `wt` or conv `weight`)."""
  for t in _topo(out):
    if t.op == "matmul" and isinstance(t.attrs.get("wt"), np.ndarray): return True
    if t.op == "conv" and isinstance(t.attrs.get("weight"), np.ndarray): return True
  return False


def _int8_candidates(out):
  """Per-weight int8 candidates (topo_index, weight_elems), largest-weight-first."""
  order = _topo(out)
  cands = []
  for i, t in enumerate(order):
    if t.op == "matmul" and isinstance(t.attrs.get("wt"), np.ndarray):
      cands.append((i, int(t.attrs["wt"].size)))
    elif t.op == "conv" and isinstance(t.attrs.get("weight"), np.ndarray):
      cands.append((i, int(t.attrs["weight"].size)))
  cands.sort(key=lambda c: c[1], reverse=True)
  return cands


def _sdpa_ids(out):
  """Stable per-sdpa-node keys: index in topo order (deterministic, cache-safe)."""
  order = _topo(out)
  return [i for i, t in enumerate(order) if t.op == "sdpa"]


def _route_ids(out):
  """Topo indices of every ROUTE-BEARING node (bridge op with a validated fused lowering)."""
  from ._rewrite import _BRIDGE_DECOMPOSERS
  order = _topo(out)
  # a causal/masked sdpa must NOT be route-flipped: the fused decomposition is unmasked.
  return [i for i, t in enumerate(order)
          if t.op in _BRIDGE_DECOMPOSERS
          and not (t.op == "sdpa" and (t.attrs.get("causal") or t.attrs.get("masked")))]


def _constfold_candidates(out):
  """Topo indices of nodes whose whole source cone is constant (foldable to one const_array)."""
  from ._rewrite import _const_subgraph
  return [i for i, t in enumerate(_topo(out))
          if t.op not in ("const_array", "input") and _const_subgraph(t) is not None]


def _scalarchain_candidates(out):
  """Topo indices of muls/adds chain heads (muls-over-muls, adds-over-adds)."""
  return [i for i, t in enumerate(_topo(out))
          if t.op in ("muls", "adds") and t.srcs[0].op == t.op]


def _apply_variant(out, cfg):
  """Materialize the rewritten DAG for variant `cfg` in ONE graph_rewrite pass; the global int8 flag is returned separately."""
  from ._rewrite import (Rule, graph_rewrite, _BRIDGE_DECOMPOSERS, _reduce_sum_as_matmul,
                         _b_fold_muls, _b_fold_adds, _b_const_fold)
  order = _topo(out)

  def ids(key, ok):
    return {id(order[i]) for i in cfg.get(key, ()) if 0 <= i < len(order) and ok(order[i])}

  sel: set[int] = set()
  rules: list = []

  rs = ids("rs_matmul", lambda t: t.op == "reduce_sum" and len(t.attrs.get("axes", ())) == 1)
  if rs:
    sel |= rs
    rules.append(Rule("rsum_mm", "numeric", lambda t: t.op == "reduce_sum", _reduce_sum_as_matmul))

  dec = ids("decomp", lambda t: t.op in _BRIDGE_DECOMPOSERS)
  if dec:
    sel |= dec
    rules.append(Rule("bridge", "lossless", lambda t: t.op in _BRIDGE_DECOMPOSERS,
                      lambda t: _BRIDGE_DECOMPOSERS[t.op](t)))

  i8 = ids("int8_nodes", lambda t: t.op in ("matmul", "conv"))
  if i8:
    sel |= i8
    rules.append(Rule("int8", "numeric", lambda t: t.op in ("matmul", "conv"),
                      lambda t: Tensor(t.shape, t.op, t.srcs, {**t.attrs, "int8": True})))

  sf = ids("scalarfold", lambda t: t.op in ("muls", "adds") and t.srcs[0].op == t.op)
  if sf:
    sel |= sf
    rules += [
      Rule("muls_chain", "numeric", lambda t: t.op == "muls" and t.srcs[0].op == "muls", _b_fold_muls),
      Rule("adds_chain", "numeric", lambda t: t.op == "adds" and t.srcs[0].op == "adds", _b_fold_adds),
    ]

  cf = ids("constfold", lambda t: t.op not in ("const_array", "input"))
  if cf:
    sel |= cf
    rules.append(Rule("const_fold", "numeric",
                      lambda t: t.op not in ("const_array", "input"), _b_const_fold))

  new_out = graph_rewrite(out, rules, select=sel) if rules else out
  return new_out, bool(cfg.get("int8", False))


def _route_configs(routes) -> list:
  """Coordinate-descent route enumeration: baseline + each single flip + all-decomposed = K+1 configs (linear, not 2^K)."""
  configs = [{"int8": False, "decomp": [], "lossy": False}]   # opt=0 baseline (all native)
  for i in routes:
    configs.append({"int8": False, "decomp": [i], "lossy": False})
  if len(routes) > 1:
    configs.append({"int8": False, "decomp": list(routes), "lossy": False})
  return configs


def _variants(out):
  """Legal variant configs: route flips (lossless) + global int8 / const-fold / scalar-fold (lossy)."""
  configs = _route_configs(_route_ids(out))
  if _has_weights(out):
    configs.append({"int8": True, "decomp": [], "lossy": True})   # PRIMARY: global int8
  cf = _constfold_candidates(out)
  if cf:
    configs.append({"int8": False, "decomp": [], "constfold": cf, "lossy": True})
  sf = _scalarchain_candidates(out)
  if sf:
    configs.append({"int8": False, "decomp": [], "scalarfold": sf, "lossy": True})
  return configs


def _route_variants(out):
  """The LOSSLESS-ONLY variant set for the default route pass: baseline + route-bearing flips."""
  return _route_configs(_route_ids(out))


def _compile_routes(out, int8: bool, build_dir=None):
  """The default compile pass: cost-model-driven, lossless route optimization (per node, cheaper of native vs fused)."""
  cfgs = _route_variants(out)
  if len(cfgs) == 1:                                          # no removable cut: opt=0 path
    return _raw_compile(out, int8=int8, build_dir=build_dir, opt=0, _check_precision=False)
  best = min(cfgs, key=lambda c: _estimate_variant(out, c))
  variant_out, _ = _apply_variant(out, best)
  return _raw_compile(variant_out, int8=int8, build_dir=build_dir, opt=0, _check_precision=False)


def _config_label(cfg: dict) -> str:
  decomp = cfg.get("decomp", ())
  parts = [f"int8={cfg.get('int8', False)}",
           f"decomp={list(decomp)}" if decomp else "route=native"]
  for key in ("int8_nodes", "rs_matmul", "paired_sub"):
    if cfg.get(key): parts.append(f"{key}={list(cfg[key])}")
  return "+".join(parts)


# measurement
def measure(out, inputs, cfg, baseline_out=None, reps: int = 20, warmup: int = 5,
            tol: float = _ACCURACY_TOL):
  """Compile under `cfg`, validate vs `baseline_out` within tol, return (min latency us, output array)."""
  arrs = [np.asarray(a, np.float16) for a in inputs]
  try:
    variant_out, int8 = _apply_variant(out, cfg)
    net = _raw_compile(variant_out, int8=int8, opt=0, _check_precision=False)
  except Exception:
    return float("inf"), None
  try:
    res = None
    for _ in range(warmup):
      res = net(*arrs)
    # correctness gate vs baseline
    if baseline_out is not None:
      ref = np.asarray(baseline_out, np.float32)
      cur = np.asarray(res, np.float32)
      if cur.shape != ref.shape: return float("inf"), None
      denom = float(np.abs(ref).max()) + 1e-6
      relerr = float(np.abs(cur - ref).max() / denom)
      if relerr > tol: return float("inf"), None
    best = float("inf")
    for _ in range(reps):
      t0 = time.perf_counter()
      net(*arrs)
      dt = (time.perf_counter() - t0) * 1e6
      best = min(best, dt)
    return best, np.asarray(res, np.float32)
  finally:
    try: net.release()
    except Exception: pass


# greedy per-weight int8 selection (coordinate descent, not 2^K)
def _greedy_int8(out, inputs, baseline_out, baseline_us, *, reps, atol,
                 min_lossy_speedup, budget, verbose=False):
  """Greedily grow the per-weight int8 set, one weight at a time (coordinate descent, capped by `budget`)."""
  cands = _int8_candidates(out)
  if not cands: return (), float("inf"), 0
  chosen: list[int] = []
  # current best = the fp16 baseline; any int8 add must beat it.
  best_us = baseline_us
  n_measured = 0
  remaining = [idx for idx, _ in cands]   # benefit-ordered topo indices

  improved = True
  while improved and remaining and n_measured < budget:
    improved = False
    best_add, best_add_us = None, best_us
    for idx in list(remaining):
      if n_measured >= budget: break
      cfg = {"int8": False, "decomp": [], "int8_nodes": tuple(chosen) + (idx,),
             "lossy": True}
      us, _ = measure(out, inputs, cfg, baseline_out=baseline_out, reps=reps, tol=atol)
      n_measured += 1
      ok = us != float("inf")
      if verbose:
        tag = (f"{us:.0f}us" if ok else "INCORRECT/FAIL(atol)")
        print(f"[greedy] try int8 node {idx} (set {chosen + [idx]}): {tag}")
      # accept only if accurate AND faster than the best so far (strict improvement).
      if ok and us < best_add_us: best_add, best_add_us = idx, us
    if best_add is not None:
      chosen.append(best_add)
      remaining.remove(best_add)
      best_us = best_add_us
      improved = True
      if verbose:
        print(f"[greedy] KEEP int8 node {best_add} -> set {chosen} "
              f"({best_us:.0f}us, baseline {baseline_us:.0f}us)")
  # lossy: return the set only if it beats the fp16 baseline by the required margin.
  if not chosen or baseline_us == float("inf") or \
      not (best_us * min_lossy_speedup <= baseline_us):
    if verbose and chosen:
      print(f"[greedy] per-weight set {chosen} ({best_us:.0f}us) does not beat "
            f"baseline {baseline_us:.0f}us by {min_lossy_speedup}x -> rejected")
    return (), float("inf"), n_measured
  return tuple(chosen), best_us, n_measured


# tune
def _gen_inputs(input_shapes):
  """Deterministic synthetic inputs (fp16 ~N(0,1), nudged off zero)."""
  rng = np.random.default_rng(0xA9E)
  def _make(sh):
    a = rng.standard_normal(tuple(sh)).astype(np.float32)
    return np.where(np.abs(a) < 0.25, a + np.sign(a + 1e-6) * 0.5, a).astype(np.float16)
  return [_make(sh) for sh in input_shapes]


def _input_shapes(out):
  inputs = (n for n in _topo(out) if n.op == "input")
  return [tuple(n.shape) for n in sorted(inputs, key=lambda n: n.attrs.get("idx", 0))]


def _estimate_variant(out, cfg):
  """Cost-model estimate for a variant: apply its rewrite, then cost the DAG."""
  variant_out, int8 = _apply_variant(out, cfg)
  return estimate(variant_out, int8=int8)


def build_variant(out, cfg):
  """Compile a variant config directly (used on a cache hit and by opt=1)."""
  variant_out, int8 = _apply_variant(out, cfg)
  return _raw_compile(variant_out, int8=int8, opt=0, _check_precision=False)


def tune(out, budget: int = 8, inputs=None, prune_factor: float = 1.5,
         reps: int = 20, atol: float = _ACCURACY_TOL,
         min_lossy_speedup: float = _MIN_LOSSY_SPEEDUP, verbose: bool = False,
         target_error: float | None = None):
  """Return the fastest CORRECT compiled Model for `out` (enumerate, prune, measure, validate, cache); `target_error` switches to the precision-aware path."""
  if target_error is not None:
    model, _report = tune_precision(out, target_error=target_error, inputs=inputs,
                                    reps=reps, verbose=verbose)
    return model

  input_shapes = _input_shapes(out)
  key = _graph_key(out, input_shapes)
  cache = _load_cache()

  configs = _variants(out)

  # cache hit: rebuild the cached winner directly (no measurement).
  if key in cache and cache[key].get("config") is not None:
    cfg = cache[key]["config"]
    if cfg in configs or cfg.get("int8_nodes"):
      if verbose:
        print(f"[tune] cache hit {key}: {_config_label(cfg)} "
              f"({cache[key].get('us', '?')} us)")
      return build_variant(out, cfg)

  if inputs is None: inputs = _gen_inputs(input_shapes)

  # rank by cost-model estimate; lossless variants first (the correctness reference).
  ranked = sorted(configs, key=lambda c: _estimate_variant(out, c))
  ranked = ([c for c in ranked if not c.get("lossy")] +
            [c for c in ranked if c.get("lossy")])

  best_cfg, best_us, baseline_out = None, float("inf"), None
  baseline_us = float("inf")     # the lossless fp16 baseline latency (the reference)
  best_est = min(_estimate_variant(out, c) for c in configs)
  n_measured = 0
  results = []
  skipped_lossy_no_baseline = False

  for cfg in ranked:
    if n_measured >= budget: break
    est = _estimate_variant(out, cfg)
    # a lossy variant must never become its own accuracy reference - skip if no baseline yet.
    if cfg.get("lossy") and baseline_out is None:
      results.append((cfg, est, None, "skipped"))
      skipped_lossy_no_baseline = True
      if verbose:
        print(f"[tune] skip {_config_label(cfg)}: no lossless baseline to "
              f"validate against")
      continue
    # prune lossy variants the model predicts far worse than the best estimate (never lossless ones).
    if cfg.get("lossy") and est > prune_factor * best_est and best_cfg is not None:
      results.append((cfg, est, None, "pruned"))
      if verbose:
        print(f"[tune] prune {_config_label(cfg)}: est {est:.0f}us > "
              f"{prune_factor}x best est {best_est:.0f}us")
      continue
    us, out_arr = measure(out, inputs, cfg, baseline_out=baseline_out, reps=reps, tol=atol)
    n_measured += 1
    if baseline_out is None and out_arr is not None:
      baseline_out = out_arr     # first successful = reference
      baseline_us = us
    results.append((cfg, est, us, "measured"))
    if verbose:
      print(f"[tune] {_config_label(cfg):28s} est {est:7.0f}us  meas "
            f"{us if us != float('inf') else 'INCORRECT/FAIL'} us")
    if us < best_us:
      # a lossy variant must beat the lossless baseline by a real margin; a lossless swap wins on raw speed.
      if cfg.get("lossy") and not (baseline_us == float("inf")
                                   or us * min_lossy_speedup <= baseline_us):
        continue
      best_us, best_cfg = us, cfg

  if skipped_lossy_no_baseline:
    warnings.warn(
      "tune(): the fp16 baseline failed to compile (no lossless variant measured "
      "successfully), so lossy variants were skipped - without a lossless "
      "reference their accuracy cannot be validated.")

  # greedy per-weight int8: only when atol is loosened past the fp16-noise default.
  # Competes with global int8 (greedy wins when only SOME weights tolerate int8).
  if (atol > _ACCURACY_TOL and _has_weights(out) and baseline_out is not None
      and n_measured < budget):
    i8_nodes, i8_us, i8_n = _greedy_int8(
      out, inputs, baseline_out, baseline_us, reps=reps, atol=atol,
      min_lossy_speedup=min_lossy_speedup, budget=budget - n_measured,
      verbose=verbose)
    n_measured += i8_n
    if i8_nodes and i8_us < best_us:
      best_us = i8_us
      best_cfg = {"int8": False, "decomp": [], "int8_nodes": list(i8_nodes),
                  "lossy": True}
      if verbose:
        print(f"[tune] per-weight int8 wins: nodes {list(i8_nodes)} "
              f"({i8_us:.0f}us vs baseline {baseline_us:.0f}us)")

  if best_cfg is None:
    best_cfg = {"int8": False, "decomp": (), "lossy": False}  # fallback

  cache[key] = {"config": best_cfg, "us": (None if best_us == float("inf") else round(best_us, 1)),
                "shapes": [list(s) for s in input_shapes]}
  _save_cache(cache)

  if verbose:
    print(f"[tune] winner: {_config_label(best_cfg)} "
          f"({best_us if best_us != float('inf') else '?'} us); cached {key}")
  return build_variant(out, best_cfg)


def tune_report(out, budget: int = 8, inputs=None, reps: int = 20):
  """Like tune() but returns a structured report dict instead of a Model (always measures, never caches)."""
  input_shapes = _input_shapes(out)
  if inputs is None: inputs = _gen_inputs(input_shapes)
  configs = _variants(out)
  ranked = ([c for c in configs if not c.get("lossy")] +
            [c for c in configs if c.get("lossy")])

  baseline_out = None
  rows = []
  for cfg in ranked:
    est = _estimate_variant(out, cfg)
    us, out_arr = measure(out, inputs, cfg, baseline_out=baseline_out, reps=reps)
    if baseline_out is None and out_arr is not None: baseline_out = out_arr
    rows.append({"config": cfg, "label": _config_label(cfg), "est_us": est,
                 "meas_us": us, "correct": us != float("inf")})

  correct = [r for r in rows if r["correct"]]
  baseline = next((r for r in rows if not r["config"].get("lossy")), None)
  winner = min(correct, key=lambda r: r["meas_us"]) if correct else baseline
  speedup = None
  if baseline and winner and baseline["meas_us"] not in (None, float("inf")) and \
      winner["meas_us"] not in (None, float("inf")):
    speedup = baseline["meas_us"] / winner["meas_us"]
  return {"rows": rows, "n_variants": len(rows), "baseline": baseline,
          "winner": winner, "speedup": speedup, "baseline_out": baseline_out}


# precision-aware tune: given an explicit error budget, select the numerics-aware #
# rewrite set that meets it at minimum cost (accuracy vs an fp32 reference).
def _f16(x):  # fp16 rounding of operands/products, wide accum
  return np.asarray(x, np.float16).astype(np.float64)


# op -> numpy evaluator for the fp32-faithful reference; an op absent here has no reference.
_FP32_EVAL = {
  "add":         lambda s, t: _f16(s[0]) + _f16(s[1]),
  "sub":         lambda s, t: _f16(s[0]) - _f16(s[1]),
  "mul":         lambda s, t: _f16(s[0]) * _f16(s[1]),
  "muls":        lambda s, t: _f16(s[0]) * np.float16(t.attrs["k"]).astype(np.float64),
  "real_div":    lambda s, t: _f16(s[0]) / _f16(s[1]),
  "maximum":     lambda s, t: np.maximum(_f16(s[0]), _f16(s[1])),
  "minimum":     lambda s, t: np.minimum(_f16(s[0]), _f16(s[1])),
  "relu":        lambda s, t: np.maximum(_f16(s[0]), 0.0),
  "square":      lambda s, t: _f16(s[0]) ** 2,
  "abs":         lambda s, t: np.abs(_f16(s[0])),
  "exp":         lambda s, t: np.exp(_f16(s[0])),
  "reduce_sum":  lambda s, t: _f16(s[0]).sum(tuple(t.attrs["axes"]), keepdims=True),
  "reduce_mean": lambda s, t: _f16(s[0]).mean(tuple(t.attrs["axes"]), keepdims=True),
  "reshape":     lambda s, t: _f16(s[0]).reshape(t.shape),
  "transpose":   lambda s, t: np.transpose(_f16(s[0]), t.attrs["perm"]),
  "matmul":      lambda s, t: _f16(s[0]) @ _f16(t.attrs["wt"].astype(np.float64)).T,
  "bmm":         lambda s, t: _f16(s[0]) @ _f16(s[1]),
}


def _fp32_reference(out, inputs):
  """fp32-faithful numpy reference using the ANE matmul's wide-accumulator semantics; None if any op lacks an evaluator."""
  order = _topo(out)
  vals: dict[int, np.ndarray] = {}
  ins = [np.asarray(a, np.float16).astype(np.float64) for a in inputs]
  in_nodes = sorted((t for t in order if t.op == "input"),
                    key=lambda t: t.attrs.get("idx", 0))
  for t, a in zip(in_nodes, ins):
    vals[id(t)] = a

  for t in order:
    if id(t) in vals: continue
    s = [vals.get(id(src)) for src in t.srcs]
    if any(v is None for v in s): return None
    fn = _FP32_EVAL.get(t.op)
    if fn is None: return None  # unknown op -> no reference
    try:
      r = fn(s, t)
    except Exception:
      return None
    vals[id(t)] = np.asarray(r, np.float64).reshape(t.shape)
  return vals.get(id(out))


def _measure_with_ref(out, inputs, cfg, ref, reps, warmup=5):
  """Compile/run a variant -> (latency_us, relerr_vs_ref, out_arr); inf / 1.0 on failure."""
  arrs = [np.asarray(a, np.float16) for a in inputs]
  try:
    variant_out, int8 = _apply_variant(out, cfg)
    net = _raw_compile(variant_out, int8=int8, opt=0, _check_precision=False)
  except Exception:
    return float("inf"), 1.0, None
  try:
    res = None
    for _ in range(warmup):
      res = net(*arrs)
    cur = np.asarray(res, np.float64)
    if ref is not None and cur.shape == ref.shape:
      denom = float(np.abs(ref).max()) + 1e-9
      relerr = float(np.abs(cur - ref).max() / denom)
    else:
      relerr = float("nan")
    best = float("inf")
    for _ in range(reps):
      t0 = time.perf_counter()
      net(*arrs)
      best = min(best, (time.perf_counter() - t0) * 1e6)
    return best, relerr, np.asarray(res, np.float32)
  finally:
    try: net.release()
    except Exception: pass


def _precision_variants(out, want_int8=True):
  """Numerics-aware variant configs in increasing op-cost order (fp16 baseline, reduce_sum->matmul, then int8)."""
  risk = precision_risk(out)
  rs_hot = [n["idx"] for n in risk["nodes"] if n["kind"] == "narrow_sum"]
  configs = [{"label_kind": "fp16-baseline", "rs_matmul": [], "lossy": False,
              "cost_rank": 0}]
  if rs_hot:
    # rewrite all flagged narrow sums to matmul: tiny extra cost, accuracy >= baseline.
    configs.append({"label_kind": "reduce_sum->matmul", "rs_matmul": list(rs_hot),
                    "lossy": False, "cost_rank": 1})
  if want_int8 and _has_weights(out):
    configs.append({"label_kind": "int8+rs_matmul", "rs_matmul": list(rs_hot),
                    "int8": True, "lossy": True, "cost_rank": 2})
  return configs, risk


def tune_precision(out, target_error: float | None = None, cost_budget_us: float | None = None,
                   inputs=None, reps: int = 20, verbose: bool = False):
  """Precision-aware tune: select the numerics-aware rewrite set under an explicit error or cost budget, returning (model, report)."""
  input_shapes = _input_shapes(out)
  if inputs is None: inputs = _gen_inputs(input_shapes)
  ref = _fp32_reference(out, inputs)
  ref_kind = "fp32" if ref is not None else None

  configs, risk = _precision_variants(out)
  rows = []
  for cfg in configs:
    est = _estimate_variant(out, cfg)
    us, relerr, out_arr = _measure_with_ref(out, inputs, cfg, ref, reps=reps)
    if ref_kind is None and cfg["label_kind"] == "fp16-baseline" and out_arr is not None:
      # no fp32 emulation: the fp16 baseline's own output becomes the reference (relerr 0.0).
      ref = np.asarray(out_arr, np.float64)
      ref_kind = "fp16-baseline"
      relerr = 0.0
    rows.append({"config": cfg, "label": cfg["label_kind"], "est_us": est,
                 "meas_us": us, "relerr": relerr, "ok": us != float("inf")})
    if verbose:
      print(f"[tune_precision] {cfg['label_kind']:22s} est {est:7.0f}us  "
            f"meas {us if us != float('inf') else 'FAIL':>9} "
            f"relerr {relerr:.3e}")

  usable = [r for r in rows if r["ok"]]
  if not usable:
    # nothing compiled - fall back to the fp16 baseline.
    model = build_variant(out, {"rs_matmul": [], "int8": False})
    return model, {"rows": rows, "risk": risk, "chosen": None,
                   "ref_available": ref is not None, "ref_kind": ref_kind}

  if ref_kind is None:
    # no reference: error-based selection is meaningless, so prefer the cheapest lossless variant.
    pool = [r for r in usable if not r["config"].get("lossy")] or usable
    chosen = min(pool, key=lambda r: r["est_us"])
    reason = ("NO accuracy reference (fp32 emulation unsupported for this graph; "
              "the fp16 baseline failed to run) - error budget NOT enforced; "
              "chose min-cost" + ("" if pool is usable else " lossless"))
    warnings.warn(f"tune_precision: {reason}")
  elif target_error is not None:
    meeting = [r for r in usable if r["relerr"] <= target_error]
    if meeting:
      chosen = min(meeting, key=lambda r: r["est_us"])   # cheapest meeting E
      reason = (f"min-cost meeting target_error={target_error:.1e} "
                f"(error vs {ref_kind} reference)")
    else:
      chosen = min(usable, key=lambda r: r["relerr"])    # none meet -> most accurate
      reason = (f"NO variant met target_error={target_error:.1e} vs the "
                f"{ref_kind} reference; chose most-accurate")
  elif cost_budget_us is not None:
    affordable = [r for r in usable if r["est_us"] <= cost_budget_us]
    pool = affordable or usable
    chosen = min(pool, key=lambda r: r["relerr"])
    reason = (f"min-error vs {ref_kind} reference within cost_budget={cost_budget_us:.0f}us"
              if affordable else
              f"NO variant under cost_budget={cost_budget_us:.0f}us; chose most-accurate")
  else:
    chosen = min(usable, key=lambda r: r["relerr"])
    reason = f"min-error vs {ref_kind} reference (no budget given)"

  if verbose:
    print(f"[tune_precision] CHOSE {chosen['label']} "
          f"(relerr {chosen['relerr']:.3e}, est {chosen['est_us']:.0f}us) - {reason}")
  model = build_variant(out, chosen["config"])
  return model, {"rows": rows, "risk": risk, "chosen": chosen, "reason": reason,
                 "ref_available": ref is not None, "ref_kind": ref_kind}
