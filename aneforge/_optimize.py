"""aneforge graph autotuner: a deterministic measurement-cache search over the
metamorphic-proven-safe variant space, validated to never change results.

The premise (see tests/fuzz_metamorphic.py): the optimizer is a rewrite engine whose
correctness rule is that a semantics-preserving rewrite never changes the output. So
the only variants this tuner enumerates are ones the metamorphic fuzzer proved safe
within tolerance:

  PRIMARY  - int8 weight selection: compile(int8=True) streams per-channel int8
             weights at half the bytes; fp16_vs_int8 is a proven-safe (PRECISION-
             class) rewrite. Global today; the per-weight hook is designed in.
  SECONDARY- sdpa<->decomposed route: an af.sdpa node can be substituted by the
             proven-equivalent decomposed attention (mha_vs_sdpa). v1 ships the
             cost-model route hint (estimate() picks by sequence length) and the
             detection/plumbing hook; the in-place graph rewrite is the next hook
             (documented below) since it requires rebuilding the graph, which the
             tune() API does not own (it receives an already-built `out` tensor).

SEARCH: enumerate the small legal variant set, prune with estimate() (skip any variant
the cost model predicts much worse than the current best), measure the rest on the ANE
(compile + warmup + MIN over reps), validate each variant's output against the opt=0
baseline within tol (a variant that breaks accuracy scores inf and is never chosen),
pick the fastest correct one, and cache the decision keyed by a deterministic
(structure-hash, shapes, dtype, variant-config) tuple.

Determinism is what makes the cache valid across runs: the same graph structure +
input shapes always hashes the same, so a cached decision is reusable. `tune` is
instant on a cache hit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from ._compile import compile as _raw_compile, _topo
from ._cost import estimate, precision_risk
from .graph import Tensor

# --------------------------------------------------------------------------- #
# tolerance for variant correctness (vs the opt=0 baseline)                    #
# --------------------------------------------------------------------------- #
# DEFAULT is accuracy-preserving (~fp16 noise): a variant must reproduce the baseline
# this closely or it is rejected. At this default a lossy rewrite like int8 (~1-2%
# quantization error, well above fp16 noise) is rejected, so tune() never silently
# trades accuracy - it returns the lossless baseline. int8 is opt-in: pass
# tune(out, atol=0.1) to admit it within a stated accuracy budget. (0.12 was the
# metamorphic fp16_vs_int8 precision tol - right to PASS for an explicit int8 budget,
# wrong as a DEFAULT.)
_ACCURACY_TOL = 5e-3        # default: fp16-noise - lossless rewrites pass, int8 fails
# A lossy variant must beat the baseline by at least this factor to be chosen, so a
# measurement-noise "win" (CV ~10-20%) never costs accuracy for no real speedup.
_MIN_LOSSY_SPEEDUP = 1.10


# --------------------------------------------------------------------------- #
# deterministic graph-structure hash                                          #
# --------------------------------------------------------------------------- #
def _graph_key(out, input_shapes) -> str:
  """A deterministic hash of the graph STRUCTURE + input shapes. Walks the graph in
    topo order, hashing each node's op, shape, source positions, and the shapes/dtypes of
    any constant-weight attrs (not their values - structure, not data). Same structure +
    shapes -> same key across runs (the cache validity premise)."""
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


# --------------------------------------------------------------------------- #
# persistent measurement cache                                                #
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
  env = os.environ.get("ANEFORGE_CACHE_DIR")
  if env:
    d = Path(env)
  else:
    # repo-local cache so it is discoverable and version-controllable if wanted;
    # falls back to ~/.cache/aneforge if the repo dir is not writable.
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


# --------------------------------------------------------------------------- #
# attention query-tile autotune                                                #
# --------------------------------------------------------------------------- #
# The query-tile count in mha/sdpa/cross_attention sets how the [H, S, T] score is
# fissioned. The S-based heuristic (max(1,(S+256)//512)) avoids the un-tiled wall but is
# not the per-shape optimum, and the optimum is bound by L2 residency / pipelining, so it
# shifts with head count and chip. This measures the best count ONCE per (chip, S, T,
# heads, head_dim) and caches it; the heuristic is the default until a tuned value exists.
_TILE_MEMO: dict = {}
_TILE_CANDIDATES = (1, 2, 3, 4, 5, 6, 8)


def _heuristic_tiles(S: int) -> int:
  return max(1, (S + 256) // 512)


def _time_attention_core(S: int, T: int, n_heads: int, dh: int, n_tiles: int, reps: int = 20) -> float:
  """Median wall time (s) of the query-tiled attention core [H,S,dh] x [H,dh,T] on the ANE."""
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
  """Query-tile count for an [n_heads, S, dh] x [n_heads, dh, T] attention score.

    Returns a chip+shape-cached tuned value if one exists, else the S-based heuristic.
    With `tune=True` (or env ANEFORGE_TUNE_ATTENTION=1) it measures the candidate counts
    once on the engine, caches the fastest (persisted in autotune.json, keyed by chip),
    and returns it. Exact either way - the tile count only changes how the score fissions."""
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


# --------------------------------------------------------------------------- #
# variant space (legal == metamorphic-proven-safe)                            #
# --------------------------------------------------------------------------- #
def _has_weights(out) -> bool:
  """True if the graph carries any int8-eligible weight: a matmul/linear streamed `wt`
    or a conv baked `weight` (per-channel int8 routes as a conv weight on the ANE). int8
    is a no-op otherwise, so we don't enumerate it."""
  for t in _topo(out):
    if t.op == "matmul" and isinstance(t.attrs.get("wt"), np.ndarray): return True
    if t.op == "conv" and isinstance(t.attrs.get("weight"), np.ndarray): return True
  return False


def _int8_candidates(out):
  """Per-weight int8 candidate nodes: the topo indices of weight-bearing nodes, paired
    with the weight's element count (the int8 benefit proxy - int8 halves these bytes, so
    a bigger weight is a bigger bandwidth win). Candidates are matmul nodes (streamed
    `wt`) AND conv nodes (baked `weight`): per-channel int8 routes as a conv weight on
    the ANE (constexpr_affine_dequantize, ~halves the conv DRAM bytes, cos~1.0 vs fp16).

    Returns a list of (topo_index, weight_elems) sorted by weight_elems DESCENDING
    (largest weight first) - the order greedy adds candidates, so the budget is spent
    where int8 helps most."""
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
  """Topo indices of every ROUTE-BEARING node - a bridge node whose op has a validated
    equivalent fused lowering in the closed route registry (sdpa, minmax_norm,
    flatten, ...). These are the axes the optimizer flips between the native bridge (a
    graph cut) and the fused decomposition (cut removed). Single-route bridge ops
    (argmax/sort/fps/the rearranges/...) are not here - they have no alternative, so the
    tuner never tries to flip them. Deterministic (topo order) so the cache key is
    stable."""
  from ._rewrite import _BRIDGE_DECOMPOSERS
  order = _topo(out)
  # A causal sdpa must NOT be route-flipped to the fused decomposition: that
  # decomposition drops the native layer's additive mask (the route registry's decomposed
  # attention is unmasked), so a routed causal sdpa would silently return unmasked
  # attention. Keep it on the native bridge route (where _run_sdpa feeds the causal mask
  # via the 5th SDPA bottom).
  return [i for i, t in enumerate(order)
          if t.op in _BRIDGE_DECOMPOSERS
          and not (t.op == "sdpa" and (t.attrs.get("causal") or t.attrs.get("masked")))]


def _constfold_candidates(out):
  """Topo indices of nodes whose whole source cone is constant (foldable to one
    const_array). Excludes the const/input leaves themselves."""
  from ._rewrite import _const_subgraph
  return [i for i, t in enumerate(_topo(out))
          if t.op not in ("const_array", "input") and _const_subgraph(t) is not None]


def _scalarchain_candidates(out):
  """Topo indices of muls/adds chain heads (a muls whose src is a muls, or adds
    over adds) - the nodes a scalar-chain fold collapses."""
  return [i for i, t in enumerate(_topo(out))
          if t.op in ("muls", "adds") and t.srcs[0].op == t.op]


def _apply_variant(out, cfg):
  """Materialize the rewritten DAG for a variant `cfg` in ONE graph_rewrite pass.
    Every axis composes over the ORIGINAL graph (indices resolved to original ids up
    front), so there is no chained-rewrite index drift. The compile-time global int8
    flag is returned separately (a compiler arg, not a graph rewrite).

    cfg axes: `decomp` (bridge topo-indices -> fused decomposition), `int8_nodes`
    (matmul/conv topo-indices -> per-node int8 tag), `rs_matmul` (reduce_sum
    topo-indices -> @ones contraction), `scalarfold` (muls/adds chain-head indices),
    `constfold` (constant-cone indices -> one const_array)."""
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


def _variants(out):
  """Return the legal variant configs for this graph as a list of dicts. Each is a
    compile-config the tuner can build, measure, and cache.

    Variant axes (separable -> coordinate-descent-ready):
      - GLOBAL int8 (legacy whole-graph): {False, True}. LOSSY (accuracy-gated).
      - ROUTE per route-bearing bridge node: {native bridge, fused decomposition}.
        Generalized over EVERY op in the closed route registry that has a validated
        equivalent lowering (sdpa, minmax_norm, flatten, ...), not just sdpa. Each
        route is LOSSLESS (bit-identical or fp16-op-noise, the metamorphic proof
        class), so it is always eligible and chosen purely by measured speed.
        Enumerated by COORDINATE DESCENT over the route-bearing nodes: the all-native
        baseline + each single node flipped to its fused decomposition + the all-
        decomposed config. That is K single-flips + (all-decomposed when K>1) = at most
        K+1 route variants on top of baseline - linear, NOT 2^K. (Single-route bridge ops
        are not in the route-id set, so the tuner never tries to flip them.)
    """
  # config values are JSON-friendly (lists, not tuples) so a cached config compares
  # equal to a freshly-enumerated one after a JSON round-trip (the cache premise).
  routes = _route_ids(out)
  configs = [{"int8": False, "decomp": [], "lossy": False}]   # opt=0 baseline (all native)
  if routes:
    # coordinate descent over the route axis: each single route-bearing node flipped
    # to its fused decomposition, plus the all-decomposed config. (Typical 1-bridge
    # graph is just {native, decomposed}; K nodes is K single-flips + all-decomposed =
    # K+1 route variants on top of baseline - linear.)
    for i in routes:
      configs.append({"int8": False, "decomp": [i], "lossy": False})
    if len(routes) > 1:
      configs.append({"int8": False, "decomp": list(routes), "lossy": False})
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
  """The LOSSLESS-ONLY variant set for the DEFAULT route pass: the all-native baseline
    plus the route-bearing bridge nodes flipped to their fused decomposition
    (sdpa/minmax_norm/flatten/lrn). No int8/lossy variant is ever included - every config
    here is cos-1.0 equivalent to opt=0, so the default pass can never change numerics.
    Same coordinate-descent shape as `_variants` (linear in #route nodes: each single flip
    + the all-decomposed config), minus the lossy int8 branch.

    (Shape-dependent equivalence is handled upstream: `af.sdpa` itself decomposes above
    the native layer's accuracy limit, so a routable native `sdpa` node only exists where
    native and decomposed actually agree - the route stays lossless.)"""
  routes = _route_ids(out)
  configs = [{"int8": False, "decomp": [], "lossy": False}]      # all-native baseline
  for i in routes:
    configs.append({"int8": False, "decomp": [i], "lossy": False})
  if len(routes) > 1:
    configs.append({"int8": False, "decomp": list(routes), "lossy": False})
  return configs


def _compile_routes(out, int8: bool, build_dir=None):
  """The DEFAULT compile pass: a cost-model-driven, lossless route optimization.

    For each route-bearing bridge node (sdpa/minmax_norm/flatten/lrn) it picks per SHAPE
    between the native bridge (a graph cut) and the fused decomposition (cut removed),
    choosing whichever `estimate()` predicts cheaper - without any on-device measurement
    (no compile-time cost beyond costing the DAGs). The route registry is lossless-only,
    so this never changes numerics (cos 1.0 vs opt=0).

    The A/B in the reverse-engineering corpus validates this: across the routable shapes,
    the cost model's argmin matches the fastest CORRECT route every time, so measurement
    would add compile cost without changing the pick. (Shape-dependent equivalence is
    handled in the frontend: `af.sdpa` decomposes above the native layer's accuracy limit,
    so a routable native sdpa node only exists where the route is genuinely lossless.)

    If the graph has NO route-bearing node (the common case), the only variant is the
    all-native baseline, so this returns exactly the same single program `opt=0` would -
    it never regresses a cut-free model. `int8` is threaded through to the eventual
    `_raw_compile` so the legacy whole-graph int8 flag is still honored."""
  cfgs = _route_variants(out)
  if len(cfgs) == 1:
    # no removable cut: identical to the byte-identical opt=0 path.
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


# --------------------------------------------------------------------------- #
# measurement                                                                 #
# --------------------------------------------------------------------------- #
def measure(out, inputs, cfg, baseline_out=None, reps: int = 20, warmup: int = 5,
            tol: float = _ACCURACY_TOL):
  """Compile the graph under `cfg`, validate vs `baseline_out` (the opt=0 output)
    within tol, and return MIN latency over `reps` (microseconds).

    Returns `inf` if the variant is incorrect (so it is never selected) or fails to
    compile/run. Also returns the variant's own baseline output array when `baseline_out`
    is None (so the caller can use the first variant as the reference)."""
  arrs = [np.asarray(a, np.float16) for a in inputs]
  try:
    variant_out, int8 = _apply_variant(out, cfg)
    net = _raw_compile(variant_out, int8=int8, opt=0, _check_precision=False)
  except Exception:
    return float("inf"), None
  try:
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


# --------------------------------------------------------------------------- #
# greedy per-weight int8 selection (coordinate descent, not 2^K)              #
# --------------------------------------------------------------------------- #
def _greedy_int8(out, inputs, baseline_out, baseline_us, *, reps, atol,
                 min_lossy_speedup, budget, verbose=False):
  """Greedily grow the per-weight int8 set, one weight at a time.

    Start from the all-fp16 baseline (empty int8 set). Walk the candidate matmul weights
    ordered by predicted benefit (largest weight first - int8 halves its bytes). For each
    candidate, tentatively add it to the int8 set and MEASURE the resulting graph: keep
    the weight int8 only if (a) accuracy stays within `atol` vs the opt=0 baseline AND
    (b) it improves measured speed over the current best config. Continue until no
    remaining candidate helps or the budget is spent. Returns `(int8_nodes_tuple, best_us,
    n_measured)` - the chosen per-weight config and its measured latency. The set is built
    on the all-native route (decomp empty); the route axis is separable and handled by the
    lossless track.

    O(#candidates^2) measurements worst-case (each pass re-scans remaining candidates), but
    capped by `budget`; the largest-weight-first order front-loads the wins so the budget
    is well spent.
    """
  cands = _int8_candidates(out)
  if not cands: return (), float("inf"), 0
  chosen: list[int] = []
  # current best = the fp16 baseline (empty int8 set). Any int8 add must beat it.
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
      # accept only if accurate AND faster than the best config found so far in
      # this pass (which starts at the running best). Strict improvement.
      if ok and us < best_add_us: best_add, best_add_us = idx, us
    if best_add is not None:
      chosen.append(best_add)
      remaining.remove(best_add)
      best_us = best_add_us
      improved = True
      if verbose:
        print(f"[greedy] KEEP int8 node {best_add} -> set {chosen} "
              f"({best_us:.0f}us, baseline {baseline_us:.0f}us)")
  # the per-weight set is lossy: return it only if it beats the fp16 baseline by the
  # required margin (same gate as global int8). Otherwise the lossless baseline wins.
  if not chosen or baseline_us == float("inf") or \
      not (best_us * min_lossy_speedup <= baseline_us):
    if verbose and chosen:
      print(f"[greedy] per-weight set {chosen} ({best_us:.0f}us) does not beat "
            f"baseline {baseline_us:.0f}us by {min_lossy_speedup}x -> rejected")
    return (), float("inf"), n_measured
  return tuple(chosen), best_us, n_measured


# --------------------------------------------------------------------------- #
# tune                                                                        #
# --------------------------------------------------------------------------- #
def _gen_inputs(input_shapes):
  """Deterministic synthetic inputs for measurement/validation (fp16 ~N(0,1),
    nudged off zero so divides/rsqrt don't blow up - same recipe as the fuzzer)."""
  rng = np.random.default_rng(0xA9E)
  def _make(sh):
    a = rng.standard_normal(tuple(sh)).astype(np.float32)
    return np.where(np.abs(a) < 0.25, a + np.sign(a + 1e-6) * 0.5, a).astype(np.float16)
  return [_make(sh) for sh in input_shapes]


def _input_shapes(out):
  inputs = (n for n in _topo(out) if n.op == "input")
  return [tuple(n.shape) for n in sorted(inputs, key=lambda n: n.attrs.get("idx", 0))]


def _estimate_variant(out, cfg):
  """Cost-model estimate for a variant: apply its graph rewrite (route/per-weight),
    then cost the resulting DAG under the global int8 flag. This is how the route decision
    (native vs decomposed) is predicted before measuring - the decomposed DAG has no
    native-SDPA cut, so estimate() costs it as one fused program."""
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
  """Return the fastest CORRECT compiled Model for the graph rooted at `out`.

    Enumerates the legal (proven-safe) variant space, prunes with the cost model, measures
    the survivors on the ANE, validates each against the opt=0 baseline, picks the fastest
    correct one, and caches the decision (instant on a cache hit).

    PRECISION axis: pass `target_error=E` to switch to the precision-aware path
    (`tune_precision`) - the optimizer then selects the numerics-aware rewrite set
    (reduce_sum->matmul, +/- int8) that meets the error budget E (measured vs an fp32
    reference, or vs the fp16 baseline's output when no fp32 emulation exists for the
    graph) at minimum cost, and can IMPROVE accuracy over the fp16 baseline. With no
    `target_error` the behavior is unchanged (speed tune; accuracy-preserving).

    Accuracy contract: by default (`atol` = fp16 noise) tune is accuracy-PRESERVING - a
    lossy rewrite (int8) is rejected, so the result matches opt=0 within fp16 noise. To
    trade accuracy for speed, pass an explicit budget, e.g. `tune(out, atol=0.1)`; even
    then a lossy variant must beat the baseline by `min_lossy_speedup` (default 1.10) to
    be chosen, so a measurement-noise "win" never costs accuracy for no real gain.

    `budget` caps the number of on-device measurements. `prune_factor`: skip any variant
    whose estimate() is > prune_factor x the best estimate so far.
    """
  if target_error is not None:
    model, _report = tune_precision(out, target_error=target_error, inputs=inputs,
                                    reps=reps, verbose=verbose)
    return model

  input_shapes = _input_shapes(out)
  key = _graph_key(out, input_shapes)
  cache = _load_cache()

  configs = _variants(out)

  # cache hit: rebuild the cached winner directly (no measurement). The winner is
  # either an enumerated variant (route/global-int8) or a greedy per-weight int8 config
  # (carrying `int8_nodes`, not in the static variant set but still buildable from the
  # same graph since the cache key pins the structure).
  if key in cache and cache[key].get("config") is not None:
    cfg = cache[key]["config"]
    if cfg in configs or cfg.get("int8_nodes"):
      if verbose:
        print(f"[tune] cache hit {key}: {_config_label(cfg)} "
              f"({cache[key].get('us', '?')} us)")
      return build_variant(out, cfg)

  if inputs is None: inputs = _gen_inputs(input_shapes)

  # rank by cost-model estimate; the lossless baseline first so it is the reference.
  ranked = sorted(configs, key=lambda c: _estimate_variant(out, c))
  # ensure a lossless variant is measured first (it is the correctness reference).
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
    # `ranked` measures every lossless variant before any lossy one, so reaching a
    # lossy variant with no baseline means the fp16 baseline failed to compile. A lossy
    # variant must never become its own accuracy reference - skip it.
    if cfg.get("lossy") and baseline_out is None:
      results.append((cfg, est, None, "skipped"))
      skipped_lossy_no_baseline = True
      if verbose:
        print(f"[tune] skip {_config_label(cfg)}: no lossless baseline to "
              f"validate against")
      continue
    # prune: skip lossy variants the model predicts far worse than the best estimate.
    # Never prune a lossless variant (route swaps are free accuracy-wise and are the
    # correctness reference / safe fallback).
    if cfg.get("lossy") and est > prune_factor * best_est and best_cfg is not None:
      results.append((cfg, est, None, "pruned"))
      if verbose:
        print(f"[tune] prune {_config_label(cfg)}: est {est:.0f}us > "
              f"{prune_factor}x best est {best_est:.0f}us")
      continue
    us, out_arr = measure(out, inputs, cfg, baseline_out=baseline_out, reps=reps, tol=atol)
    n_measured += 1
    if baseline_out is None and out_arr is not None:
      baseline_out = out_arr     # first successful = reference (the fp16 baseline)
      baseline_us = us
    results.append((cfg, est, us, "measured"))
    if verbose:
      print(f"[tune] {_config_label(cfg):28s} est {est:7.0f}us  meas "
            f"{us if us != float('inf') else 'INCORRECT/FAIL'} us")
    if us < best_us:
      # a lossy variant (int8) must beat the lossless baseline by a real margin -
      # never swap accuracy for a measurement-noise "win". A lossless route swap is
      # chosen on raw speed (no margin needed; it's bit-identical).
      if cfg.get("lossy") and not (baseline_us == float("inf")
                                   or us * min_lossy_speedup <= baseline_us):
        continue
      best_us, best_cfg = us, cfg

  if skipped_lossy_no_baseline:
    warnings.warn(
      "tune(): the fp16 baseline failed to compile (no lossless variant measured "
      "successfully), so lossy variants were skipped - without a lossless "
      "reference their accuracy cannot be validated.")

  # greedy per-weight int8 (coordinate descent). int8 is lossy, so only enumerate it
  # when the user has opted into a loose accuracy budget (atol > the fp16-noise default).
  # At the tight default it fails the accuracy gate anyway - skip it so the budget is not
  # wasted and default tune stays the lossless baseline (byte-identical decision). The
  # result competes with GLOBAL int8 already in the variant set: greedy wins when only
  # SOME weights tolerate int8, global when all do (greedy then selects every candidate
  # and ties global).
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
    best_cfg = {"int8": False, "decomp": (), "lossy": False}  # safe fallback

  cache[key] = {"config": best_cfg, "us": (None if best_us == float("inf") else round(best_us, 1)),
                "shapes": [list(s) for s in input_shapes]}
  _save_cache(cache)

  if verbose:
    print(f"[tune] winner: {_config_label(best_cfg)} "
          f"({best_us if best_us != float('inf') else '?'} us); cached {key}")
  return build_variant(out, best_cfg)


def tune_report(out, budget: int = 8, inputs=None, reps: int = 20):
  """Like tune() but returns a structured report dict instead of a Model - used by the
    validator to report per-program speedup and cost-model ranking accuracy. Does not
    consult/write the cache (always measures), so the report is fresh."""
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


# =========================================================================== #
# precision-aware tune  -  the synthesis of the cost model with the fp16        #
# envelope. Speed is no longer the only axis: given an EXPLICIT error budget,    #
# the optimizer selects the numerics-aware rewrite set (reduce_sum->matmul,      #
# paired-fp16, +/- int8) that MEETS the budget at minimum predicted cost - or    #
# minimizes error within a cost budget. Accuracy is measured vs an fp32          #
# reference where one can be emulated (else the fp16 baseline's own output), so   #
# a rewrite can be selected for improving accuracy, which speed-only tune()       #
# cannot do.                                                                      #
# =========================================================================== #
def _f16(x):  # fp16 rounding of operands/products (the ANE storage), wide accum
  return np.asarray(x, np.float16).astype(np.float64)


# op -> numpy evaluator for the fp32-faithful reference. Each takes the node's
# fp16-rounded source values `s` and the node `t`. Built once (module scope), not
# per-node. An op absent here has no faithful reference (the graph falls back).
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
  """Compute a fp64/fp32-faithful reference for the graph on numpy, so accuracy can be
    measured as IMPROVEMENT (not just preservation). Uses the same wide-accumulator
    semantics the ANE matmul offers (products fp16-rounded, accumulation wide) so the
    reference is the achievable target, not an unreachable fp64 ideal.

    Returns the reference array (fp64) or None if any op lacks a numpy evaluator (then
    precision tune falls back to measuring vs the fp16 baseline, like speed tune)."""
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
    if fn is None: return None  # unknown op -> no faithful reference
    try:
      r = fn(s, t)
    except Exception:
      return None
    vals[id(t)] = np.asarray(r, np.float64).reshape(t.shape)
  return vals.get(id(out))


def _measure_with_ref(out, inputs, cfg, ref, reps, warmup=5):
  """Compile a variant, run it, and return (latency_us, relerr_vs_ref, out_arr).
    `relerr_vs_ref` is max-abs relative error vs the fp32 reference (the accuracy
    metric the budget is stated in). inf latency / 1.0 relerr on failure."""
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
  """The numerics-aware variant configs for precision tune, in increasing op-cost
    order: fp16 baseline, then reduce_sum->matmul on the flagged narrow sums, then
    (optionally) int8 (a LOSSY speed lever the budget may still admit). paired-fp16 is a
    region rewrite (it changes the dataflow type), so it is not auto-enumerated here - it
    is an explicit opt-in (see precision_rewrite / the demo); automatic cancel-sub
    detection is a CANDIDATE flag only (data-dependent)."""
  risk = precision_risk(out)
  rs_hot = [n["idx"] for n in risk["nodes"] if n["kind"] == "narrow_sum"]
  configs = [{"label_kind": "fp16-baseline", "rs_matmul": [], "lossy": False,
              "cost_rank": 0}]
  if rs_hot:
    # the flagship safe rewrite: rewrite all flagged narrow sums to matmul. A tiny
    # extra op cost (one matmul vs one reduce), accuracy strictly >= baseline.
    configs.append({"label_kind": "reduce_sum->matmul", "rs_matmul": list(rs_hot),
                    "lossy": False, "cost_rank": 1})
  if want_int8 and _has_weights(out):
    configs.append({"label_kind": "int8+rs_matmul", "rs_matmul": list(rs_hot),
                    "int8": True, "lossy": True, "cost_rank": 2})
  return configs, risk


def tune_precision(out, target_error: float | None = None, cost_budget_us: float | None = None,
                   inputs=None, reps: int = 20, verbose: bool = False):
  """PRECISION-AWARE tune: select the numerics-aware rewrite set under an explicit
    ERROR BUDGET (or a cost budget), measuring accuracy vs an fp32 reference.

    Two modes (give one):
      * `target_error=E`    : among variants whose measured relerr-vs-reference <= E,
                                pick the one with MINIMUM predicted cost. (A tight E
                                forces the accurate rewrites; a loose E lets a cheaper/
                                lossy variant in.)
      * `cost_budget_us=C`  : among variants whose predicted cost <= C, pick the one
                                with MINIMUM measured error. (Best accuracy you can buy.)

    Default (neither given) == minimize error at any cost (the most-accurate variant).

    Returns `(model, report)` where report carries the per-variant (cost, error) table,
    the precision-risk flags, and the chosen config - so a caller can SEE the accuracy/cost
    frontier, not just the winner. Unlike speed-tune(), a variant here can be chosen for
    IMPROVING accuracy over the fp16 baseline.

    SCOPE: automatic hotspot detection covers the reduce_sum->matmul case
    (structurally detectable). The CFG-style paired-fp16 fix is opt-in (use
    `precision_rewrite` / pass a marked region) because near-equal cancellation is
    data-dependent and cannot be confirmed at graph-build time.

    REFERENCE: `_fp32_reference` emulates only the simple elementwise/matmul ops; a graph
    it cannot evaluate (conv/softmax/norms/...) falls back to the fp16 baseline's own
    measured output as the reference - `target_error` then bounds divergence FROM the fp16
    baseline (which has relerr 0.0 by definition). The report's `ref_kind` ("fp32" |
    "fp16-baseline" | None) records which reference was used; None means no reference of
    any kind existed (the baseline failed too), in which case the budget is NOT enforced
    and a warning is emitted."""
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
      # no fp32-faithful emulation for this graph: the fp16 baseline's own measured
      # output becomes the reference (the docstring contract), so remaining variants
      # are gated on divergence from it. The baseline is the reference: relerr 0.0 by
      # definition, not NaN.
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
    # nothing compiled - fall back to the plain fp16 baseline.
    model = build_variant(out, {"rs_matmul": [], "int8": False})
    return model, {"rows": rows, "risk": risk, "chosen": None,
                   "ref_available": ref is not None, "ref_kind": ref_kind}

  if ref_kind is None:
    # NO reference of any kind: fp32 emulation is unsupported for this graph AND the
    # fp16 baseline failed, so no variant's error was measured. Error-based selection
    # would be meaningless - prefer the cheapest LOSSLESS variant and say so rather than
    # silently pretending the budget was enforced.
    pool = [r for r in usable if not r["config"].get("lossy")] or usable
    chosen = min(pool, key=lambda r: r["est_us"])
    reason = ("NO accuracy reference (fp32 emulation unsupported for this graph; "
              "the fp16 baseline failed to run) - error budget NOT enforced; "
              "chose min-cost" + ("" if pool is usable else " lossless"))
    warnings.warn(f"tune_precision: {reason}")
  elif target_error is not None:
    meeting = [r for r in usable if r["relerr"] <= target_error]
    if meeting:
      chosen = min(meeting, key=lambda r: r["est_us"])   # cheapest that meets E
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
