"""Graph-rewrite engine. Tensors are immutable, so a rewrite re-derives the output DAG (memoized by identity, rules bottom-up): route decompositions, per-node int8, reduce_sum->matmul, canon/numeric folds."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

import numpy as np

from .graph import Tensor


@dataclass(frozen=True)
class Rule:
  """One graph-rewrite rule. `match`/`build` see the REBUILT node (sources already
  rewritten). `kind` is "lossless" (always-on canon) or "numeric" (tuner-gated)."""
  name: str; kind: str
  match: Callable[[Tensor], bool]; build: Callable[[Tensor], Tensor]


def graph_rewrite(out: Tensor, rules: list["Rule"], select: set[int] | None = None) -> Tensor:
  """Return a new output DAG, rules applied bottom-up in one memoized pass (first matching rule fires). `select` (original-node ids) gates eligibility; None = all. A no-op rule-set yields the SAME object."""
  from ._compile import _topo                 # iterative post-order: stack-safe on deep graphs
  memo: dict[int, Tensor] = {}
  for t in _topo(out):
    new_srcs = [memo[id(s)] for s in t.srcs]   # every src precedes t in _topo
    rebuilt = t if all(a is b for a, b in zip(new_srcs, t.srcs)) else Tensor(t.shape, t.op, new_srcs, dict(t.attrs))
    res = rebuilt
    if select is None or id(t) in select:
      for r in rules:
        if r.match(rebuilt): res = r.build(rebuilt); break
    memo[id(t)] = res
  return memo[id(out)]


def rewrite(out: Tensor, rule: Callable[[Tensor], Optional[Tensor]]) -> Tensor:
  """Return a new output DAG with `rule(node) -> Tensor | None` applied bottom-up; a no-op rule yields the SAME object."""
  # `rule` must be a pure node-local transform (called in both match and build); the
  # match-guard makes the None return safe to narrow (cast).
  r = Rule("adhoc", "lossless", lambda t: rule(t) is not None,
       cast(Callable[[Tensor], Tensor], lambda t: rule(t)))
  return graph_rewrite(out, [r])


def _decompose_sdpa(node: Tensor) -> Tensor:
  """Decomposed `sdpa`: `((q @ k^T) * scale).softmax(-1) @ v`. Fused MIL, no cut; bit-identical to native sdpa."""
  q, k, v = node.srcs
  scale = node.attrs["scale"]
  scores = (q @ k.transpose([0, 1, 3, 2])) * float(scale)
  a = scores.softmax(-1)
  return a @ v


def sdpa_to_decomposed(out: Tensor, only_id: Optional[int] = None) -> Tensor:
  """Rewrite native `sdpa` node(s) to the decomposed form. `only_id` (an original-graph id) targets one; None = all."""
  sel = None if only_id is None else {only_id}
  r = Rule("sdpa", "lossless", lambda t: t.op == "sdpa", _decompose_sdpa)
  return graph_rewrite(out, [r], select=sel)


def list_sdpa_nodes(out: Tensor) -> list[Tensor]:
  """The sdpa nodes in topo order (the coordinate-descent route axes)."""
  from ._compile import _topo
  return [t for t in _topo(out) if t.op == "sdpa"]


def set_node_int8(out: Tensor, node_ids: set[int]) -> Tensor:
  """Tag each weight-bearing node in `node_ids` with `int8=True` (per-weight override of the global flag)."""
  if not node_ids: return out
  r = Rule("int8", "numeric", lambda t: t.op in ("matmul", "conv"),
       lambda t: Tensor(t.shape, t.op, t.srcs, {**t.attrs, "int8": True}))
  return graph_rewrite(out, [r], select=set(node_ids))


def list_weight_nodes(out: Tensor) -> list[Tensor]:
  """Weight-bearing (int8-eligible) nodes in topo order: matmul (`wt`) and conv (`weight`)."""
  from ._compile import _topo
  return [t for t in _topo(out)
      if (t.op == "matmul" and isinstance(t.attrs.get("wt"), np.ndarray))
      or (t.op == "conv" and isinstance(t.attrs.get("weight"), np.ndarray))]


# --- numerics-aware rewrites (accuracy-improving, validated vs fp32) ---
def _reduce_sum_as_matmul(t: Tensor) -> Tensor:
  """Rebuild a single-axis `reduce_sum` as `x @ ones[K,1]` (WIDE matmul accumulator; >= accuracy under cancellation). A non-last axis is transposed to the end and back."""
  src = t.srcs[0]
  axes = tuple(t.attrs.get("axes", ()))
  if len(axes) != 1:
    return t                       # multi-axis: leave as-is (not offered)
  (ax,) = axes
  rank = len(src.shape)
  ax = ax % rank
  K = int(src.shape[ax])
  ones = np.ones((K, 1), dtype=np.float16)

  if ax == rank - 1:
    return src @ ones
  # move `ax` last, contract, move the size-1 dim back
  perm = [i for i in range(rank) if i != ax] + [ax]
  inv = [0] * rank
  for new_pos, old in enumerate(perm):
    inv[old] = new_pos
  contracted = src.transpose(perm) @ ones
  return contracted.transpose(inv)


def reduce_sum_to_matmul(out: Tensor, only_idx: set[int] | None = None) -> Tensor:
  """Rewrite single-axis `reduce_sum` node(s) to a `@ ones` contraction (lossless-or-better). `only_idx` selects by original-graph topo index; None = all."""
  from ._compile import _topo
  order = _topo(out)
  elig = lambda t: t.op == "reduce_sum" and len(t.attrs.get("axes", ())) == 1
  if only_idx is None: sel = {id(t) for t in order if elig(t)}
  else: sel = {id(order[i]) for i in only_idx if i < len(order) and elig(order[i])}
  if not sel: return out
  r = Rule("rsum_mm", "numeric", elig, _reduce_sum_as_matmul)
  return graph_rewrite(out, [r], select=sel)


# --- bridge-elimination rewrites: replace a netplist cut with a fused decomposition ---
# each is lossless (validated on-device vs the bridge); removing the cut lets the region fuse.
def _decompose_minmax_norm(node: Tensor) -> Tensor:
  """Fused minmax_norm: `(x - amin) / (amax - amin + eps)` over the reduction axis. Lossless (max==min -> 0/eps, no NaN)."""
  (x,) = node.srcs
  ax = -1 if node.attrs["dimension"] == "Width" else -2
  eps = float(node.attrs["eps"])
  mn = x.amin(ax)
  mx = x.amax(ax)
  return (x - mn) / (mx - mn).adds(eps)


def _flatten_to_reshape(node: Tensor) -> Tensor:
  """Rebuild a `flatten` bridge as a fused `reshape` (bit-identical contiguous collapse; removes the cut)."""
  (x,) = node.srcs
  return x.reshape(node.shape)


def _decompose_lrn(node: Tensor) -> Tensor:
  """Rebuild an `lrn` bridge as fused local_response_norm: `lrn(alpha,beta,k) == local_response_norm(size=C, alpha=alpha*C, beta, k)`. Bit-equivalent; removes the cut."""
  (x,) = node.srcs
  C = node.shape[1]
  a = node.attrs
  from .graph import local_response_norm
  return local_response_norm(x, size=C, alpha=a["alpha"] * C, beta=a["beta"], k=a["k"])


# op-name -> decomposition builder (the route registry in _capabilities.py decides WHICH are selectable)
_BRIDGE_DECOMPOSERS = {
  "sdpa": _decompose_sdpa,
  "minmax_norm": _decompose_minmax_norm,
  "flatten": _flatten_to_reshape,
  "lrn": _decompose_lrn,
}


def decompose_bridge(out: Tensor, decomp_idx: set[int]) -> Tensor:
  """Rewrite bridge nodes at the given original-graph topo indices to their fused decomposition (see `_BRIDGE_DECOMPOSERS`); others left as-is."""
  from ._compile import _topo
  order = _topo(out)
  sel = {id(order[i]) for i in decomp_idx
       if 0 <= i < len(order) and order[i].op in _BRIDGE_DECOMPOSERS}
  if not sel: return out
  r = Rule("bridge", "lossless", lambda t: t.op in _BRIDGE_DECOMPOSERS,
       lambda t: _BRIDGE_DECOMPOSERS[t.op](t))
  return graph_rewrite(out, [r], select=sel)


def paired_subtract(a, b):
  """Carry `a - b` through paired-fp16 (compensated TwoSum) -> a plain fp16 Tensor (cancellation fix). `a`/`b` may be Tensor or `Paired` (pass Paired for regime B)."""
  from ._paired import Paired, paired as _mk_paired
  pa = a if isinstance(a, Paired) else _mk_paired(a)
  pb = b if isinstance(b, Paired) else _mk_paired(b)
  return (pa - pb).to_tensor()


# --- CANON_RULES: lossless identity/redundancy elimination ---

def _compose_perm(outer: tuple, inner: tuple) -> tuple:
  """perm of (transpose(outer) o transpose(inner)): result[i] = inner[outer[i]]."""
  return tuple(inner[i] for i in outer)

# Each builder gets the REBUILT node; returns the simplified replacement.
def _b_drop_to_src(t: Tensor) -> Tensor: return t.srcs[0]                       # identity node -> its src

def _b_collapse_reshape(t: Tensor) -> Tensor:
  return Tensor(t.shape, "reshape", t.srcs[0].srcs, dict(t.srcs[0].attrs))      # reshape o reshape -> one reshape

def _b_collapse_transpose(t: Tensor) -> Tensor:
  inner = t.srcs[0]; perm = _compose_perm(t.attrs["perm"], inner.attrs["perm"])
  src = inner.srcs[0]
  if perm == tuple(range(len(perm))): return src                                # composes to identity -> drop both
  return Tensor(t.shape, "transpose", [src], {"perm": perm})

CANON_RULES: list[Rule] = [
  Rule("cast_same", "lossless",
       lambda t: t.op == "cast" and t.srcs[0].op != "input" and t.attrs.get("dtype", "fp16") == "fp16",
       _b_drop_to_src),                                                          # cast fp16->fp16 on a compute tensor (NOT the uint8 input port)
  Rule("reshape_same", "lossless", lambda t: t.op == "reshape" and t.shape == t.srcs[0].shape, _b_drop_to_src),
  Rule("reshape_chain", "lossless", lambda t: t.op == "reshape" and t.srcs[0].op == "reshape", _b_collapse_reshape),
  Rule("transpose_chain", "lossless", lambda t: t.op == "transpose" and t.srcs[0].op == "transpose", _b_collapse_transpose),
  Rule("mul_one", "lossless", lambda t: t.op == "muls" and t.attrs.get("k") == 1.0, _b_drop_to_src),
  Rule("add_zero", "lossless", lambda t: t.op == "adds" and t.attrs.get("k") == 0.0, _b_drop_to_src),
  Rule("not_not", "lossless", lambda t: t.op == "logical_not" and t.srcs[0].op == "logical_not", lambda t: t.srcs[0].srcs[0]),
]

def canonicalize(out: Tensor) -> Tensor:
  """Apply the lossless CANON_RULES to a fixed point; returns the SAME object when nothing matches."""
  while (nxt := graph_rewrite(out, CANON_RULES)) is not out: out = nxt
  return out


# --- NUMERIC_RULES: accuracy-affecting scalar-chain folding (fp16-gated) ---

def _b_fold_muls(t: Tensor) -> Tensor:
  """muls(b) o muls(a) -> muls(a*b). Folded in fp16 to match device semantics."""
  inner = t.srcs[0]; k = float(np.float16(inner.attrs["k"]) * np.float16(t.attrs["k"]))
  return Tensor(t.shape, "muls", inner.srcs, {"k": k})

def _b_fold_adds(t: Tensor) -> Tensor:
  """adds(b) o adds(a) -> adds(a+b). Folded in fp16 to match device semantics."""
  inner = t.srcs[0]; k = float(np.float16(inner.attrs["k"]) + np.float16(t.attrs["k"]))
  return Tensor(t.shape, "adds", inner.srcs, {"k": k})

NUMERIC_RULES: list[Rule] = [
  Rule("muls_chain", "numeric", lambda t: t.op == "muls" and t.srcs[0].op == "muls", _b_fold_muls),
  Rule("adds_chain", "numeric", lambda t: t.op == "adds" and t.srcs[0].op == "adds", _b_fold_adds),
]

# fp16 NumPy kernels for foldable ops; not bit-identical to the engine, so const_fold is NUMERIC-gated
def _f16(x: "np.ndarray") -> "np.ndarray": return np.asarray(x, np.float16)
_EvalFn = Callable[["list[np.ndarray]", "dict[str, Any]", "tuple[int, ...]"], "np.ndarray"]
_EVAL: dict[str, _EvalFn] = {
  "muls":    lambda s, a, sh=(): _f16(s[0] * _f16(a["k"])),
  "adds":    lambda s, a, sh=(): _f16(s[0] + _f16(a["k"])),
  "add":     lambda s, a, sh=(): _f16(s[0] + s[1]),
  "sub":     lambda s, a, sh=(): _f16(s[0] - s[1]),
  "mul":     lambda s, a, sh=(): _f16(s[0] * s[1]),
  "relu":    lambda s, a, sh=(): _f16(np.maximum(s[0], 0)),
  "reshape": lambda s, a, sh=(): _f16(s[0].reshape(sh)),
  "cast":    lambda s, a, sh=(): _f16(s[0]),
}

def _const_subgraph(t: Tensor, max_elems: int = 1 << 16) -> "np.ndarray | None":
  """fp16 value of `t` if its whole source cone is constant, foldable, and size-bounded; else None."""
  if math.prod(t.shape) > max_elems: return None
  memo: dict[int, "np.ndarray | None"] = {}
  def ev(n: Tensor) -> "np.ndarray | None":
    if id(n) in memo: return memo[id(n)]
    if n.op == "const_array": memo[id(n)] = _f16(n.attrs["value"]); return memo[id(n)]
    if n.op == "input" or n.op not in _EVAL: memo[id(n)] = None; return None
    srcs_maybe = [ev(s) for s in n.srcs]
    if any(s is None for s in srcs_maybe): memo[id(n)] = None; return None
    srcs = cast("list[np.ndarray]", srcs_maybe)
    fn = _EVAL[n.op]
    v: "np.ndarray" = fn(srcs, n.attrs, n.shape)
    memo[id(n)] = v; return v
  return ev(t)

def _b_const_fold(t: Tensor) -> Tensor:
  v = _const_subgraph(t); assert v is not None  # match guard makes None unreachable
  return Tensor(t.shape, "const_array", [], {"value": v})

NUMERIC_RULES.append(
  Rule("const_fold", "numeric",
       lambda t: t.op != "const_array" and t.op != "input" and _const_subgraph(t) is not None,
       _b_const_fold))
