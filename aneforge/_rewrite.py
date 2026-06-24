"""Graph-rewrite infrastructure for the aneforge optimizer.

Tensors are immutable/pure (op + srcs + attrs), so a "rewrite" is not in-place
mutation - it RE-DERIVES the output DAG, substituting new nodes at match sites
and rebuilding every node on the path from a match up to `out`. Shared subgraphs
are cloned once (memoized by node identity) so a diamond stays a diamond, not two
copies.

The public entry point is `rewrite(out, rule)`: `rule(t) -> Tensor | None` is a
node-local rewrite (return a replacement Tensor for `t`, or None to leave it). It
is applied bottom-up: a node's sources are rewritten first, then the node itself is
rebuilt on the (possibly new) sources, then `rule` is offered the rebuilt node.

On top of that, two concrete rewrites the optimizer uses:

  - `sdpa_to_decomposed` : replace one (or all) native-`sdpa` node(s) with the
    metamorphic-PROVEN bit-identical decomposed form
    `((q @ k^T) * scale).softmax(-1) @ v` - built from aneforge ops (bmm/muls/
    softmax), which fuse into ONE e5rt program (no native-SDPA graph cut). Being
    bit-identical (see the reverse-engineering corpus + tests/fuzz_metamorphic
    `mha_vs_sdpa`), it is LOSSLESS: the optimizer picks native-vs-decomposed
    purely by speed.

  - `set_node_int8` : tag a specific weight-bearing node with an `int8` attr so
    the compiler streams just that node's weight as int8 (a per-weight, LOSSY
    rewrite - opt-in, accuracy-gated by the tuner).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .graph import Tensor


@dataclass(frozen=True)
class Rule:
  """One graph-rewrite rule. `match`/`build` both see the REBUILT node (sources
  already rewritten); identity-based selection is delegated to graph_rewrite's
  `select`. `kind` is "lossless" (always-on canon) or "numeric" (tuner-gated)."""
  name: str; kind: str
  match: Callable[[Tensor], bool]; build: Callable[[Tensor], Tensor]


def graph_rewrite(out: Tensor, rules: list["Rule"], select: set[int] | None = None) -> Tensor:
  """Return a new output DAG, rules applied bottom-up in ONE memoized pass.

  Sources are rewritten first; the node is rebuilt on the (possibly new) sources;
  then the first rule whose `match` accepts the rebuilt node fires (its `build`
  replaces it). `select` (original-node id()s) gates eligibility; None = all
  eligible. A node with no applied rule and unchanged sources is returned
  unchanged (same object), so a no-op rule-set yields the SAME object."""
  memo: dict[int, Tensor] = {}
  def visit(t: Tensor) -> Tensor:
    if (c := memo.get(id(t))) is not None: return c
    new_srcs = [visit(s) for s in t.srcs]
    rebuilt = t if all(a is b for a, b in zip(new_srcs, t.srcs)) else Tensor(t.shape, t.op, new_srcs, dict(t.attrs))
    res = rebuilt
    if select is None or id(t) in select:
      for r in rules:
        if r.match(rebuilt): res = r.build(rebuilt); break
    memo[id(t)] = res; return res
  return visit(out)


def rewrite(out: Tensor, rule: Callable[[Tensor], Optional[Tensor]]) -> Tensor:
    """Return a NEW output DAG with `rule` applied bottom-up to every node.

    `rule(node) -> Tensor | None`: a replacement for `node` (already built on the
    rewritten sources), or None to keep the rebuilt node as-is. The clone is memoized
    by source-node identity, so shared subgraphs are rewritten exactly once and stay
    shared. Inputs (and any node `rule` leaves unchanged with unchanged sources) are
    returned unchanged, so a no-op rule yields the SAME object - the optimizer relies
    on that to keep opt=0 byte-identical.
    """
    # `rule` must be a pure node-local transform: it is called once in `match` and
    # again in `build`, so a stateful/non-deterministic rule would be unsafe. The
    # match-guard ensures `build` fires only when `rule` returns a Tensor (not None),
    # which is why the build-lambda's None-able return is type-ignored as safe.
    r = Rule("adhoc", "lossless", lambda t: rule(t) is not None, lambda t: rule(t))  # type: ignore[arg-type]
    return graph_rewrite(out, [r])


def _decompose_sdpa(node: Tensor) -> Tensor:
    """Build the decomposed equivalent of an `sdpa` node from aneforge ops.

    Matches the reverse-engineering corpus's `dec` exactly:
        scores = (q @ k.transpose([0,1,3,2])) * scale
        a      = scores.softmax(-1)
        out    = a @ v
    q/k/v are [1, H, S, D]; the result is [1, H, S, D]. Pure fused MIL - no cut."""
    q, k, v = node.srcs
    scale = node.attrs["scale"]
    scores = (q @ k.transpose([0, 1, 3, 2])) * float(scale)
    a = scores.softmax(-1)
    return a @ v


def sdpa_to_decomposed(out: Tensor, only_id: Optional[int] = None) -> Tensor:
    """Rewrite native `sdpa` node(s) to the decomposed (fused-MIL) form.

    `only_id` (an `id()` of a node in the ORIGINAL graph) rewrites just that one
    sdpa node; None rewrites every sdpa node. Returns a new output DAG (the same
    object if there was nothing to rewrite)."""
    sel = None if only_id is None else {only_id}
    r = Rule("sdpa", "lossless", lambda t: t.op == "sdpa", _decompose_sdpa)
    return graph_rewrite(out, [r], select=sel)


def list_sdpa_nodes(out: Tensor) -> list[Tensor]:
    """The sdpa nodes in topo order (the coordinate-descent route axes)."""
    from ._compile import _topo
    return [t for t in _topo(out) if t.op == "sdpa"]


def set_node_int8(out: Tensor, node_ids: set[int]) -> Tensor:
    """Return a new DAG where each weight-bearing node whose `id()` is in
    `node_ids` carries an `int8=True` attr (per-weight int8 override). The
    compiler's `weight()` honors `t.attrs['int8']` over the global flag."""
    if not node_ids: return out
    r = Rule("int8", "numeric", lambda t: t.op in ("matmul", "conv"),
             lambda t: Tensor(t.shape, t.op, t.srcs, {**t.attrs, "int8": True}))
    return graph_rewrite(out, [r], select=set(node_ids))


def list_weight_nodes(out: Tensor) -> list[Tensor]:
    """Weight-bearing (int8-eligible) nodes in topo order: matmul (streamed `wt`) and
    conv (baked `weight`). Both honor per-channel int8 (constexpr_affine_dequantize)
    as a routable weight operand on the ANE."""
    from ._compile import _topo
    return [t for t in _topo(out)
            if (t.op == "matmul" and isinstance(t.attrs.get("wt"), np.ndarray))
            or (t.op == "conv" and isinstance(t.attrs.get("weight"), np.ndarray))]


# --------------------------------------------------------------------------- #
# NUMERICS-aware rewrites (accuracy-improving, each validated vs fp32)          #
# --------------------------------------------------------------------------- #
def _reduce_sum_as_matmul(t: Tensor) -> Tensor:
    """Rebuild a single `reduce_sum` node as a contraction against a ones-vector,
    so the sum runs through the WIDE matmul accumulator instead of the NARROW
    reduce_sum accumulator. Mathematically identical (sum_k x_k == x @ 1), strictly
    >= accuracy under cancellation (fp16_envelope: comp_mm beats comp_rsum).

    Only the single-axis, last-axis case is rewritten here (the contraction matmul
    expresses directly): `x[..., K].sum(-1, keepdims) == x @ ones[K,1]`. For a
    non-last single axis we transpose the axis to the end, contract, and transpose
    back. Multi-axis sums are left to reduce_sum (the rewrite would need a reshape
    chain; not worth it, and the optimizer won't offer it)."""
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
        # x[..., K] @ ones[K, 1] -> [..., 1]   (keepdims, matches reduce_sum keep_dims)
        return src @ ones
    # move `ax` to the last position, contract, move the resulting size-1 dim back.
    perm = [i for i in range(rank) if i != ax] + [ax]
    inv = [0] * rank
    for new_pos, old in enumerate(perm):
        inv[old] = new_pos
    contracted = src.transpose(perm) @ ones          # [..., 1] with ax-data last
    return contracted.transpose(inv)                  # restore original axis order


def reduce_sum_to_matmul(out: Tensor, only_idx: set[int] | None = None) -> Tensor:
    """Rewrite single-axis `reduce_sum` node(s) to a `@ ones` matmul contraction
    (the WIDE-accumulator sum). LOSSLESS-or-better: identical math, >= accuracy under
    cancellation. `only_idx` selects nodes by topo index in the ORIGINAL graph;
    None rewrites every eligible reduce_sum. Returns a new output DAG (same object if
    nothing matched)."""
    from ._compile import _topo
    order = _topo(out)
    elig = lambda t: t.op == "reduce_sum" and len(t.attrs.get("axes", ())) == 1
    if only_idx is None: sel = {id(t) for t in order if elig(t)}
    else: sel = {id(order[i]) for i in only_idx if i < len(order) and elig(order[i])}
    if not sel: return out
    r = Rule("rsum_mm", "numeric", elig, _reduce_sum_as_matmul)
    return graph_rewrite(out, [r], select=sel)


# --------------------------------------------------------------------------- #
# BRIDGE-ELIMINATION rewrites (replace a native netplist cut with a fused        #
# decomposition built from _EMIT ops). Each is lossless - validated on-device     #
# to agree with the bridge within fp16 op-noise (tests/test_routes.py), same       #
# proof class as sdpa<->decomposed. Removing the cut lets the region fuse into     #
# one e5rt program (estimate() costs it cheaper accordingly).                       #
# --------------------------------------------------------------------------- #
def _decompose_minmax_norm(node: Tensor) -> Tensor:
    """Build the fused equivalent of a `minmax_norm` node from _EMIT ops:
        y = (x - amin(x, dim)) / (amax(x, dim) - amin(x, dim) + eps)
    over the reduction axis (Width=-1, Height=-2). reduce_min/reduce_max/sub/
    real_div are all fused; the `+ eps` is the scalar-add `adds` op. The native
    layer is bit-faithful to this formula (aneforge/_bridges/minmax_norm_fused.py header), so
    the rewrite is LOSSLESS: the two agree within fp16 op-noise (test_routes.py), and
    the degenerate max==min row yields 0/eps==0 on BOTH sides (no NaN divergence)."""
    (x,) = node.srcs
    ax = -1 if node.attrs["dimension"] == "Width" else -2
    eps = float(node.attrs["eps"])
    mn = x.amin(ax)
    mx = x.amax(ax)
    return (x - mn) / (mx - mn).adds(eps)


def _flatten_to_reshape(node: Tensor) -> Tensor:
    """Rebuild a `flatten` node (native Flatten bridge, [C,H,W] -> [prod]) as a plain
    fused `reshape` to the same 1-D shape. BIT-IDENTICAL (both are a contiguous
    row-major collapse - test_routes.py measures relerr 0.0), so it is LOSSLESS and
    removes the netplist cut entirely."""
    (x,) = node.srcs
    return x.reshape(node.shape)


def _decompose_lrn(node: Tensor) -> Tensor:
    """Rebuild an `lrn` node (native LocalResponseNormalization bridge, a graph cut)
    as the fused MIL `local_response_norm` op (one program, no cut). BIT-EQUIVALENT
    (test_routes.py measures cos 1.000000 / relerr 0.0 across C/alpha/beta/k): the
    bridge fixes the layer's channel window to N = C and divides the window sum by
    KernelChannel internally, so the fused op reproduces it with `size=C` and `alpha`
    pre-scaled by C (the fused op uses `alpha/size`; size=C cancels the C, recovering
    the bridge's TRUE effective alpha):
        lrn(alpha, beta, k)  ==  local_response_norm(size=C, alpha=alpha*C, beta, k)
    where C = x.shape[1]. LOSSLESS, so the optimizer picks native-vs-fused purely by
    speed - and the fused route removes the cut (~1250x faster on a conv->lrn->relu
    block). The fused op also drops the C<16 cap the bridge carries."""
    (x,) = node.srcs
    C = node.shape[1]
    a = node.attrs
    from .graph import local_response_norm
    return local_response_norm(x, size=C, alpha=a["alpha"] * C, beta=a["beta"], k=a["k"])


# rewrite op-name -> (matched bridge op, decomposition builder). The route registry
# in _capabilities.py is the single source OF truth for *which* bridge ops are route-
# selectable; this table is just the executable builders, keyed identically.
# Reconciled by tests/test_routes.py.
_BRIDGE_DECOMPOSERS = {
    "sdpa": _decompose_sdpa,
    "minmax_norm": _decompose_minmax_norm,
    "flatten": _flatten_to_reshape,
    "lrn": _decompose_lrn,
}


def decompose_bridge(out: Tensor, decomp_idx: set[int]) -> Tensor:
    """Rewrite the bridge nodes at the given ORIGINAL-graph topo indices to their fused
    decomposition (sdpa/minmax_norm/flatten/lrn - see `_BRIDGE_DECOMPOSERS`). Walks the
    original graph so the indices are stable, rebuilding bottom-up. A node whose op has
    no registered decomposer is left as-is (single-route). Returns a new output DAG."""
    from ._compile import _topo
    order = _topo(out)
    sel = {id(order[i]) for i in decomp_idx
           if 0 <= i < len(order) and order[i].op in _BRIDGE_DECOMPOSERS}
    if not sel: return out
    r = Rule("bridge", "lossless", lambda t: t.op in _BRIDGE_DECOMPOSERS,
             lambda t: _BRIDGE_DECOMPOSERS[t.op](t))
    return graph_rewrite(out, [r], select=sel)


def paired_subtract(a, b):
    """Carry a single `a - b` through paired-fp16 (compensated TwoSum), returning a
    plain fp16 Tensor whose value is the best-fp16 result of the compensated subtract.

    This is the CFG-style cancellation fix from fp16_envelope: the compensated
    subtract captures the subtract's own rounding (and, in regime B, the carried lo of
    paired inputs). `a` / `b` may each be a plain Tensor or a `Paired` (pass a Paired
    to exploit regime B, the recovering case). Higher op cost (~6 fp16 ops/elem) -> the
    optimizer gates this behind the error budget.

    Returns a Tensor (`.to_tensor()` of the Paired difference) so it drops straight
    into an existing fp16 graph at the hotspot."""
    from ._paired import Paired, paired as _mk_paired
    pa = a if isinstance(a, Paired) else _mk_paired(a)
    pb = b if isinstance(b, Paired) else _mk_paired(b)
    return (pa - pb).to_tensor()


# --------------------------------------------------------------------------- #
# CANON_RULES: lossless identity/redundancy elimination                        #
# --------------------------------------------------------------------------- #

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
  """Apply the lossless CANON_RULES to a fixed point (a rule may expose a fresh
  redundancy for the next pass). Returns the SAME object when nothing matches, so
  a clean graph is unchanged."""
  while (nxt := graph_rewrite(out, CANON_RULES)) is not out: out = nxt
  return out


# --------------------------------------------------------------------------- #
# NUMERIC_RULES: accuracy-affecting scalar-chain folding (fp16-gated)         #
# --------------------------------------------------------------------------- #

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

# fp16 NumPy kernels for the foldable ops. Compute reproduces device fp16 where it can,
# but is NEVER assumed bit-identical to the engine -> const_fold is gated (NUMERIC).
def _f16(x: "np.ndarray") -> "np.ndarray": return np.asarray(x, np.float16)
_EVAL: dict[str, "object"] = {
  "muls":    lambda s, a: _f16(s[0] * _f16(a["k"])),
  "adds":    lambda s, a: _f16(s[0] + _f16(a["k"])),
  "add":     lambda s, a: _f16(s[0] + s[1]),
  "sub":     lambda s, a: _f16(s[0] - s[1]),
  "mul":     lambda s, a: _f16(s[0] * s[1]),
  "relu":    lambda s, a: _f16(np.maximum(s[0], 0)),
  "reshape": lambda s, a, shape=None: _f16(s[0].reshape(shape)),
  "cast":    lambda s, a: _f16(s[0]),
}

def _const_subgraph(t: Tensor, max_elems: int = 1 << 16) -> "np.ndarray | None":
  """fp16 value of `t` if its entire source cone is constant (const_array leaves,
  no `input`) and the op set is foldable and the size is bounded; else None."""
  if math.prod(t.shape) > max_elems: return None
  memo: dict[int, "np.ndarray | None"] = {}
  def ev(n: Tensor) -> "np.ndarray | None":
    if id(n) in memo: return memo[id(n)]
    if n.op == "const_array": memo[id(n)] = _f16(n.attrs["value"]); return memo[id(n)]
    if n.op == "input" or n.op not in _EVAL: memo[id(n)] = None; return None
    srcs = [ev(s) for s in n.srcs]
    if any(s is None for s in srcs): memo[id(n)] = None; return None
    fn = _EVAL[n.op]  # type: ignore[index]
    v: "np.ndarray" = fn(srcs, n.attrs, n.shape) if n.op == "reshape" else fn(srcs, n.attrs)  # type: ignore[operator]
    memo[id(n)] = v; return v
  return ev(t)

def _b_const_fold(t: Tensor) -> Tensor:
  v = _const_subgraph(t); assert v is not None  # match guard makes None unreachable
  return Tensor(t.shape, "const_array", [], {"value": v})

NUMERIC_RULES.append(
  Rule("const_fold", "numeric",
       lambda t: t.op != "const_array" and t.op != "input" and _const_subgraph(t) is not None,
       _b_const_fold))
