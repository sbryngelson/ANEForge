"""Import an ONNX model and run it on the ANE. CNN-classifier op subset; see
docs. Public: load_onnx / onnx_to_tensor."""
from __future__ import annotations
from typing import Callable
from math import prod
from functools import reduce
import numpy as np
from .graph import Tensor, input as _input, conv as _conv, batch_norm as _bn, concat as _concat
from .graph import instance_norm as _instnorm, space_to_depth as _s2d
from .graph import local_response_norm as _lrnorm, depth_to_space as _d2s
from .graph import resize_bilinear as _rbilin, resize_nearest_neighbor as _rnn
from .graph import gather as _gather, topk as _topk, _const as _baked
from . import _compile

_ONNX: dict[str, Callable] = {}
def onnx_op(*names):
  """Register a handler `fn(node, ins, attrs, inits)` for the given ONNX op names."""
  def reg(fn):
    for n in names: _ONNX[n] = fn
    return fn
  return reg

def _attrs(node) -> dict:
  """ONNX node attributes as a name->value dict."""
  from onnx import helper
  return {a.name: helper.get_attribute_value(a) for a in node.attribute}

def _inits(graph) -> dict:
  """Graph initializers (weights) as name->np.ndarray."""
  from onnx import numpy_helper
  return {t.name: numpy_helper.to_array(t) for t in graph.initializer}

def _shape(vi) -> tuple:
  """Static shape tuple from a value_info; rejects dynamic/symbolic dims."""
  d = vi.type.tensor_type.shape.dim
  out = []
  for x in d:
    if x.HasField("dim_value"): out.append(int(x.dim_value))
    else: raise ValueError(f"onnx: dynamic/symbolic dim in '{vi.name}' - static shapes only")
  return tuple(out)

def _fold_conv_bn(m):
  """Fold each `Conv -> BatchNormalization` into the Conv (exact: the BN becomes per-output-channel weight
  scale + bias). This removes a standalone fp16 BN op, which on the ANE is the main precision loss for
  TF-exported models: the Conv reduces in a wide fp32-class accumulator, but a following BN runs in the fp16
  datapath on the large pre-BN values and its rounding then amplifies through the network. No-op unless the BN
  directly follows a single-consumer Conv whose weight (and bias, if any) are initializers. Returns the count."""
  from onnx import numpy_helper as nh
  g = m.graph
  init = {i.name: i for i in g.initializer}
  producer = {o: n for n in g.node for o in n.output}
  consumers: dict = {}
  for n in g.node:
    for inp in n.input:
      consumers.setdefault(inp, []).append(n)
  graph_outs = {o.name for o in g.output}
  folded = 0
  for bn in list(g.node):
    if bn.op_type != "BatchNormalization":
      continue
    conv = producer.get(bn.input[0])
    if conv is None or conv.op_type != "Conv" or bn.output[0] in graph_outs:
      continue
    if conv.output[0] in graph_outs or len(consumers.get(conv.output[0], [])) != 1:
      continue                                              # conv feeds only this BN
    if conv.input[1] not in init or any(bn.input[k] not in init for k in (1, 2, 3, 4)):
      continue
    if len(conv.input) > 2 and conv.input[2] not in init:  # dynamic conv bias -> skip
      continue
    W = nh.to_array(init[conv.input[1]])
    gm, be, mean, var = (nh.to_array(init[bn.input[k]]) for k in (1, 2, 3, 4))
    eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
    s = (gm / np.sqrt(var + eps)).astype(np.float32)
    Wf = (W * s.reshape([-1] + [1] * (W.ndim - 1))).astype(W.dtype)
    b0 = nh.to_array(init[conv.input[2]]) if len(conv.input) > 2 else 0.0
    bf = (be + (b0 - mean) * s).astype(np.float32)
    g.initializer.remove(init[conv.input[1]]); g.initializer.append(nh.from_array(Wf, conv.input[1]))
    if len(conv.input) > 2:
      g.initializer.remove(init[conv.input[2]]); g.initializer.append(nh.from_array(bf, conv.input[2]))
    else:
      bname = conv.output[0] + "_bnbias"; conv.input.append(bname); g.initializer.append(nh.from_array(bf, bname))
    for n in consumers.get(bn.output[0], []):               # rewire BN's consumers to the conv output
      for i, inp in enumerate(n.input):
        if inp == bn.output[0]:
          n.input[i] = conv.output[0]
    g.node.remove(bn); folded += 1
  return folded


def _load(path):
  """Load an ONNX model (or accept one in-memory), fold Conv->BN, and run shape inference."""
  import onnx
  m = path if hasattr(path, "graph") else onnx.load(path)
  _fold_conv_bn(m)                                          # exact; removes standalone fp16 BN (ANE precision)
  return onnx.shape_inference.infer_shapes(m)

_APPROX_RESIZE = False                          # opt-in: map a half-pixel Resize to the closest ANE bilinear

def onnx_to_tensor(path, approx_resize=False):
  """Build an aneforge graph from an ONNX model; returns (graph_inputs, output). `approx_resize` maps a
  half-pixel Resize (no exact ANE match) to the closest bilinear (~0.99 on smooth maps) instead of raising."""
  global _APPROX_RESIZE
  prev = _APPROX_RESIZE; _APPROX_RESIZE = approx_resize
  try:
    m = _load(path); g = m.graph
    inits = _inits(g)
    vals: dict[str, object] = dict(inits)        # name -> Tensor | np.ndarray (initializers as arrays)
    graph_inputs = []
    for vi in g.input:
      if vi.name in inits: continue              # initializers also listed as inputs in some exporters
      t = _input(_shape(vi)); vals[vi.name] = t; graph_inputs.append(t)
    _SCOPES.append(vals)                         # visible to If/Loop subgraphs (ONNX outer-scope names)
    try:
      _run_nodes(g, vals, inits)
    finally:
      _SCOPES.pop()
    name = g.output[0].name
    if name not in vals: raise ValueError(f"onnx: graph output '{name}' was never produced")
    out = vals[name]
    if not isinstance(out, Tensor): raise TypeError("onnx: graph output is not a Tensor")
    return graph_inputs, out
  finally:
    _APPROX_RESIZE = prev

_SCOPES: list[dict] = []                         # enclosing-graph value environments, innermost last

def _run_nodes(g, vals, inits):
  """Run a (sub)graph's topologically-ordered node list against the value environment `vals`."""
  for node in g.node:
    if node.op_type not in _ONNX:
      raise NotImplementedError(f"ONNX op '{node.op_type}' not supported")
    ins = [vals.get(n) if n else None for n in node.input]        # "" marks an omitted optional input
    outs = _ONNX[node.op_type](node, ins, _attrs(node), inits)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    for name, val in zip(node.output, outs):
      if name: vals[name] = val                                   # "" marks an unused optional output

def _run_subgraph(sub, binding, inits):
  """Import a subgraph (If branch / Loop body): outer-scope names resolve per ONNX scoping, the
  subgraph's formal inputs bind to `binding`, and its output values are returned in order."""
  from onnx import numpy_helper
  vals = dict(_SCOPES[-1]) if _SCOPES else {}
  vals.update({t.name: numpy_helper.to_array(t) for t in sub.initializer})
  vals.update(binding)
  _SCOPES.append(vals)
  try:
    _run_nodes(sub, vals, inits)
  finally:
    _SCOPES.pop()
  outs = []
  for o in sub.output:
    if o.name not in vals: raise ValueError(f"onnx: subgraph output '{o.name}' was never produced")
    outs.append(vals[o.name])
  return outs

@onnx_op("If")
def _if(node, ins, a, i):
  """If with a CONSTANT condition: the taken branch imports inline (the other is never built).
  A data-dependent condition has no ANE path (programs are single static graphs)."""
  if isinstance(ins[0], Tensor):
    raise NotImplementedError("ONNX If: data-dependent condition not supported (constant conditions fold at import)")
  cond = bool(np.asarray(ins[0]).ravel()[0])
  return _run_subgraph(a["then_branch"] if cond else a["else_branch"], {}, i)

@onnx_op("Loop")
def _loop(node, ins, a, i):
  """Loop with a STATIC trip count: the body unrolls at import (M copies of its ops in one program).
  Requires constant M, a constant-true (or absent) condition, and a body that keeps it true;
  scan outputs stack along a new leading axis, per the ONNX spec."""
  M = ins[0] if len(ins) > 0 else None
  if isinstance(M, Tensor): raise NotImplementedError("ONNX Loop: data-dependent trip count not supported")
  if M is None: raise NotImplementedError("ONNX Loop: a trip count is required (while-style loops cannot unroll)")
  M = int(np.asarray(M).ravel()[0])
  if M > 1024: raise NotImplementedError(f"ONNX Loop: trip count {M} too large to unroll into one program")
  cond = ins[1] if len(ins) > 1 else None
  if cond is not None and (isinstance(cond, Tensor) or not bool(np.asarray(cond).ravel()[0])):
    raise NotImplementedError("ONNX Loop: only a constant-true (or absent) condition unrolls")
  body = a["body"]; carried = list(ins[2:])
  n_car = len(carried); n_scan = len(body.output) - 1 - n_car
  scans: list[list] = [[] for _ in range(n_scan)]
  for it in range(M):
    binding = {body.input[0].name: np.array(it, np.int64), body.input[1].name: np.array(True)}
    for k, v in enumerate(carried): binding[body.input[2 + k].name] = v
    outs = _run_subgraph(body, binding, i)
    cond_out = outs[0]
    if isinstance(cond_out, Tensor) or not bool(np.asarray(cond_out).ravel()[0] if cond_out is not None else True):
      raise NotImplementedError("ONNX Loop: the body's condition output must stay constant-true to unroll")
    carried = outs[1:1 + n_car]
    for k in range(n_scan): scans[k].append(outs[1 + n_car + k])
  from .graph import concat as _cat
  stacked = []
  for parts in scans:                            # scan outputs concatenate along a new leading axis
    if all(_isc(p) for p in parts): stacked.append(np.stack([np.asarray(p) for p in parts])); continue
    ts = [(p if isinstance(p, Tensor) else _baked(np.asarray(p, np.float16))).expand_dims((0,)) for p in parts]
    stacked.append(_cat(ts, axis=0) if len(ts) > 1 else ts[0])
  return carried + stacked

def onnx_to_features(path):
  """Import a classifier and return `(inputs, features)` where `features` is the input to the final
  linear layer (a trailing softmax is peeled). Compile it for a frozen feature extractor — train a
  fresh head on the features for on-ANE transfer learning."""
  inputs, out = onnx_to_tensor(path)
  if out.op == "softmax" and out.srcs: out = out.srcs[0]   # peel a trailing softmax to the logits
  if out.op not in ("matmul", "bmm") or not out.srcs:
    raise ValueError(f"onnx_to_features: expected the model to end in a linear classifier (matmul/bmm); got '{out.op}'")
  return inputs, out.srcs[0]

def load_onnx(path, fuse_attention=False, approx_resize=False, **compile_kwargs):
  """Import an ONNX model and compile it to a runnable ANE Model. `fuse_attention` rewrites
  the `softmax(Q@K^T*scale)@V` pattern onto the native fused-attention layer (a graph cut).
  `approx_resize` allows a half-pixel Resize (no exact ANE match) via the closest bilinear."""
  _, out = onnx_to_tensor(path, approx_resize=approx_resize)
  if fuse_attention:
    from ._rewrite import graph_rewrite, Rule
    out = graph_rewrite(out, [Rule("fuse_sdpa", "numeric", _is_attention, _build_sdpa)])
  return _compile.compile(out, **compile_kwargs)

def _attn_scores(scaled):
  """The `scores * scale` step of attention: `muls(scores, k)` or `mul(scores, scalar const_array)`; return scores or None."""
  if scaled.op == "muls": return scaled.srcs[0]
  if scaled.op == "mul" and len(scaled.srcs) == 2:
    a, b = scaled.srcs
    if b.op == "const_array" and b.shape == (): return a
    if a.op == "const_array" and a.shape == (): return b
  return None

def _split_mask(pre):
  """softmax input is the scaled scores, or `add(scaled, mask_const)` (causal/additive mask). Returns (scaled, mask_const|None)."""
  if pre.op == "add" and len(pre.srcs) == 2:
    a, b = pre.srcs
    if b.op == "const_array": return a, b
    if a.op == "const_array": return b, a
  return pre, None

def _is_causal_mask(mc):
  """True if a const additive mask is the causal upper-triangular -inf (strict upper << 0, diagonal+lower ~ 0)."""
  m = np.asarray(mc.attrs["value"]).astype(np.float32)
  while m.ndim > 2 and m.shape[0] == 1: m = m[0]
  if m.ndim != 2 or m.shape[0] != m.shape[1]: return False
  return bool((m[np.triu_indices(m.shape[0], 1)] < -1e3).all() and (m[np.tril_indices(m.shape[0], 0)] > -1e3).all())

def _attn_parts(t):
  """Decompose a candidate attention `bmm` into (q, k, v, scaled, mask) or None if it is not the pattern."""
  if t.op != "bmm" or len(t.srcs) != 2: return None
  attn, v = t.srcs
  if attn.op != "softmax" or attn.attrs.get("axis") != len(attn.shape) - 1: return None
  scaled, mask = _split_mask(attn.srcs[0])
  scores = _attn_scores(scaled)
  if scores is None or scores.op != "bmm" or len(scores.srcs) != 2: return None
  q, kt = scores.srcs
  if kt.op != "transpose" or tuple(kt.attrs.get("perm", ())) != (0, 1, 3, 2): return None
  k = kt.srcs[0]
  if not (len(q.shape) == 4 and q.shape[0] == 1 and k.shape == v.shape
          and q.shape[1] == k.shape[1] and q.shape[3] == k.shape[3]): return None
  return q, k, v, scaled, mask

def _is_attention(t):
  """Match `softmax(q@kᵀ*scale [+ causal_mask]) @ v`; a non-causal additive mask is left to the generic path."""
  p = _attn_parts(t)
  return p is not None and (p[4] is None or _is_causal_mask(p[4]))

def _build_sdpa(t):
  from .graph import sdpa as _sdpa
  parts = _attn_parts(t); assert parts is not None    # guaranteed by _is_attention
  q, k, v, scaled, mask = parts
  scale = float(scaled.attrs["k"]) if scaled.op == "muls" else float(np.asarray(
    (scaled.srcs[1] if scaled.srcs[1].op == "const_array" else scaled.srcs[0]).attrs["value"]))
  return _sdpa(q, k, v, scale=scale, is_causal=mask is not None)   # causal mask -> native fused-attention layer

@onnx_op("Relu")
def _relu(node, ins, attrs, inits): return ins[0].relu()
@onnx_op("Sigmoid")
def _sig(node, ins, a, i): return ins[0].sigmoid()
@onnx_op("Tanh")
def _tanh(node, ins, a, i): return ins[0].tanh()
@onnx_op("Erf")
def _erf(node, ins, a, i): return ins[0].erf()
@onnx_op("Exp")
def _exp(node, ins, a, i): return ins[0].exp()
@onnx_op("Log")
def _log(node, ins, a, i): return ins[0].log()
@onnx_op("Sqrt")
def _sqrt(node, ins, a, i): return ins[0].sqrt()
@onnx_op("Abs")
def _abs(node, ins, a, i): return ins[0].abs()
@onnx_op("Sign")
def _sign(node, ins, a, i): return ins[0].sign()
@onnx_op("Sin")
def _sin(node, ins, a, i): return ins[0].sin()
@onnx_op("Cos")
def _cos(node, ins, a, i): return ins[0].cos()
@onnx_op("Softplus")
def _softplus(node, ins, a, i): return ins[0].softplus()
@onnx_op("Floor")
def _floor(node, ins, a, i): return ins[0].floor()
@onnx_op("Ceil")
def _ceil(node, ins, a, i): return ins[0].ceil()
@onnx_op("Round")
def _round(node, ins, a, i): return ins[0].round()
@onnx_op("Reciprocal")
def _recip(node, ins, a, i): return ins[0].inverse()
@onnx_op("Neg")
def _neg(node, ins, a, i): return ins[0] * -1.0
@onnx_op("Max")
def _max(node, ins, a, i):
  from .graph import maximum as _maximum
  ts = [v if isinstance(v, Tensor) else _baked(v) for v in ins]
  return reduce(_maximum, ts)
@onnx_op("Min")
def _min(node, ins, a, i):
  from .graph import minimum as _minimum
  ts = [v if isinstance(v, Tensor) else _baked(v) for v in ins]
  return reduce(_minimum, ts)
@onnx_op("Where")
def _where(node, ins, a, i):
  from .graph import select as _select
  return _select(ins[0], ins[1] if isinstance(ins[1], Tensor) else _baked(ins[1]),
                 ins[2] if isinstance(ins[2], Tensor) else _baked(ins[2]))
@onnx_op("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual")
def _compare(node, ins, a, i):
  """Elementwise comparison -> a BOOL tensor (feeds Where/Not); a constant operand bakes in."""
  x, y = _binops(ins)
  meth = {"Equal": "equal", "Greater": "greater", "GreaterOrEqual": "greater_equal",
          "Less": "less", "LessOrEqual": "less_equal"}[node.op_type]
  return getattr(x, meth)(y)
@onnx_op("Not")
def _not(node, ins, a, i): return ins[0].logical_not()
@onnx_op("Tile")
def _tile(node, ins, a, i): return ins[0].tile([int(v) for v in np.asarray(ins[1])])
@onnx_op("Expand")
def _expand(node, ins, a, i):
  """Broadcast-to-shape via expand_dims + tile; the target shape (input 1) must be constant. ONNX
  semantics: the output shape is broadcast(input.shape, shape), so a target dim of 1 keeps the input dim."""
  if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Expand: data-dependent shape not supported")
  shape = [int(v) for v in np.asarray(ins[1]).ravel()]
  if _isc(ins[0]):                                   # constant (shape-arithmetic) expand folds
    x0 = np.asarray(ins[0]); return np.broadcast_to(x0, np.broadcast_shapes(x0.shape, tuple(shape)))
  x = ins[0]; xs = list(x.shape)
  if len(shape) < len(xs): shape = [1] * (len(xs) - len(shape)) + shape
  if len(shape) > len(xs):
    x = x.expand_dims(tuple(range(len(shape) - len(xs)))); xs = [1] * (len(shape) - len(xs)) + xs
  reps = []
  for dx, dt in zip(xs, shape):
    if dt in (1, dx): reps.append(1)
    elif dx == 1: reps.append(dt)
    else: raise ValueError(f"ONNX Expand: cannot broadcast {tuple(xs)} to {tuple(shape)}")
  return x.tile(reps) if any(r != 1 for r in reps) else x
@onnx_op("Elu")
def _elu(node, ins, a, i): return ins[0].elu(float(a.get("alpha", 1.0)))
@onnx_op("LeakyRelu")
def _lrelu(node, ins, a, i): return ins[0].leaky_relu(float(a.get("alpha", 0.01)))
@onnx_op("Gelu")
def _gelu(node, ins, a, i):
  """ONNX Gelu (opset 20); preserve exact gelu or decompose the tanh approximation."""
  ap = a.get("approximate")
  if ap is not None and (ap.decode() if isinstance(ap, bytes) else ap) == "tanh":
    x = ins[0]
    inner = (x + x.pow(3.0) * 0.044715) * np.sqrt(2.0 / np.pi)
    return (x * 0.5) * inner.tanh().adds(1.0)
  return ins[0].gelu()
@onnx_op("PRelu")
def _prelu(node, ins, a, i): return ins[0].prelu(np.asarray(ins[1]).reshape(-1))   # slope=ins[1], [C,1,1]->[C]
@onnx_op("Selu")
def _selu(node, ins, a, i):
  """selu = gamma * elu(x, alpha) exactly (ONNX defaults are the Klambauer et al. constants)."""
  alpha = float(a.get("alpha", 1.6732632423543772)); gamma = float(a.get("gamma", 1.0507009873554805))
  return ins[0].elu(alpha) * gamma
@onnx_op("Celu")
def _celu(node, ins, a, i):
  """celu(x, alpha) = alpha * elu(x/alpha, 1): the positive branch recovers x, the negative alpha*(exp(x/alpha)-1)."""
  alpha = float(a.get("alpha", 1.0))
  return (ins[0] * (1.0 / alpha)).elu(1.0) * alpha
@onnx_op("Mish")
def _mish(node, ins, a, i): return ins[0] * ins[0].softplus().tanh()
@onnx_op("Softsign")
def _softsign(node, ins, a, i): return ins[0].softsign()
@onnx_op("ThresholdedRelu")
def _thresholded_relu(node, ins, a, i): return ins[0].thresholded_relu(float(a.get("alpha", 1.0)))
@onnx_op("Atan")
def _atan(node, ins, a, i): return ins[0].atan()
@onnx_op("Pow")
def _pow(node, ins, a, i): x, y = _binops(ins); return x.pow(y)
@onnx_op("InstanceNormalization")
def _instance_norm(node, ins, a, i):
  return _instnorm(ins[0], np.asarray(ins[1]), np.asarray(ins[2]), eps=float(a.get("epsilon", 1e-5)))
@onnx_op("SpaceToDepth")
def _space_to_depth(node, ins, a, i): return _s2d(ins[0], int(a["blocksize"]))
def _isc(v): return not isinstance(v, Tensor)      # a constant (numpy/scalar from a folded shape subgraph) vs a graph tensor
def _binops(ins):                                  # a constant operand bakes into the fused program (const_array -> GOC fold)
  return (ins[0] if isinstance(ins[0], Tensor) else _baked(ins[0]),
          ins[1] if isinstance(ins[1], Tensor) else _baked(ins[1]))
@onnx_op("Add")
def _add(node, ins, a, i):
  if _isc(ins[0]) and _isc(ins[1]): return np.asarray(ins[0]) + np.asarray(ins[1])   # shape arithmetic folds to a constant
  x, y = _binops(ins); return x + y
@onnx_op("Sub")
def _sub(node, ins, a, i):
  if _isc(ins[0]) and _isc(ins[1]): return np.asarray(ins[0]) - np.asarray(ins[1])
  x, y = _binops(ins); return x - y
@onnx_op("Mul")
def _mul(node, ins, a, i):
  if _isc(ins[0]) and _isc(ins[1]): return np.asarray(ins[0]) * np.asarray(ins[1])
  x, y = _binops(ins); return x * y
@onnx_op("Div")
def _div(node, ins, a, i):
  if _isc(ins[0]) and _isc(ins[1]):
    x0, x1 = np.asarray(ins[0]), np.asarray(ins[1])
    return x0 // x1 if np.issubdtype(x0.dtype, np.integer) else x0 / x1     # ONNX integer Div truncates
  x, y = _binops(ins); return x / y
@onnx_op("Mod")
def _mod(node, ins, a, i):
  """Mod with a constant divisor c: x - floor(x/c)*c (integer Mod, sign follows the divisor); the fmod=1
  variant truncates the quotient toward zero instead (sign follows the dividend), as in the Cast handler."""
  fmod = int(a.get("fmod", 0))
  if _isc(ins[0]) and _isc(ins[1]):
    x0, x1 = np.asarray(ins[0]), np.asarray(ins[1])
    return np.fmod(x0, x1) if fmod else np.mod(x0, x1)
  if isinstance(ins[1], Tensor):
    raise NotImplementedError("ONNX Mod: only a constant divisor is supported (got a tensor divisor)")
  x, c = _binops(ins)
  q = x / c
  t = q.sign() * q.abs().floor() if fmod else q.floor()
  return (q - t) * c        # frac(q)*c, not x - t*c: a const mul directly after floor is dropped by the fold
@onnx_op("Sum", "Mean")
def _sum_mean(node, ins, a, i):
  """Variadic elementwise Sum/Mean: folded with add; Mean scales by 1/N (a constant, exact)."""
  if all(_isc(v) for v in ins):                                                # shape arithmetic folds
    s = reduce(lambda p, q: p + q, (np.asarray(v) for v in ins))
    return s / len(ins) if node.op_type == "Mean" else s
  y = reduce(lambda p, q: p + q, (v if isinstance(v, Tensor) else _baked(v) for v in ins))
  return y * (1.0 / len(ins)) if node.op_type == "Mean" else y
@onnx_op("HardSigmoid")
def _hardsigmoid(node, ins, a, i):
  """ONNX HardSigmoid = clip(alpha*x + beta, 0, 1); defaults alpha=0.2, beta=0.5."""
  return (ins[0] * float(a.get("alpha", 0.2)) + float(a.get("beta", 0.5))).clip(0.0, 1.0)
@onnx_op("HardSwish")
def _hardswish(node, ins, a, i):
  """ONNX HardSwish(x) = x * relu6(x + 3) / 6 (the fixed-point hard-swish)."""
  return ins[0] * ((ins[0] + 3.0).relu6() * (1.0 / 6.0))
@onnx_op("Clip")
def _clip(node, ins, a, i):
  """Clip; opset<11 reads min/max attrs, opset>=11 reads inputs 2/3. (0,6)->relu6, (0,inf)->relu."""
  lo = a.get("min", -3.4e38); hi = a.get("max", 3.4e38)
  if len(ins) >= 2 and ins[1] is not None: lo = float(np.asarray(ins[1]))
  if len(ins) >= 3 and ins[2] is not None: hi = float(np.asarray(ins[2]))
  if lo == 0.0 and hi == 6.0: return ins[0].relu6()
  if lo == 0.0 and hi >= 3.4e38: return ins[0].relu()
  return ins[0].clip(float(lo), float(hi))

def _sattr(a, name, default):                 # ONNX string attrs arrive as bytes
  v = a.get(name)
  if v is None: return default
  return v.decode() if isinstance(v, bytes) else v

def _uniform(vals, op, default=1):            # ONNX gives per-axis lists; ANE takes a scalar
  if vals is None: return default             # spec default for strides/dilations is 1 per axis
  v = list(vals)
  if len(set(v)) != 1: raise NotImplementedError(f"onnx {op}: non-uniform {v} not supported")
  return int(v[0])

def _sympad(pads, op):                         # pads = [top,left,bottom,right]; require symmetric+uniform
  if pads is None: return 0
  p = list(pads)
  if len(set(p)) != 1: raise NotImplementedError(f"onnx {op}: asymmetric pads {p} not supported")
  return int(p[0])


def _pad_hw(x, top, bottom, left, right, value):
  """Explicitly pad an NCHW tensor by (top,bottom,left,right) with `value` -- for asymmetric pads the ANE
  pool's scalar pad can't express (e.g. TF SAME emits [0,0,1,1]). Padding is a concat of const borders."""
  from .graph import concat as _cat, _const
  N, C, H, W = x.shape
  if top or bottom:
    parts = ([_const(np.full((N, C, top, W), value, np.float16))] if top else []) + [x] + \
            ([_const(np.full((N, C, bottom, W), value, np.float16))] if bottom else [])
    x = _cat(parts, axis=2); H += top + bottom
  if left or right:
    parts = ([_const(np.full((N, C, H, left), value, np.float16))] if left else []) + [x] + \
            ([_const(np.full((N, C, H, right), value, np.float16))] if right else [])
    x = _cat(parts, axis=3)
  return x


def _pool_pad(x, pads, op, value):
  """(x, scalar_pad) for a pool: symmetric uniform pads pass through as a scalar; asymmetric pads are applied
  explicitly to `x` with `value` so the pool then runs with pad=0. `value` is a large negative for max_pool
  (pad cells never win the max); avg_pool with asymmetric pads is left unsupported (count_include_pad ambiguity)."""
  if pads is None: return x, 0
  p = [int(v) for v in pads]                    # [top, left, bottom, right]
  if len(set(p)) == 1: return x, p[0]
  if op != "MaxPool": raise NotImplementedError(f"onnx {op}: asymmetric pads {p} not supported")
  return _pad_hw(x, p[0], p[2], p[1], p[3], value), 0

def _pad4(pads):                               # ONNX conv pads=[top,left,bottom,right] -> scalar, or (top,bottom,left,right)
  if pads is None: return 0
  p = [int(v) for v in pads]
  return p[0] if len(set(p)) == 1 else (p[0], p[2], p[1], p[3])

def _pad_axis(x, ax, before, after, mode, value):
  """Pad one axis by concatenating border blocks: const blocks, repeated edge slices, a reversed
  interior slice (reflect), or the opposite end (wrap). All static; reflect requires pad < dim."""
  from .graph import concat as _cat
  def _sl(begin, size):
    b = [0] * len(x.shape); s = list(x.shape); b[ax] = begin; s[ax] = size
    return x.slice_by_size(b, s)
  def _blk(n):
    shp = list(x.shape); shp[ax] = n                 # only [ax] differs, so the pre-pad shape is right for both borders
    return _baked(np.full(shp, value, np.float16))
  dim = x.shape[ax]
  if mode == "constant":
    # nested 2-part concats: a 3-part concat with two const borders fails to lower once a second
    # padded axis stacks on top (Espresso "not implemented"); pairwise concats compile and are exact
    y = x
    if after: y = _cat([y, _blk(after)], axis=ax)
    if before: y = _cat([_blk(before), y], axis=ax)
    return y
  if mode == "edge":
    parts = [_sl(0, 1)] * before + [x] + [_sl(dim - 1, 1)] * after
  elif mode == "reflect":
    if before >= dim or after >= dim: raise NotImplementedError(f"ONNX Pad reflect: pad {before}/{after} >= dim {dim}")
    parts = ([_sl(1, before).reverse(ax)] if before else []) + [x] + ([_sl(dim - 1 - after, after).reverse(ax)] if after else [])
  elif mode == "wrap":
    parts = ([_sl(dim - before, before)] if before else []) + [x] + ([_sl(0, after)] if after else [])
  else:
    raise NotImplementedError(f"ONNX Pad: mode={mode!r} not supported")
  return _cat(parts, axis=ax) if len(parts) > 1 else x

@onnx_op("Pad")
def _pad_h(node, ins, a, i):
  """Pad (opset 13+): constant/edge/reflect/wrap via static concat of border blocks; pads/axes must be constant."""
  mode = _sattr(a, "mode", "constant")
  if len(ins) > 1 and isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Pad: data-dependent pads not supported")
  pads = [int(v) for v in np.asarray(ins[1]).ravel()] if len(ins) > 1 and ins[1] is not None \
    else [int(v) for v in a.get("pads", [])]
  value = float(a.get("value", 0.0))                 # opset<11 attr form
  if len(ins) > 2 and ins[2] is not None:
    if isinstance(ins[2], Tensor): raise NotImplementedError("ONNX Pad: data-dependent constant_value not supported")
    value = float(np.asarray(ins[2]).ravel()[0])
  if _isc(ins[0]):                                   # constant pad folds
    d = np.asarray(ins[0]); r = d.ndim
    axes = [int(v) % r for v in np.asarray(ins[3]).ravel()] if len(ins) > 3 and ins[3] is not None else list(range(r))
    width = [(0, 0)] * r
    for k, ax in enumerate(axes): width[ax] = (pads[k], pads[k + len(axes)])
    if mode == "constant": return np.pad(d, width, constant_values=value)
    return np.pad(d, width, mode)  # type: ignore[arg-type]  # np.pad stubs over-constrain mode to a callable protocol
  x = ins[0]; r = len(x.shape)
  axes = [int(v) % r for v in np.asarray(ins[3]).ravel()] if len(ins) > 3 and ins[3] is not None else list(range(r))
  if len(pads) != 2 * len(axes): raise ValueError(f"ONNX Pad: pads {pads} does not match axes {axes}")
  for k, ax in enumerate(axes):
    before, after = pads[k], pads[k + len(axes)]
    if before or after: x = _pad_axis(x, ax, before, after, mode, value)
  return x

@onnx_op("Cast")
def _cast(node, ins, a, i):
  """Cast at import level: constants convert; a float->float cast on an activation is identity (the engine
  computes fp16 regardless); float->int truncates toward zero as sign(x)*floor(|x|) (exact within fp16 range)."""
  from onnx import helper
  to = helper.tensor_dtype_to_np_dtype(int(a["to"]))
  if _isc(ins[0]): return np.asarray(ins[0]).astype(to)
  if np.issubdtype(to, np.floating): return ins[0]
  if np.issubdtype(to, np.integer): return ins[0].sign() * ins[0].abs().floor()
  raise NotImplementedError(f"ONNX Cast: target dtype {to} not supported on the ANE (fp16 datapath)")

@onnx_op("Shrink")
def _shrink(node, ins, a, i):
  """Shrink: x-bias above lambd, x+bias below -lambd, else 0 - nested select against baked thresholds."""
  from .graph import select as _select
  x = ins[0]; bias = float(a.get("bias", 0.0)); lambd = float(a.get("lambd", 0.5))
  lc = _baked(np.full(x.shape, lambd, np.float16)); nlc = _baked(np.full(x.shape, -lambd, np.float16))
  zero = _baked(np.zeros(x.shape, np.float16))
  return _select(x.greater(lc), x + (-bias), _select(x.less(nlc), x + bias, zero))

@onnx_op("OneHot")
def _onehot(node, ins, a, i):
  """OneHot: compare the index tensor against a baked arange along the inserted axis, then select
  on/off values. depth and values must be constant; a TENSOR index must be non-negative and in range
  (out-of-range simply never matches -> all off), and indices are exact in fp16 up to 2048."""
  from .graph import select as _select
  if isinstance(ins[1], Tensor) or isinstance(ins[2], Tensor):
    raise NotImplementedError("ONNX OneHot: data-dependent depth/values not supported")
  depth = int(np.asarray(ins[1]).ravel()[0]); off_v, on_v = (float(v) for v in np.asarray(ins[2]).ravel())
  if depth > 2048: raise NotImplementedError(f"ONNX OneHot: depth {depth} exceeds exact fp16 integer range")
  if _isc(ins[0]):
    idx = np.asarray(ins[0]).astype(np.int64)
    pos0 = int(a.get("axis", -1)) % (idx.ndim + 1)
    idxw = np.where(idx < 0, idx + depth, idx)               # ONNX: negatives wrap, out-of-range -> all off
    valid = ((idxw >= 0) & (idxw < depth)).astype(np.float32)
    hot = np.eye(depth, dtype=np.float32)[np.clip(idxw, 0, depth - 1)] * valid[..., None]
    return (off_v + (on_v - off_v) * np.moveaxis(hot, -1, pos0)).astype(np.float32)
  x = ins[0]; r = len(x.shape); pos = int(a.get("axis", -1)) % (r + 1)
  xe = x.expand_dims((pos,))
  rng_shape = [1] * (r + 1); rng_shape[pos] = depth
  rng = _baked(np.arange(depth, dtype=np.float16).reshape(rng_shape))
  out_shape = tuple(depth if k == pos else d for k, d in enumerate(xe.shape))
  on = _baked(np.full(out_shape, on_v, np.float16)); off = _baked(np.full(out_shape, off_v, np.float16))
  return _select(xe.equal(rng), on, off)

@onnx_op("DequantizeLinear")
def _dequant(node, ins, a, i):
  """int8/uint8 weight -> dequantized fp32 const (per-channel along `axis`); on an activation it is identity (the ANE computes in fp16)."""
  if isinstance(ins[0], Tensor): return ins[0]                       # activation Q/DQ pair is a fp16 passthrough
  x = np.asarray(ins[0]).astype(np.float32); scale = np.asarray(ins[1]).astype(np.float32)
  zp = np.asarray(ins[2]).astype(np.float32) if len(ins) > 2 and ins[2] is not None else np.float32(0)
  if scale.ndim and scale.size > 1:                                  # per-channel: broadcast scale/zp along `axis`
    shp = [1] * x.ndim; shp[int(a.get("axis", 1)) % x.ndim] = scale.size
    scale = scale.reshape(shp); zp = zp.reshape(shp) if zp.ndim else zp
  return (x - zp) * scale
@onnx_op("QuantizeLinear")
def _quant(node, ins, a, i):
  """On an activation: clip to the quant range (keeps fp16, skips int8 rounding) — this folds a relu/saturation the QDQ encoded (zp pinning qmin to 0). On a const: quantize."""
  scale = np.asarray(ins[1]); zp = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else np.array(0, np.int8)
  if isinstance(ins[0], Tensor):
    qmin, qmax = (0, 255) if zp.dtype == np.uint8 else (-128, 127)
    s, z = float(scale), float(np.asarray(zp))
    return ins[0].clip((qmin - z) * s, (qmax - z) * s)
  return np.clip(np.round(np.asarray(ins[0]) / scale) + zp, -128, 127).astype(zp.dtype)
@onnx_op("Conv")
def _conv_h(node, ins, a, i):
  ap = a.get("auto_pad")
  if ap is not None and (ap.decode() if isinstance(ap, bytes) else ap) != "NOTSET":
    raise NotImplementedError(f"ONNX Conv: auto_pad={ap!r} not supported (use explicit pads)")
  x = ins[0]; w = np.asarray(ins[1]); b = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  return _conv(x, w, stride=_uniform(a.get("strides"), "Conv"), pad=_pad4(a.get("pads")),
               dilation=_uniform(a.get("dilations"), "Conv"), groups=int(a.get("group", 1)), bias=b)

@onnx_op("MaxPool")
def _maxpool(node, ins, a, i):
  if _uniform(a.get("dilations"), "MaxPool") != 1: raise NotImplementedError("ONNX MaxPool: dilations != 1 not supported")
  x, pad = _pool_pad(ins[0], a.get("pads"), "MaxPool", -65504.0)   # -inf-ish const pad for asymmetric (TF SAME)
  return x.max_pool(_uniform(a.get("kernel_shape"), "MaxPool"), _uniform(a.get("strides"), "MaxPool"),
                    pad, ceil_mode=bool(int(a.get("ceil_mode", 0))))
@onnx_op("AveragePool")
def _avgpool(node, ins, a, i):
  return ins[0].avg_pool(_uniform(a.get("kernel_shape"), "AveragePool"), _uniform(a.get("strides"), "AveragePool"),
                         _sympad(a.get("pads"), "AveragePool"), ceil_mode=bool(int(a.get("ceil_mode", 0))),
                         exclude_pad=not int(a.get("count_include_pad", 0)))   # ONNX default count_include_pad=0 -> exclude pad cells
@onnx_op("GlobalAveragePool")
def _gap(node, ins, a, i): return ins[0].mean((2, 3))     # keepdims -> [N,C,1,1]
@onnx_op("GlobalMaxPool")
def _gmp(node, ins, a, i): return ins[0].amax((2, 3))     # keepdims -> [N,C,1,1]

@onnx_op("Gemm")
def _gemm(node, ins, a, i):
  """Y = alpha*op(A)@op(B) + beta*C. alpha folds into the weight and beta into the bias (both constants),
  so the general form still lowers to one linear; transA is a transpose on the activation."""
  x = ins[0]
  if int(a.get("transA", 0)): x = x.transpose((1, 0))
  W = np.asarray(ins[1]); B = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  if not int(a.get("transB", 0)): W = W.T            # x.linear expects [out,in]
  alpha, beta = float(a.get("alpha", 1.0)), float(a.get("beta", 1.0))
  if alpha != 1.0: W = (W * alpha).astype(W.dtype)
  if beta != 1.0 and B is not None: B = (B * beta).astype(B.dtype)
  return x.linear(W, B)
@onnx_op("MatMul")
def _matmul(node, ins, a, i):
  if not isinstance(ins[0], Tensor): raise NotImplementedError("ONNX MatMul: a constant first operand is not supported")
  b = ins[1]
  return ins[0] @ (np.asarray(b) if not isinstance(b, Tensor) else b)
@onnx_op("Einsum")
def _einsum_h(node, ins, a, i):
  """Routes to aneforge.einsum (matmul-reducible specs; diagonal/ellipsis reject as EinsumUnsupported)."""
  from .einsum import einsum as _es
  eq = _sattr(a, "equation", None)
  if not eq: raise ValueError("ONNX Einsum: missing equation attribute")
  ts = [v if isinstance(v, Tensor) else _baked(np.asarray(v, np.float32)) for v in ins]
  return _es(eq, *ts)  # type: ignore[arg-type]  # einsum.py imports Tensor absolutely; same class at runtime
@onnx_op("BatchNormalization")
def _bnorm(node, ins, a, i):
  x, s, bb, mean, var = ins[0], np.asarray(ins[1]), np.asarray(ins[2]), np.asarray(ins[3]), np.asarray(ins[4])
  return _bn(x, s, bb, mean, var, eps=float(a.get("epsilon", 1e-5)))
@onnx_op("LayerNormalization")
def _layernorm(node, ins, a, i):
  """LayerNorm over the trailing axes [axis:] (the transformer/ViT case); folds to native layer_norm on a 2-D view."""
  x = ins[0]; ax = int(a.get("axis", -1)) % len(x.shape)
  scale = np.asarray(ins[1]).reshape(-1)
  bias = np.asarray(ins[2]).reshape(-1) if len(ins) > 2 and ins[2] is not None else np.zeros_like(scale)
  m, d = prod(x.shape[:ax]) or 1, prod(x.shape[ax:])
  return x.reshape(m, d).layer_norm(scale, bias, float(a.get("epsilon", 1e-5))).reshape(*x.shape)
@onnx_op("LpNormalization")
def _lpnorm(node, ins, a, i):
  """p=2 maps to the native per-axis l2_norm; p=1 divides by the keepdims L1 reduce."""
  x = ins[0]; ax = int(a.get("axis", -1)) % len(x.shape); p = int(a.get("p", 2))
  if p == 2: return x.l2_norm(ax)
  if p == 1: return x / x.l1_norm((ax,))
  raise NotImplementedError(f"ONNX LpNormalization: p={p} not supported (only 1/2)")
@onnx_op("GroupNormalization")
def _groupnorm(node, ins, a, i):
  """GroupNormalization (opset 21 semantics: per-channel scale/bias) on NCHW via native group_norm."""
  x = ins[0]
  if len(x.shape) != 4: raise NotImplementedError("ONNX GroupNormalization: 4D NCHW inputs only")
  return x.group_norm(np.asarray(ins[1]).reshape(-1), np.asarray(ins[2]).reshape(-1),
                      int(a["num_groups"]), eps=float(a.get("epsilon", 1e-5)))
@onnx_op("RMSNormalization")
def _rmsnorm(node, ins, a, i):
  """RMSNorm over the trailing axes [axis:], like the LayerNormalization handler: 2-D view, native rms_norm, reshape back."""
  x = ins[0]; ax = int(a.get("axis", -1)) % len(x.shape)
  scale = np.asarray(ins[1]).reshape(-1)
  m, d = prod(x.shape[:ax]) or 1, prod(x.shape[ax:])
  return x.reshape(m, d).rms_norm(scale, eps=float(a.get("epsilon", 1e-5))).reshape(*x.shape)
@onnx_op("Reshape")
def _reshape(node, ins, a, i):
  shape = [int(v) for v in np.asarray(ins[1])]
  shape = [ins[0].shape[k] if d == 0 else d for k, d in enumerate(shape)]   # 0 = keep input dim (before -1)
  n = prod(ins[0].shape); known = prod(d for d in shape if d != -1)
  shape = [n // known if d == -1 else d for d in shape]                     # -1 = infer remaining
  return ins[0].reshape(*shape)
@onnx_op("Flatten")
def _flatten(node, ins, a, i): return ins[0].flatten2d(int(a.get("axis", 1)))
@onnx_op("Transpose")
def _transpose(node, ins, a, i):
  perm = a.get("perm"); perm = list(perm) if perm is not None else list(reversed(range(len(ins[0].shape))))
  return ins[0].transpose(perm)
@onnx_op("Squeeze")
def _squeeze(node, ins, a, i):
  axes = a.get("axes")                               # opset<13 attr; opset>=13 axes input
  if axes is None and len(ins) > 1 and ins[1] is not None: axes = [int(v) for v in np.asarray(ins[1])]
  if axes is None:                                   # squeeze-all: static shapes make it well-defined
    axes = [k for k, d in enumerate(ins[0].shape) if d == 1]
    if not axes: return ins[0]
  return ins[0].squeeze(tuple(axes))
@onnx_op("Unsqueeze")
def _unsqueeze(node, ins, a, i):
  axes = a.get("axes") or [int(v) for v in np.asarray(ins[1])]
  if _isc(ins[0]): return np.expand_dims(np.asarray(ins[0]), tuple(axes))   # shape-vector unsqueeze folds
  return ins[0].expand_dims(tuple(axes))
@onnx_op("Concat")
def _concat_h(node, ins, a, i):
  if all(_isc(v) for v in ins):                                             # concat of constant shape pieces folds
    return np.concatenate([np.atleast_1d(np.asarray(v)) for v in ins], axis=int(a["axis"]))
  return _concat(list(ins), axis=int(a["axis"]))
@onnx_op("Split")
def _split_h(node, ins, a, i):
  """Split along `axis`: equal parts use the native split; uneven sizes (the `split` attr/input) slice."""
  x = ins[0]; ax = int(a.get("axis", 0)) % len(x.shape)
  sizes = a.get("split")
  if sizes is None and len(ins) > 1 and ins[1] is not None:
    if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Split: data-dependent split sizes not supported")
    sizes = [int(v) for v in np.asarray(ins[1]).ravel()]
  if sizes is None:                                  # equal parts from the output count (last chunk may be smaller)
    n = int(a.get("num_outputs", len(node.output))); chunk = -(-x.shape[ax] // n)
    sizes = [chunk] * (n - 1) + [x.shape[ax] - chunk * (n - 1)]
  if sum(sizes) != x.shape[ax]: raise ValueError(f"ONNX Split: sizes {sizes} do not cover axis {ax} of {x.shape}")
  if len(set(sizes)) == 1:
    from .graph import split as _gsplit
    return _gsplit(x, len(sizes), axis=ax)
  outs, off = [], 0
  for s in sizes:
    begin = [0] * len(x.shape); size = list(x.shape); begin[ax] = off; size[ax] = s
    outs.append(x.slice_by_size(begin, size)); off += s
  return outs
@onnx_op("Shape")
def _shape_op(node, ins, a, i):
  """Static shape as an int64 constant (the channel-shuffle shape subgraph folds at import)."""
  s = np.array(ins[0].shape, dtype=np.int64); r = len(s)
  st = int(a.get("start", 0)); en = int(a.get("end", r))
  return s[(st + r if st < 0 else st):(en + r if en < 0 else en)]
@onnx_op("Slice")
def _slice(node, ins, a, i):
  starts = [int(v) for v in np.asarray(ins[1])]; ends = [int(v) for v in np.asarray(ins[2])]
  axes = [int(v) for v in np.asarray(ins[3])] if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
  steps = [int(v) for v in np.asarray(ins[4])] if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
  if _isc(ins[0]):                                   # constant (shape-arithmetic) slice folds
    d = np.asarray(ins[0]); sl = [slice(None)] * d.ndim
    for ax, s0, e0, st0 in zip(axes, starts, ends, steps): sl[ax % d.ndim] = slice(s0, e0, st0)
    return d[tuple(sl)]
  x = ins[0]; rank = len(x.shape); begin = [0] * rank; size = list(x.shape)   # activation slice -> static slice_by_size
  rev = []
  for ax, s0, e0, st0 in zip(axes, starts, ends, steps):
    ax %= rank; dim = x.shape[ax]
    if st0 == -1:                                    # full-extent flip -> native reverse (partial reversed slices stay unsupported)
      full = (s0 == -1 or s0 >= dim - 1) and e0 < -dim
      if not full: raise NotImplementedError("ONNX Slice: step=-1 is supported only as a full-axis flip")
      rev.append(ax); continue
    if st0 != 1: raise NotImplementedError(f"ONNX Slice: step={st0} on a tensor not supported")
    s0 = s0 + dim if s0 < 0 else min(s0, dim); e0 = e0 + dim if e0 < 0 else min(e0, dim)
    begin[ax] = max(0, s0); size[ax] = max(0, e0 - begin[ax])
  if rev: x = x.reverse(tuple(rev))
  if begin == [0] * rank and size == list(x.shape): return x
  return x.slice_by_size(begin, size)
@onnx_op("Softmax")
def _softmax(node, ins, a, i): return ins[0].softmax(int(a.get("axis", -1)))
@onnx_op("LogSoftmax")
def _logsoftmax(node, ins, a, i):
  """log_softmax = x - logsumexp(x, axis): the native stable reduce, where log(softmax(x)) underflows fp16."""
  x = ins[0]; ax = int(a.get("axis", -1)) % len(x.shape)
  return x - x.reduce_log_sum_exp((ax,))
@onnx_op("Constant")
def _const(node, ins, a, i):
  from onnx import numpy_helper
  return numpy_helper.to_array(a["value"])         # returns np.ndarray (folded by consumers)
@onnx_op("Identity")
def _identity(node, ins, a, i): return ins[0]      # pass-through (Tensor or array)
@onnx_op("ConstantOfShape")
def _const_of_shape(node, ins, a, i):
  """Constant fill from a static shape input; folds like Constant (consumers bake it)."""
  if isinstance(ins[0], Tensor): raise NotImplementedError("ONNX ConstantOfShape: data-dependent shape not supported")
  shape = tuple(int(v) for v in np.asarray(ins[0]).ravel())
  v = a.get("value")
  if v is None: return np.zeros(shape, np.float32)
  from onnx import numpy_helper
  arr = numpy_helper.to_array(v)
  return np.full(shape, arr.ravel()[0], dtype=arr.dtype)
@onnx_op("Range")
def _range(node, ins, a, i):
  """Constant-input Range folds to np.arange (shape-arithmetic plumbing)."""
  if any(isinstance(v, Tensor) for v in ins[:3]): raise NotImplementedError("ONNX Range: data-dependent bounds not supported")
  s, l, d = (np.asarray(v).ravel()[0] for v in ins[:3])
  return np.arange(s, l, d)
@onnx_op("EyeLike")
def _eyelike(node, ins, a, i):
  """Identity matrix of the input's (static) shape; only the shape is consumed, so a Tensor input is fine."""
  shape = ins[0].shape if isinstance(ins[0], Tensor) else np.asarray(ins[0]).shape
  if len(shape) != 2: raise NotImplementedError("ONNX EyeLike: 2D inputs only")
  return np.eye(shape[0], shape[1], k=int(a.get("k", 0)), dtype=np.float32)
@onnx_op("Trilu")
def _trilu(node, ins, a, i):
  """Upper/lower triangle. A constant folds; a Tensor multiplies by the baked 0/1 triangle mask."""
  k = 0
  if len(ins) > 1 and ins[1] is not None:
    if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Trilu: data-dependent k not supported")
    k = int(np.asarray(ins[1]).ravel()[0])
  tri = np.triu if int(a.get("upper", 1)) else np.tril
  if _isc(ins[0]): return tri(np.asarray(ins[0]), k)
  x = ins[0]
  if len(x.shape) < 2: raise NotImplementedError("ONNX Trilu: rank >= 2 required")
  mask = tri(np.ones(x.shape[-2:], np.float16), k)
  return x * _baked(np.ascontiguousarray(np.broadcast_to(mask, x.shape)))
@onnx_op("Dropout")
def _dropout(node, ins, a, i):
  """Inference no-op. Rejects training mode and a declared mask output (both training-time artifacts)."""
  if len(node.output) > 1: raise NotImplementedError("ONNX Dropout: mask output not supported")
  if len(ins) > 2 and ins[2] is not None and bool(np.asarray(ins[2])):
    raise NotImplementedError("ONNX Dropout: training_mode=1 not supported")
  return ins[0]
@onnx_op("LRN")
def _lrn(node, ins, a, i):
  """LRN; ANE local_response_norm folds alpha/size internally, so ONNX alpha maps raw and k=bias (validated on-device)."""
  return _lrnorm(ins[0], size=int(a["size"]), alpha=float(a.get("alpha", 1e-4)),
                 beta=float(a.get("beta", 0.75)), k=float(a.get("bias", 1.0)))
@onnx_op("DepthToSpace")
def _depth_to_space(node, ins, a, i):
  """DepthToSpace; ANE matches ONNX 'DCR' (the default) only - 'CRD' channel order is unsupported."""
  mode = _sattr(a, "mode", "DCR")
  if mode != "DCR": raise NotImplementedError(f"ONNX DepthToSpace: mode={mode!r} not supported (ANE matches 'DCR' only)")
  return _d2s(ins[0], int(a["blocksize"]))
@onnx_op("Resize")
def _resize(node, ins, a, i):
  """Resize (opset 11+): nearest/linear at the coord modes the ANE matches; cubic + half-pixel raise (validated on-device)."""
  x = ins[0]
  sizes = ins[3] if len(ins) > 3 and ins[3] is not None else None
  if sizes is not None and np.asarray(sizes).size:
    s = [int(v) for v in np.asarray(sizes)]; th, tw = s[-2], s[-1]
  else:
    scales = ins[2] if len(ins) > 2 and ins[2] is not None else None
    if scales is None or not np.asarray(scales).size:
      raise NotImplementedError("ONNX Resize: needs a non-empty scales or sizes input")
    sc = [float(v) for v in np.asarray(scales)]
    th = int(round(sc[-2] * x.shape[-2])); tw = int(round(sc[-1] * x.shape[-1]))
  mode = _sattr(a, "mode", "nearest"); ctm = _sattr(a, "coordinate_transformation_mode", "half_pixel")
  if mode == "cubic": raise NotImplementedError("ONNX Resize: mode='cubic' not supported")
  if mode == "nearest":
    if ctm != "asymmetric":
      raise NotImplementedError(f"ONNX Resize nearest: coordinate_transformation_mode={ctm!r} not supported (ANE matches 'asymmetric')")
    nm = _sattr(a, "nearest_mode", "round_prefer_floor")   # ONNX default; ANE samples with floor
    if nm != "floor": raise NotImplementedError(f"ONNX Resize: nearest_mode={nm!r} not supported (only 'floor')")
    return _rnn(x, th, tw)
  if mode == "linear":
    if ctm == "asymmetric": return _rbilin(x, th, tw, align_corners=False)
    if ctm == "align_corners": return _rbilin(x, th, tw, align_corners=True)
    if _APPROX_RESIZE and ctm in ("half_pixel", "pytorch_half_pixel"):
      return _rbilin(x, th, tw, align_corners=True)   # closest ANE convention (~0.99 on smooth maps; opt-in approx)
    raise NotImplementedError(f"ONNX Resize linear: coordinate_transformation_mode={ctm!r} not supported "
                              "(ANE matches 'asymmetric'/'align_corners'; pass approx_resize=True for half-pixel)")
  raise NotImplementedError(f"ONNX Resize: mode={mode!r} not supported")

def _reduce_op(ins, a, method):
  """ONNX reduce: axes attr (opset<18) or input (>=18); empty/absent -> all axes; keepdims=0 squeezes after."""
  x = ins[0]; axes = a.get("axes")
  if axes is None and len(ins) > 1 and ins[1] is not None:
    if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Reduce: data-dependent axes (non-constant) not supported")
    axes = [int(v) for v in np.asarray(ins[1]).ravel()]
  if not axes:
    if int(a.get("noop_with_empty_axes", 0)):
      raise NotImplementedError("ONNX Reduce: noop_with_empty_axes=1 with empty axes (identity) not supported")
    axes = list(range(len(x.shape)))                 # empty/absent -> reduce over all axes
  axes = tuple(int(v) % len(x.shape) for v in axes)
  y = method(x, axes)
  if not int(a.get("keepdims", 1)): y = y.squeeze(axes)   # ANE reduce keeps dims; drop them ourselves
  return y
@onnx_op("ReduceMax")
def _rmax(node, ins, a, i): return _reduce_op(ins, a, Tensor.amax)
@onnx_op("ReduceMin")
def _rmin(node, ins, a, i): return _reduce_op(ins, a, Tensor.amin)
@onnx_op("ReduceSum")
def _rsum(node, ins, a, i): return _reduce_op(ins, a, Tensor.sum)
@onnx_op("ReduceMean")
def _rmean(node, ins, a, i): return _reduce_op(ins, a, Tensor.mean)
@onnx_op("ReduceL1")
def _rl1(node, ins, a, i): return _reduce_op(ins, a, Tensor.l1_norm)
@onnx_op("ReduceL2")
def _rl2(node, ins, a, i): return _reduce_op(ins, a, lambda x, axes: x.sum_square(axes).sqrt())
@onnx_op("ReduceLogSum")
def _rlogsum(node, ins, a, i): return _reduce_op(ins, a, Tensor.log_sum)
@onnx_op("ReduceLogSumExp")
def _rlse(node, ins, a, i): return _reduce_op(ins, a, Tensor.reduce_log_sum_exp)
@onnx_op("ReduceSumSquare")
def _rss(node, ins, a, i): return _reduce_op(ins, a, Tensor.sum_square)
@onnx_op("CumSum")
def _cumsum(node, ins, a, i):
  """CumSum along `axis` (input 1, constant), via the tril-ones matmul lowering; a non-last axis is
  transposed to last and back (the swap perm is its own inverse)."""
  if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX CumSum: data-dependent axis not supported")
  if int(a.get("exclusive", 0)) or int(a.get("reverse", 0)):
    raise NotImplementedError("ONNX CumSum: exclusive/reverse not supported")
  x = ins[0]; r = len(x.shape); ax = int(np.asarray(ins[1])) % r
  if ax == r - 1: return x.cumsum(-1)
  perm = list(range(r)); perm[ax], perm[-1] = perm[-1], perm[ax]
  return x.transpose(perm).cumsum(-1).transpose(perm)
@onnx_op("Gather")
def _gather_h(node, ins, a, i):
  """Static-index gather along `axis`; constant scalar/1-D integer indices only (data-dependent indices have no ANE path)."""
  if _isc(ins[0]):                                   # gather over a constant shape vector folds (np.take preserves scalar-index rank reduction)
    return np.take(np.asarray(ins[0]), np.asarray(ins[1]).astype(int), axis=int(a.get("axis", 0)))
  if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX Gather: only constant integer indices supported")
  idx = np.asarray(ins[1])
  if idx.ndim > 1: raise NotImplementedError("ONNX Gather: only scalar or 1-D indices supported")
  ax = int(a.get("axis", 0)) % len(ins[0].shape)
  y = _gather(ins[0], [int(v) for v in idx.ravel()], axis=ax)
  if idx.ndim == 0: y = y.squeeze(ax)                # scalar index drops the gathered axis (ONNX rank rule)
  return y
@onnx_op("ArgMax")
def _argmax_h(node, ins, a, i):
  """ArgMax -> ANE argmax (2D [C,W] only, keepdims, indices fp16-encoded; GlobalArgMinMax bridge cuts the graph)."""
  x = ins[0]
  if len(x.shape) != 2: raise NotImplementedError("ONNX ArgMax: only 2D inputs supported on the ANE")
  if int(a.get("select_last_index", 0)): raise NotImplementedError("ONNX ArgMax: select_last_index=1 not supported")
  ax = int(a.get("axis", 0)) % 2
  y = x.argmax(ax)
  if not int(a.get("keepdims", 1)): y = y.squeeze(ax)
  return y
@onnx_op("ArgMin")
def _argmin_h(node, ins, a, i):
  """ArgMin as argmax(-x) (same 2D [C,W] bridge and index encoding as ArgMax)."""
  x = ins[0]
  if len(x.shape) != 2: raise NotImplementedError("ONNX ArgMin: only 2D inputs supported on the ANE")
  if int(a.get("select_last_index", 0)): raise NotImplementedError("ONNX ArgMin: select_last_index=1 not supported")
  ax = int(a.get("axis", 0)) % 2
  y = (x * -1.0).argmax(ax)
  if not int(a.get("keepdims", 1)): y = y.squeeze(ax)
  return y
@onnx_op("TopK")
def _topk_h(node, ins, a, i):
  """TopK -> ANE topk (2D [C,W], last axis only); returns VALUES only - ONNX's indices output is unsupported."""
  x = ins[0]
  if len(x.shape) != 2: raise NotImplementedError("ONNX TopK: only 2D inputs supported on the ANE")
  if int(a.get("axis", -1)) not in (-1, 1): raise NotImplementedError("ONNX TopK: only last-axis (width) topk supported")
  if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX TopK: data-dependent k (non-constant) not supported")
  k = int(np.asarray(ins[1]).ravel()[0])
  return _topk(x, k, largest=bool(int(a.get("largest", 1))))


# -- recurrent layers: LSTM / GRU / RNN unrolled over the static sequence length -- #

_RNN_DEFAULT_ACTS = {"LSTM": ["sigmoid", "tanh", "tanh"], "GRU": ["sigmoid", "tanh"], "RNN": ["tanh"]}

def _rnn_prep(node, ins, a, n_gates):
  """Shared RNN-family validation/unpacking. Returns (x, dirs, W, R, B, h0, c0, seq, batch, isz, H)."""
  op = node.op_type; x = ins[0]
  if not isinstance(x, Tensor): raise NotImplementedError(f"ONNX {op}: constant X not supported")
  if int(a.get("layout", 0)): raise NotImplementedError(f"ONNX {op}: layout=1 not supported")
  if a.get("clip") is not None: raise NotImplementedError(f"ONNX {op}: clip not supported")
  acts = a.get("activations")
  if acts is not None:
    want = _RNN_DEFAULT_ACTS[op]
    got = [(v.decode() if isinstance(v, bytes) else v).lower() for v in acts]
    if got != want * (len(got) // len(want)) or len(got) % len(want):
      raise NotImplementedError(f"ONNX {op}: non-default activations {got} not supported")
  for k, what in ((5, "initial_h"), (6, "initial_c")):
    if len(ins) > k and isinstance(ins[k], Tensor):
      raise NotImplementedError(f"ONNX {op}: tensor {what} not supported (constant or absent only)")
  seq, batch, isz = x.shape
  H = int(a["hidden_size"])
  d = _sattr(a, "direction", "forward")
  dirs = {"forward": [False], "reverse": [True], "bidirectional": [False, True]}.get(d)
  if dirs is None: raise NotImplementedError(f"ONNX {op}: direction={d!r} not supported")
  W = np.asarray(ins[1], np.float32); R = np.asarray(ins[2], np.float32)
  B = np.asarray(ins[3], np.float32) if len(ins) > 3 and ins[3] is not None else np.zeros((len(dirs), 2 * n_gates * H), np.float32)
  if len(ins) > 4 and ins[4] is not None:
    sl = np.asarray(ins[4]).ravel()
    if not (sl == seq).all(): raise NotImplementedError(f"ONNX {op}: per-sample sequence_lens not supported")
  h0 = np.asarray(ins[5], np.float32) if len(ins) > 5 and ins[5] is not None else np.zeros((len(dirs), batch, H), np.float32)
  c0 = np.asarray(ins[6], np.float32) if len(ins) > 6 and ins[6] is not None else np.zeros((len(dirs), batch, H), np.float32)
  if op == "LSTM" and len(ins) > 7 and ins[7] is not None and np.any(np.asarray(ins[7])):
    raise NotImplementedError("ONNX LSTM: peephole connections (P) not supported")
  return x, dirs, W, R, B, h0, c0, seq, batch, isz, H

def _rnn_assemble(ys_dirs, h_dirs, c_dirs):
  """Stack per-direction results: Y [seq, nd, b, H], Y_h [nd, b, H], Y_c [nd, b, H]."""
  from .graph import concat as _cat
  ydirs = []
  for ys in ys_dirs:                             # ys: per-timestep [b, H] in time order
    steps = [h.expand_dims((0, 1)) for h in ys]  # -> [1, 1, b, H]
    ydirs.append(_cat(steps, axis=0) if len(steps) > 1 else steps[0])
  Y = _cat(ydirs, axis=1) if len(ydirs) > 1 else ydirs[0]
  hs = [h.expand_dims((0,)) for h in h_dirs]
  Yh = _cat(hs, axis=0) if len(hs) > 1 else hs[0]
  if c_dirs is None: return Y, Yh
  cs = [c.expand_dims((0,)) for c in c_dirs]
  return Y, Yh, _cat(cs, axis=0) if len(cs) > 1 else cs[0]

def _xt(x, t, batch, isz):
  return x.slice_by_size([t, 0, 0], [1, batch, isz]).reshape(batch, isz)

@onnx_op("LSTM")
def _lstm(node, ins, a, i):
  """LSTM unrolled: per step two fused linears ([b,4H] gates, iofc order), sliced per gate. Forward,
  reverse, and bidirectional; default activations; constant weights/initial states; no peepholes/clip."""
  x, dirs, W, R, B, h0, c0, seq, batch, isz, H = _rnn_prep(node, ins, a, 4)
  def g(t2, k): return t2.slice_by_size([0, k * H], [batch, H])
  ys_dirs, h_dirs, c_dirs = [], [], []
  for d, rev in enumerate(dirs):
    bsum = B[d][:4 * H] + B[d][4 * H:]
    h, c = _baked(h0[d].astype(np.float16)), _baked(c0[d].astype(np.float16))
    ys: list = [None] * seq
    for t in (range(seq - 1, -1, -1) if rev else range(seq)):
      gt = _xt(x, t, batch, isz).linear(W[d], bsum) + h.linear(R[d], None)
      it, ot, ft = g(gt, 0).sigmoid(), g(gt, 1).sigmoid(), g(gt, 2).sigmoid()
      c = ft * c + it * g(gt, 3).tanh()
      h = ot * c.tanh()
      ys[t] = h
    ys_dirs.append(ys); h_dirs.append(h); c_dirs.append(c)
  return list(_rnn_assemble(ys_dirs, h_dirs, c_dirs))

@onnx_op("GRU")
def _gru(node, ins, a, i):
  """GRU unrolled (zrh gate order); both linear_before_reset forms; same scope as LSTM."""
  x, dirs, W, R, B, h0, _, seq, batch, isz, H = _rnn_prep(node, ins, a, 3)
  lbr = int(a.get("linear_before_reset", 0))
  def g(t2, k): return t2.slice_by_size([0, k * H], [batch, H])
  ys_dirs, h_dirs = [], []
  for d, rev in enumerate(dirs):
    Wb, Rb = B[d][:3 * H], B[d][3 * H:]
    h = _baked(h0[d].astype(np.float16))
    ys: list = [None] * seq
    for t in (range(seq - 1, -1, -1) if rev else range(seq)):
      xg = _xt(x, t, batch, isz).linear(W[d], Wb)
      hg = h.linear(R[d], Rb if lbr else None)
      z = (g(xg, 0) + g(hg, 0)).sigmoid()
      r = (g(xg, 1) + g(hg, 1)).sigmoid()
      if lbr:
        hh = (g(xg, 2) + r * g(hg, 2)).tanh()
      else:
        hh = (g(xg, 2) + (r * h).linear(R[d][2 * H:], Rb[2 * H:])).tanh()
      h = z * h + (z * -1.0 + 1.0) * hh
      ys[t] = h
    ys_dirs.append(ys); h_dirs.append(h)
  return list(_rnn_assemble(ys_dirs, h_dirs, None))

@onnx_op("RNN")
def _rnn_h(node, ins, a, i):
  """Vanilla tanh RNN unrolled; same scope as LSTM."""
  x, dirs, W, R, B, h0, _, seq, batch, isz, H = _rnn_prep(node, ins, a, 1)
  ys_dirs, h_dirs = [], []
  for d, rev in enumerate(dirs):
    bsum = B[d][:H] + B[d][H:]
    h = _baked(h0[d].astype(np.float16))
    ys: list = [None] * seq
    for t in (range(seq - 1, -1, -1) if rev else range(seq)):
      h = (_xt(x, t, batch, isz).linear(W[d], bsum) + h.linear(R[d], None)).tanh()
      ys[t] = h
    ys_dirs.append(ys); h_dirs.append(h)
  return list(_rnn_assemble(ys_dirs, h_dirs, None))
