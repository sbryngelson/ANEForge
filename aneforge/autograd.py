"""Reverse-mode autograd over the aneforge graph; forward and backward both run
on the ANE. See docs/developer/autograd.md."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import graph
from ._compile import Model as _Model
from .graph import Tensor

VJP: dict[str, Callable] = {}


def vjp(*names: str):
  """Register a vjp rule `fn(node, g) -> list[grad|None]` (one per node.srcs)."""
  def reg(fn):
    for n in names: VJP[n] = fn
    return fn
  return reg


def parameter(init) -> Tensor:
  """A trainable leaf: a graph input tagged trainable, holding an fp32 master
    value in attrs['value']; fed each eval and updated by the optimizer."""
  init = np.asarray(init, dtype=np.float32)
  t = graph.input(init.shape)
  t.attrs["trainable"] = True
  t.attrs["value"] = init
  return t


def _topo(out: Tensor, stop: set | None = None) -> list[Tensor]:
  stop = stop or set()
  seen, order = set(), []
  stack = [(out, False)]                      # iterative post-order (deep-graph safe)
  while stack:
    t, processed = stack.pop()
    if processed:
      order.append(t)
      continue
    if id(t) in seen: continue
    seen.add(id(t))
    stack.append((t, True))
    if id(t) not in stop:                   # treat stop tensors as leaves: don't recurse
      for s in t.srcs:
        if id(s) not in seen: stack.append((s, False))
  return order


def _const_like(t: Tensor, c: float) -> Tensor:
  """Constant `c` shaped like `t`, built from existing ops (no const-tensor op).
    `(t - t)` not `(t * 0.0)`: mul-by-zero on a reduce output trips an ANECCompile
    fusion wall; an exact-zero sub sidesteps it. See docs/developer/autograd.md."""
  return (t - t).adds(float(c))


def _ones_like(t: Tensor) -> Tensor:
  """A tensor of 1.0 shaped like t, built from existing ops (no const-tensor op)."""
  return _const_like(t, 1.0)


def _reverse(order, g, params, stop: set | None = None):
  """Reverse-accumulate over topo `order` from seeded grads `g` (id->Tensor) ->
    {param: grad}. Tensors in `stop` accumulate but don't propagate (detach)."""
  stop = stop or set()
  for t in reversed(order):
    gt = g.get(id(t))
    if gt is None or not t.srcs or id(t) in stop: continue
    rule = VJP.get(t.op)
    if rule is None:
      raise NotImplementedError(f"autograd: no vjp for op {t.op!r}")
    for src, gs in zip(t.srcs, rule(t, gt)):
      if gs is None: continue
      g[id(src)] = gs if id(src) not in g else (g[id(src)] + gs)
  return {p: g[id(p)] for p in params}


def backward(loss: Tensor, params, loss_scale: float = 1.0, stop=None) -> dict:
  """Reverse-mode grads of scalar `loss` wrt each Tensor in `params` ->
    {param: grad}. `stop` is the stop-gradient (detach) frontier (defaults to
    `params`); it matters for unrolled training. See docs/developer/autograd.md."""
  stop_ids = {id(t) for t in (params if stop is None else stop)}
  order = _topo(loss, stop_ids)
  return _reverse(order, {id(loss): _const_like(loss, float(loss_scale))}, params, stop_ids)


def backward_from(grad_root, root, params, stop=None) -> dict:
  """Reverse-mode from an explicit gradient `grad_root` at `root` (e.g. logits),
    not from a scalar loss + ones-seed. `stop` as in `backward`."""
  stop_ids = {id(t) for t in (params if stop is None else stop)}
  return _reverse(_topo(root, stop_ids), {id(root): grad_root}, params, stop_ids)


@vjp("muls")
def _vjp_muls(t, g):
  return [g * float(t.attrs["k"])]


@vjp("reduce_sum")
def _vjp_reduce_sum(t, g):
  x = t.srcs[0]
  return [g * _ones_like(x)]                   # broadcast g back to the source shape


def _unbroadcast(g, shape):
  """Sum g down to `shape` (reverse of numpy broadcasting)."""
  shape = tuple(shape)
  while len(g.shape) > len(shape):             # sum leading extra dims
    g = g.sum((0,))            # reduce axis 0 (keepdims -> shape[0]=1)
    g = g.reshape(g.shape[1:])
  axes = tuple(i for i, (d, s) in enumerate(zip(g.shape, shape)) if s == 1 and d != 1)
  if axes: g = g.sum(axes)
  if g.shape != shape: g = g.reshape(shape)
  return g


@vjp("add")
def _vjp_add(t, g):
  a, b = t.srcs
  return [_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)]


@vjp("sub")
def _vjp_sub(t, g):
  a, b = t.srcs
  return [_unbroadcast(g, a.shape), _unbroadcast(g * -1.0, b.shape)]


@vjp("mul")
def _vjp_mul(t, g):
  a, b = t.srcs
  return [_unbroadcast(g * b, a.shape), _unbroadcast(g * a, b.shape)]


@vjp("adds")
def _vjp_adds(t, g):
  return [g]


@vjp("square")
def _vjp_square(t, g):
  x = t.srcs[0]
  return [g * (x * 2.0)]


def _T(t):
  """Transpose the last two axes of a 2-D/N-D tensor."""
  perm = list(range(len(t.shape)))
  perm[-1], perm[-2] = perm[-2], perm[-1]
  return t.transpose(perm)


@vjp("bmm")
def _vjp_bmm(t, g):
  # ga = g @ b^T ; gb = a^T @ g. _unbroadcast handles a/b broadcasting over the
  # batch dim (trainable conv); a no-op for equal-batch bmm (MLP/attention).
  a, b = t.srcs
  ga = g @ _T(b)
  gb = _T(a) @ g
  return [_unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)]


@vjp("matmul")
def _vjp_matmul(t, g):
  # baked weight (attrs['wt']=W^T [N,K]); only the activation gets a grad: gx = g @ wt.
  wt = t.attrs["wt"]
  gx = g @ wt
  return [gx]


@vjp("reduce_mean")
def _vjp_reduce_mean(t, g):
  # tensor*tensor broadcasts g to x and scales by 1/n in one wall-proof op.
  x = t.srcs[0]
  n = math.prod(x.shape[a] for a in t.attrs["axes"])
  return [g * _const_like(x, 1.0 / n)]


@vjp("softmax")
def _vjp_softmax(t, g):
  x = t.srcs[0]
  axis = t.attrs["axis"]
  y = x.softmax(axis)                 # recompute the forward softmax (stable native op)
  s = (g * y).sum((axis,))            # keepdims sum over the softmax axis
  return [y * (g - s)]                # broadcasts s back over `axis`


@vjp("gelu")
def _vjp_gelu(t, g):
  x = t.srcs[0]
  c1 = 1.0 / math.sqrt(2.0)              # erf(x/sqrt2)
  c2 = 1.0 / math.sqrt(2.0 * math.pi)
  cdf = ((x * c1).erf()).adds(1.0) * 0.5          # 0.5*(1+erf(x/sqrt2))
  pdf = ((x.square() * -0.5).exp()) * c2          # phi(x)
  dz = cdf + x * pdf                              # gelu'(x)
  return [g * dz]


@vjp("tanh")
def _vjp_tanh(t, g):
  th = t.srcs[0].tanh()
  return [g * (th.square() * -1.0).adds(1.0)]      # g*(1 - tanh^2)


@vjp("sigmoid")
def _vjp_sigmoid(t, g):
  s = t.srcs[0].sigmoid()
  return [g * (s * (s * -1.0).adds(1.0))]          # g*s*(1-s)


@vjp("silu")
def _vjp_silu(t, g):
  x = t.srcs[0]
  s = x.sigmoid()
  one_minus_s = (s * -1.0).adds(1.0)               # 1 - s
  dz = s * (x * one_minus_s).adds(1.0)             # silu'(x) = s*(1 + x*(1-s))
  return [g * dz]


# normalization vjps: gamma (a per-channel const) is re-injected as a fed
# value-input; closed-form grad_x. See docs/developer/autograd.md.
def _gamma_input(t):
  """gamma from a norm node's attrs, shaped [1,...,1,D] as a fed value-input."""
  gamma = np.asarray(t.attrs["gamma"], np.float32)
  gshape = (1,) * (len(t.shape) - 1) + (t.shape[-1],)
  gt = graph.input(gshape)
  gt.attrs["value"] = gamma.reshape(gshape)
  return gt


@vjp("rms_norm")
def _vjp_rms_norm(t, g):
  x = t.srcs[0]; last = (len(x.shape) - 1,)
  eps = float(t.attrs["eps"]); gn = g * _gamma_input(t)        # g * gamma
  rinv = x.square().mean(last).adds(eps).rsqrt()               # 1/rms  [..,1]
  n = x * rinv                                                 # x/rms
  m = (gn * n).mean(last)                                      # mean(gn*n) [..,1]
  return [(gn - n * m) * rinv]                                 # rinv*(gn - n*mean(gn*n))


@vjp("layer_norm")
def _vjp_layer_norm(t, g):
  x = t.srcs[0]; last = (len(x.shape) - 1,)
  eps = float(t.attrs["eps"]); gn = g * _gamma_input(t)        # g * gamma (beta drops out)
  xc = x - x.mean(last)                                        # x - mean
  rstd = xc.square().mean(last).adds(eps).rsqrt()              # 1/std  [..,1]
  n = xc * rstd                                                # normalized
  grad = gn - gn.mean(last) - n * (gn * n).mean(last)          # gn - mean(gn) - n*mean(gn*n)
  return [grad * rstd]


@vjp("channel_layer_norm")
def _vjp_channel_layer_norm(t, g):
  x = t.srcs[0]; ax = (1,)                                     # LayerNorm over the channel axis
  eps = float(t.attrs["eps"]); C = x.shape[1]
  gamma = np.asarray(t.attrs["gamma"], np.float32)            # per-channel gamma as [1,C,1,1]
  gt = graph.input((1, C, 1, 1)); gt.attrs["value"] = gamma.reshape(1, C, 1, 1)
  gn = g * gt                                                  # g * gamma (beta drops out)
  xc = x - x.mean(ax)                                          # x - channel mean
  rstd = xc.square().mean(ax).adds(eps).rsqrt()              # 1/std  [N,1,1,S]
  n = xc * rstd                                                # normalized
  grad = gn - gn.mean(ax) - n * (gn * n).mean(ax)            # same LayerNorm backward, over C
  return [grad * rstd]


# unary math / activation vjps: closed-form derivatives from existing ops; the
# forward output `t` is reused where it IS the derivative term (exp/inverse/rsqrt).
@vjp("exp")
def _vjp_exp(t, g):
  return [g * t]                                   # exp'(x) = exp(x) = t


@vjp("sqrt")
def _vjp_sqrt(t, g):
  return [g * (t.srcs[0].rsqrt() * 0.5)]           # 0.5/sqrt(x)


@vjp("rsqrt")
def _vjp_rsqrt(t, g):
  return [g * (t * t * t * -0.5)]                  # -0.5*rsqrt(x)^3, t = rsqrt(x)


@vjp("inverse")
def _vjp_inverse(t, g):
  return [g * (t * t * -1.0)]                      # -1/x^2 = -t^2, t = 1/x


@vjp("log")
def _vjp_log(t, g):
  return [g * t.srcs[0].inverse()]                 # 1/x


@vjp("erf")
def _vjp_erf(t, g):
  c = 2.0 / math.sqrt(math.pi)
  return [g * ((t.srcs[0].square() * -1.0).exp() * c)]   # 2/sqrt(pi)*exp(-x^2)


@vjp("cos")
def _vjp_cos(t, g):
  return [g * (t.srcs[0].sin() * -1.0)]            # -sin(x)


@vjp("abs")
def _vjp_abs(t, g):
  x = t.srcs[0]
  sign = graph.select(x.greater(_const_like(x, 0.0)), _ones_like(x), _const_like(x, -1.0))
  return [g * sign]


@vjp("leaky_relu")
def _vjp_leaky_relu(t, g):
  x = t.srcs[0]; a = float(t.attrs["alpha"])
  return [g * graph.select(x.greater(_const_like(x, 0.0)), _ones_like(x), _const_like(x, a))]


@vjp("prelu")
def _vjp_prelu(t, g):
  x = t.srcs[0]; C = x.shape[1]
  alpha = np.asarray(t.attrs["alpha"], np.float32)
  ashape = (1, C) + (1,) * (len(x.shape) - 2)
  at = graph.input(ashape); at.attrs["value"] = alpha.reshape(ashape)
  ab = at + _const_like(x, 0.0)                     # broadcast per-channel alpha to x's shape
  return [g * graph.select(x.greater(_const_like(x, 0.0)), _ones_like(x), ab)]


@vjp("elu")
def _vjp_elu(t, g):
  x = t.srcs[0]; a = float(t.attrs["alpha"])
  return [g * graph.select(x.greater(_const_like(x, 0.0)), _ones_like(x), x.exp() * a)]


@vjp("relu6")
def _vjp_relu6(t, g):
  x = t.srcs[0]; z = _const_like(x, 0.0); o = _ones_like(x)
  inner = graph.select(x.greater(_const_like(x, 6.0)), z, o)    # x>6 -> 0 else 1
  return [g * graph.select(x.greater(z), inner, z)]             # 1 iff 0<x<6


@vjp("clip")
def _vjp_clip(t, g):
  x = t.srcs[0]; lo = float(t.attrs["lo"]); hi = float(t.attrs["hi"])
  z = _const_like(x, 0.0); o = _ones_like(x)
  inner = graph.select(x.greater(_const_like(x, hi)), z, o)
  return [g * graph.select(x.greater(_const_like(x, lo)), inner, z)]


@vjp("scaled_tanh")
def _vjp_scaled_tanh(t, g):
  x = t.srcs[0]; a = float(t.attrs["alpha"]); b = float(t.attrs["beta"])
  th = (x * b).tanh()
  return [g * ((th.square() * -1.0).adds(1.0) * (a * b))]       # a*b*(1-tanh^2(b*x))


@vjp("sigmoid_hard")
def _vjp_sigmoid_hard(t, g):
  x = t.srcs[0]; a = float(t.attrs["alpha"]); b = float(t.attrs["beta"])
  u = (x * a).adds(b); z = _const_like(x, 0.0); oa = _const_like(x, a)
  inner = graph.select(u.greater(_const_like(x, 1.0)), z, oa)   # hard-sigmoid: a in (0,1)
  return [g * graph.select(u.greater(z), inner, z)]


@vjp("l2_norm")
def _vjp_l2_norm(t, g):
  x = t.srcs[0]
  axis = int(t.attrs.get("axis", -1)); ax = (axis if axis >= 0 else len(x.shape) + axis,)
  eps = float(t.attrs.get("eps", 1e-12))
  rinv = x.square().sum(ax).adds(eps).rsqrt()      # 1/||x||
  n = x * rinv
  return [(g - n * (g * n).sum(ax)) * rinv]        # rinv*(g - n*sum(g*n))


@vjp("reverse")
def _vjp_reverse(t, g):
  return [g.reverse(t.attrs["axes"])]                # reverse the cotangent back


@vjp("reduce_log_sum_exp")
def _vjp_rlse(t, g):
  x = t.srcs[0]
  return [g * (x - t).exp()]                         # g * softmax(x) (t = LSE, keepdims)


@vjp("pow")
def _vjp_pow(t, g):
  x, p = t.srcs
  gx = g * (p * x.pow(p.adds(-1.0)))                 # g * p * x^(p-1)
  gp = g * (t * x.log())                             # g * x^p * ln(x)
  return [gx, gp]


@vjp("group_norm")
def _vjp_group_norm(t, g):
  x = t.srcs[0]                                     # [N, C, H, W]
  N, C, H, W = x.shape
  G = int(t.attrs["groups"]); eps = float(t.attrs["eps"]); M = (C // G) * H * W
  gamma = np.asarray(t.attrs["gamma"], np.float32)
  gt = graph.input((1, C, 1, 1)); gt.attrs["value"] = gamma.reshape(1, C, 1, 1)
  gn = (g * gt).reshape(N, G, M)                    # (g*gamma) grouped
  xc = x.reshape(N, G, M); xc = xc - xc.mean((2,))  # per-group center
  rstd = xc.square().mean((2,)).adds(eps).rsqrt()   # per-group 1/std
  n = xc * rstd
  grad = gn - gn.mean((2,)) - n * (gn * n).mean((2,))   # LN backward within group
  return [(grad * rstd).reshape(N, C, H, W)]


# structural vjps (transpose / reshape): pure index re-arrangements, fp16-exact;
# unlock transformer (af.mha) training. See docs/developer/autograd.md.

@vjp("transpose")
def _vjp_transpose(t, g):
  """dx = transpose(g, inverse-perm)."""
  perm = t.attrs["perm"]
  inv = [0] * len(perm)
  for i, p in enumerate(perm): inv[p] = i
  return [g.transpose(tuple(inv))]


@vjp("reshape")
def _vjp_reshape(t, g):
  """dx = reshape(g, x.shape)."""
  x = t.srcs[0]
  return [g.reshape(x.shape)]


@vjp("flatten2d")
def _vjp_flatten2d(t, g):
  """flatten2d is a reshape; dx reshapes back to the source."""
  x = t.srcs[0]
  return [g.reshape(x.shape)]


@vjp("slice_by_size")
def _vjp_slice_by_size(t, g):
  """dx scatters g back into a zeros-like-x at the slice offset, built on-ANE by
    concatenating zero-slices around g (no scatter op). See docs/developer/autograd.md."""
  x = t.srcs[0]
  begin, size, full = t.attrs["begin"], t.attrs["size"], x.shape
  z = _const_like(x, 0.0)                          # zeros shaped like x
  out, cur = g, list(size)
  for ax in range(len(full)):
    before, after = begin[ax], full[ax] - begin[ax] - size[ax]
    if before == 0 and after == 0: continue
    pieces = []
    if before > 0:
      pieces.append(z.slice_by_size([0] * len(full),
                    [before if i == ax else cur[i] for i in range(len(full))]))
    pieces.append(out)
    if after > 0:
      pieces.append(z.slice_by_size([0] * len(full),
                    [after if i == ax else cur[i] for i in range(len(full))]))
    out = graph.concat(pieces, axis=ax)
    cur[ax] = full[ax]
  return [out]


@vjp("concat")
def _vjp_concat(t, g):
  """Each source receives the slice of g spanning its own extent along the axis."""
  axis = t.attrs["axis"]
  outs, off = [], 0
  for src in t.srcs:
    begin = [0] * len(g.shape)
    size = list(g.shape)
    begin[axis] = off
    size[axis] = src.shape[axis]
    outs.append(g.slice_by_size(begin, size))
    off += src.shape[axis]
  return outs


@vjp("relu")
def _vjp_relu(t, g):
  """dx = select(x > 0, g, 0). `greater` needs a Tensor rhs, so compare vs (x - x)."""
  x = t.srcs[0]
  zero = _const_like(x, 0.0)
  cond = x.greater(zero)
  return [graph.select(cond, g, _const_like(g, 0.0))]


# spatial vjps. Native conv needs a baked weight, so only the conv INPUT grad is
# defined here (transposed-conv backward, stride=1); for a trainable conv weight
# use conv2d below. See docs/developer/autograd.md.

@vjp("conv")
def _vjp_conv(t, g):
  x = t.srcs[0]
  a = t.attrs
  st, dl = a["stride"], a["dilation"]
  if st != 1:
    raise NotImplementedError(
      "autograd: conv backward is verified for stride=1 only (stride>1 needs an "
      "output_padding the conv_transpose op does not expose). Use stride-1 convs "
      "with avg_pool for downsampling, or conv_trainable.")
  gx = graph.conv_transpose(g, a["weight"], stride=st, pad=a["pad"],
                            dilation=dl, groups=a["groups"])
  if gx.shape != x.shape:                         # guard the dim bookkeeping
    raise NotImplementedError(
      f"autograd: conv backward shape {gx.shape} != input {x.shape} for this "
      f"config (pad={a['pad']}, dilation={dl}); restrict to verified configs.")
  return [gx]


@vjp("avg_pool")
def _vjp_avg_pool(t, g):
  """Uniform spread over each window: nearest-neighbour upsample by k, scaled 1/k^2
    (non-overlapping case, stride == k)."""
  k, st = t.attrs["k"], t.attrs["stride"]
  if st != k or t.attrs["pad"] != 0:
    raise NotImplementedError(
      "autograd: avg_pool backward is verified for the non-overlapping, unpadded "
      "case (stride == k, pad == 0); other configs are not shipped.")
  return [g.upsample(k) * (1.0 / (k * k))]


@vjp("max_pool")
def _vjp_max_pool(t, g):
  """Route g only to each window's max (no gather/scatter): grad_x =
    upsample(g, k) * mask, mask = select(y_up.greater(x), 0, 1) with y_up the
    upsampled window max. Ties over-route (1 per tied cell) vs torch's one; rare
    on continuous fp16. Non-overlapping, unpadded only. See docs/developer/autograd.md."""
  x = t.srcs[0]
  k, st = t.attrs["k"], t.attrs["stride"]
  if st != k or t.attrs["pad"] != 0:
    raise NotImplementedError(
      "autograd: max_pool backward is verified for the non-overlapping, unpadded "
      "case (stride == k, pad == 0); other configs are not shipped.")
  y_up = t.upsample(k)                       # window max repeated over each k*k block
  is_max = graph.select(y_up.greater(x), _const_like(x, 0.0), _ones_like(x))  # 1 at argmax
  return [g.upsample(k) * is_max]


# trainable conv (weight is a real graph parameter): native conv needs a baked
# weight, so build the conv from primitives (im2col via slice_by_size + concat,
# then broadcast bmm) where every op has a vjp. See docs/developer/autograd.md.

def conv_param(weight_init) -> Tensor:
  """A trainable conv weight parameter. `weight_init` is [Cout, Cin, kH, kW]
    (PyTorch layout), stored as the flat patch matrix [Cin*kH*kW, Cout] that
    `conv2d` consumes (row order ci*(kH*kW) + (u*kW + v))."""
  W = np.asarray(weight_init, dtype=np.float32)
  Cout, Cin, kH, kW = W.shape
  flat = W.reshape(Cout, Cin * kH * kW).T.copy()        # [Cin*kH*kW, Cout]
  p = parameter(flat)
  p.attrs["conv_shape"] = (Cout, Cin, kH, kW)
  return p


def conv2d(x: Tensor, weight: Tensor, stride: int = 1, pad: int = 0) -> Tensor:
  """A trainable stride-1 2-D conv built from primitives so `weight` is a real
    graph parameter (see `conv_param`). `x` is [N, Cin, H, W]; `weight` a
    `conv_param`; returns [N, Cout, Hout, Wout]. `stride` must be 1; `pad` >= 0
    zero-pads in-graph (concat before im2col, differentiated through concat's VJP).

    Compile time grows with batch N (the im2col materialises [N, Cin*kH*kW,
    Hout*Wout]); train in MINI-BATCHES (N <= ~128). See docs/developer/autograd.md."""
  if stride != 1:
    raise NotImplementedError("conv2d (trainable) supports stride=1 only; "
                              "downsample with avg_pool/max_pool.")
  if pad < 0:
    raise ValueError(f"conv2d: pad must be >= 0, got {pad}")
  if "conv_shape" not in weight.attrs:
    raise ValueError("conv2d weight must come from af.conv_param([Cout,Cin,kH,kW])")
  N, Cin, H, W = x.shape
  Cout, Cin_w, kH, kW = weight.attrs["conv_shape"]
  if Cin_w != Cin:
    raise ValueError(f"conv2d: weight Cin {Cin_w} != input Cin {Cin}")
  if pad:
    # In-graph zero padding: concat zero-constant borders onto H then W (the
    # constant is a non-trainable leaf; its grad slice is discarded).
    zh = graph.input((N, Cin, pad, W)); zh.attrs["value"] = np.zeros((N, Cin, pad, W), np.float32)
    x = graph.concat([zh, x, zh], axis=2)            # [N, Cin, H+2pad, W]
    H = H + 2 * pad
    zw = graph.input((N, Cin, H, pad)); zw.attrs["value"] = np.zeros((N, Cin, H, pad), np.float32)
    x = graph.concat([zw, x, zw], axis=3)            # [N, Cin, H+2pad, W+2pad]
    W = W + 2 * pad
  Hout, Wout = H - kH + 1, W - kW + 1
  L, K = Hout * Wout, Cin * kH * kW
  parts = []
  for u in range(kH):
    for v in range(kW):
      # patch index on axis 2 (not last): A13's x16 crop-DMA saturation (>4094)
      # fires only on a nonzero last-axis slice offset, so keeping the large
      # loss-scaled grad off the width axis keeps M1 conv training correct.
      parts.append(x.slice_by_size([0, 0, u, v], [N, Cin, Hout, Wout]).reshape(N, Cin, 1, L))
  patches = graph.concat(parts, axis=2).transpose([0, 3, 1, 2]).reshape(N, L, K)   # [N,L,K]
  y = patches @ weight.reshape(1, K, Cout)                  # broadcast bmm -> [N,L,Cout]
  return y.transpose([0, 2, 1]).reshape(N, Cout, Hout, Wout)


# --------------------------------------------------------------------------- #
# loss, optimizer, train loop                                                 #
# --------------------------------------------------------------------------- #

class CEHandle:
  """A softmax-cross-entropy objective carrying the logits and one-hot target.
    The logit gradient is the analytic fused form (softmax(logits) - target)/N,
    fp16-stable (no log); the loss value + accuracy are host-side fp32."""
  def __init__(self, logits: Tensor, target: Tensor):
    if len(logits.shape) != 2 or logits.shape != target.shape:
      raise ValueError(f"softmax_cross_entropy expects 2-D logits/target [N,K]; "
                       f"got {logits.shape}, {target.shape}")
    self.logits, self.target, self.n = logits, target, int(logits.shape[0])

  def seed(self, loss_scale: float) -> Tensor:
    """dL/dlogits * loss_scale = (softmax(logits) - target) * (loss_scale / n)."""
    return (self.logits.softmax(-1) - self.target) * (float(loss_scale) / self.n)


def softmax_cross_entropy(logits: Tensor, target: Tensor) -> CEHandle:
  return CEHandle(logits, target)


def mse(y: Tensor, target: Tensor) -> Tensor:
  """Mean squared error over all axes (a scalar loss)."""
  diff = y - target
  return diff.square().mean(tuple(range(len(y.shape))))


# on-ANE optimizer update graph builders: the update arithmetic runs as graph
# ops. `lr_t` is a fed [1,1] input (broadcast tensor-mul, not a baked `muls`) so
# per-step bias correction / loss-scale folding varies each step.

def _sgd_update(w: Tensor, g: Tensor, lr_t: Tensor) -> Tensor:
  """w' = w - lr_t * g  (lr_t a fed [1,1] input, broadcast tensor-mul)."""
  return w - (lr_t * g)


def _adam_update(w: Tensor, m: Tensor, v: Tensor, g: Tensor, lr_t: Tensor,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
  """Adam update as ANE ops; b1/b2/eps baked, lr_t fed (bias correction folded in
    host-side). Returns (w', m', v')."""
  m2 = (m * float(b1)) + (g * float(1 - b1))
  v2 = (v * float(b2)) + (g.square() * float(1 - b2))
  w2 = w - ((lr_t * m2) / (v2.sqrt().adds(float(eps))))
  return w2, m2, v2


def adam_step(params, m, v, grads: dict, lr_t, betas=(0.9, 0.999), eps: float = 1e-8):
  """One Adam update as graph ops over lists `params`/`m`/`v`, returning the new
    (params, m, v) tensor lists. Used to UNROLL K steps into one program (thread the
    returned lists into the next step's forward); `conv_shape` is propagated so a
    conv weight still works as a `conv2d` weight next step."""
  b1, b2 = betas
  nP, nM, nV = [], [], []
  for p, mi, vi in zip(params, m, v):
    w2, m2, v2 = _adam_update(p, mi, vi, grads[p], lr_t, b1, b2, eps)
    if "conv_shape" in p.attrs: w2.attrs["conv_shape"] = p.attrs["conv_shape"]
    nP.append(w2); nM.append(m2); nV.append(v2)
  return nP, nM, nV


def _stack3(w: Tensor, m: Tensor, v: Tensor) -> Tensor:
  """Concat (w, m, v) along axis 0 in their natural row width -> [3*rows, cols],
    so a 3-output Adam update compiles as a SINGLE-output program (split host-side
    via `_split3`). Natural-width axis-0 is load-bearing: reshaping to a wide row
    first trips the ANECCompile wide-row wall. See docs/developer/autograd.md."""
  w2 = w.reshape(1, int(np.prod(w.shape))) if len(w.shape) == 1 else w
  m2 = m.reshape(1, int(np.prod(m.shape))) if len(m.shape) == 1 else m
  v2 = v.reshape(1, int(np.prod(v.shape))) if len(v.shape) == 1 else v
  return graph.concat([w2, m2, v2], axis=0)


def _split3(out, shape):
  """Inverse of `_stack3`: split [3*rows, cols] back into (w', m', v') in `shape`."""
  out = np.asarray(out)
  rows = shape[0] if len(shape) >= 2 else 1
  return (out[0 * rows:1 * rows].reshape(shape),
          out[1 * rows:2 * rows].reshape(shape),
          out[2 * rows:3 * rows].reshape(shape))


def _check_finite_grads(opt, grads) -> bool:
  """True iff every gradient is finite. Otherwise bump the skip counter, warn
    (1st skip then every 100th), and skip the whole step (loss-scaling overflow
    idiom: masters + Adam state stay untouched)."""
  bad = [i for i, g in enumerate(grads) if not np.isfinite(np.asarray(g)).all()]
  if not bad:
    opt._nonfinite_skips = 0
    return True
  opt._nonfinite_skips += 1
  if opt._nonfinite_skips == 1 or opt._nonfinite_skips % 100 == 0:
    import warnings
    warnings.warn(
      f"aneforge.{type(opt).__name__}: {len(bad)} of {len(grads)} parameter gradients "
      f"are non-finite (inf/nan) at param indices {bad}; skipping this optimizer step "
      f"(fp32 masters and optimizer state unchanged; {opt._nonfinite_skips} consecutive "
      f"step(s) skipped so far). On the ANE this means a loss-scaled fp16 gradient "
      f"overflowed; if you see inf/nan weight-grads, lower loss_scale.",
      stacklevel=3)
  return False


class SGD:
  """Host fp32 SGD over the parameters' master values (loss-scaled grads divided
    out before the step)."""
  def __init__(self, params, lr: float, loss_scale: float = 1.0):
    self.params, self.lr, self.scale = list(params), float(lr), float(loss_scale)
    self._nonfinite_skips = 0

  def step(self, grads):
    if not _check_finite_grads(self, grads): return
    for p, g in zip(self.params, grads):
      p.attrs["value"] = p.attrs["value"] - self.lr * (g.astype(np.float32) / self.scale)


class Adam:
  """Host fp32 Adam over the parameters' master values (loss-scaled grads divided
    out before the moment update; sibling to SGD)."""
  def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
               eps: float = 1e-8, loss_scale: float = 1.0):
    self.params = list(params)
    self.lr, (self.b1, self.b2), self.eps, self.scale = float(lr), betas, float(eps), float(loss_scale)
    self.m = [np.zeros(p.attrs["value"].shape, np.float32) for p in self.params]
    self.v = [np.zeros_like(x) for x in self.m]
    self.t = 0
    self._nonfinite_skips = 0

  def step(self, grads):
    if not _check_finite_grads(self, grads): return
    self.t += 1
    bc1, bc2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
    for i, (p, g) in enumerate(zip(self.params, grads)):
      g = g.astype(np.float32) / self.scale
      self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
      self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
      mhat, vhat = self.m[i] / bc1, self.v[i] / bc2
      p.attrs["value"] = p.attrs["value"] - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# A13 (M1) conv weight-grad saturation ceiling: a nonzero width-axis im2col slice
# offset routes through A13's Q.4 x16 crop-DMA, clamping |value| > 4094 to +/-inf,
# so loss_scale x |backward activation| must stay under 4094. Real normalized CNNs
# clear it; a WARN, never an auto-cap (M5/A16 unaffected). See docs/developer/autograd.md.
_A13_CONV_WGRAD_LOSS_SCALE_MAX = 512.0


def _has_conv_wgrad(params, objective) -> bool:
  """True iff a param is a trainable conv weight with kW>1 (only those hit the
    A13 width-offset saturation path)."""
  for p in params:
    cs = p.attrs.get("conv_shape")
    if cs is not None and int(cs[3]) > 1: return True   # conv_shape = (Cout, Cin, kH, kW)
  return False


def _guard_a13_conv_loss_scale(params, objective, loss_scale: float) -> float:
  """On A13/M1 with a trainable conv weight, warn when loss_scale could push
    backward activations past 4094 (width-offset im2col saturation). WARN only;
    returns loss_scale unchanged. Other families unaffected."""
  from . import _targets as TG
  try:
    fam = TG.detect_family()
  except Exception:
    return loss_scale
  if int(fam) != int(TG.Family.A13): return loss_scale
  if not _has_conv_wgrad(params, objective): return loss_scale
  if loss_scale >= _A13_CONV_WGRAD_LOSS_SCALE_MAX:
    import warnings
    warnings.warn(
      f"aneforge.Trainer: on the A13/M1 ANE, conv weight-grads saturate to +/-inf if "
      f"loss_scale x |backward activation| exceeds 4094 (=65504/16): the trainable "
      f"conv's width-offset im2col slices route through A13's Q.4 x16 crop-DMA. "
      f"loss_scale={loss_scale} clears that bound only if backward activations stay "
      f"under {4094.0 / loss_scale:.3g}. Real normalized CNNs train correctly at any "
      f"loss_scale (measured on M1); if you see inf/nan weight-grads, lower loss_scale. "
      f"M5/A16 has no such path.",
      stacklevel=3)
  return loss_scale


class Trainer:
  """Compiles a forward program ONCE plus one backward program PER PARAMETER (each
    emitting that param's grad in natural shape); `step` evals them and applies the
    optimizer host-side (no recompile). Per-param programs avoid an ANECCompile
    wide-row wall.

    Accepts a scalar `loss` Tensor (regression; ones-seed) or a `CEHandle`
    (classification; analytic on-ANE logit seed, host-side fp32 `loss()`/`accuracy`).
    `optimizer="sgd"|"adam"`.

    `device_optimizer=True` runs the optimizer step on the ANE (per-param UPDATE
    programs); Adam moments held host-side as fp16, fed/read each step. Default
    False keeps the host fp32 path byte-for-byte. See docs/developer/autograd.md."""
  def __init__(self, objective, params, lr: float, loss_scale: float = 1.0,
               data_inputs: dict | None = None, optimizer: str = "sgd",
               betas=(0.9, 0.999), eps: float = 1e-8, device_optimizer: bool = False,
               resident_state: bool = False):
    from . import _compile as _c
    self.params = list(params)
    # A13/M1 conv weight-grad saturation guard (before scale is consumed).
    loss_scale = _guard_a13_conv_loss_scale(self.params, objective, float(loss_scale))
    self.data = dict(data_inputs or {})       # {input Tensor: numpy value}
    self.scale = float(loss_scale)
    self.lr = float(lr)
    self.b1, self.b2 = float(betas[0]), float(betas[1])
    self.eps = float(eps)
    self.optimizer = optimizer
    # resident_state forces device_optimizer on (it IS the on-device update).
    self.resident_state = bool(resident_state)
    self.device_optimizer = bool(device_optimizer) or self.resident_state
    self.opt = (Adam(self.params, lr, betas, eps=eps, loss_scale=loss_scale)
                if optimizer == "adam" else SGD(self.params, lr, loss_scale))
    if isinstance(objective, CEHandle):
      self.ce = objective
      grads = backward_from(objective.seed(loss_scale), objective.logits, self.params)
      fwd_out = objective.logits                 # forward program -> logits
    else:
      self.ce = None
      grads = backward(objective, self.params, loss_scale=loss_scale)
      fwd_out = objective                        # forward program -> loss scalar
    # ONE backward program per param, in natural shape: concatenating all grads
    # into one wide row trips an ANECCompile wide-row wall. _check_precision is off
    # because the training kernels' structural subtracts (loss/axpy/update) are
    # vouched, not user-data choices. See docs/developer/autograd.md.
    self._fwd = _c.compile(fwd_out, _check_precision=False)
    if self.resident_state:
      self._build_resident(_c, grads)            # _fwd kept only for checkpoint accuracy
    else:
      self._bwd = [_c.compile(grads[p], _check_precision=False) for p in self.params]
      if self.device_optimizer: self._build_device_optimizer(_c)

  def _build_device_optimizer(self, _c):
    """Compile a per-param UPDATE program so optimizer arithmetic runs on the ANE.
        SGD: (w, g, lr_t) -> w'. Adam: -> stack(w', m', v') via `_stack3`, split
        host-side. g/m/v/lr_t are fed leaves; w is the param leaf itself."""
    self._upd, self._upd_g, self._upd_lr = [], [], []
    if self.optimizer == "adam":
      self._m = [np.zeros(p.shape, np.float16) for p in self.params]
      self._v = [np.zeros(p.shape, np.float16) for p in self.params]
      self._upd_m, self._upd_v = [], []
      self._t = 0
    for p in self.params:
      g_in = graph.input(p.shape)
      lr_in = graph.input((1, 1))
      self._upd_g.append(g_in); self._upd_lr.append(lr_in)
      if self.optimizer == "adam":
        m_in = graph.input(p.shape); v_in = graph.input(p.shape)
        self._upd_m.append(m_in); self._upd_v.append(v_in)
        w2, m2, v2 = _adam_update(p, m_in, v_in, g_in, lr_in, self.b1, self.b2, self.eps)
        out = _stack3(w2, m2, v2)
      else:
        out = _sgd_update(p, g_in, lr_in)
      self._upd.append(_c.compile(out, _check_precision=False))

  def _build_resident(self, _c, grads):
    """Assemble the whole step as ONE fused multi-output program with optimizer
        state RESIDENT on-device: each updated-state output is aliased back onto its
        input port via `share_buffer`, so state lives on the engine across steps and
        the host feeds only the minibatch + lr_t. Within one execute, FIFO ordering
        has the forward read the pre-step param and the update overwrite it last.
        See docs/developer/autograd.md."""
    lr_in = graph.input((1, 1))
    self._res_lr = lr_in
    self._t = 0
    outs, alias, self._res_state = [], [], []
    for p in self.params:
      g = grads[p]
      entry = {"p": p}
      if self.optimizer == "adam":
        m_in, v_in = graph.input(p.shape), graph.input(p.shape)
        w2, m2, v2 = _adam_update(p, m_in, v_in, g, lr_in, self.b1, self.b2, self.eps)
        outs += [w2, m2, v2]
        alias += [(w2, p), (m2, m_in), (v2, v_in)]
        entry.update(m_in=m_in, v_in=v_in, w_out=w2)
      else:
        w2 = _sgd_update(p, g, lr_in)
        outs += [w2]
        alias += [(w2, p)]
        entry.update(w_out=w2)
      self._res_state.append(entry)

    mm = _c.compile_multi(outs)
    self._res = mm
    prog = mm.prog
    self._res_in_name = dict(mm.input_ports)
    self._res_out_name = dict(mm.output_ports)
    # alias each updated-state output onto its input port, then seed once.
    for out_t, in_t in alias:
      prog.share_buffer(0, self._res_out_name[out_t], 0, self._res_in_name[in_t])
    for entry in self._res_state:
      p = entry["p"]
      prog.set_input(self._res_in_name[p], p.attrs["value"].astype(np.float16))
      if self.optimizer == "adam":
        prog.set_input(self._res_in_name[entry["m_in"]], np.zeros(p.shape, np.float16))
        prog.set_input(self._res_in_name[entry["v_in"]], np.zeros(p.shape, np.float16))
    # the remaining inputs (neither state nor lr) are the data ports (x, target)
    state_ids = set()
    for entry in self._res_state:
      state_ids.add(id(entry["p"]))
      if self.optimizer == "adam":
        state_ids.update({id(entry["m_in"]), id(entry["v_in"])})
    self._res_data_inputs = [t for t, _ in mm.input_ports
                             if id(t) not in state_ids and t is not lr_in]
    self._res_dirty = False

  def _resident_step(self) -> None:
    """One resident step: feed ONLY the minibatch + lr_t, execute (state stays
        on-device; no host tensor math or state copy)."""
    if getattr(self, "_ds", None) is not None:
      xin, X, tin, Y = self._ds
      idx = self._next_batch()
      self.data[xin] = X[idx]; self.data[tin] = Y[idx]
    prog = self._res.prog
    for t in self._res_data_inputs:
      prog.set_input(self._res_in_name[t], np.asarray(self.data[t], np.float16))
    if self.optimizer == "adam":
      self._t += 1
      lr_t = self.lr * math.sqrt(1.0 - self.b2 ** self._t) / (1.0 - self.b1 ** self._t)
    else:
      lr_t = self.lr / self.scale
    prog.set_input(self._res_in_name[self._res_lr], np.full((1, 1), lr_t, np.float16))
    prog.execute()
    self._res_dirty = True

  def _sync_params_from_device(self) -> None:
    """Checkpoint read: copy resident params off-device into the host masters (for
        accuracy/loss via the forward program); moments stay resident."""
    prog = self._res.prog
    for i, entry in enumerate(self._res_state):
      w = prog.read_output(self._res_out_name[entry["w_out"]]).astype(np.float32)
      if not np.isfinite(w).all():
        import warnings
        warnings.warn(
          f"aneforge.Trainer: resident param {i} (shape {tuple(entry['p'].shape)}) read "
          f"back non-finite values (inf/nan) at checkpoint - the on-device update was "
          f"poisoned by an overflowed fp16 gradient; if you see inf/nan weight-grads, "
          f"lower loss_scale.",
          stacklevel=3)
      entry["p"].attrs["value"] = w.reshape(entry["p"].shape)
    self._res_dirty = False

  def _feed(self, model, override: dict | None = None):
    """Map each compiled input Tensor to a fp16 array: trainable -> master value;
        in `override` -> override[t] (device optimizer grads/lr); else the baked
        value-input attrs['value'] (e.g. a norm VJP's re-injected gamma)."""
    lk = self.data if override is None else override
    return [
      t.attrs["value"].astype(np.float16) if t.attrs.get("trainable")
      else np.asarray(lk[t] if t in lk else t.attrs["value"]).astype(np.float16)
      for t in model._input_tensors
    ]

  def _feed_update(self, net, p, extra):
    """Feed a per-param update program (delegates to _feed with `extra`)."""
    return self._feed(net, extra)

  def set_dataset(self, x_input, X_full, target_input, Y_onehot, seed: int = 0):
    """Provide the full dataset for mini-batch sampling. `x_input`/`target_input`
        are the batch-B graph input placeholders the objective was built from."""
    self._ds = (x_input, np.asarray(X_full, np.float32), target_input, np.asarray(Y_onehot, np.float32))
    self._ds_B = int(x_input.shape[0])
    self._ds_rng = np.random.default_rng(seed)
    self._ds_perm = self._ds_rng.permutation(len(self._ds[1]))
    self._ds_pos = 0

  def _next_batch(self):
    B = self._ds_B
    if self._ds_pos + B > len(self._ds_perm):       # reshuffle each epoch
      self._ds_perm = self._ds_rng.permutation(len(self._ds[1]))
      self._ds_pos = 0
    idx = self._ds_perm[self._ds_pos:self._ds_pos + B]
    self._ds_pos += B
    return idx

  def step(self) -> None:
    if self.resident_state:
      self._resident_step()
      return
    if getattr(self, "_ds", None) is not None:
      xin, X, tin, Y = self._ds
      idx = self._next_batch()
      self.data[xin] = X[idx]
      self.data[tin] = Y[idx]
    grads = [np.asarray(net(*self._feed(net))).reshape(p.shape)
             for net, p in zip(self._bwd, self.params)]
    if not self.device_optimizer:
      self.opt.step(grads)
      return
    # On-ANE optimizer: update arithmetic runs as graph ops; host only computes lr_t.
    if self.optimizer == "adam":
      self._device_adam_step(grads)
    else:
      self._device_sgd_step(grads)

  def _device_sgd_step(self, grads):
    lr_t = self.lr / self.scale          # fold grad-unscale into lr_t
    lr_arr = np.full((1, 1), lr_t, np.float16)
    for i, p in enumerate(self.params):
      net, g_in, lr_in = self._upd[i], self._upd_g[i], self._upd_lr[i]
      extra = {g_in: grads[i], lr_in: lr_arr}
      w2 = np.asarray(net(*self._feed_update(net, p, extra))).reshape(p.shape)
      p.attrs["value"] = w2.astype(np.float32)

  def _device_adam_step(self, grads):
    self._t += 1
    # lr_t = lr * sqrt(1-b2^t)/(1-b1^t). Loss-scale is NOT divided out: it cancels
    # in Adam's m/sqrt(v) ratio (m'=scale*m, v'=scale^2*v); unscaling would
    # double-unscale and collapse the step. See docs/developer/autograd.md.
    lr_t = self.lr * math.sqrt(1.0 - self.b2 ** self._t) / (1.0 - self.b1 ** self._t)
    lr_arr = np.full((1, 1), lr_t, np.float16)
    for i, p in enumerate(self.params):
      net = self._upd[i]
      extra = {self._upd_g[i]: grads[i], self._upd_m[i]: self._m[i],
               self._upd_v[i]: self._v[i], self._upd_lr[i]: lr_arr}
      out = np.asarray(net(*self._feed_update(net, p, extra)))
      w2, m2, v2 = _split3(out, p.shape)
      p.attrs["value"] = w2.astype(np.float32)
      self._m[i] = m2.astype(np.float16)       # fp16 optimizer state held host-side
      self._v[i] = v2.astype(np.float16)

  def accuracy(self, X, y_labels) -> float:
    """Argmax accuracy over X (any length) via the batch-B forward program,
        chunking X into B-row pieces (last padded then truncated)."""
    assert self.ce is not None, "accuracy() is for classification objectives"
    if self.resident_state and getattr(self, "_res_dirty", False):
      self._sync_params_from_device()
    X = np.asarray(X, np.float32); y = np.asarray(y_labels)
    # the feature input is the non-trainable, non-target input whose per-sample
    # shape matches X (handles 2-D [B,D] MLP and N-D [B,...] CNN inputs).
    feat_shape = X.shape[1:]
    assert isinstance(self._fwd, _Model)  # Trainer always compiles non-sdpa graphs
    xin = next(t for t in self._fwd._input_tensors
               if not t.attrs.get("trainable") and t is not getattr(self.ce, "target", None)
               and tuple(t.shape[1:]) == tuple(feat_shape))
    B = xin.shape[0]
    saved = self.data.get(xin)
    preds = []
    for s in range(0, X.shape[0], B):
      chunk = X[s:s + B]
      m = chunk.shape[0]
      if m < B:
        pad = np.zeros((B - m,) + tuple(feat_shape), np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
      self.data[xin] = chunk
      logits = np.asarray(self._fwd(*self._feed(self._fwd)))
      preds.append(logits[:m].argmax(1))
    if saved is not None: self.data[xin] = saved
    return float((np.concatenate(preds) == y).mean())

  def loss(self) -> float:
    if self.resident_state and getattr(self, "_res_dirty", False):
      self._sync_params_from_device()
    out = np.asarray(self._fwd(*self._feed(self._fwd)))
    if self.ce is None: return float(out.reshape(-1)[0])
    logits = out.reshape(self.ce.logits.shape)
    t = np.asarray(self.data[self.ce.target])
    z = logits - logits.max(1, keepdims=True)
    logsm = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-(t * logsm).sum(1).mean())

  def release(self) -> None:
    self._fwd.release()
    if getattr(self, "_res", None) is not None: self._res.release()
    for net in getattr(self, "_bwd", []): net.release()
    for net in getattr(self, "_upd", []): net.release()


class UnrolledTrainer:
  """Train with `K` Adam steps UNROLLED into ONE fused ANE program: each `step()`
    runs K forward->backward->update steps in a single dispatch. Bounded-K
    fully-on-engine analogue of `Trainer`, enabled by the stop-gradient frontier in
    `backward` (each step treats current weights as leaves). See docs/developer/autograd.md.

    Args:
      params: trainable leaves (`af.parameter` / `af.conv_param`).
      forward: `forward(P, x) -> output` building the model from current-step weight
        tensors `P` and data input `x`; used per step and for `predict`.
      kind: `"ce"` (logits + softmax-cross-entropy) or `"mse"`.
      x_inputs, t_inputs: K data / target input placeholders, one per step.
      dataset: `(X, Y)` arrays; `Y` one-hot [N, C] for `"ce"`.
      resident: if True (default) optimizer state stays RESIDENT on-device across
        dispatches (aliased via `share_buffer`); host feeds only minibatches + lr.
    """
  def __init__(self, params, forward, kind, x_inputs, t_inputs, dataset, lr,
               loss_scale: float = 1.0, betas=(0.9, 0.999), eps: float = 1e-8,
               seed: int = 0, resident: bool = True):
    from . import _compile as _c
    if kind not in ("ce", "mse"):
      raise ValueError("kind must be 'ce' or 'mse'")
    self.params = list(params)
    self.forward = forward
    self.kind = kind
    self.K = len(x_inputs)
    self.lr = float(lr); self.scale = float(loss_scale)
    self.b1, self.b2 = float(betas[0]), float(betas[1]); self.eps = float(eps)
    self.X = np.asarray(dataset[0], np.float32)
    self.Y = np.asarray(dataset[1], np.float32)
    self.B = int(x_inputs[0].shape[0])
    self.m = [np.zeros(p.shape, np.float16) for p in self.params]
    self.v = [np.zeros(p.shape, np.float16) for p in self.params]
    self.t = 0
    self.rng = np.random.default_rng(seed)
    self._perm = self.rng.permutation(len(self.X)); self._pos = 0

    # build the unrolled graph: thread (P, m, v) through K Adam steps
    m_in = [graph.input(p.shape) for p in self.params]
    v_in = [graph.input(p.shape) for p in self.params]
    lr_ins = [graph.input((1, 1)) for _ in range(self.K)]
    P, M, V = list(self.params), list(m_in), list(v_in)
    for k in range(self.K):
      out = forward(P, x_inputs[k])
      if kind == "ce":
        g = backward_from(softmax_cross_entropy(out, t_inputs[k]).seed(self.scale), out, P)
      else:
        g = backward(mse(out, t_inputs[k]), P, loss_scale=self.scale)
      P, M, V = adam_step(P, M, V, g, lr_ins[k], (self.b1, self.b2), self.eps)
    self._net = _c.compile_multi([*P, *M, *V])
    self._oname = dict(self._net.output_ports)
    self._P_out, self._M_out, self._V_out = P, M, V
    self._m_in, self._v_in, self._lr_ins = m_in, v_in, lr_ins
    # map each data input tensor -> (step k, 'x'|'t') for feeding
    self._data_map = {}
    for k in range(self.K):
      self._data_map[id(x_inputs[k])] = (k, "x")
      self._data_map[id(t_inputs[k])] = (k, "t")

    # separate single-batch forward program for checkpoint predict, with its OWN
    # weight leaves: compile mutates Tensor names, so sharing params would clobber
    # the training program's ports.
    ev_w = [graph.input(p.shape) for p in self.params]
    for ew, p in zip(ev_w, self.params):
      if "conv_shape" in p.attrs: ew.attrs["conv_shape"] = p.attrs["conv_shape"]
    xe = graph.input(x_inputs[0].shape)
    self._ev_w = ev_w
    self._eval = _c.compile(forward(ev_w, xe), _check_precision=False)

    self.resident = bool(resident)
    if self.resident:
      # Alias each final state OUTPUT onto its own initial INPUT port, so params/
      # m/v live on-device across dispatches; seed the shared buffers once.
      prog = self._net.prog
      inm = {id(t): n for t, n in self._net.input_ports}
      self._res_inm = inm
      self._res_lr_names = [inm[id(t)] for t in lr_ins]
      self._res_data = ([(inm[id(x_inputs[k])], k, "x") for k in range(self.K)] +
                        [(inm[id(t_inputs[k])], k, "t") for k in range(self.K)])
      pairs = (list(zip(self._P_out, self.params)) +
               list(zip(self._M_out, m_in)) + list(zip(self._V_out, v_in)))
      for out_t, in_t in pairs:
        prog.share_buffer(0, self._oname[out_t], 0, inm[id(in_t)])
      for i, p in enumerate(self.params):
        prog.set_input(inm[id(p)], p.attrs["value"].astype(np.float16))
        prog.set_input(inm[id(m_in[i])], np.zeros(p.shape, np.float16))
        prog.set_input(inm[id(v_in[i])], np.zeros(p.shape, np.float16))
      self._res_dirty = False

  def _next(self):
    if self._pos + self.B > len(self._perm):
      self._perm = self.rng.permutation(len(self.X)); self._pos = 0
    idx = self._perm[self._pos:self._pos + self.B]; self._pos += self.B
    return idx

  def step(self) -> None:
    """Run K training steps on the ANE in ONE dispatch. Resident: feed only the K
        minibatches + per-step lr; state stays on-device. Else: shuttle params/m/v."""
    batches = [self._next() for _ in range(self.K)]
    if self.resident:
      prog = self._net.prog
      for name, k, which in self._res_data:
        idx = batches[k]
        prog.set_input(name, (self.X[idx] if which == "x" else self.Y[idx]).astype(np.float16))
      for k, name in enumerate(self._res_lr_names):
        gt = self.t + k + 1
        prog.set_input(name, np.full((1, 1), self.lr * math.sqrt(1.0 - self.b2 ** gt) /
                                     (1.0 - self.b1 ** gt), np.float16))
      prog.execute()
      self.t += self.K
      self._res_dirty = True
      return
    vals = []
    for t in self._net.input_tensors:
      if t.attrs.get("trainable"):
        vals.append(t.attrs["value"].astype(np.float16))
      elif t in self._m_in:
        vals.append(self.m[self._m_in.index(t)])
      elif t in self._v_in:
        vals.append(self.v[self._v_in.index(t)])
      elif t in self._lr_ins:
        gt = self.t + self._lr_ins.index(t) + 1          # global step for bias correction
        lr_t = self.lr * math.sqrt(1.0 - self.b2 ** gt) / (1.0 - self.b1 ** gt)
        vals.append(np.full((1, 1), lr_t, np.float16))
      else:
        k, which = self._data_map[id(t)]
        idx = batches[k]
        vals.append((self.X[idx] if which == "x" else self.Y[idx]).astype(np.float16))
    out = self._net(*vals)
    for i, p in enumerate(self.params):
      p.attrs["value"] = out[self._oname[self._P_out[i]]].reshape(p.shape)
      self.m[i] = out[self._oname[self._M_out[i]]].astype(np.float16).reshape(p.shape)
      self.v[i] = out[self._oname[self._V_out[i]]].astype(np.float16).reshape(p.shape)
    self.t += self.K

  def _sync_from_device(self) -> None:
    """Checkpoint read: copy the resident params off-device into the host masters."""
    prog = self._net.prog
    for i, p in enumerate(self.params):
      w = prog.read_output(self._oname[self._P_out[i]]).astype(np.float32)
      p.attrs["value"] = w.reshape(p.shape)
    self._res_dirty = False

  def predict(self, X) -> np.ndarray:
    """Run the trained weights forward on the ANE in B-sized chunks; returns the
        model output (logits for 'ce', prediction for 'mse')."""
    if self.resident and getattr(self, "_res_dirty", False):
      self._sync_from_device()
    X = np.asarray(X, np.float32)
    feeds_w = [p.attrs["value"].astype(np.float16) for p in self.params]
    outs = []
    for s in range(0, len(X), self.B):
      chunk = X[s:s + self.B]
      pad = self.B - len(chunk)
      if pad:
        chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), np.float32)])
      # eval inputs are [weight leaves..., xe]; feed masters then the chunk
      assert isinstance(self._eval, _Model)  # UnrolledTrainer eval never uses sdpa nodes
      args = []
      for t in self._eval._input_tensors:
        args.append(feeds_w[self._ev_w.index(t)] if t in self._ev_w
                    else chunk.astype(np.float16))
      o = np.asarray(self._eval(*args), np.float32)
      outs.append(o[:len(chunk)] if pad else o)
    return np.concatenate(outs)

  def release(self) -> None:
    self._net.release(); self._eval.release()
