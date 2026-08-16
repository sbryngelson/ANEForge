"""Lower a graph into ONE fused e5rt program (one MIL op per node, weights in one BLOBFILE)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
  from ._netplist_worker import _Worker

from ._runtime import E5RT, ANE_MASK

from ._blob import (BlobWriter, fp16_bytes, quantize_per_row, palettize_lut4, sparsify,
          quantize_blockwise, FP16, INT8, UINT4, UINT1)
from .graph import Tensor

_BUILD_INFO = (
  '[buildInfo = dict<string, string>('
  '{{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}})]'
)


def _topo_multi(*outs: Tensor) -> list[Tensor]:
  # iterative post-order DFS (deep-graph safe; single-root order == seeding from that root).
  seen: set[int] = set()
  order: list[Tensor] = []
  stack: list = [(o, False) for o in reversed(outs)]
  while stack:
    t, processed = stack.pop()
    if processed:
      order.append(t)
      continue
    if id(t) in seen: continue
    seen.add(id(t))
    stack.append((t, True))
    for src in t.srcs:
      if id(src) not in seen: stack.append((src, False))
  return order


def _topo(out: Tensor) -> list[Tensor]:
  return _topo_multi(out)


# per-op MIL emitters (registry)

_EMIT: dict[str, Callable[..., Any]] = {}


def op(*names: str):
  def register(fn):
    for name in names: _EMIT[name] = fn
    return fn
  return register


class _Emitter:
  """Accumulates MIL lines + the weight BLOBFILE while walking the graph."""

  def __init__(self, int8: bool, compress: str | None = None, compress_atol: float = 0.05,
        block_size: int = 32, family: int | None = None) -> None:
    # compress: None (fp16), "int8", "int4", "sparse", "blockwise", "auto"; int8=True aliases "int8".
    self.compress = compress if compress is not None else ("int8" if int8 else None)
    self.int8 = self.compress == "int8"
    self.compress_atol = compress_atol
    self.block_size = block_size       # inner-dim block for compress="blockwise"
    # auto is family-aware: only natively-streaming encodings are candidates.
    if self.compress == "auto" and family is not None:
      from . import _targets as TG
      self._auto_streams = TG.native_streams(family)
    else:
      self._auto_streams = frozenset({"int4", "int8", "sparse", "blockwise"})
    self.blob = BlobWriter()
    self.lines: list[str] = []
    self._split_cache: dict = {}       # (src, num_splits, axis) -> emitted part names, so split is emitted once

  def line(self, text: str) -> None:
    self.lines.append("        " + text)

  @staticmethod
  def ty(shape) -> str:
    return "tensor<fp16, [" + ", ".join(str(d) for d in shape) + "]>"

  @staticmethod
  def ty_dt(shape, dt: str) -> str:
    return f"tensor<{dt}, [" + ", ".join(str(d) for d in shape) + "]>"

  @staticmethod
  def _lut4_rel_error(W2: np.ndarray, packed: bytes, lut: bytes) -> float:
    cb = np.frombuffer(lut, dtype=np.float16).astype(np.float32)
    raw = np.frombuffer(packed, dtype=np.uint8)
    idx = np.empty(raw.size * 2, dtype=np.uint8)
    idx[0::2], idx[1::2] = raw & 0x0F, (raw >> 4) & 0x0F
    recon = cb[idx][: W2.size].reshape(W2.shape)
    W2 = W2.astype(np.float32)   # norm in fp32: it overflows fp16 for large weights
    return float(np.linalg.norm(recon - W2) / (np.linalg.norm(W2) + 1e-12))

  @staticmethod
  def _blockwise_rel_error(W2: np.ndarray, data: bytes, scale: bytes, nblocks: int) -> float:
    OUT, IN = W2.shape
    q = np.frombuffer(data, dtype=np.int8).astype(np.float32).reshape(OUT, nblocks, IN // nblocks)
    sc = np.frombuffer(scale, dtype=np.float16).astype(np.float32).reshape(OUT, nblocks)
    recon = (q * sc[:, :, None]).reshape(OUT, IN)
    W2 = W2.astype(np.float32)   # norm in fp32: it overflows fp16 for large weights
    return float(np.linalg.norm(recon - W2) / (np.linalg.norm(W2) + 1e-12))

  def weight(self, name: str, W: np.ndarray, allow_int8: bool,
       int8: bool | None = None, allow_int4: bool = False,
       allow_sparse: bool = False) -> str:
    """Declare a constant weight; precedence sparse -> int4-LUT -> per-channel int8 -> fp16."""
    path = '@model_path/weights.bin'
    W2 = W.reshape(W.shape[0], -1)
    _auto = self.compress == "auto"
    if (self.compress == "sparse" or (_auto and "sparse" in self._auto_streams)) and allow_sparse:
      mask, vals = sparsify(W2)
      nnz = len(vals) // 2
      if 0 < nnz <= (W2.size // 2):          # only when genuinely sparse
        om, on = self.blob.add(mask, UINT1), self.blob.add(vals, FP16)
        shp = list(W.shape)
        self.line(f'tensor<uint1, {shp}> {name}_mask = const()[name = string("{name}_mask"), '
             f'val = tensor<uint1, {shp}>(BLOBFILE(path = string("{path}"), offset = uint64({om})))];')
        self.line(f'tensor<fp16, [{nnz}]> {name}_nz = const()[name = string("{name}_nz"), '
             f'val = tensor<fp16, [{nnz}]>(BLOBFILE(path = string("{path}"), offset = uint64({on})))];')
        self.line(f'{self.ty(W.shape)} {name} = constexpr_sparse_to_dense('
             f'nonzero_data = {name}_nz, mask = {name}_mask)[name = string("{name}")];')
        return name
      # else: fall through to int8/fp16
    if (self.compress == "int4" or (_auto and "int4" in self._auto_streams)) \
        and allow_int4 and W2.shape[1] % 2 == 0:
      packed, lut = palettize_lut4(W2)
      if self._lut4_rel_error(W2, packed, lut) <= self.compress_atol:
        oi, ol = self.blob.add(packed, UINT4), self.blob.add(lut, FP16)
        # ND indices (lut.rank == indices.rank + 2): constexpr_ reshape is not ANE-routable.
        ishp = list(W.shape)
        lut_shp = [1] * W.ndim + [16, 1]
        lut_shp_str = "[" + ", ".join(str(d) for d in lut_shp) + "]"
        ishp_str = "[" + ", ".join(str(d) for d in ishp) + "]"
        self.line(f'tensor<uint4, {ishp_str}> {name}_idx = const()[name = string("{name}_idx"), '
             f'val = tensor<uint4, {ishp_str}>(BLOBFILE(path = string("{path}"), offset = uint64({oi})))];')
        self.line(f'tensor<fp16, {lut_shp_str}> {name}_lut = const()[name = string("{name}_lut"), '
             f'val = tensor<fp16, {lut_shp_str}>(BLOBFILE(path = string("{path}"), offset = uint64({ol})))];')
        self.line(f'{self.ty(W.shape)} {name} = constexpr_lut_to_dense('
             f'indices = {name}_idx, lut = {name}_lut)[name = string("{name}")];')
        return name
      # else: fall through to int8/fp16
    if self.compress == "blockwise" and allow_int8:
      data, scale, nblocks = quantize_blockwise(W2, self.block_size)
      if self._blockwise_rel_error(W2, data, scale, nblocks) <= self.compress_atol:
        od, osc = self.blob.add(data, INT8), self.blob.add(scale, FP16)
        # constexpr output not routable as a weight operand; bridge it via +0 add to dense fp16.
        shp = list(W.shape)
        self.line(f'tensor<int8, {shp}> {name}_d = const()[name = string("{name}_d"), '
             f'val = tensor<int8, {shp}>(BLOBFILE(path = string("{path}"), offset = uint64({od})))];')
        self.line(f'tensor<fp16, [{W.shape[0]}, {nblocks}]> {name}_sc = const()[name = string("{name}_sc"), '
             f'val = tensor<fp16, [{W.shape[0]}, {nblocks}]>(BLOBFILE(path = string("{path}"), offset = uint64({osc})))];')
        self.line(f'{self.ty(W.shape)} {name}_bw = constexpr_blockwise_shift_scale('
             f'data = {name}_d, scale = {name}_sc)[name = string("{name}_bw")];')
        self.line(f'fp16 {name}_bz = const()[name = string("{name}_bz"), val = fp16(0x0p+0)];')
        self.line(f'{self.ty(W.shape)} {name} = add(x = {name}_bw, y = {name}_bz)[name = string("{name}")];')
        return name
      # else: fall through to int8/fp16
    use_int8 = int8 if int8 is not None else (
      self.int8 or self.compress == "int4" or (_auto and "int8" in self._auto_streams))
    if use_int8 and allow_int8:
      q, sc = quantize_per_row(W2)
      oq, os = self.blob.add(q, INT8), self.blob.add(sc, FP16)
      self.line(
        f'{self.ty(W.shape)} {name} = constexpr_affine_dequantize()[axis = int32(0), name = string("{name}"), '
        f'quantized_data = tensor<int8, {list(W.shape)}>(BLOBFILE(path = string("{path}"), offset = uint64({oq}))), '
        f'scale = tensor<fp16, [{W.shape[0]}]>(BLOBFILE(path = string("{path}"), offset = uint64({os}))), zero_point = int8(0)];')
    else:
      o = self.blob.add(fp16_bytes(W), FP16)
      self.line(f'{self.ty(W.shape)} {name} = const()[name = string("{name}"), '
           f'val = {self.ty(W.shape)}(BLOBFILE(path = string("{path}"), offset = uint64({o})))];')
    return name


# param-free unary ops: graph op name == MIL op name, signature op(x = ...)
@op("relu", "silu", "sigmoid", "tanh", "exp", "sqrt", "abs",
  "sin", "cos", "erf", "relu6", "softsign", "atan", "exp2",
  "floor", "ceil", "sign")
def _e_unary(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(x = {s[0]})[name = string("{n}")];')


@op("round")
def _e_round(em, t, n, s):
  """The ANE round kernel corrupts values with |x| >= 1024 (measured on M5/H17s:
  round(1024)=1025, round(1025)=1026, round(-2047)=-2048 - it adds 0.5 in fp16, where the tie
  is unrepresentable). Every fp16 value with |x| >= 1024 is already an integer, so route them
  through unchanged: select(|x| < 1024, round(x), x). Exact on the full fp16 range, and safe on
  every family regardless of whether its kernel shares the bug (the select form is semantically
  identity there); cross-compiles for H13-H16s (see test_espresso_workarounds)."""
  thr = float(np.float16(1024.0)).hex()
  em.line(f'fp16 {n}_c = const()[name = string("{n}_c"), val = fp16({thr})];')
  em.line(f'{em.ty(t.shape)} {n}_a = abs(x = {s[0]})[name = string("{n}_a")];')
  em.line(f'{em.ty_dt(t.shape, "bool")} {n}_lt = less(x = {n}_a, y = {n}_c)[name = string("{n}_lt")];')
  em.line(f'{em.ty(t.shape)} {n}_r = round(x = {s[0]})[name = string("{n}_r")];')
  em.line(f'{em.ty(t.shape)} {n} = select(cond = {n}_lt, a = {n}_r, b = {s[0]})[name = string("{n}")];')


@op("softplus")
def _e_softplus(em, t, n, s):
  """Espresso's ANE softplus computes exp(x) in the fp16 datapath and returns 0 for every
  x >= ln(65504) ~= 11.09 (measured on M5/H17s, macOS 26.5; other families may differ but the
  stable form is exact everywhere: sp(10)=10, sp(11)=0 - silent, not an
  error). Lower the stable split instead: softplus(x) = softplus(min(x, 10)) + relu(x - 10),
  whose error is <= log1p(e^-10) ~= 4.5e-5 and whose exp argument never overflows."""
  ten = float(np.float16(10.0)).hex(); nten = float(np.float16(-10.0)).hex()
  em.line(f'fp16 {n}_c = const()[name = string("{n}_c"), val = fp16({ten})];')
  em.line(f'{em.ty(t.shape)} {n}_m = minimum(x = {s[0]}, y = {n}_c)[name = string("{n}_m")];')
  em.line(f'{em.ty(t.shape)} {n}_s = softplus(x = {n}_m)[name = string("{n}_s")];')
  em.line(f'fp16 {n}_nc = const()[name = string("{n}_nc"), val = fp16({nten})];')
  em.line(f'{em.ty(t.shape)} {n}_t = add(x = {s[0]}, y = {n}_nc)[name = string("{n}_t")];')
  em.line(f'{em.ty(t.shape)} {n}_r = relu(x = {n}_t)[name = string("{n}_r")];')
  em.line(f'{em.ty(t.shape)} {n} = add(x = {n}_s, y = {n}_r)[name = string("{n}")];')


@op("prelu")
def _e_prelu(em, t, n, s):
  C = t.shape[1]
  a = em.weight(f"{n}_a", np.asarray(t.attrs["alpha"], np.float32).reshape(C), allow_int8=False)
  em.line(f'{em.ty(t.shape)} {n} = prelu(x = {s[0]}, alpha = {a})[name = string("{n}")];')


@op("less", "equal", "not_equal", "less_equal", "greater_equal")
def _e_cmp(em, t, n, s):
  em.line(f'{em.ty_dt(t.shape, "bool")} {n} = {t.op}(x = {s[0]}, y = {s[1]})[name = string("{n}")];')


@op("logical_not")
def _e_logical_not(em, t, n, s):
  em.line(f'{em.ty_dt(t.shape, "bool")} {n} = logical_not(x = {s[0]})[name = string("{n}")];')


@op("reverse")
def _e_reverse(em, t, n, s):
  axes = list(t.attrs["axes"])
  em.line(f'tensor<int32, [{len(axes)}]> {n}_ax = const()[name = string("{n}_ax"), val = tensor<int32, [{len(axes)}]>({axes})];')
  em.line(f'{em.ty(t.shape)} {n} = reverse(x = {s[0]}, axes = {n}_ax)[name = string("{n}")];')


@op("tile")
def _e_tile(em, t, n, s):
  reps = list(t.attrs["reps"])
  em.line(f'tensor<int32, [{len(reps)}]> {n}_r = const()[name = string("{n}_r"), val = tensor<int32, [{len(reps)}]>({reps})];')
  em.line(f'{em.ty(t.shape)} {n} = tile(x = {s[0]}, reps = {n}_r)[name = string("{n}")];')


@op("inverse")  # unary reciprocal with an epsilon floor
def _e_inverse(em, t, n, s):
  eps = float(np.float16(t.attrs.get("eps", 0.0))).hex()
  em.line(f'{em.ty(t.shape)} {n} = inverse(epsilon = fp16({eps}), x = {s[0]})[name = string("{n}")];')


@op("threshold", "thresholded_relu")  # activation with a single alpha param
def _e_act_alpha(em, t, n, s):
  a = float(np.float16(t.attrs["alpha"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(alpha = fp16({a}), x = {s[0]})[name = string("{n}")];')


@op("scaled_tanh", "clamped_relu", "sigmoid_hard", "linear_activation")  # activation with alpha+beta
def _e_act_alpha_beta(em, t, n, s):
  a = float(np.float16(t.attrs["alpha"])).hex()
  b = float(np.float16(t.attrs["beta"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(alpha = fp16({a}), beta = fp16({b}), ' f'x = {s[0]})[name = string("{n}")];')


@op("square")
def _e_square(em, t, n, s):
  # mul(x,x) not the `square` opcode: square fused with a nonlinearity fails ANECCompile when (Width % 128) > 64.
  em.line(f'{em.ty(t.shape)} {n} = mul(x = {s[0]}, y = {s[0]})[name = string("{n}")];')


@op("log", "rsqrt")  # unary with an epsilon param
def _e_unary_eps(em, t, n, s):
  eps = float(np.float16(t.attrs.get("eps", 0.0))).hex()
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(epsilon = fp16({eps}), x = {s[0]})[name = string("{n}")];')


@op("elu", "leaky_relu")  # unary with an alpha param
def _e_unary_alpha(em, t, n, s):
  a = float(np.float16(t.attrs["alpha"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(alpha = fp16({a}), x = {s[0]})[name = string("{n}")];')


@op("clip")
def _e_clip(em, t, n, s):
  lo, hi = float(np.float16(t.attrs["lo"])).hex(), float(np.float16(t.attrs["hi"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = clip(alpha = fp16({lo}), beta = fp16({hi}), x = {s[0]})[name = string("{n}")];')


@op("gelu")
def _e_gelu(em, t, n, s):
  em.line(f'string {n}_m = const()[name = string("{n}_m"), val = string("EXACT")];')
  em.line(f'{em.ty(t.shape)} {n} = gelu(x = {s[0]}, mode = {n}_m)[name = string("{n}")];')


@op("add", "sub", "real_div", "maximum", "minimum", "pow")
def _e_binary(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(x = {s[0]}, y = {s[1]})[name = string("{n}")];')


# Espresso fuses a SCALAR multiplier into a preceding floor/ceil/round's scale slot, and the
# rounding kernel drops the scale (measured on M5 / macOS 26.5: floor(x)*k returns floor(x);
# scalar ADDS are honored, and no other op family is affected - see issue #112). A FULL-shape
# constant blocks the mis-fusion; a (1,1) broadcast const does not.
_ROUNDING_OPS = ("floor", "ceil", "round")

def _e_mul_full_const(em, t, n, src_name, k):
  w = em.weight(f"{n}_kf", np.full(t.shape, k, np.float16), allow_int8=False)
  em.line(f'{em.ty(t.shape)} {n} = mul(x = {src_name}, y = {w})[name = string("{n}")];')

@op("mul")
def _e_mul(em, t, n, s):
  a, b = t.srcs
  def scalar_const(v): return v.op == "const_array" and np.asarray(v.attrs["value"]).size == 1
  if (a.op in _ROUNDING_OPS and scalar_const(b)) or (b.op in _ROUNDING_OPS and scalar_const(a)):
    const_t, src_name = (b, s[0]) if scalar_const(b) else (a, s[1])
    _e_mul_full_const(em, t, n, src_name, float(np.asarray(const_t.attrs["value"]).ravel()[0]))
    return
  _e_binary(em, t, n, s)


@op("muls")
def _e_muls(em, t, n, s):
  if float(t.attrs["k"]) == 0.0:
    # mul-by-zero after a reduce crashes Espresso (ANECCompile FAILED, measured on M5); x - x is the
    # same zeros for every finite x and lowers everywhere
    em.line(f'{em.ty(t.shape)} {n} = sub(x = {s[0]}, y = {s[0]})[name = string("{n}")];')
    return
  if t.srcs[0].op in _ROUNDING_OPS:                # the same Espresso scale-slot mis-fusion
    _e_mul_full_const(em, t, n, s[0], float(np.float16(t.attrs["k"])))
    return
  k = float(np.float16(t.attrs["k"])).hex()
  em.line(f'fp16 {n}_k = const()[name = string("{n}_k"), val = fp16({k})];')
  em.line(f'{em.ty(t.shape)} {n} = mul(x = {s[0]}, y = {n}_k)[name = string("{n}")];')


@op("adds")
def _e_adds(em, t, n, s):
  # scalar add (x + k): fp16 const broadcast-added to x.
  k = float(np.float16(t.attrs["k"])).hex()
  em.line(f'fp16 {n}_k = const()[name = string("{n}_k"), val = fp16({k})];')
  em.line(f'{em.ty(t.shape)} {n} = add(x = {s[0]}, y = {n}_k)[name = string("{n}")];')


@op("cast")
def _e_cast(em, t, n, s):
  # dtype conversion (e.g. uint8 image port -> fp16 compute type).
  dt = t.attrs.get("dtype", "fp16")
  em.line(f'string {n}_d = const()[name = string("{n}_d"), val = string("{dt}")];')
  em.line(f'{em.ty_dt(t.shape, dt)} {n} = cast(dtype = {n}_d, x = {s[0]})[name = string("{n}")];')


@op("const_array")
def _e_const_array(em, t, n, s):
  # a small baked fp16 const tensor from the weights blob.
  W = np.asarray(t.attrs["value"], dtype=np.float16)
  path = '@model_path/weights.bin'
  o = em.blob.add(fp16_bytes(W), FP16)
  em.line(f'{em.ty(W.shape)} {n} = const()[name = string("{n}"), '
      f'val = {em.ty(W.shape)}(BLOBFILE(path = string("{path}"), offset = uint64({o})))];')


@op("matmul")
def _e_matmul(em, t, n, s):
  # per-node int8 override wins over global self.int8; None defers.
  w = em.weight(f"{n}_w", t.attrs["wt"], allow_int8=True, int8=t.attrs.get("int8"), allow_int4=True, allow_sparse=True)  # wt is [N, K]
  em.line(f'{em.ty(t.shape)} {n} = matmul(transpose_x = bool(false), transpose_y = bool(true), '
      f'x = {s[0]}, y = {w})[name = string("{n}")];')
  if "bias" in t.attrs:
    b = em.weight(f"{n}_b", t.attrs["bias"].reshape(1, -1), allow_int8=False)
    em.line(f'{em.ty(t.shape)} {n}_bb = add(x = {n}, y = {b})[name = string("{n}_bb")];')
    t._name = f"{n}_bb"


@op("bmm")
def _e_bmm(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = matmul(transpose_x = bool(false), transpose_y = bool(false), '
      f'x = {s[0]}, y = {s[1]})[name = string("{n}")];')


@op("bmm_w")
def _e_bmm_w(em, t, n, s):
  # batched matmul x @ W with a baked, quantizable weight W [B, K, N] (per-batch int8 scale via em.weight).
  w = em.weight(f"{n}_w", t.attrs["wt"], allow_int8=True, int8=t.attrs.get("int8"), allow_int4=True)
  em.line(f'{em.ty(t.shape)} {n} = matmul(transpose_x = bool(false), transpose_y = bool(false), '
      f'x = {s[0]}, y = {w})[name = string("{n}")];')


# shared const/param preambles for the conv & pool emitters.
def _const_pt(em, n):
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')


def _const_i32_pair(em, n, suf, v):
  em.line(f'tensor<int32, [2]> {n}_{suf} = const()[name = string("{n}_{suf}"), val = tensor<int32, [2]>([{v}, {v}])];')


def _const_pad4(em, n, p):
  v = [p, p, p, p] if isinstance(p, int) else list(p)    # scalar (symmetric) or (top, bottom, left, right)
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{v[0]}, {v[1]}, {v[2]}, {v[3]}])];')


def _emit_conv_params(em, n, p, st, dl, g):   # pt, st, pd, dl, g (conv/dynamic_conv/conv_transpose)
  _const_pt(em, n); _const_i32_pair(em, n, "st", st); _const_pad4(em, n, p); _const_i32_pair(em, n, "dl", dl)
  em.line(f'int32 {n}_g = const()[name = string("{n}_g"), val = int32({g})];')


def _emit_pool_params(em, n, k, st, p):       # pt, ks, st, pd (max_pool/avg_pool)
  _const_pt(em, n); _const_i32_pair(em, n, "ks", k); _const_i32_pair(em, n, "st", st); _const_pad4(em, n, p)


@op("conv")
def _e_conv(em, t, n, s):
  a = t.attrs
  # per-channel int8 is routable as a conv weight directly; per-node int8 attr overrides the global flag.
  w = em.weight(f"{n}_w", a["weight"], allow_int8=True, int8=a.get("int8"), allow_int4=True, allow_sparse=True)
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  _emit_conv_params(em, n, p, st, dl, g)
  em.line(f'{em.ty(t.shape)} {n} = conv(dilations = {n}_dl, groups = {n}_g, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, weight = {w}, x = {s[0]})[name = string("{n}")];')
  if "bias" in a:
    b = em.weight(f"{n}_bw", a["bias"].reshape(1, t.shape[1], 1, 1), allow_int8=False)
    em.line(f'{em.ty(t.shape)} {n}_b = add(x = {n}, y = {b})[name = string("{n}_b")];')
    t._name = f"{n}_b"


@op("dynamic_conv")
def _e_dynamic_conv(em, t, n, s):
  # conv with a dynamic weight (kernel s[1] is a graph value, not const); batch is 1 by construction.
  a = t.attrs
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  _emit_conv_params(em, n, p, st, dl, g)
  em.line(f'{em.ty(t.shape)} {n} = conv(dilations = {n}_dl, groups = {n}_g, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, weight = {s[1]}, x = {s[0]})[name = string("{n}")];')


@op("max_pool")
def _e_max_pool(em, t, n, s):
  k, st, p = t.attrs["k"], t.attrs["stride"], t.attrs["pad"]
  cm = "true" if t.attrs.get("ceil_mode") else "false"
  _emit_pool_params(em, n, k, st, p)
  em.line(f'bool {n}_cm = const()[name = string("{n}_cm"), val = bool({cm})];')
  em.line(f'{em.ty(t.shape)} {n} = max_pool(ceil_mode = {n}_cm, kernel_sizes = {n}_ks, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, x = {s[0]})[name = string("{n}")];')


@op("avg_pool")
def _e_avg_pool(em, t, n, s):
  k, st, p = t.attrs["k"], t.attrs["stride"], t.attrs["pad"]
  cm = "true" if t.attrs.get("ceil_mode") else "false"
  ep = "true" if t.attrs.get("exclude_pad") else "false"
  _emit_pool_params(em, n, k, st, p)
  em.line(f'bool {n}_ep = const()[name = string("{n}_ep"), val = bool({ep})];')
  em.line(f'bool {n}_cm = const()[name = string("{n}_cm"), val = bool({cm})];')
  em.line(f'{em.ty(t.shape)} {n} = avg_pool(ceil_mode = {n}_cm, exclude_padding_from_average = {n}_ep, '
      f'kernel_sizes = {n}_ks, pad = {n}_pd, pad_type = {n}_pt, strides = {n}_st, x = {s[0]})[name = string("{n}")];')


@op("conv_transpose")
def _e_conv_transpose(em, t, n, s):
  a = t.attrs
  w = em.weight(f"{n}_w", a["weight"], allow_int8=False)         # [Cin, Cout, kH, kW]; int4 unverified here -> fp16
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  _emit_conv_params(em, n, p, st, dl, g)
  em.line(f'{em.ty(t.shape)} {n} = conv_transpose(dilations = {n}_dl, groups = {n}_g, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, weight = {w}, x = {s[0]})[name = string("{n}")];')
  if "bias" in a:
    b = em.weight(f"{n}_bw", a["bias"].reshape(1, t.shape[1], 1, 1), allow_int8=False)
    em.line(f'{em.ty(t.shape)} {n}_b = add(x = {n}, y = {b})[name = string("{n}_b")];')
    t._name = f"{n}_b"


@op("batch_norm")
def _e_batch_norm(em, t, n, s):
  C = t.shape[1]
  g = em.weight(f"{n}_g", t.attrs["gamma"].reshape(C), allow_int8=False)
  b = em.weight(f"{n}_b", t.attrs["beta"].reshape(C), allow_int8=False)
  m = em.weight(f"{n}_m", t.attrs["mean"].reshape(C), allow_int8=False)
  v = em.weight(f"{n}_v", t.attrs["var"].reshape(C), allow_int8=False)
  eps = float(np.float16(t.attrs["eps"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = batch_norm(beta = {b}, epsilon = fp16({eps}), gamma = {g}, '
      f'mean = {m}, variance = {v}, x = {s[0]})[name = string("{n}")];')


@op("upsample")
def _e_upsample(em, t, n, s):
  sc = t.attrs["scale"]
  em.line(f'{em.ty(t.shape)} {n} = upsample_nearest_neighbor(scale_factor_height = int32({sc}), '
      f'scale_factor_width = int32({sc}), x = {s[0]})[name = string("{n}")];')


@op("concat")
def _e_concat(em, t, n, s):
  em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({t.attrs["axis"]})];')
  em.line(f'bool {n}_il = const()[name = string("{n}_il"), val = bool(false)];')
  em.line(f'{em.ty(t.shape)} {n} = concat(axis = {n}_ax, interleave = {n}_il, '
      f'values = ({", ".join(s)}))[name = string("{n}")];')


@op("reduce_mean", "reduce_sum", "reduce_max", "reduce_min",
  "reduce_l1_norm", "reduce_log_sum", "reduce_sum_square", "reduce_log_sum_exp")
def _e_reduce(em, t, n, s):
  axes = list(t.attrs["axes"])
  em.line(f'tensor<int32, [{len(axes)}]> {n}_ax = const()[name = string("{n}_ax"), val = tensor<int32, [{len(axes)}]>({axes})];')
  em.line(f'bool {n}_kd = const()[name = string("{n}_kd"), val = bool(true)];')
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(axes = {n}_ax, keep_dims = {n}_kd, x = {s[0]})[name = string("{n}")];')


@op("reshape")
def _e_reshape(em, t, n, s):
  sh = list(t.shape)
  em.line(f'tensor<int32, [{len(sh)}]> {n}_sh = const()[name = string("{n}_sh"), val = tensor<int32, [{len(sh)}]>({sh})];')
  em.line(f'{em.ty(t.shape)} {n} = reshape(shape = {n}_sh, x = {s[0]})[name = string("{n}")];')


@op("softmax")
def _e_softmax(em, t, n, s):
  em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({t.attrs["axis"]})];')
  em.line(f'{em.ty(t.shape)} {n} = softmax(axis = {n}_ax, x = {s[0]})[name = string("{n}")];')


@op("l2_norm")
def _e_l2_norm(em, t, n, s):
  # x / max(reduce_l2_norm(x), eps); eps is a safe-divide floor.
  ax, eps = t.attrs["axis"], float(np.float16(t.attrs["eps"] ** 0.5)).hex()
  red_shape = tuple(1 if i == ax else d for i, d in enumerate(t.shape))
  em.line(f'tensor<int32,[1]> {n}_ax = const()[name=string("{n}_ax"), val=tensor<int32,[1]>([{ax}])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'{em.ty(red_shape)} {n}_nrm = reduce_l2_norm(axes={n}_ax, keep_dims={n}_kd, x={s[0]})[name=string("{n}_nrm")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({eps})];')
  em.line(f'{em.ty(red_shape)} {n}_sn = maximum(x={n}_nrm, y={n}_ep)[name=string("{n}_sn")];')
  em.line(f'{em.ty(t.shape)} {n} = real_div(x={s[0]}, y={n}_sn)[name=string("{n}")];')


@op("pixel_shuffle")
def _e_pixel_shuffle(em, t, n, s):
  r = t.attrs["r"]
  em.line(f'int32 {n}_r = const()[name=string("{n}_r"), val=int32({r})];')
  em.line(f'{em.ty(t.shape)} {n} = pixel_shuffle(upscale_factor={n}_r, x={s[0]})[name=string("{n}")];')


@op("pixel_unshuffle")
def _e_pixel_unshuffle(em, t, n, s):
  r = t.attrs["r"]
  em.line(f'uint32 {n}_r = const()[name=string("{n}_r"), val=uint32({r})];')
  em.line(f'{em.ty(t.shape)} {n} = pixel_unshuffle(downscale_factor={n}_r, x={s[0]})[name=string("{n}")];')


@op("transpose")
def _e_transpose(em, t, n, s):
  perm = list(t.attrs["perm"])
  em.line(f'tensor<int32, [{len(perm)}]> {n}_p = const()[name = string("{n}_p"), val = tensor<int32, [{len(perm)}]>({perm})];')
  em.line(f'{em.ty(t.shape)} {n} = transpose(perm = {n}_p, x = {s[0]})[name = string("{n}")];')


@op("squeeze", "expand_dims")
def _e_axes_reshape(em, t, n, s):
  axes = list(t.attrs["axes"])
  em.line(f'tensor<int32, [{len(axes)}]> {n}_ax = const()[name = string("{n}_ax"), '
      f'val = tensor<int32, [{len(axes)}]>({axes})];')
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(axes = {n}_ax, x = {s[0]})[name = string("{n}")];')


@op("flatten2d")
def _e_flatten2d(em, t, n, s):
  em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({t.attrs["axis"]})];')
  em.line(f'{em.ty(t.shape)} {n} = flatten2d(axis = {n}_ax, x = {s[0]})[name = string("{n}")];')


@op("slice_by_size")
def _e_slice_by_size(em, t, n, s):
  b, sz = list(t.attrs["begin"]), list(t.attrs["size"])
  em.line(f'tensor<int32, [{len(b)}]> {n}_b = const()[name = string("{n}_b"), val = tensor<int32, [{len(b)}]>({b})];')
  em.line(f'tensor<int32, [{len(sz)}]> {n}_sz = const()[name = string("{n}_sz"), val = tensor<int32, [{len(sz)}]>({sz})];')
  em.line(f'{em.ty(t.shape)} {n} = slice_by_size(begin = {n}_b, size = {n}_sz, x = {s[0]})[name = string("{n}")];')


@op("stack")
def _e_stack(em, t, n, s):
  em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({t.attrs["axis"]})];')
  em.line(f'{em.ty(t.shape)} {n} = stack(axis = {n}_ax, values = ({", ".join(s)}))[name = string("{n}")];')


@op("split")
def _e_split(em, t, n, s):
  # emit the N-output split once; later split nodes alias its ports by name.
  ns, ax, which = t.attrs["num_splits"], t.attrs["axis"], t.attrs["which"]
  src = s[0]
  key = (src, ns, ax)
  names = em._split_cache.get(key)
  if names is None:
    names = [f"{n}_part{i}" for i in range(ns)]
    decl = ", ".join(f"{em.ty(t.shape)} {names[i]}" for i in range(ns))
    em.line(f'int32 {n}_ns = const()[name = string("{n}_ns"), val = int32({ns})];')
    em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({ax})];')
    em.line(f'{decl} = split(axis = {n}_ax, num_splits = {n}_ns, x = {src})[name = string("{n}_split")];')
    em._split_cache[key] = names
  t._name = names[which]


@op("greater")
def _e_greater(em, t, n, s):
  em.line(f'{em.ty_dt(t.shape, "bool")} {n} = greater(x = {s[0]}, y = {s[1]})[name = string("{n}")];')


@op("select")
def _e_select(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = select(cond = {s[0]}, a = {s[1]}, b = {s[2]})[name = string("{n}")];')


@op("instance_norm")
def _e_instance_norm(em, t, n, s):
  C = t.shape[1]
  g = em.weight(f"{n}_g", t.attrs["gamma"].reshape(C), allow_int8=False)
  b = em.weight(f"{n}_b", t.attrs["beta"].reshape(C), allow_int8=False)
  eps = float(np.float16(t.attrs["eps"])).hex()
  em.line(f'{em.ty(t.shape)} {n} = instance_norm(beta = {b}, epsilon = fp16({eps}), '
      f'gamma = {g}, x = {s[0]})[name = string("{n}")];')


@op("local_response_norm")
def _e_local_response_norm(em, t, n, s):
  a = t.attrs
  al, bt = float(np.float16(a["alpha"])).hex(), float(np.float16(a["beta"])).hex()
  k = float(np.float16(a["k"])).hex()
  em.line(f'int32 {n}_sz = const()[name = string("{n}_sz"), val = int32({a["size"]})];')
  em.line(f'{em.ty(t.shape)} {n} = local_response_norm(alpha = fp16({al}), beta = fp16({bt}), '
      f'k = fp16({k}), size = {n}_sz, x = {s[0]})[name = string("{n}")];')


@op("einsum")
def _e_einsum(em, t, n, s):
  # only the verified 'nchw,nwhu->nchu' equation is reachable; b is a streamed weight.
  b = em.weight(f"{n}_b", t.attrs["b"], allow_int8=False)
  em.line(f'string {n}_eq = const()[name = string("{n}_eq"), val = string("nchw,nwhu->nchu")];')
  em.line(f'{em.ty(t.shape)} {n} = einsum(equation = {n}_eq, values = ({s[0]}, {b}))[name = string("{n}")];')


@op("space_to_depth", "depth_to_space")
def _e_depth_space(em, t, n, s):
  em.line(f'int32 {n}_bs = const()[name = string("{n}_bs"), val = int32({t.attrs["block_size"]})];')
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(block_size = {n}_bs, x = {s[0]})[name = string("{n}")];')


@op("crop")
def _e_crop(em, t, n, s):
  ch, cw = t.attrs["crop_h"], t.attrs["crop_w"]
  em.line(f'tensor<int32, [2]> {n}_ch = const()[name = string("{n}_ch"), val = tensor<int32, [2]>([{ch[0]}, {ch[1]}])];')
  em.line(f'tensor<int32, [2]> {n}_cw = const()[name = string("{n}_cw"), val = tensor<int32, [2]>([{cw[0]}, {cw[1]}])];')
  em.line(f'{em.ty(t.shape)} {n} = crop(crop_height = {n}_ch, crop_width = {n}_cw, x = {s[0]})[name = string("{n}")];')


@op("resize_nearest_neighbor")
def _e_resize_nn(em, t, n, s):
  a = t.attrs
  em.line(f'int32 {n}_th = const()[name = string("{n}_th"), val = int32({a["target_h"]})];')
  em.line(f'int32 {n}_tw = const()[name = string("{n}_tw"), val = int32({a["target_w"]})];')
  em.line(f'{em.ty(t.shape)} {n} = resize_nearest_neighbor(target_size_height = {n}_th, '
      f'target_size_width = {n}_tw, x = {s[0]})[name = string("{n}")];')


@op("resize_bilinear")
def _e_resize_bilinear(em, t, n, s):
  a = t.attrs
  mode = "ALIGN_CORNERS" if a["align_corners"] else "DEFAULT"
  em.line(f'int32 {n}_th = const()[name = string("{n}_th"), val = int32({a["target_h"]})];')
  em.line(f'int32 {n}_tw = const()[name = string("{n}_tw"), val = int32({a["target_w"]})];')
  em.line(f'string {n}_sm = const()[name = string("{n}_sm"), val = string("{mode}")];')
  em.line(f'{em.ty(t.shape)} {n} = resize_bilinear(sampling_mode = {n}_sm, '
      f'target_size_height = {n}_th, target_size_width = {n}_tw, x = {s[0]})[name = string("{n}")];')


@op("upsample_bilinear")
def _e_upsample_bilinear(em, t, n, s):
  a = t.attrs
  sc = float(a["scale"])
  ac = "true" if a["align_corners"] else "false"
  em.line(f'bool {n}_ac = const()[name = string("{n}_ac"), val = bool({ac})];')
  em.line(f'{em.ty(t.shape)} {n} = upsample_bilinear(align_corners = {n}_ac, '
      f'scale_factor_height = fp32({sc}), scale_factor_width = fp32({sc}), '
      f'x = {s[0]})[name = string("{n}")];')


@op("affine")
def _e_affine(em, t, n, s):
  a = t.attrs
  tm = em.weight(f"{n}_tm", a["transform"], allow_int8=False)
  ac = "true" if a["align_corners"] else "false"
  em.line(f'int32 {n}_oh = const()[name = string("{n}_oh"), val = int32({a["output_h"]})];')
  em.line(f'int32 {n}_ow = const()[name = string("{n}_ow"), val = int32({a["output_w"]})];')
  em.line(f'string {n}_sm = const()[name = string("{n}_sm"), val = string("bilinear")];')
  em.line(f'string {n}_pm = const()[name = string("{n}_pm"), val = string("constant")];')
  em.line(f'fp16 {n}_pv = const()[name = string("{n}_pv"), val = fp16(0x0p+0)];')
  em.line(f'string {n}_cm = const()[name = string("{n}_cm"), val = string("normalized_minus_one_to_one")];')
  em.line(f'bool {n}_ac = const()[name = string("{n}_ac"), val = bool({ac})];')
  em.line(f'{em.ty(t.shape)} {n} = affine(align_corners = {n}_ac, coordinates_mode = {n}_cm, '
      f'output_height = {n}_oh, output_width = {n}_ow, padding_mode = {n}_pm, '
      f'padding_value = {n}_pv, sampling_mode = {n}_sm, transform_matrix = {tm}, '
      f'x = {s[0]})[name = string("{n}")];')


@op("rms_norm")
def _e_rms_norm(em, t, n, s):
  # RMS-norm via fused reduce_l2_norm over the last axis: rms(x) = x*sqrt(D)/sqrt(sum(x^2))*g
  # (sqrt(D) folds into gamma, eps is a safe-divide floor).
  D = t.shape[-1]
  ax = len(t.shape) - 1
  gw = (t.attrs["gamma"].reshape(1, D).astype(np.float32) * float(np.sqrt(D)))
  g2 = em.weight(f"{n}_g", gw, allow_int8=False)
  flo = float(np.float16((D * float(t.attrs["eps"])) ** 0.5)).hex()
  red_shape = tuple(1 if i == ax else d for i, d in enumerate(t.shape))
  em.line(f'tensor<int32,[1]> {n}_ax = const()[name=string("{n}_ax"), val=tensor<int32,[1]>([{ax}])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'{em.ty(red_shape)} {n}_nrm = reduce_l2_norm(axes={n}_ax, keep_dims={n}_kd, x={s[0]})[name=string("{n}_nrm")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({flo})];')
  em.line(f'{em.ty(red_shape)} {n}_sn = maximum(x={n}_nrm, y={n}_ep)[name=string("{n}_sn")];')
  em.line(f'{em.ty(t.shape)} {n}_dv = real_div(x={s[0]}, y={n}_sn)[name=string("{n}_dv")];')
  em.line(f'{em.ty(t.shape)} {n} = mul(x={n}_dv, y={g2})[name=string("{n}")];')


@op("layer_norm")
def _e_layer_norm(em, t, n, s):
  D = t.shape[-1]
  g2 = em.weight(f"{n}_g", t.attrs["gamma"].reshape(1, D), allow_int8=False)
  b2 = em.weight(f"{n}_b", t.attrs["beta"].reshape(1, D), allow_int8=False)
  eps = float(np.float16(t.attrs["eps"])).hex()
  ax = len(t.shape) - 1
  red_shape = tuple(1 if i == ax else d for i, d in enumerate(t.shape))
  em.line(f'tensor<int32,[1]> {n}_ax = const()[name=string("{n}_ax"), val=tensor<int32,[1]>([{ax}])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'{em.ty(red_shape)} {n}_mu = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={s[0]})[name=string("{n}_mu")];')
  em.line(f'{em.ty(t.shape)} {n}_xc = sub(x={s[0]}, y={n}_mu)[name=string("{n}_xc")];')
  em.line(f'{em.ty(t.shape)} {n}_sq = mul(x={n}_xc, y={n}_xc)[name=string("{n}_sq")];')
  em.line(f'{em.ty(red_shape)} {n}_var = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={n}_sq)[name=string("{n}_var")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({eps})];')
  em.line(f'{em.ty(red_shape)} {n}_ve = add(x={n}_var, y={n}_ep)[name=string("{n}_ve")];')
  em.line(f'{em.ty(red_shape)} {n}_rr = rsqrt(epsilon=fp16(0.0), x={n}_ve)[name=string("{n}_rr")];')
  em.line(f'{em.ty(t.shape)} {n}_xn = mul(x={n}_xc, y={n}_rr)[name=string("{n}_xn")];')
  em.line(f'{em.ty(t.shape)} {n}_gg = mul(x={n}_xn, y={g2})[name=string("{n}_gg")];')
  em.line(f'{em.ty(t.shape)} {n} = add(x={n}_gg, y={b2})[name=string("{n}")];')


@op("channel_layer_norm")
def _e_channel_layer_norm(em, t, n, s):
  # LayerNorm over the channel axis of a channels-first [N,C,1,S] tensor, no reshape in/out.
  N, C, _, S = t.shape
  ty = em.ty(t.shape)
  rty = f"tensor<fp16,[{N},1,1,{S}]>"
  g4 = em.weight(f"{n}_g", t.attrs["gamma"].reshape(1, C, 1, 1), allow_int8=False)
  b4 = em.weight(f"{n}_b", t.attrs["beta"].reshape(1, C, 1, 1), allow_int8=False)
  eps = float(np.float16(t.attrs["eps"])).hex()
  em.line(f'tensor<int32,[1]> {n}_ax = const()[name=string("{n}_ax"), val=tensor<int32,[1]>([1])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'{rty} {n}_mu = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={s[0]})[name=string("{n}_mu")];')
  em.line(f'{ty} {n}_xc = sub(x={s[0]}, y={n}_mu)[name=string("{n}_xc")];')
  em.line(f'{ty} {n}_sq = mul(x={n}_xc, y={n}_xc)[name=string("{n}_sq")];')
  em.line(f'{rty} {n}_var = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={n}_sq)[name=string("{n}_var")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({eps})];')
  em.line(f'{rty} {n}_ve = add(x={n}_var, y={n}_ep)[name=string("{n}_ve")];')
  em.line(f'{rty} {n}_rr = rsqrt(epsilon=fp16(0.0), x={n}_ve)[name=string("{n}_rr")];')
  em.line(f'{ty} {n}_xn = mul(x={n}_xc, y={n}_rr)[name=string("{n}_xn")];')
  em.line(f'{ty} {n}_gg = mul(x={n}_xn, y={g4})[name=string("{n}_gg")];')
  em.line(f'{ty} {n} = add(x={n}_gg, y={b4})[name=string("{n}")];')


@op("group_norm")
def _e_group_norm(em, t, n, s):
  # Rank-4 tiling: reshape to [1,G,C/G,H*W] and reduce the trailing two axes (keeps each axis under the cap).
  _, C, H, W = t.shape
  G = t.attrs["groups"]; D = C // G; HW = H * W
  g4 = em.weight(f"{n}_g", t.attrs["gamma"].reshape(1, C, 1, 1), allow_int8=False)
  b4 = em.weight(f"{n}_b", t.attrs["beta"].reshape(1, C, 1, 1), allow_int8=False)
  invd, eps = float(np.float16(1.0 / (D * HW))).hex(), float(np.float16(t.attrs["eps"])).hex()
  em.line(f'tensor<int32,[4]> {n}_gs = const()[name=string("{n}_gs"), val=tensor<int32,[4]>([1,{G},{D},{HW}])];')
  em.line(f'tensor<int32,[4]> {n}_os = const()[name=string("{n}_os"), val=tensor<int32,[4]>([1,{C},{H},{W}])];')
  em.line(f'tensor<int32,[1]> {n}_a2 = const()[name=string("{n}_a2"), val=tensor<int32,[1]>([2])];')
  em.line(f'tensor<int32,[1]> {n}_a3 = const()[name=string("{n}_a3"), val=tensor<int32,[1]>([3])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'tensor<fp16,[1,{G},{D},{HW}]> {n}_xg = reshape(shape={n}_gs, x={s[0]})[name=string("{n}_xg")];')
  # mean = sum over both group axes * 1/(D*HW).
  em.line(f'tensor<fp16,[1,{G},{D},1]> {n}_s3 = reduce_sum(axes={n}_a3, keep_dims={n}_kd, x={n}_xg)[name=string("{n}_s3")];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_ss = reduce_sum(axes={n}_a2, keep_dims={n}_kd, x={n}_s3)[name=string("{n}_ss")];')
  em.line(f'fp16 {n}_id = const()[name=string("{n}_id"), val=fp16({invd})];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_mu = mul(x={n}_ss, y={n}_id)[name=string("{n}_mu")];')
  em.line(f'tensor<fp16,[1,{G},{D},{HW}]> {n}_xc = sub(x={n}_xg, y={n}_mu)[name=string("{n}_xc")];')
  em.line(f'tensor<fp16,[1,{G},{D},{HW}]> {n}_sq = mul(x={n}_xc, y={n}_xc)[name=string("{n}_sq")];')
  em.line(f'tensor<fp16,[1,{G},{D},1]> {n}_v3 = reduce_sum(axes={n}_a3, keep_dims={n}_kd, x={n}_sq)[name=string("{n}_v3")];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_vs = reduce_sum(axes={n}_a2, keep_dims={n}_kd, x={n}_v3)[name=string("{n}_vs")];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_var = mul(x={n}_vs, y={n}_id)[name=string("{n}_var")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({eps})];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_ve = add(x={n}_var, y={n}_ep)[name=string("{n}_ve")];')
  em.line(f'tensor<fp16,[1,{G},1,1]> {n}_rr = rsqrt(epsilon=fp16(0.0), x={n}_ve)[name=string("{n}_rr")];')
  em.line(f'tensor<fp16,[1,{G},{D},{HW}]> {n}_xn = mul(x={n}_xc, y={n}_rr)[name=string("{n}_xn")];')
  em.line(f'{em.ty(t.shape)} {n}_n4 = reshape(shape={n}_os, x={n}_xn)[name=string("{n}_n4")];')
  em.line(f'{em.ty(t.shape)} {n}_gg = mul(x={n}_n4, y={g4})[name=string("{n}_gg")];')
  em.line(f'{em.ty(t.shape)} {n} = add(x={n}_gg, y={b4})[name=string("{n}")];')


# compiled model

class Model:
  """A compiled, fused ANE program. Call it with the input array(s), in `af.input` order."""

  def __init__(self, prog, inputs: list[tuple[str, tuple]], out_name: str, out_shape, n_ops: int,
        input_tensors: list | None = None):
    self._prog = prog
    self._inputs = inputs
    # ordered input Tensors: lets autograd.Trainer map each input to its source Tensor.
    self._input_tensors = input_tensors if input_tensors is not None else []
    self._out_name, self._out_shape = out_name, out_shape
    self.n_ops = n_ops  # graph ops fused into this single program

  def __call__(self, *arrays: np.ndarray) -> np.ndarray:
    if len(arrays) != len(self._inputs): raise ValueError(f"expected {len(self._inputs)} input(s), got {len(arrays)}")
    dts = getattr(self._prog, "_input_dtypes", {})
    feed = {}
    for (name, shape), a in zip(self._inputs, arrays):
      a = np.asarray(a)
      if tuple(a.shape) != tuple(shape): raise ValueError(f"input '{name}' shape {a.shape} != compiled {shape}")
      if dts.get(name, "fp16") == "uint8":
        if not np.issubdtype(a.dtype, np.integer):
          raise TypeError(f"input '{name}' is a uint8 image port; pass an integer/uint8 "
                  f"array (got dtype {a.dtype})")
        feed[name] = a.astype(np.uint8)              # raw bytes
      else:
        feed[name] = a.astype(np.float16)
    return self._prog.eval(feed)[self._out_name].astype(np.float32)

  # ---- zero-copy hot-loop API: input_view (write) / execute / output_view (read), no memcpy ----
  def input_view(self, name: str | None = None) -> np.ndarray:
    """Writable fp16 view onto an input buffer (defaults to the sole input port)."""
    return self._prog.input_view(name or self._inputs[0][0])

  def output_view(self) -> np.ndarray:
    """fp16 view onto the output buffer, valid after `execute()`."""
    return self._prog.output_view(self._out_name)

  def execute(self) -> None:
    """Run once without binding inputs / reading outputs (pair with input_view/output_view)."""
    self._prog.execute()

  def release(self) -> None:
    self._prog.release()


def _in_dtype(t: Tensor) -> str:
  """Wire dtype of an input port ("fp16" default; "uint8" for image inputs)."""
  return t.attrs.get("dtype", "fp16")


def _input_dtypes(inputs) -> dict:
  """{port_name: dtype} for the non-fp16 input ports."""
  return {t._name: _in_dtype(t) for t in inputs if _in_dtype(t) != "fp16"}


def _sig_ty(em, t: Tensor) -> str:
  """MIL type for an input in the function signature (honors the port dtype)."""
  dt = _in_dtype(t)
  return em.ty(t.shape) if dt == "fp16" else em.ty_dt(t.shape, dt)


def cache_root() -> Path:
  """Persistent content-addressed build/compile cache (default ~/Models/.aneforge-cache, override
  ANEFORGE_CACHE_DIR) -- replaces per-compile $TMPDIR mkdtemp dirs, which leaked and re-emitted every call."""
  root = os.environ.get("ANEFORGE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), "Models", ".aneforge-cache")
  p = Path(root); p.mkdir(parents=True, exist_ok=True)
  return p


def _emit_program_dir(em, inputs, out_var, out_shape, build_dir):
  """Write one e5rt program (MIL + weights) to a dir. Without an explicit `build_dir` it's content-addressed
  under `cache_root()`: identical graph+weights is written and compiled ONCE, reused on every later compile."""
  sig = ", ".join(f"{_sig_ty(em, t)} {t._name}" for t in inputs)
  mil = (f"program(1.3)\n{_BUILD_INFO}\n{{\n    func main<ios18>({sig}) {{\n"
     + "\n".join(em.lines) + f"\n    }} -> ({out_var});\n}}\n")
  weights = em.blob.build()
  if build_dir:
    d = Path(build_dir)
  else:
    key = hashlib.sha256(mil.encode() + weights).hexdigest()[:24]
    d = cache_root() / key
    if (d / "model.mil").is_file() and (d / "weights.bin").is_file():
      return d                                   # reuse: identical program already emitted (cache/ compiled too)
  d.mkdir(parents=True, exist_ok=True)
  (d / "model.mil").write_text(mil)
  (d / "weights.bin").write_bytes(weights)
  return d


def _assemble_and_compile(em, inputs, out_var, out_shape, build_dir):
  """Write one e5rt program (MIL + weights) for a set of input tensors and compile it."""
  d = _emit_program_dir(em, inputs, out_var, out_shape, build_dir)
  return E5RT.compile(mil_path=d / "model.mil", cache_dir=str(d / "cache"),
            inputs={t._name: t.shape for t in inputs}, outputs={out_var: out_shape},
            device_mask=ANE_MASK, input_dtypes=_input_dtypes(inputs))


def _lower_fused_to_dir(out: Tensor, build_dir=None, int8: bool = False):
  """Lower a fused graph to a MIL+weights dir WITHOUT compiling it (bridge graphs raise)."""
  order = _topo(out)
  if any(t.op in NETPLIST_OPS for t in order):
    raise NotImplementedError("cross_compile_check: bridge/segmented graphs not supported")
  bad = sorted({t.op for t in order if t.op != "input" and t.op not in _EMIT})
  if bad: raise NotImplementedError(f"aneforge: ops not reachable on the ANE: {bad}")
  inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
  for i, t in enumerate(order): t._name = f"t{i}"
  em = _Emitter(int8)
  for t in order:
    if t.op != "input": _EMIT[t.op](em, t, t._name, [src._name for src in t.srcs])
  return _emit_program_dir(em, inputs, out._name, out.shape, build_dir)


def cross_compile_check(out: Tensor, target, int8: bool = False) -> bool:
  """Does the graph compile for another ANE family, checked from this host (compile-level only)?"""
  from . import _runtime
  from . import _targets as TG
  if out.op == "input": out = out * 1.0            # same empty-program guard as compile()
  if isinstance(target, str):
    arch = target.strip().lower()
    # e5rt silently falls back to the host target on an unknown arch string (false pass); gate first.
    if arch not in TG._ARCH_FAMILY:
      raise ValueError(f"unknown ANE target arch {target!r}; known: " f"{sorted(TG._ARCH_FAMILY)}")
  else:
    arch = TG.arch_for_family(int(target))
  # Static pre-gate: the host cross-compiler doesn't enforce another family's caps (false pass).
  fam = TG.family_of_arch(arch)
  rep = TG.preflight(out, fam)
  if not rep.ok:
    import warnings
    bad = sorted({r.op for r in rep.reject}) or \
      [f"{r.op}{tuple(r.shape)}" for r in rep.oversize]
    warnings.warn(
      f"aneforge.cross_compile_check: graph does not fit ANE family {fam} "
      f"({arch}): {bad}. Rejected statically (preflight); the compiler was not run.",
      stacklevel=2)
    return False
  # warn-only cross-chip fp16 divergence annotation before compiling.
  try:
    _warn_fp16_cross_chip(out, TG.detect_family(), fam)
  except Exception as e:
    import warnings
    warnings.warn(
      f"aneforge.cross_compile_check: cross-chip fp16 divergence prediction was "
      f"unavailable ({e!r}); validate numerics on the target silicon.",
      stacklevel=2)
  d = _lower_fused_to_dir(out, None, int8=int8)
  rc = _runtime.compile_check(d / "model.mil", d / f"cache_{arch}", custom_ane_options=f"TargetArchitecture={arch}")
  return rc == 0


class MultiModel:
  """A compiled fused program with N named outputs (e.g. a resident training step)."""

  def __init__(self, prog, inputs: list, outputs: list):
    self.prog = prog
    self.input_tensors = list(inputs)
    self.output_tensors = list(outputs)
    # snapshot port names: a later compile sharing these Tensors reassigns t._name.
    self.input_ports = [(t, t._name) for t in inputs]
    self.output_ports = [(t, t._name) for t in outputs]

  def __call__(self, *arrays: np.ndarray) -> dict:
    if len(arrays) != len(self.input_ports):
      raise ValueError(f"expected {len(self.input_ports)} input(s), got {len(arrays)}")
    dts = getattr(self.prog, "_input_dtypes", {})
    feed = {name: (np.asarray(a).astype(np.uint8)
           if dts.get(name, "fp16") == "uint8" else
           np.asarray(a).astype(np.float16))
        for (_, name), a in zip(self.input_ports, arrays)}
    out = self.prog.eval(feed)
    return {name: out[name].astype(np.float32) for _, name in self.output_ports}

  def release(self) -> None:
    self.prog.release()


def compile_multi(outs, build_dir=None, int8: bool = False, compress: str | None = None,
          compress_atol: float = 0.05, block_size: int = 32) -> MultiModel:
  """Lower a multi-output graph into one fused program with N named output ports (fp16 I/O). `int8`/`compress`
  quantize the constant weights as in `compile` (per-channel int8 / int4-LUT / blockwise); I/O stays fp16."""
  order = _topo_multi(*outs)
  if any(t.op in NETPLIST_OPS for t in order):
    raise NotImplementedError("compile_multi: native-SDPA/netplist ops not supported")
  bad = sorted({t.op for t in order if t.op != "input" and t.op not in _EMIT})
  if bad: raise NotImplementedError(f"aneforge: ops not reachable on the ANE: {bad}")
  inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
  if not inputs: raise ValueError("aneforge.compile_multi: graph has no inputs")
  for i, t in enumerate(order): t._name = f"t{i}"

  em = _Emitter(int8, compress=compress, compress_atol=compress_atol, block_size=block_size)
  for t in order:
    if t.op != "input": _EMIT[t.op](em, t, t._name, [src._name for src in t.srcs])

  out_vars = ", ".join(o._name for o in outs)
  d = _emit_program_dir(em, inputs, out_vars, None, build_dir)
  prog = E5RT.compile(mil_path=d / "model.mil", cache_dir=str(d / "cache"),
            inputs={t._name: t.shape for t in inputs},
            outputs={o._name: o.shape for o in outs}, device_mask=ANE_MASK,
            input_dtypes=_input_dtypes(inputs))
  return MultiModel(prog, inputs, list(outs))


class PrecisionWarning(UserWarning):
  """Emitted by `compile` when a graph has fp16-cancellation-risk nodes (results may be inaccurate)."""


class DispatchFloorWarning(UserWarning):
  """Emitted by `compile` when a program is dispatch-floor-bound."""


def _dispatch_floor_signal(out: Tensor) -> None:
  """Warn once if the program is dispatch-floor-bound (predicted cost ~= the per-call floor)."""
  try:
    from ._cost import estimate, _constants
    pred = float(estimate(out))
    floor = float(_constants()["floor_us"])
  except Exception as e:
    import warnings
    warnings.warn(
      f"aneforge.compile: dispatch-floor estimate was unavailable ({e!r}); a "
      f"floor-bound program would not be flagged.", stacklevel=3)
    return
  if pred > floor * 1.5:          # already compute/bytes-bound
    return
  import warnings
  msg = ("aneforge.compile: dispatch-floor-bound program (predicted "
     f"~{pred:.0f}us, ~all of it the fixed per-call cost). Every ANE call pays a fixed "
     "dispatch + firmware round-trip (~0.2 ms on M1) regardless of how small the work "
     "is, and dispatch is single-in-flight (threads/concurrency do NOT amortize it). "
     "To amortize, do more work per call: increase batch N (per-sample cost falls ~linearly "
     "toward the compute rate) or chain more ops into one program (share_buffer keeps state "
     "resident). Silence: filter aneforge.DispatchFloorWarning.")
  warnings.warn(msg, DispatchFloorWarning, stacklevel=3)


def _precision_signal(out: Tensor, strict: bool) -> None:
  """Warn (or, with strict, raise) on the graph's fp16 cancellation / narrow-reduce hotspots."""
  from ._cost import precision_risk
  risk = precision_risk(out)
  if not risk["hotspots"]: return
  flagged = [n for n in risk["nodes"] if n["idx"] in risk["hotspots"]]
  items = "; ".join(
    f"node {n['idx']} {n['op']}[{n['kind']}]: {n['reason']}"
    + (f" (fix: {n['fixable']})" if n["fixable"] else " (avoid)")
    for n in flagged)
  msg = (f"aneforge.compile: {len(flagged)} fp16-precision-risk node(s) "
     f"(graph_error~{risk['graph_error']:.1e}); results may be inaccurate in fp16. "
     f"{items}. Inspect: precision_risk(out, verbose=True); mitigate: tune_precision() "
     f"or paired-fp16.")
  if strict: raise ValueError(msg)
  import warnings
  warnings.warn(msg, PrecisionWarning, stacklevel=3)


def _warn_h13_slice_saturation(out: Tensor, family: int) -> None:
  """Warn on the pre-A16 last-axis-offset slice_by_size Q.4-crop-DMA quirk (wrong elements / saturation)."""
  import warnings

  from . import _targets as TG
  if int(family) >= int(TG.Family.A16):       # A16+ (M5/M4) is unaffected
    return
  sat_family = int(family) <= int(TG.Family.A14)   # saturation measured on A13 + A14
  for t in _topo(out):
    # Mode (1): concat of >= 2 nonzero-width-offset slices -> wrong elements (pre-A16).
    if t.op == "concat":
      n_w = sum(1 for s in t.srcs if s.op == "slice_by_size" and (s.attrs.get("begin") or [0])[-1] > 0)
      if n_w >= 2:
        warnings.warn(
          f"aneforge: concat of {n_w} width-offset slice_by_size on a pre-A16 ANE "
          f"(family {int(family)}) routes through the Q.4 crop-DMA and returns WRONG "
          f"ELEMENTS (the gather-axis-1 failure mode). Slice on a non-last axis "
          f"(transpose the sliced axis off the last position).",
          stacklevel=3)
    # Mode (2): magnitude saturation on any single width-offset slice (A13 + A14).
    if sat_family and t.op == "slice_by_size" and (t.attrs.get("begin") or [0])[-1] > 0:
      warnings.warn(
        f"aneforge: last-axis-offset slice_by_size(begin={t.attrs['begin']}) on a "
        f"pre-A16 ANE (family {int(family)}) saturates any sliced |value| > 4094 "
        f"(=65504/16) to +/-inf via the Q.4 crop-DMA. Safe if the data stays under "
        f"4094; otherwise slice with a zero last-axis begin or on the host.",
        stacklevel=3)


class CrossChipFP16Warning(UserWarning):
  """Emitted by `cross_compile_check` when a graph compiles for a different-family target but an op's fp16 value can diverge."""


def _node_max_abs(t: Tensor):
  """Static bound on a node's value magnitude: a baked const's max, a declared input bound, else None."""
  v = t.attrs.get("value")
  if v is not None:
    try:
      import numpy as _np
      return float(_np.abs(_np.asarray(v)).max())
    except Exception:
      return None
  return t.attrs.get("max_abs")            # af.input(..., max_abs=) declaration, or None


# Ops whose output magnitude is bounded regardless of input: a normalization or a saturating
# activation. Value is the bound on |out|. Keeps propagation useful past a norm, which is where most
# real activations become small.
_BOUNDED_OUT = {
  "sigmoid": 1.0, "tanh": 1.0, "softmax": 1.0, "softsign": 1.0, "erf": 1.0, "relu6": 6.0,
  "sigmoid_hard": 1.0, "l2_norm": 1.0,
}
# Elementwise unary ops that cannot increase |value|; the input bound carries through.
_NONEXPANDING = frozenset({"relu", "abs", "floor", "ceil", "round", "sign", "reverse", "transpose",
                           "reshape", "squeeze", "expand_dims", "flatten2d", "slice_by_size", "tile",
                           "identity", "clamped_relu", "thresholded_relu", "threshold"})


def _propagate_max_abs(t: Tensor, seen: "dict | None" = None):
  """Conservative upper bound on `|t|`, or None when it cannot be bounded.

  Over-estimating is always safe for the caller (a loose bound declines a lossy variant more often,
  it never wrongly keeps one), so anything not modelled returns None rather than a guess. Covers the
  ops that actually feed matmuls; see #155."""
  import numpy as _np
  if seen is None: seen = {}
  key = id(t)
  if key in seen: return seen[key]
  seen[key] = None                                   # cycles cannot happen, but stay safe
  b = _bound_of(t, seen, _np)
  seen[key] = b
  return b


def _bound_of(t: Tensor, seen: dict, _np):
  op = t.op
  if not t.srcs:                                     # leaf: const value or declared input bound
    return _node_max_abs(t)
  if op in _BOUNDED_OUT:
    return _BOUNDED_OUT[op]
  if op == "clip":
    lo, hi = t.attrs.get("lo"), t.attrs.get("hi")
    if lo is None or hi is None: return None
    return max(abs(float(lo)), abs(float(hi)))
  if op in ("layer_norm", "rms_norm", "group_norm", "instance_norm"):
    # normalized to ~unit scale, then scaled by gamma and shifted by beta. The normalized part is
    # data-dependent (fp16 keeps it O(1) but not provably <=1), so only bound it when the affine is
    # baked: |out| <= max|gamma| * NORM_SIGMA + max|beta|.
    g, b = t.attrs.get("gamma"), t.attrs.get("beta")
    if g is None: return None
    gmax = float(_np.abs(_np.asarray(g)).max())
    bmax = float(_np.abs(_np.asarray(b)).max()) if b is not None else 0.0
    return gmax * _NORM_SIGMA + bmax
  srcs = [_propagate_max_abs(s, seen) for s in t.srcs]
  if any(s is None for s in srcs) and op not in ("matmul", "linear"):
    return None
  if op in _NONEXPANDING:
    return srcs[0]
  if op in ("add", "sub"):
    return srcs[0] + srcs[1] if len(srcs) == 2 else None
  if op == "adds":
    return srcs[0] + abs(float(t.attrs.get("k", 0.0)))
  if op in ("mul", "minimum"):                       # |x*y| <= |x||y|; min is bounded by either
    return srcs[0] * srcs[1] if len(srcs) == 2 else None
  if op == "muls":
    return srcs[0] * abs(float(t.attrs.get("k", 1.0)))
  if op == "maximum":
    return max(srcs) if len(srcs) == 2 else None
  if op in ("matmul", "linear"):                     # |x @ W| <= max|x| * sum_k max_n |W[k,n]|
    if srcs[0] is None: return None
    w = t.attrs.get("wt")
    if w is None: return None
    aw = _np.abs(_np.asarray(w, _np.float32))
    # wt is stored [N,K] for matmul (consumed transposed) and [out,in] for linear: both reduce over
    # the last axis to give, per output, the sum of |weight| along the contraction.
    return srcs[0] * float(aw.sum(axis=-1).max())
  if op == "bmm":                                    # activation x activation: K is the contraction
    if any(s is None for s in srcs): return None
    K = t.srcs[0].shape[-1]
    return srcs[0] * srcs[1] * float(K)
  return None                                        # not modelled -> unbounded, fail closed


# Normalized activations are O(1) but not provably bounded, since fp16 division by a small variance
# can overshoot. 8 sigma is a deliberately loose cap: a real normalized value past it would be an
# outlier far outside anything a trained network produces.
_NORM_SIGMA = 8.0


def _act_max_abs(out: Tensor, ops=("matmul", "linear", "bmm", "conv")):
  """Bound on the largest activation feeding any `ops` node, or None if any of them is unbounded.

  This is the quantity a weight-quantizing variant needs: the activation it has to encode, not the
  graph's output. A node whose activation input cannot be bounded makes the whole answer None, so the
  caller stays fail-closed."""
  seen, stack, visited, worst = {}, [out], set(), 0.0
  found = False
  while stack:
    t = stack.pop()
    if id(t) in visited: continue
    visited.add(id(t))
    if t.op in ops and t.srcs:
      found = True
      b = _propagate_max_abs(t.srcs[0], seen)        # srcs[0] is the activation; weights are attrs
      if b is None: return None
      worst = max(worst, b)
    stack.extend(t.srcs)
  return worst if found else 0.0


def _fp16_risk_kind(t: Tensor) -> str:
  """Predictor op-kind: 'slice', 'reduce_square' (reduce->square/mul), 'reduce', or ''."""
  op = t.op
  if op == "slice_by_size": return "slice"
  if op in ("square", "mul"):
    # reduce -> square/mul: flag only when a source is a reduction.
    if any(s.op.startswith("reduce") or s.op in ("l2_norm", "rms_norm") for s in t.srcs): return "reduce_square"
    return ""
  if op.startswith("reduce") or op in ("softmax", "l2_norm", "rms_norm", "layer_norm", "group_norm", "instance_norm"):
    return "reduce"
  return ""


def _warn_fp16_cross_chip(out: Tensor, host_family: int, target_family: int) -> None:
  """Warn which ops carry cross-chip fp16 divergence risk for a different-family target. Never rejects."""
  import warnings

  from . import _targets as TG
  if int(host_family) == int(target_family): return
  risks: dict[tuple, tuple] = {}
  for t in _topo(out):
    kind = _fp16_risk_kind(t)
    if not kind: continue
    begin = t.attrs.get("begin") if t.op == "slice_by_size" else None
    verdict = TG.predict_fp16_divergence(
      kind, tuple(t.shape), host_family, target_family,
      begin=begin, max_abs=_node_max_abs(t))
    if verdict != "none": risks.setdefault((t.op, verdict), tuple(t.shape))
  if not risks: return
  _CLASS = {
    "saturation": "saturation (finite->inf; |value|>4094 on A13's x16 crop-DMA slice)",
    "round1": "~1 fp16 round (0x494 reduce->square fusion differs A13<->A14+)",
    "ulp1": "<=1 ULP (0x3f0 reduction route threshold differs)",
  }
  items = "; ".join(f"{op}{shape}: {_CLASS[v]}" for (op, v), shape in sorted(risks.items()))
  warnings.warn(
    f"aneforge.cross_compile_check: graph compiles for family {target_family} but "
    f"{len(risks)} op(s) may diverge in fp16 vs the host (family {host_family}): "
    f"{items}. Compile is valid; numeric correctness on that chip still needs the "
    f"silicon. (Direction B HAL-field predictor.)",
    CrossChipFP16Warning, stacklevel=3)


def _resolve_family(target) -> int:
  """target: None (auto-detect host) | int family | arch string -> compiler family."""
  from . import _targets as TG
  if target is None: return TG.detect_family()
  if isinstance(target, str): return TG.family_of_arch(target)
  return int(target)


def _retarget_for(out: Tensor, target) -> Tensor:
  """Gate the graph for a target family before lowering (raises on floor/unreachable/oversize; decomposes below-floor ops)."""
  from . import _targets as TG
  from . import special
  fam = _resolve_family(target)
  if not TG.supports_mil(fam):
    raise NotImplementedError(
      f"aneforge requires an H13+ (A13+) ANE: 'MIL is only supported for H13+ ANE "
      f"architectures'. Target family {fam} is below the floor.")
  _warn_h13_slice_saturation(out, fam)
  rep = TG.preflight(out, fam)
  if rep.reject:
    ops = sorted({r.op for r in rep.reject})
    raise NotImplementedError(
      f"aneforge: ops not runnable on ANE family {fam}: {ops}. No in-graph "
      f"decomposition is available; restructure the graph or target a higher chip.")
  if rep.oversize:
    worst = max(rep.oversize, key=lambda r: max(r.shape))
    raise NotImplementedError(
      f"aneforge: tensor {worst.shape} (op {worst.op!r}) exceeds ANE family {fam}'s "
      f"max dimension {TG.limit('max_tensor_dim', fam)}; tile the graph or target a "
      f"higher chip.")
  if not rep.decompose: return out
  decomp = {"sin": special.sin, "cos": special.cos}
  memo: dict[int, Tensor] = {}
  for t in _topo(out):
    if not t.srcs:
      memo[id(t)] = t                                   # input / leaf
      continue
    new_srcs = [memo[id(s)] for s in t.srcs]
    if TG.op_status(t.op, fam) == "decompose":
      if t.op not in decomp:
        raise NotImplementedError(
          f"aneforge: op {t.op!r} needs host-side handling on family {fam} "
          f"(no in-graph decomposition).")
      memo[id(t)] = decomp[t.op](new_srcs[0])
    else:
      memo[id(t)] = Tensor(t.shape, t.op, new_srcs, t.attrs)
  return memo[id(out)]


def compile(out: Tensor, int8: bool = False, build_dir=None, opt: "str | int | None" = "routes",
      compress: str | None = None, compress_atol: float = 0.05,
      block_size: int = 32, validate: bool = False, target=None,
      _check_precision: bool = True):
  """Lower `out` into ONE fused ANE program (or a segmented plan if it has `af.sdpa` nodes)."""
  if out.op == "input":
    # A graph whose output IS an input lowers to an empty MIL body, which crashes Espresso's
    # ANE compiler ("unordered_map::at: key not found"). Wrap in an exact scalar mul(1.0) so
    # the program always has at least one op (and downstream port naming stays consistent).
    out = out * 1.0
  if _check_precision:                 # once per user compile (internal re-entries pass False)
    _precision_signal(out, strict=validate)
    _dispatch_floor_signal(out)
    out = _retarget_for(out, target)  # gate ops/shapes for the target ANE family
  _OPT0 = (0, None, False)
  family = None
  if compress is not None:
    if opt not in _OPT0 and opt != "routes":
      raise NotImplementedError("compress= is not yet supported with opt>=1; " "use opt=0 for compressed weights")
    opt = 0          # compressed weights take the byte-identical lowering
    if compress == "auto": family = _resolve_family(target)
  if opt not in _OPT0:                  # lossless canon for routes/1/2/max; never opt=0 or compress
    from ._rewrite import canonicalize
    out = canonicalize(out)
  if opt == "routes":
    from . import _optimize
    return _optimize._compile_routes(out, int8=int8, build_dir=build_dir)
  if opt not in _OPT0: return _compile_opt(out, int8=int8, opt=opt)
  order = _topo(out)
  if any(t.op in NETPLIST_OPS for t in order):
    return _compile_segmented(out, int8, build_dir, compress, compress_atol, block_size, family=family)
  bad = sorted({t.op for t in order if t.op != "input" and t.op not in _EMIT})
  if bad: raise NotImplementedError(f"aneforge: ops not reachable on the ANE: {bad}")
  inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
  if not inputs: raise ValueError("aneforge.compile: graph has no inputs")
  for i, t in enumerate(order): t._name = f"t{i}"

  em = _Emitter(int8, compress=compress, compress_atol=compress_atol, block_size=block_size, family=family)
  for t in order:
    if t.op != "input": _EMIT[t.op](em, t, t._name, [src._name for src in t.srcs])

  prog = _assemble_and_compile(em, inputs, out._name, out.shape, build_dir)
  n_ops = sum(1 for t in order if t.op != "input")
  return Model(prog, [(t._name, t.shape) for t in inputs], out._name, out.shape, n_ops, input_tensors=list(inputs))


def _compile_opt(out: Tensor, int8: bool, opt):
  """opt=1/2/'max' dispatch (lazy-imported so aneforge imports without the optimizer)."""
  from . import _optimize
  if opt in (1,):
    # cost-model pick: cheaper-predicted variant, no measurement. Since nothing here validates the
    # result, drop lossy variants whose activation-encoding ceiling this graph can exceed (#153); the
    # bound comes from declared input ranges and propagation (#155), so a provably-safe graph keeps
    # the fast path.
    cfgs = _optimize._variants(out, drop_unsafe=True)
    best = min(cfgs, key=lambda c: _optimize._estimate_variant(out, c))
    return _optimize.build_variant(out, best)
  if opt in (2, "max", "2"): return _optimize.tune(out)
  raise ValueError(f"compile: unknown opt={opt!r} (use 0, 1, 2, or 'max')")


# segmented compile: native-SDPA graph cuts
def _import_bridge(module: str, attr: str, op_name: str, filename: str):
  """Lazily import `attr` from the in-package `aneforge._bridges.<module>` bridge."""
  import importlib
  try:
    mod = importlib.import_module(f"{__package__}._bridges.{module}")
    return getattr(mod, attr)
  except (ImportError, AttributeError) as e:
    raise RuntimeError(f"af.{op_name} needs aneforge/_bridges/{filename} (the native "
             "bridge) to import cleanly") from e


def _sdpa_fused(*args, **kwargs):
  """Lazy bridge to the verified native-SDPA entry point in aneforge._bridges."""
  return _import_bridge("ane_sdpa_fused", "sdpa_fused", "sdpa", "ane_sdpa_fused.py")(*args, **kwargs)


# netplist-bridge ops: graph nodes that run as a native-ANE Path-A sub-program
# (a graph cut). Each NETPLIST_OPS entry is a legal cut point (_compile_segmented)#

def _run_sdpa(srcs, attrs):
  q, k, v = srcs[0], srcs[1], srcs[2]
  mask = None
  if attrs.get("causal"):
    # additive causal mask [S,S]: 0 on/below diagonal, -inf above.
    S = q.shape[2]
    mask = np.triu(np.full((S, S), -1e4, np.float32), k=1)
  elif attrs.get("masked"):
    # runtime attn_mask (4th src) -> [Sq, Skv].
    mask = np.asarray(srcs[3], np.float32).reshape(q.shape[2], k.shape[2])
  return _sdpa_fused(q, k, v, scale=attrs["scale"], mask=mask).astype(np.float16)


def _run_argmax(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "global_argminmax", "argmax", "ane_rank_fused.py")
  (x,) = srcs
  # GlobalArgMinMax over Width (last axis) or Channel (axis 0); x is [C, W] fp16.
  dim = "Width" if attrs["axis"] == 1 else "Channel"
  out = np.asarray(fn(x, dimension=dim, largest=True), dtype=np.float16)
  C, W = x.shape
  return out.reshape((C, 1) if attrs["axis"] == 1 else (1, W))


def _run_topk(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "topk", "topk", "ane_rank_fused.py")
  (x,) = srcs
  k, largest = attrs["k"], attrs["largest"]
  # native TopK ranks Width keyed by one channel lane; dispatch each row as a 1-channel tile.
  rows = [np.asarray(fn(x[c:c + 1], k, largest=largest), dtype=np.float16).reshape(k) for c in range(x.shape[0])]
  return np.stack(rows, axis=0)


def _run_sort(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "sort", "sort", "ane_rank_fused.py")
  (x,) = srcs
  W = x.shape[1]
  desc, idx = attrs["descending"], attrs["return_indices"]
  # like TopK: native Sort orders Width keyed by one channel lane; dispatch each row as a 1-channel tile.
  rows = [np.asarray(fn(x[c:c + 1], descending=desc, return_indices=idx),
           dtype=np.float16).reshape(W)
      for c in range(x.shape[0])]
  return np.stack(rows, axis=0)


def _run_cross_product(srcs, attrs):
  fn = _import_bridge("ane_cross_product_fused", "cross_product_fused", "cross_product", "ane_cross_product_fused.py")
  a, b = srcs
  return np.asarray(fn(a.reshape(3), b.reshape(3)), dtype=np.float16).reshape(3)


def _run_cross_correlation(srcs, attrs):
  fn = _import_bridge("ane_cross_correlation_fused", "cross_correlation_fused",
          "cross_correlation", "ane_cross_correlation_fused.py")
  x, template = srcs
  return np.asarray(fn(x, template), dtype=np.float16)


def _run_cost_volume(srcs, attrs):
  fn = _import_bridge("ane_cost_volume_fused", "cost_volume_fused", "cost_volume", "ane_cost_volume_fused.py")
  aux, ref = srcs
  return np.asarray(fn(aux.reshape(-1), ref.reshape(-1), attrs["disparity_range"]), dtype=np.float16)


def _run_fps(srcs, attrs):
  fn = _import_bridge("ane_fps_fused", "fps_fused", "fps", "ane_fps_fused.py")
  (points,) = srcs
  # DistanceMetric is L2-only on this arch (see the bridge); pass "L2".
  return np.asarray(fn(points, attrs["k"], metric="L2"), dtype=np.float16)


def _run_radius_search(srcs, attrs):
  fn = _import_bridge("ane_radius_search_fused", "radius_search_fused", "radius_search", "ane_radius_search_fused.py")
  points, centroids = srcs
  # bridge returns a (Np, Nc) uint8 membership matrix; cast to fp16 (0/1 exact).
  return np.asarray(fn(points, centroids, attrs["radius"]), dtype=np.float16)


def _run_minmax_norm(srcs, attrs):
  fn = _import_bridge("minmax_norm_fused", "minmax_norm_fused", "minmax_norm", "minmax_norm_fused.py")
  (x,) = srcs  # [1, C, H, W]
  out = np.asarray(fn(x, dimension=attrs["dimension"], epsilon=attrs["eps"]),
          dtype=np.float16)              # bridge returns [C, H, W]
  return out.reshape(x.shape)


def _run_lrn(srcs, attrs):
  fn = _import_bridge("lrn_fused", "lrn_fused", "lrn", "lrn_fused.py")
  (x,) = srcs  # [1, C, H, W]
  out = np.asarray(fn(x, alpha=attrs["alpha"], beta=attrs["beta"], k=attrs["k"]),
          dtype=np.float16)              # bridge returns [C, H, W]
  return out.reshape(x.shape)


def _run_rearrange(layer_fn, *index_keys):
  """Runner for an ane_rearrange_fused.py op (index_keys -> positional args after x)."""
  def run(srcs, attrs):
    fn = _import_bridge("ane_rearrange_fused", layer_fn, layer_fn, "ane_rearrange_fused.py")
    (x,) = srcs
    args = [attrs[k] for k in index_keys]
    return np.asarray(fn(x, *args), dtype=np.float16)
  return run


def _run_flatten(srcs, attrs):
  fn = _import_bridge("ane_structural_fused", "flatten", "flatten", "ane_structural_fused.py")
  (x,) = srcs  # bridge wants (C, H, W); graph guarantees a 3D input
  return np.asarray(fn(x), dtype=np.float16).reshape(-1)


def _run_input_view(srcs, attrs):
  fn = _import_bridge("ane_input_view_fused", "input_view_fused", "input_view", "ane_input_view_fused.py")
  (x,) = srcs
  return np.asarray(fn(x.reshape(-1), attrs["offset"], attrs["size"]), dtype=np.float16)


def _run_dynamic_slice(srcs, attrs):
  fn = _import_bridge("ane_dynamic_slice_fused", "dynamic_slice_fused", "dynamic_slice", "ane_dynamic_slice_fused.py")
  (x,) = srcs
  return np.asarray(fn(x.reshape(-1), attrs["start"], slice_size=attrs["size"]), dtype=np.float16)


def _run_scaled_elementwise(srcs, attrs):
  fn = _import_bridge("scaled_elementwise_fused", "scaled_elementwise_fused",
          "scaled_elementwise", "scaled_elementwise_fused.py")
  x, z = srcs
  return np.asarray(fn(x.reshape(-1), z.reshape(-1), op=attrs["op"], scale=attrs["scale"]), dtype=np.float16)


# op name -> (runner, n_srcs); these are the legal graph-cut boundaries.
NETPLIST_OPS = {
  "sdpa": (_run_sdpa, 3),
  "argmax": (_run_argmax, 1),
  "topk": (_run_topk, 1),
  "sort": (_run_sort, 1),
  "cross_product": (_run_cross_product, 2),
  "cross_correlation": (_run_cross_correlation, 2),
  "cost_volume": (_run_cost_volume, 2),
  "fps": (_run_fps, 1),
  "radius_search": (_run_radius_search, 2),
  "minmax_norm": (_run_minmax_norm, 1),
  "lrn": (_run_lrn, 1),
  "space_to_channel": (_run_rearrange("space_to_channel", "r"), 1),
  "channel_to_space": (_run_rearrange("channel_to_space", "r"), 1),
  "space_to_batch": (_run_rearrange("space_to_batch", "bh", "bw"), 1),
  "batch_to_space": (_run_rearrange("batch_to_space", "bh", "bw"), 1),
  "flatten": (_run_flatten, 1),
  "input_view": (_run_input_view, 1),
  "dynamic_slice": (_run_dynamic_slice, 1),
  "scaled_elementwise": (_run_scaled_elementwise, 2),
}


def _subgraph(target, source_ids):
  """Emit-order nodes to compute `target`, stopping at source tensors; returns (emit_order, input_tensors)."""
  seen, order, inputs = set(), [], []
  stack = [(target, False)]                              # iterative post-order (deep-graph safe)
  while stack:
    t, processed = stack.pop()
    if processed:
      order.append(t)
      continue
    if id(t) in seen: continue
    seen.add(id(t))
    if id(t) in source_ids:
      inputs.append(t)
      continue
    stack.append((t, True))
    for src in t.srcs:
      if id(src) not in seen: stack.append((src, False))
  return order, inputs


class SegmentedModel:
  """A compiled plan: e5rt program segments interleaved with native-ANE sub-program calls (NETPLIST_OPS)."""

  def __init__(self, stages, inputs, out_id, out_shape, n_ops, n_netplist):
    self._stages = stages
    self._inputs = inputs  # list of (id, name, shape) in creation order
    self._out_id, self._out_shape = out_id, out_shape
    self.n_ops = n_ops
    self.n_netplist = self.n_sdpa = n_netplist  # n_sdpa kept for backward compat
    # A2: persistent Path-A workers, built lazily and cached. ANEFORGE_NETPLIST_WORKER=0 forces A1.
    self._workers: dict[int, "_Worker"] = {}
    self._worker_runs: dict[int, Callable[..., Any]] = {}
    self._worker_warned: set[str] = set()

  def __call__(self, *arrays: np.ndarray) -> np.ndarray:
    if len(arrays) != len(self._inputs): raise ValueError(f"expected {len(self._inputs)} input(s), got {len(arrays)}")
    env = {}
    for (iid, name, shape), a in zip(self._inputs, arrays):
      a = np.asarray(a)
      if tuple(a.shape) != tuple(shape): raise ValueError(f"input '{name}' shape {a.shape} != compiled {shape}")
      env[iid] = a.astype(np.float16)
    for st in self._stages:
      if st["kind"] == "region":
        feed = {name: env[sid] for sid, name in st["srcs"]}
        env[st["tid"]] = st["prog"].eval(feed)[st["out_var"]].astype(np.float16)
      else:  # native netplist-bridge sub-program (sdpa / argmax / topk / ...)
        runner = self._netplist_runner(st)
        src_arrays = [env[i] for i in st["src_ids"]]
        env[st["tid"]] = np.asarray(runner(src_arrays, st["attrs"]), dtype=np.float16)
    return env[self._out_id].astype(np.float32)

  def _netplist_runner(self, st) -> Callable[..., Any]:
    """Resolve a netplist stage's runner: prefer a persistent Path-A worker, else the subprocess bridge."""
    import os
    if os.environ.get("ANEFORGE_NETPLIST_WORKER", "1") == "0": return NETPLIST_OPS[st["op"]][0]
    # masked/causal SDPA needs per-call mask injection; the worker is mask-less, so use the bridge.
    if st["op"] == "sdpa" and (st["attrs"].get("causal") or st["attrs"].get("masked")): return NETPLIST_OPS[st["op"]][0]
    sid = st["tid"]
    run = self._worker_runs.get(sid)
    if run is not None: return run
    try:
      from . import _netplist_worker as nw
      if not nw.has_worker(st["op"]):
        # no worker route - subprocess bridge is the normal path.
        run = NETPLIST_OPS[st["op"]][0]
        self._worker_runs[sid] = run
        return run
      worker, run = nw.build_worker(st["op"], st["src_shapes"][0], st["attrs"])
      self._workers[sid] = worker
      self._worker_runs[sid] = run
      return run
    except Exception as e:
      # worker failure -> fall back to the subprocess bridge; remember it (signal once per op).
      if st["op"] not in self._worker_warned:
        self._worker_warned.add(st["op"])
        import warnings
        warnings.warn(
          f"aneforge: persistent worker for {st['op']!r} unavailable ({e!r}); "
          f"falling back to the slower per-call subprocess bridge. Set "
          f"ANEFORGE_NETPLIST_WORKER=0 to force (and silence) this path.",
          stacklevel=2)
      run = NETPLIST_OPS[st["op"]][0]
      self._worker_runs[sid] = run
      return run

  def release(self) -> None:
    for st in self._stages:
      if st["kind"] == "region": st["prog"].release()
    for w in self._workers.values():
      try:
        w.release()
      except Exception:
        pass
    self._workers.clear()
    self._worker_runs.clear()


def _compile_segmented(out: Tensor, int8: bool, build_dir,
           compress: str | None = None, compress_atol: float = 0.05,
           block_size: int = 32, family: int | None = None):
  order = _topo(out)
  bad = sorted({t.op for t in order if t.op not in NETPLIST_OPS and t.op != "input" and t.op not in _EMIT})
  if bad: raise NotImplementedError(f"aneforge: ops not reachable on the ANE: {bad}")
  for i, t in enumerate(order): t._name = f"t{i}"
  graph_inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
  cuts = [t for t in order if t.op in NETPLIST_OPS]      # graph-cut nodes (run as native sub-programs)
  source_ids = {id(t) for t in graph_inputs} | {id(t) for t in cuts}

  stages, built = [], set()

  def build_region(target):
    # value already materialized (graph input / prior cut output) -> no program
    if id(target) in source_ids or id(target) in built: return
    built.add(id(target))
    emit_order, inputs_r = _subgraph(target, source_ids)
    em = _Emitter(int8, compress=compress, compress_atol=compress_atol, block_size=block_size, family=family)
    for t in emit_order: _EMIT[t.op](em, t, t._name, [s._name for s in t.srcs])
    out_var = target._name  # captured post-emit (bias-adding emitters rename in place)
    prog = _assemble_and_compile(em, inputs_r, out_var, target.shape, None)
    stages.append({"kind": "region", "prog": prog, "tid": id(target), "out_var": out_var,
           "srcs": [(id(t), t._name) for t in inputs_r]})

  for cut in cuts:                                       # cuts appear in topo order
    for src in cut.srcs: build_region(src)
    stages.append({"kind": "netplist", "op": cut.op, "tid": id(cut),
           "src_ids": [id(x) for x in cut.srcs], "attrs": cut.attrs,
           "src_shapes": [tuple(x.shape) for x in cut.srcs]})
  build_region(out)

  n_ops = sum(1 for t in order if t.op not in NETPLIST_OPS and t.op != "input")
  return SegmentedModel(stages, [(id(t), t._name, t.shape) for t in graph_inputs], id(out), out.shape, n_ops, len(cuts))
