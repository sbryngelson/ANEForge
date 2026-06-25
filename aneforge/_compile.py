"""Lower a graph into ONE fused e5rt program.

`compile(out)` topologically orders the graph, emits a single MIL function (one op
per graph node, weights packed into one BLOBFILE), and hands it to e5rt on the ANE.
Each op's MIL is produced by a small handler registered with `@op(...)` - the
registry doubles as the set of ops aneforge can reach on the device.
"""
from __future__ import annotations

import tempfile
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


def _topo(out: Tensor) -> list[Tensor]:
  # iterative post-order DFS (explicit stack) so deep unrolled graphs don't blow
  # Python's recursion limit. Sources are appended before their consumers (valid topo).
  seen: set[int] = set()
  order: list[Tensor] = []
  stack: list = [(out, False)]
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


# --------------------------------------------------------------------------- #
# per-op MIL emitters (registry)                                              #
# --------------------------------------------------------------------------- #

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
    # `compress` is the unified knob: None (fp16), "int8", "int4", "sparse",
    # "blockwise", or "auto". `int8=True` is the back-compat alias for compress="int8".
    self.compress = compress if compress is not None else ("int8" if int8 else None)
    self.int8 = self.compress == "int8"
    self.compress_atol = compress_atol
    self.block_size = block_size       # inner-dim block for compress="blockwise"
    # compress="auto" is family-aware: only encodings that stream natively on the
    # target family are auto candidates - a folding encoding costs accuracy for zero
    # bandwidth win, so fp16 dominates it (h13/M1 streams int4-LUT + sparse; A14+ stream
    # all four). family=None keeps every branch enabled (historical behavior);
    # explicit single-mode knobs are never filtered - the user's call.
    if self.compress == "auto" and family is not None:
      from . import _targets as TG
      self._auto_streams = TG.native_streams(family)
    else:
      self._auto_streams = frozenset({"int4", "int8", "sparse", "blockwise"})
    self.blob = BlobWriter()
    self.lines: list[str] = []

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
    W2 = W2.astype(np.float32)   # norm in fp32: W2.dot(W2) overflows fp16 for large weights
    return float(np.linalg.norm(recon - W2) / (np.linalg.norm(W2) + 1e-12))

  @staticmethod
  def _blockwise_rel_error(W2: np.ndarray, data: bytes, scale: bytes, nblocks: int) -> float:
    OUT, IN = W2.shape
    q = np.frombuffer(data, dtype=np.int8).astype(np.float32).reshape(OUT, nblocks, IN // nblocks)
    sc = np.frombuffer(scale, dtype=np.float16).astype(np.float32).reshape(OUT, nblocks)
    recon = (q * sc[:, :, None]).reshape(OUT, IN)
    W2 = W2.astype(np.float32)   # norm in fp32: W2.dot(W2) overflows fp16 for large weights
    return float(np.linalg.norm(recon - W2) / (np.linalg.norm(W2) + 1e-12))

  def weight(self, name: str, W: np.ndarray, allow_int8: bool,
       int8: bool | None = None, allow_int4: bool = False,
       allow_sparse: bool = False) -> str:
    """Declare a constant weight `name`. Encoding precedence: sparse (when
        compress='sparse', allowed, >=50% zeros) -> int4-LUT (when compress='int4',
        allowed, inner dim even, within the accuracy budget) -> per-channel int8
        (compress='int8'/int8 override, allowed) -> fp16. The int4 fallback chain is
        int4 -> int8 (per-channel, where allowed) -> fp16: when int4-LUT is rejected
        (gate miss or odd inner dim), the weight falls to int8 if allow_int8 (denser
        than fp16, and the user opted into compression), else fp16. int4 is gated
        by per-tensor relative-L2 reconstruction error vs self.compress_atol, falling
        back rather than emitting a too-lossy weight. compress=None/opt=0 stays
        byte-identical (the fp16 branch). Under compress='auto' each compressed branch
        is additionally gated on the encoding streaming natively on the target family
        (self._auto_streams): on h13/M1 that is int4-LUT + sparse, so auto streams those
        (sparse for >=50%-zero weights) but skips int8/blockwise (they fold), and a rejected
        int4 falls to fp16, not int8."""
    path = '@model_path/weights.bin'
    W2 = W.reshape(W.shape[0], -1)
    # Branch order is the "auto" precedence: sparse -> int4 -> int8 -> fp16.
    # Single-mode knobs ("sparse"/"int4"/"int8") enable exactly their one compressed
    # branch; "auto" enables the branches that stream natively on the target family
    # and takes the most-aggressive that stays correct (sparse lossless, int4
    # accuracy-gated, int8 the dense fallback).
    _auto = self.compress == "auto"
    if (self.compress == "sparse" or (_auto and "sparse" in self._auto_streams)) and allow_sparse:
      mask, vals = sparsify(W2)
      nnz = len(vals) // 2
      if 0 < nnz <= (W2.size // 2):          # only when genuinely sparse (>=50% zeros, not all-zero)
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
        # Indices always have the same shape as the weight (lut rank = W.ndim + 2).
        # For 2-D weights this is [1,1,16,1] exactly as before (byte-identical).
        # For ND weights (e.g. conv) the lut is [1]*N+[16,1]; Espresso accepts any
        # rank as long as lut.rank == indices.rank + 2.  A reshape of a constexpr_
        # output is not routable on the ANE backend, so we must use ND indices.
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
        # data has the weight's shape; scale is [OUT, nblocks]. On-device
        # constexpr_blockwise_shift_scale reconstructs data.reshape(OUT,nblocks,bs)
        # * scale[:,:,None] (zero offset). Verified multi-block on M5 (relerr ~3e-4).
        # wall: the constexpr output is not routable as a matmul/conv weight operand
        # (Espresso "Not implemented"); it only feeds elementwise consumers, so we
        # bridge it through a +0 add to a dense fp16 result that can back a
        # matmul/conv. Net: dequantized in-program, not streamed as int8 (disk ~2x
        # smaller, no bandwidth win - unlike int4-LUT/sparse).
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
  "sin", "cos", "erf", "softplus", "relu6", "softsign", "atan", "exp2",
  "floor", "ceil", "round", "sign")
def _e_unary(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(x = {s[0]})[name = string("{n}")];')


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
  # emit mul(x,x) instead of the `square` opcode: the native square opcode fused
  # with a following nonlinearity fails ANECCompile when (Width % 128) > 64
  # (mapped in the reverse-engineering corpus); mul(x,x) is identical math and
  # compiles everywhere.
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


@op("add", "sub", "mul", "real_div", "maximum", "minimum", "pow")
def _e_binary(em, t, n, s):
  em.line(f'{em.ty(t.shape)} {n} = {t.op}(x = {s[0]}, y = {s[1]})[name = string("{n}")];')


@op("muls")
def _e_muls(em, t, n, s):
  k = float(np.float16(t.attrs["k"])).hex()
  em.line(f'fp16 {n}_k = const()[name = string("{n}_k"), val = fp16({k})];')
  em.line(f'{em.ty(t.shape)} {n} = mul(x = {s[0]}, y = {n}_k)[name = string("{n}")];')


@op("adds")
def _e_adds(em, t, n, s):
  # scalar add (x + k), the additive sibling of muls. A const fp16 broadcast-added
  # to x - the only fused way to inject a scalar offset (e.g. a normalization eps),
  # since _binary requires two graph Tensors. Bit-faithful: the const is fp16-rounded
  # exactly as muls rounds its scalar.
  k = float(np.float16(t.attrs["k"])).hex()
  em.line(f'fp16 {n}_k = const()[name = string("{n}_k"), val = fp16({k})];')
  em.line(f'{em.ty(t.shape)} {n} = add(x = {s[0]}, y = {n}_k)[name = string("{n}")];')


@op("cast")
def _e_cast(em, t, n, s):
  # dtype conversion (dequantises a uint8 image input to fp16): the source port's
  # declared dtype is uint8; cast lifts it to the fp16 compute type.
  dt = t.attrs.get("dtype", "fp16")
  em.line(f'string {n}_d = const()[name = string("{n}_d"), val = string("{dt}")];')
  em.line(f'{em.ty_dt(t.shape, dt)} {n} = cast(dtype = {n}_d, x = {s[0]})[name = string("{n}")];')


@op("const_array")
def _e_const_array(em, t, n, s):
  # a small baked fp16 const tensor (e.g. per-channel image scale/bias), streamed
  # from the weights blob like any other const.
  W = np.asarray(t.attrs["value"], dtype=np.float16)
  path = '@model_path/weights.bin'
  o = em.blob.add(fp16_bytes(W), FP16)
  em.line(f'{em.ty(W.shape)} {n} = const()[name = string("{n}"), '
      f'val = {em.ty(W.shape)}(BLOBFILE(path = string("{path}"), offset = uint64({o})))];')


@op("matmul")
def _e_matmul(em, t, n, s):
  # per-node int8 override (set by the rewriter) takes precedence over the global
  # self.int8; None defers to the global flag (keeps opt=0 byte-identical).
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


@op("conv")
def _e_conv(em, t, n, s):
  a = t.attrs
  # per-channel int8 is routable as a conv weight on the ANE: constexpr_affine_dequantize
  # feeds the conv weight operand directly (no +0 bridge - unlike constexpr_blockwise_
  # shift_scale). Measured on h13/M1: int8 conv compiles+runs at cos~1.0 vs fp16,
  # relerr ~0.6% vs fp32 ref (the expected per-channel int8 error), and ~halves the
  # DRAM weight bytes (dequant-in-program at the MAC port). int4-LUT/sparse stay as
  # higher-precedence branches (int4 streams natively 2.37x on M1; int8 folds/dequants
  # but still halves bytes). The per-node `int8` attr (set by the auto-rewriter) overrides
  # the global flag, like matmul; None defers (keeps opt=0 byte-identical).
  w = em.weight(f"{n}_w", a["weight"], allow_int8=True, int8=a.get("int8"), allow_int4=True, allow_sparse=True)
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')
  em.line(f'tensor<int32, [2]> {n}_st = const()[name = string("{n}_st"), val = tensor<int32, [2]>([{st}, {st}])];')
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];')
  em.line(f'tensor<int32, [2]> {n}_dl = const()[name = string("{n}_dl"), val = tensor<int32, [2]>([{dl}, {dl}])];')
  em.line(f'int32 {n}_g = const()[name = string("{n}_g"), val = int32({g})];')
  em.line(f'{em.ty(t.shape)} {n} = conv(dilations = {n}_dl, groups = {n}_g, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, weight = {w}, x = {s[0]})[name = string("{n}")];')
  if "bias" in a:
    b = em.weight(f"{n}_bw", a["bias"].reshape(1, t.shape[1], 1, 1), allow_int8=False)
    em.line(f'{em.ty(t.shape)} {n}_b = add(x = {n}, y = {b})[name = string("{n}_b")];')
    t._name = f"{n}_b"


@op("dynamic_conv")
def _e_dynamic_conv(em, t, n, s):
  # native conv with a dynamic weight: the kernel is the second src (s[1]), a graph value,
  # not a baked BLOBFILE const. Lowers to the ANE dynamic-kernel path. Batch is guaranteed 1
  # by the graph builder (batch>=2 dynamic-weight conv is unsupported).
  a = t.attrs
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')
  em.line(f'tensor<int32, [2]> {n}_st = const()[name = string("{n}_st"), val = tensor<int32, [2]>([{st}, {st}])];')
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];')
  em.line(f'tensor<int32, [2]> {n}_dl = const()[name = string("{n}_dl"), val = tensor<int32, [2]>([{dl}, {dl}])];')
  em.line(f'int32 {n}_g = const()[name = string("{n}_g"), val = int32({g})];')
  em.line(f'{em.ty(t.shape)} {n} = conv(dilations = {n}_dl, groups = {n}_g, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, weight = {s[1]}, x = {s[0]})[name = string("{n}")];')


@op("max_pool")
def _e_max_pool(em, t, n, s):
  k, st, p = t.attrs["k"], t.attrs["stride"], t.attrs["pad"]
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')
  em.line(f'tensor<int32, [2]> {n}_ks = const()[name = string("{n}_ks"), val = tensor<int32, [2]>([{k}, {k}])];')
  em.line(f'tensor<int32, [2]> {n}_st = const()[name = string("{n}_st"), val = tensor<int32, [2]>([{st}, {st}])];')
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];')
  em.line(f'bool {n}_cm = const()[name = string("{n}_cm"), val = bool(false)];')
  em.line(f'{em.ty(t.shape)} {n} = max_pool(ceil_mode = {n}_cm, kernel_sizes = {n}_ks, pad = {n}_pd, '
      f'pad_type = {n}_pt, strides = {n}_st, x = {s[0]})[name = string("{n}")];')


@op("avg_pool")
def _e_avg_pool(em, t, n, s):
  k, st, p = t.attrs["k"], t.attrs["stride"], t.attrs["pad"]
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')
  em.line(f'tensor<int32, [2]> {n}_ks = const()[name = string("{n}_ks"), val = tensor<int32, [2]>([{k}, {k}])];')
  em.line(f'tensor<int32, [2]> {n}_st = const()[name = string("{n}_st"), val = tensor<int32, [2]>([{st}, {st}])];')
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];')
  em.line(f'bool {n}_ep = const()[name = string("{n}_ep"), val = bool(false)];')
  em.line(f'bool {n}_cm = const()[name = string("{n}_cm"), val = bool(false)];')
  em.line(f'{em.ty(t.shape)} {n} = avg_pool(ceil_mode = {n}_cm, exclude_padding_from_average = {n}_ep, '
      f'kernel_sizes = {n}_ks, pad = {n}_pd, pad_type = {n}_pt, strides = {n}_st, x = {s[0]})[name = string("{n}")];')


@op("conv_transpose")
def _e_conv_transpose(em, t, n, s):
  a = t.attrs
  w = em.weight(f"{n}_w", a["weight"], allow_int8=False)         # [Cin, Cout, kH, kW]; int4-LUT unverified for conv_transpose -> fp16/int8
  p, st, dl, g = a["pad"], a["stride"], a["dilation"], a["groups"]
  em.line(f'string {n}_pt = const()[name = string("{n}_pt"), val = string("custom")];')
  em.line(f'tensor<int32, [2]> {n}_st = const()[name = string("{n}_st"), val = tensor<int32, [2]>([{st}, {st}])];')
  em.line(f'tensor<int32, [4]> {n}_pd = const()[name = string("{n}_pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];')
  em.line(f'tensor<int32, [2]> {n}_dl = const()[name = string("{n}_dl"), val = tensor<int32, [2]>([{dl}, {dl}])];')
  em.line(f'int32 {n}_g = const()[name = string("{n}_g"), val = int32({g})];')
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
  # x / sqrt(sum(x^2, axis) + eps), built from reduce_l2_norm (verified on e5rt)
  # over the requested axis. reduce_l2_norm computes sqrt(sum(x^2)); eps enters via
  # a safe-divide guard (max with eps) to avoid div-by-zero.
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
  # split emits N equal outputs once (keyed on which==0); later splits reuse the
  # already-emitted multi-output statement by name (n0_<i>). Guard against re-emit
  # via a per-source marker stored on the emitter.
  ns, ax, which = t.attrs["num_splits"], t.attrs["axis"], t.attrs["which"]
  src = s[0]
  key = f"_split_{src}_{ns}_{ax}"
  names = getattr(em, key, None)
  if names is None:
    names = [f"{n}_part{i}" for i in range(ns)]
    decl = ", ".join(f"{em.ty(t.shape)} {names[i]}" for i in range(ns))
    em.line(f'int32 {n}_ns = const()[name = string("{n}_ns"), val = int32({ns})];')
    em.line(f'int32 {n}_ax = const()[name = string("{n}_ax"), val = int32({ax})];')
    em.line(f'{decl} = split(axis = {n}_ax, num_splits = {n}_ns, x = {src})[name = string("{n}_split")];')
    setattr(em, key, names)
  # alias this split node's output name to the right port
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
  # RMS-norm via the fused `reduce_l2_norm` over the last axis - the same fast
  # reduction `l2_norm` uses. The previous lowering reshaped to [M,D,1,1] and ran
  # `reduce_sum` over the *channel* axis, which falls off the ANE's fast reduction
  # tile past ~256 rows and ran ~48x slower than this path at [1024,1024]
  # (e.g. 6882us -> 142us), and 16-27x slower than layer_norm despite RMS being
  # structurally cheaper. Mathematically identical:
  #     rms(x) = x / sqrt(mean(x^2)+eps) * g
  #            = x * sqrt(D) / sqrt(sum(x^2)) * g        (reduce_l2_norm = sqrt(sum x^2))
  # eps enters as a safe-divide floor on the norm (as in l2_norm); the sqrt(D)
  # rescale is folded into the gamma weight so no extra op is emitted.
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
  M, D = t.shape[0], t.shape[-1]
  g4 = em.weight(f"{n}_g", t.attrs["gamma"].reshape(1, D, 1, 1), allow_int8=False)
  b4 = em.weight(f"{n}_b", t.attrs["beta"].reshape(1, D, 1, 1), allow_int8=False)
  eps = float(np.float16(t.attrs["eps"])).hex()
  em.line(f'tensor<int32,[4]> {n}_rs = const()[name=string("{n}_rs"), val=tensor<int32,[4]>([{M},{D},1,1])];')
  em.line(f'tensor<int32,[{len(t.shape)}]> {n}_ro = const()[name=string("{n}_ro"), val=tensor<int32,[{len(t.shape)}]>({list(t.shape)})];')
  em.line(f'tensor<int32,[1]> {n}_ax = const()[name=string("{n}_ax"), val=tensor<int32,[1]>([1])];')
  em.line(f'bool {n}_kd = const()[name=string("{n}_kd"), val=bool(true)];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_x4 = reshape(shape={n}_rs, x={s[0]})[name=string("{n}_x4")];')
  em.line(f'tensor<fp16,[{M},1,1,1]> {n}_mu = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={n}_x4)[name=string("{n}_mu")];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_xc = sub(x={n}_x4, y={n}_mu)[name=string("{n}_xc")];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_sq = mul(x={n}_xc, y={n}_xc)[name=string("{n}_sq")];')
  em.line(f'tensor<fp16,[{M},1,1,1]> {n}_var = reduce_mean(axes={n}_ax, keep_dims={n}_kd, x={n}_sq)[name=string("{n}_var")];')
  em.line(f'fp16 {n}_ep = const()[name=string("{n}_ep"), val=fp16({eps})];')
  em.line(f'tensor<fp16,[{M},1,1,1]> {n}_ve = add(x={n}_var, y={n}_ep)[name=string("{n}_ve")];')
  em.line(f'tensor<fp16,[{M},1,1,1]> {n}_rr = rsqrt(epsilon=fp16(0.0), x={n}_ve)[name=string("{n}_rr")];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_xn = mul(x={n}_xc, y={n}_rr)[name=string("{n}_xn")];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_gg = mul(x={n}_xn, y={g4})[name=string("{n}_gg")];')
  em.line(f'tensor<fp16,[{M},{D},1,1]> {n}_bb = add(x={n}_gg, y={b4})[name=string("{n}_bb")];')
  em.line(f'{em.ty(t.shape)} {n} = reshape(shape={n}_ro, x={n}_bb)[name=string("{n}")];')


@op("channel_layer_norm")
def _e_channel_layer_norm(em, t, n, s):
  # LayerNorm over the channel axis (axis 1) of a channels-first [N,C,1,S] tensor.
  # Same reduction as layer_norm but with no reshape in/out: the ANE keeps the data in
  # its native [N,C,1,S] layout, so the surrounding 1x1-conv projections stay efficient.
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
  # Rank-4 tiling: reshape to [1,G,C/G,H*W] and reduce over the trailing two axes
  # (per-group, over C/G and H*W) rather than the flattened [1,G,(C/G)*H*W].
  # Keeps every internal axis under the per-axis cap so big SD-1.5 maps (640@64,
  # 512@128) compile, where the flat (C/G)*H*W extent overflowed.
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
  # mean = sum over both group axes * 1/(D*HW), via chained reduce_sum (D->1, then HW->1).
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


# --------------------------------------------------------------------------- #
# compiled model                                                              #
# --------------------------------------------------------------------------- #

class Model:
  """A compiled, fused ANE program. Call it with the input array(s), in the order
    they were created with `af.input`."""

  def __init__(self, prog, inputs: list[tuple[str, tuple]], out_name: str, out_shape, n_ops: int,
        input_tensors: list | None = None):
    self._prog = prog
    self._inputs = inputs
    # The ordered input Tensor objects (same order as `inputs`); lets callers
    # (e.g. autograd.Trainer) map each compiled input back to its source Tensor -
    # trainable parameter vs provided data. Additive; no effect on inference.
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
        feed[name] = a.astype(np.uint8)              # Program._feed sends the raw bytes
      else:
        feed[name] = a.astype(np.float16)
    return self._prog.eval(feed)[self._out_name].astype(np.float32)

  # ---- zero-copy hot-loop API (skip the per-call host<->device memcpy) ----
  # Pattern:  v = model.input_view(); ... write fp16 into v ...; model.execute();
  #           out = model.output_view()   # fp16 view onto the result, no copy
  # For tight inference/training loops; for one-off calls just use __call__.
  def input_view(self, name: str | None = None) -> np.ndarray:
    """Writable fp16 view onto an input buffer (defaults to the sole input port).
        See `Program.input_view`."""
    return self._prog.input_view(name or self._inputs[0][0])

  def output_view(self) -> np.ndarray:
    """fp16 view onto the output buffer, valid after `execute()`. See
        `Program.output_view`."""
    return self._prog.output_view(self._out_name)

  def execute(self) -> None:
    """Run once without binding inputs / reading outputs - pair with
        `input_view`/`output_view` for a zero-copy loop."""
    self._prog.execute()

  def release(self) -> None:
    self._prog.release()


def _in_dtype(t: Tensor) -> str:
  """Wire dtype of an input port ("fp16" default; "uint8" for image inputs)."""
  return t.attrs.get("dtype", "fp16")


def _input_dtypes(inputs) -> dict:
  """{port_name: dtype} for the non-fp16 input ports (passed to the runtime so the
    port buffer is sized and fed in the right dtype)."""
  return {t._name: _in_dtype(t) for t in inputs if _in_dtype(t) != "fp16"}


def _sig_ty(em, t: Tensor) -> str:
  """MIL type for an input in the function signature (honors the port dtype)."""
  dt = _in_dtype(t)
  return em.ty(t.shape) if dt == "fp16" else em.ty_dt(t.shape, dt)


def _emit_program_dir(em, inputs, out_var, out_shape, build_dir):
  """Write one e5rt program (MIL + weights) for a set of input tensors to a dir; return
    it. Shared by the device-compile path and the cross-target compile check."""
  sig = ", ".join(f"{_sig_ty(em, t)} {t._name}" for t in inputs)
  mil = (f"program(1.3)\n{_BUILD_INFO}\n{{\n    func main<ios18>({sig}) {{\n"
     + "\n".join(em.lines) + f"\n    }} -> ({out_var});\n}}\n")
  d = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="aneforge_"))
  d.mkdir(parents=True, exist_ok=True)
  (d / "model.mil").write_text(mil)
  (d / "weights.bin").write_bytes(em.blob.build())
  return d


def _assemble_and_compile(em, inputs, out_var, out_shape, build_dir):
  """Write one e5rt program (MIL + weights) for a set of input tensors and compile it."""
  d = _emit_program_dir(em, inputs, out_var, out_shape, build_dir)
  return E5RT.compile(mil_path=d / "model.mil", cache_dir=str(d / "cache"),
            inputs={t._name: t.shape for t in inputs}, outputs={out_var: out_shape},
            device_mask=ANE_MASK, input_dtypes=_input_dtypes(inputs))


def _lower_fused_to_dir(out: Tensor, build_dir=None, int8: bool = False):
  """Lower a single-program fused graph to a MIL+weights dir WITHOUT compiling it.
    Bridge/segmented graphs are out of scope for the cross-target check (raises)."""
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
  """Does the graph rooted at `out` compile for another ANE family, checked from this
    host? `target` is a compiler arch string ('h13') or a family int. Returns True iff
    the e5rt compiler produces a library for that TargetArchitecture - compile-level
    validation only (it does not, and on a different-family host cannot, execute there).

    The keystone of cross-chip CI: validate that the op corpus compiles for every chip
    family on one box. Numeric correctness still needs the real silicon."""
  from . import _runtime
  from . import _targets as TG
  if isinstance(target, str):
    arch = target.strip().lower()
    # The e5rt compiler silently falls back to the host target on an unrecognized
    # TargetArchitecture string (measured: 'zzz' compiles fine), turning a typo into
    # a false cross-target pass. Gate on the known-target table first.
    if arch not in TG._ARCH_FAMILY:
      raise ValueError(f"unknown ANE target arch {target!r}; known: " f"{sorted(TG._ARCH_FAMILY)}")
  else:
    arch = TG.arch_for_family(int(target))
  # Static pre-gate: a target's per-family caps (conv kernel width, max tensor dim, ops
  # below their MinimumFamily floor) are HAL-data gated, and the host cross-compiler does
  # not reliably enforce a different family's caps when emitting for its TargetArchitecture
  # - so a cap violation would compile here and only fail on the real silicon, a false CI
  # pass. preflight() answers the same question statically and family-aware (it is what
  # compile(target=) gates on), so reject up front and skip the compiler entirely.
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
  # Annotate cross-chip fp16 divergence risk (Direction B) before compiling: warn-only,
  # so a typo'd/unreachable graph still surfaces its compile error.
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
  """A compiled fused program with N named outputs (e.g. a resident training
    step: `(x, y, lr, w, m, v, ...) -> (w', m', v', ...)`). Unlike `Model` it
    keeps the ordered input/output Tensors and exposes the raw `Program` so
    callers (autograd.Trainer's resident path) can alias state outputs onto their
    inputs via `share_buffer` and drive the resident loop. `__call__` evals
    once and returns the outputs by name (for verification / non-resident use)."""

  def __init__(self, prog, inputs: list, outputs: list):
    self.prog = prog
    self.input_tensors = list(inputs)
    self.output_tensors = list(outputs)
    # Port names are snapshotted here (like Model): compile()/compile_multi()
    # reassign t._name on shared Tensor objects, so a later compile of another
    # graph sharing these tensors must not break this model's feed/read names.
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


def compile_multi(outs, build_dir=None) -> MultiModel:
  """Lower a graph with multiple output Tensors into one fused ANE program with
    N named output ports (the multi-output analogue of `compile`). Used for the
    resident training step, where each state tensor needs its own output port to
    alias back onto its input. fp16 only; no opt/compress/segmented path."""
  seen: set[int] = set()
  order: list[Tensor] = []
  stack: list = [(o, False) for o in reversed(outs)]    # iterative post-order (deep-graph safe)
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
  if any(t.op in NETPLIST_OPS for t in order):
    raise NotImplementedError("compile_multi: native-SDPA/netplist ops not supported")
  bad = sorted({t.op for t in order if t.op != "input" and t.op not in _EMIT})
  if bad: raise NotImplementedError(f"aneforge: ops not reachable on the ANE: {bad}")
  inputs = sorted((t for t in order if t.op == "input"), key=lambda t: t.attrs.get("idx", 0))
  if not inputs: raise ValueError("aneforge.compile_multi: graph has no inputs")
  for i, t in enumerate(order): t._name = f"t{i}"

  em = _Emitter(False)
  for t in order:
    if t.op != "input": _EMIT[t.op](em, t, t._name, [src._name for src in t.srcs])

  sig = ", ".join(f"{_sig_ty(em, t)} {t._name}" for t in inputs)
  out_vars = ", ".join(o._name for o in outs)
  mil = (f"program(1.3)\n{_BUILD_INFO}\n{{\n    func main<ios18>({sig}) {{\n"
     + "\n".join(em.lines) + f"\n    }} -> ({out_vars});\n}}\n")
  d = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="aneforge_"))
  d.mkdir(parents=True, exist_ok=True)
  (d / "model.mil").write_text(mil)
  (d / "weights.bin").write_bytes(em.blob.build())
  prog = E5RT.compile(mil_path=d / "model.mil", cache_dir=str(d / "cache"),
            inputs={t._name: t.shape for t in inputs},
            outputs={o._name: o.shape for o in outs}, device_mask=ANE_MASK,
            input_dtypes=_input_dtypes(inputs))
  return MultiModel(prog, inputs, list(outs))


class PrecisionWarning(UserWarning):
  """Emitted by `compile` when a graph has fp16-cancellation-risk nodes whose
    results may be inaccurate. Silence with
    `warnings.filterwarnings('ignore', category=aneforge.PrecisionWarning)`."""


class DispatchFloorWarning(UserWarning):
  """Emitted by `compile` when a program is dispatch-floor-bound: its predicted ANE
    time is dominated by the fixed per-call dispatch + firmware round-trip, so each call
    costs about the same however small the work is. Dispatch is single-in-flight, so
    threads/concurrency do not amortize it - only larger batches or more ops per program do.
    Silence with `warnings.filterwarnings('ignore', category=aneforge.DispatchFloorWarning)`."""


def _dispatch_floor_signal(out: Tensor) -> None:
  """Warn once if the program is dispatch-floor-bound, so a caller looping per-sample
    isn't silently paying the full fixed per-call cost every time. The ANE per-call latency
    is a fixed dispatch + firmware round-trip (~0.2 ms on M1-class parts) that batching and
    op-chaining amortize but concurrency does not. Cheap: a structural estimate, no device
    work; fires only when the predicted cost is dominated by the dispatch floor."""
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
  if pred > floor * 1.5:          # already compute/bytes-bound - amortizing buys little
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
  """Run the structural fp16-precision-risk check and tell the user. Warns (or, with
    `strict`/`validate=True`, raises) when the graph has cancellation / narrow-reduce
    hotspots, so an unstable computation can never run silently. Cheap: a static pass
    over the graph, no device work. See `precision_risk` for the (heuristic) model."""
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
  """Pre-A16 ANE quirk on nonzero-width (last-axis) `slice_by_size` offsets, which
    lower through a Q.4 fixed-point crop-DMA (an implied x16 scale). Two distinct,
    silicon-pinned failure modes:

      (1) wrong elements when multiple width-offset slices are concatenated - confirmed
          on A14 (the gather axis-1 bug) and A13, at any magnitude. A single width-offset
          slice selects the correct elements (a 180-config bare-slice sweep is numpy-exact
          on A14), so this mode is specific to the concat-of-width-slices pattern.
      (2) saturation: a sliced element with |value| > 4094 (= 65504/16) clamps to +/-inf,
          magnitude-driven, on a single width-offset slice. Silicon-confirmed on A13 (conv
          weight-grad) and A14 (M2 probe: 4094 finite -> 4100 inf, bit-exact); A15 pending
          M3 data.

    A zero last-axis begin, or an offset on any non-last axis, avoids the path and is
    exact; A16/M5 is unaffected. `gather` and the conv im2col backward already route off
    the last axis, so this is a safety net for any remaining hand-written occurrence.
    """
  import warnings

  from . import _targets as TG
  if int(family) >= int(TG.Family.A16):       # A16+ (M5/M4) is unaffected
    return
  sat_family = int(family) <= int(TG.Family.A14)   # saturation measured on A13 + A14
  for t in _topo(out):
    # Mode (1): a concat of >= 2 nonzero-width-offset slices (the gather / im2col
    # pattern) - wrong elements on any pre-A16 family.
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
  """Emitted by `cross_compile_check` when a graph compiles for a different-family
    target but carries an op whose fp16 VALUE can diverge from the host's on that chip
    (per the Direction B HAL-field predictor). The compile is still valid - this is a
    numeric heads-up, not a rejection. Silence with
    `warnings.filterwarnings('ignore', category=aneforge.CrossChipFP16Warning)`."""


def _node_max_abs(t: Tensor):
  """Best-effort static bound on a node's value magnitude (for the slice saturation
    gate): a baked const's actual max, else unknown (None = treat as possibly-large)."""
  v = t.attrs.get("value")
  if v is not None:
    try:
      import numpy as _np
      return float(_np.abs(_np.asarray(v)).max())
    except Exception:
      return None
  return None


def _fp16_risk_kind(t: Tensor) -> str:
  """Map a graph node to the predictor's op-kind string: 'slice', 'reduce_square'
    (a reduce feeding a square/mul = variance/L2/RMS), 'reduce' (any reduction/softmax/
    norm), or '' (no fp16-route axis)."""
  op = t.op
  if op == "slice_by_size": return "slice"
  if op in ("square", "mul"):
    # reduce -> square/mul: the 0x494 fuse axis. Flag only when a source is a reduction
    # (variance/L2/RMSNorm shape), not every elementwise square.
    if any(s.op.startswith("reduce") or s.op in ("l2_norm", "rms_norm") for s in t.srcs): return "reduce_square"
    return ""
  if op.startswith("reduce") or op in ("softmax", "l2_norm", "rms_norm", "layer_norm", "group_norm", "instance_norm"):
    return "reduce"
  return ""


def _warn_fp16_cross_chip(out: Tensor, host_family: int, target_family: int) -> None:
  """Annotate, as warnings, which ops in the graph carry a cross-chip fp16 divergence
    risk when validating for a different-family target (Direction B predictor). Groups by
    risk class so the message is one line per (op, class). Never rejects."""
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
  """Gate the graph for a target ANE family before lowering. Raises a clear
    compile-time error on the H13+ hard floor, on an unreachable op, or on a tensor
    exceeding the family's limits; substitutes below-floor ops that have an in-graph
    decomposition (sin/cos -> aneforge.special). Returns the (possibly rewritten) graph.
    On the host family with an all-native graph this is a no-op fast path (returns out)."""
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
      memo[id(t)] = t                                   # input / leaf - reuse
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
  """Lower the graph rooted at `out` into ONE fused ANE program (or, if it
    contains `af.sdpa` nodes, a segmented plan: e5rt programs split around each
    native-SDPA sub-program).

    `opt` selects the graph optimizer (default `'routes'`):
      - `opt='routes'` (default) : the lossless route pass - cost-model-driven, per
                      shape, choosing for each route-bearing bridge node (sdpa,
                      minmax_norm, flatten, lrn) between the native bridge (a graph
                      cut) and the proven-equivalent fused decomposition (cut removed),
                      without on-device measurement. The route registry is lossless
                      (cos 1.0), so this never changes numerics; a cut-free graph
                      compiles to exactly the `opt=0` program (no regression). For
                      `sdpa` the cost model keeps native where it wins (long
                      sequences), so the default never blindly removes a cut that
                      helps.
      - `opt=0`   : no optimization - the historical, byte-identical path
                      (`int8` honored as given). Use this for byte-identity tests.
      - `opt=1`   : cost-model pick over the full variant set (route swaps and the
                      lossy whole-graph int8 variant), without measuring on-device.
      - `opt=2` / `opt='max'` : autotune - measure the legal proven-safe variants
                      on the ANE, validate each vs the opt=0 baseline, return the
                      fastest correct one (cached; instant on a cache hit).

    `compress` selects weight encoding: None (fp16, default, byte-identical at
    `opt=0`), 'int8' (per-channel), 'int4' (LUT, accuracy-gated), 'sparse' (bitmask,
    when the weight is >=50% zeros), 'blockwise' (per-inner-block int8 via
    `constexpr_blockwise_shift_scale`, `block_size` columns per scale,
    accuracy-gated -> int8 -> fp16), or 'auto' (per-weight: sparse if sparse, else int4
    if accurate, else int8, else fp16 - the most aggressive encoding that stays
    correct). 'auto' is family-aware: only encodings that stream natively on the
    `target` family (host-detected when `target=None`) are candidates - on h13/M1
    that is int4-LUT + sparse, so auto streams those (sparse for >=50%-zero weights) but
    skips int8/blockwise (they fold to dense fp16: accuracy cost, no bandwidth win) and a
    rejected int4 falls to fp16. Explicit single-mode knobs are never filtered.
    `compress_atol` is the int4/blockwise fallback budget (relative L2);
    `block_size` is the inner-dim block width for 'blockwise'. Compressed weights
    stay on the byte-identical (opt=0) lowering: passing `compress` with an explicit
    `opt>=1` is rejected, and the default route pass is skipped for compressed
    compiles (they take the no-op path).
    """
  if _check_precision:                 # once per user compile (internal re-entries pass False)
    _precision_signal(out, strict=validate)
    _dispatch_floor_signal(out)
    out = _retarget_for(out, target)  # gate ops/shapes for the target ANE family
  _OPT0 = (0, None, False)
  family = None
  if compress is not None:
    if opt not in _OPT0 and opt != "routes":
      raise NotImplementedError("compress= is not yet supported with opt>=1; " "use opt=0 for compressed weights")
    opt = 0          # compressed weights always take the byte-identical lowering
    if compress == "auto": family = _resolve_family(target)   # auto is family-aware: stream-only candidates
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
  """opt=1/2/'max' dispatch (lazy-imported so aneforge stays importable without the
    optimizer/cost-model present)."""
  from . import _optimize
  if opt in (1,):
    # cost-model pick: no measurement, just the cheaper-predicted variant (this
    # now spans route swaps too, since _estimate_variant costs the rewritten DAG).
    cfgs = _optimize._variants(out)
    best = min(cfgs, key=lambda c: _optimize._estimate_variant(out, c))
    return _optimize.build_variant(out, best)
  if opt in (2, "max", "2"): return _optimize.tune(out)
  raise ValueError(f"compile: unknown opt={opt!r} (use 0, 1, 2, or 'max')")


# --------------------------------------------------------------------------- #
# segmented compile: native-SDPA graph cuts                                   #
# --------------------------------------------------------------------------- #
def _import_bridge(module: str, attr: str, op_name: str, filename: str):
  """Lazily import `attr` from the in-package `aneforge._bridges.<module>` bridge
    (keeps aneforge self-contained: the netplist-bridge closure lives inside the
    package, with no runtime dependency on the reverse-engineering corpus)."""
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


# --------------------------------------------------------------------------- #
# netplist-bridge ops: graph nodes that run as a native-ANE Path-A sub-program #
# (a graph cut), like af.sdpa. Each entry maps an op name to a runner that,     #
# given the materialized fp16 inputs + the node attrs, returns the fp16 output. #
# Adding an op here makes it a legal cut point in the segmented plan (see        #
# _compile_segmented). The fused e5rt-MIL emitters above are always preferred;   #
# these are for ops the ANE backend doesn't expose through MIL.                  #
# --------------------------------------------------------------------------- #

def _run_sdpa(srcs, attrs):
  q, k, v = srcs[0], srcs[1], srcs[2]
  mask = None
  if attrs.get("causal"):
    # additive causal mask [S_q, S_k]: 0 on/below the diagonal, -inf above (fed as the
    # native SDPA layer's optional 5th "mask" bottom). q is [B, H, S, D].
    S = q.shape[2]
    mask = np.triu(np.full((S, S), -1e4, np.float32), k=1)
  elif attrs.get("masked"):
    # runtime attn_mask (4th src): a [1,1,Sq|1,Skv] additive bias -> [Sq, Skv] for the bridge.
    mask = np.asarray(srcs[3], np.float32).reshape(q.shape[2], k.shape[2])
  return _sdpa_fused(q, k, v, scale=attrs["scale"], mask=mask).astype(np.float16)


def _run_argmax(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "global_argminmax", "argmax", "ane_rank_fused.py")
  (x,) = srcs
  # GlobalArgMinMax over Width (last axis) or Channel (axis 0); x is [C, W] fp16.
  dim = "Width" if attrs["axis"] == 1 else "Channel"
  out = np.asarray(fn(x, dimension=dim, largest=True), dtype=np.float16)
  # bridge returns a vector (one index per surviving position); reshape to keepdims
  C, W = x.shape
  return out.reshape((C, 1) if attrs["axis"] == 1 else (1, W))


def _run_topk(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "topk", "topk", "ane_rank_fused.py")
  (x,) = srcs
  k, largest = attrs["k"], attrs["largest"]
  # The native TopK sorts Width keyed by one channel lane and permutes all
  # channels by that single order. For PyTorch's per-row top-k (each row
  # ranked independently) we dispatch each row as its own 1-channel tile.
  rows = [np.asarray(fn(x[c:c + 1], k, largest=largest), dtype=np.float16).reshape(k) for c in range(x.shape[0])]
  return np.stack(rows, axis=0)


def _run_sort(srcs, attrs):
  fn = _import_bridge("ane_rank_fused", "sort", "sort", "ane_rank_fused.py")
  (x,) = srcs
  W = x.shape[1]
  desc, idx = attrs["descending"], attrs["return_indices"]
  # Like TopK, the native Sort orders Width keyed by one channel lane and
  # permutes all channels by that single order. For a per-row independent sort
  # (numpy-like) we dispatch each row as its own 1-channel tile.
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
  """Build a runner for an ane_rearrange_fused.py op. `layer_fn` is the bridge
    function name; `index_keys` are the attr names whose values are passed
    positionally after the input array (e.g. ('r',) or ('bh','bw'))."""
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
  """Nodes to emit to compute `target`, stopping at source tensors (which become
    program inputs). Returns (emit_order, input_tensors)."""
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
  """A compiled plan: e5rt program segments interleaved with native-ANE
    sub-program calls (sdpa / argmax / topk - see NETPLIST_OPS). Tensors thread
    between segments as host fp16 arrays (correctness-first; a persistent IOSurface
    worker is the throughput follow-up)."""

  def __init__(self, stages, inputs, out_id, out_shape, n_ops, n_netplist):
    self._stages = stages
    self._inputs = inputs  # list of (id, name, shape) in creation order
    self._out_id, self._out_shape = out_id, out_shape
    self.n_ops = n_ops
    # count of native sub-programs; kept as `n_sdpa` for backward compat
    self.n_netplist = self.n_sdpa = n_netplist
    # A2: persistent Path-A workers, one per netplist stage with a worker
    # route. Built lazily on first call (keyed by the stage), reused for the
    # rest of this model's lifetime, released in release(). Set
    # ANEFORGE_NETPLIST_WORKER=0 to force the A1 (subprocess-per-call) path.
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
    """Resolve the runner for a netplist stage. Prefer a persistent Path-A worker
        (load-once-eval-many), built lazily on first use and cached for this model's
        lifetime. Fall back to the subprocess-per-call bridge when no worker route
        exists for the op or when the worker is disabled / fails to start."""
    import os
    if os.environ.get("ANEFORGE_NETPLIST_WORKER", "1") == "0": return NETPLIST_OPS[st["op"]][0]
    # Causal SDPA needs the per-call additive-mask injection, which lives on the
    # subprocess bridge path (_run_sdpa); the persistent Path-A worker pre-builds a
    # mask-less netplist, so route masked SDPA to the bridge (correctness over the worker).
    if st["op"] == "sdpa" and (st["attrs"].get("causal") or st["attrs"].get("masked")): return NETPLIST_OPS[st["op"]][0]
    sid = st["tid"]
    run = self._worker_runs.get(sid)
    if run is not None: return run
    try:
      from . import _netplist_worker as nw
      if not nw.has_worker(st["op"]):
        # No worker route exists for this op - the subprocess bridge IS the
        # normal path, not a degradation; stay silent.
        run = NETPLIST_OPS[st["op"]][0]
        self._worker_runs[sid] = run
        return run
      worker, run = nw.build_worker(st["op"], st["src_shapes"][0], st["attrs"])
      self._workers[sid] = worker
      self._worker_runs[sid] = run
      return run
    except Exception as e:
      # Genuine worker failure -> fall back to the verified subprocess-bridge path
      # (correctness-preserving, slower per call) and remember it so we don't
      # retry the worker every call. Signal once per op.
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
    # value already materialized (a graph input or a prior cut output) -> no program
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
