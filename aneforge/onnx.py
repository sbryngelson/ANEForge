"""Import an ONNX model and run it on the ANE. CNN-classifier op subset; see
docs. Public: load_onnx / onnx_to_tensor."""
from __future__ import annotations
from typing import Callable
from math import prod
import numpy as np
from .graph import Tensor, input as _input, conv as _conv, batch_norm as _bn, concat as _concat
from .graph import instance_norm as _instnorm, space_to_depth as _s2d
from .graph import local_response_norm as _lrnorm, depth_to_space as _d2s
from .graph import resize_bilinear as _rbilin, resize_nearest_neighbor as _rnn
from .graph import gather as _gather, topk as _topk
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

def _load(path):
  """Load an ONNX model (or accept one in-memory) and run shape inference."""
  import onnx
  m = path if hasattr(path, "graph") else onnx.load(path)
  return onnx.shape_inference.infer_shapes(m)

def onnx_to_tensor(path):
  """Build an aneforge graph from an ONNX model; returns (graph_inputs, output)."""
  m = _load(path); g = m.graph
  inits = _inits(g)
  vals: dict[str, object] = dict(inits)        # name -> Tensor | np.ndarray (initializers as arrays)
  graph_inputs = []
  for vi in g.input:
    if vi.name in inits: continue              # initializers also listed as inputs in some exporters
    t = _input(_shape(vi)); vals[vi.name] = t; graph_inputs.append(t)
  for node in g.node:                          # ONNX node list is topologically ordered
    if node.op_type not in _ONNX:
      raise NotImplementedError(f"ONNX op '{node.op_type}' not supported")
    ins = [vals.get(n) for n in node.input]
    outs = _ONNX[node.op_type](node, ins, _attrs(node), inits)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    for name, val in zip(node.output, outs): vals[name] = val
  name = g.output[0].name
  if name not in vals: raise ValueError(f"onnx: graph output '{name}' was never produced")
  out = vals[name]
  if not isinstance(out, Tensor): raise TypeError("onnx: graph output is not a Tensor")
  return graph_inputs, out

def load_onnx(path, **compile_kwargs):
  """Import an ONNX model and compile it to a runnable ANE Model."""
  _, out = onnx_to_tensor(path)
  return _compile.compile(out, **compile_kwargs)

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
@onnx_op("Elu")
def _elu(node, ins, a, i): return ins[0].elu(float(a.get("alpha", 1.0)))
@onnx_op("LeakyRelu")
def _lrelu(node, ins, a, i): return ins[0].leaky_relu(float(a.get("alpha", 0.01)))
@onnx_op("Gelu")
def _gelu(node, ins, a, i):
  """ONNX Gelu (opset 20); exact erf-gelu only ('approximate=tanh' unsupported)."""
  ap = a.get("approximate")
  if ap is not None and (ap.decode() if isinstance(ap, bytes) else ap) == "tanh":
    raise NotImplementedError("ONNX Gelu: approximate='tanh' not supported (only exact/erf)")
  return ins[0].gelu()
@onnx_op("PRelu")
def _prelu(node, ins, a, i): return ins[0].prelu(np.asarray(ins[1]).reshape(-1))   # slope=ins[1], [C,1,1]->[C]
@onnx_op("Pow")
def _pow(node, ins, a, i): x, y = _two(ins, "Pow"); return x.pow(y)   # tensor-tensor only
@onnx_op("InstanceNormalization")
def _instance_norm(node, ins, a, i):
  return _instnorm(ins[0], np.asarray(ins[1]), np.asarray(ins[2]), eps=float(a.get("epsilon", 1e-5)))
@onnx_op("SpaceToDepth")
def _space_to_depth(node, ins, a, i): return _s2d(ins[0], int(a["blocksize"]))
def _two(ins, op):                                 # aneforge elementwise is tensor-tensor only
  if not isinstance(ins[0], Tensor) or not isinstance(ins[1], Tensor):
    raise NotImplementedError(f"ONNX {op}: a constant operand is not supported (tensor-tensor elementwise only)")
  return ins[0], ins[1]
@onnx_op("Add")
def _add(node, ins, a, i): x, y = _two(ins, "Add"); return x + y
@onnx_op("Sub")
def _sub(node, ins, a, i): x, y = _two(ins, "Sub"); return x - y
@onnx_op("Mul")
def _mul(node, ins, a, i): x, y = _two(ins, "Mul"); return x * y
@onnx_op("Div")
def _div(node, ins, a, i): x, y = _two(ins, "Div"); return x / y
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

@onnx_op("Conv")
def _conv_h(node, ins, a, i):
  ap = a.get("auto_pad")
  if ap is not None and (ap.decode() if isinstance(ap, bytes) else ap) != "NOTSET":
    raise NotImplementedError(f"ONNX Conv: auto_pad={ap!r} not supported (use explicit pads)")
  x = ins[0]; w = np.asarray(ins[1]); b = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  return _conv(x, w, stride=_uniform(a.get("strides"), "Conv"), pad=_sympad(a.get("pads"), "Conv"),
               dilation=_uniform(a.get("dilations"), "Conv"), groups=int(a.get("group", 1)), bias=b)

@onnx_op("MaxPool")
def _maxpool(node, ins, a, i):
  if int(a.get("ceil_mode", 0)): raise NotImplementedError("ONNX MaxPool: ceil_mode=1 not supported")
  if _uniform(a.get("dilations"), "MaxPool") != 1: raise NotImplementedError("ONNX MaxPool: dilations != 1 not supported")
  return ins[0].max_pool(_uniform(a.get("kernel_shape"), "MaxPool"),
                         _uniform(a.get("strides"), "MaxPool"), _sympad(a.get("pads"), "MaxPool"))
@onnx_op("AveragePool")
def _avgpool(node, ins, a, i):
  if int(a.get("ceil_mode", 0)): raise NotImplementedError("ONNX AveragePool: ceil_mode=1 not supported")
  if int(a.get("count_include_pad", 0)): raise NotImplementedError("ONNX AveragePool: count_include_pad=1 not supported")
  return ins[0].avg_pool(_uniform(a.get("kernel_shape"), "AveragePool"),
                         _uniform(a.get("strides"), "AveragePool"), _sympad(a.get("pads"), "AveragePool"))
@onnx_op("GlobalAveragePool")
def _gap(node, ins, a, i): return ins[0].mean((2, 3))     # keepdims -> [N,C,1,1]

@onnx_op("Gemm")
def _gemm(node, ins, a, i):
  if float(a.get("alpha", 1.0)) != 1.0 or float(a.get("beta", 1.0)) != 1.0 or int(a.get("transA", 0)):
    raise NotImplementedError("ONNX Gemm: only alpha=1, beta=1, transA=0 supported")
  x = ins[0]; W = np.asarray(ins[1]); B = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  if not int(a.get("transB", 0)): W = W.T            # x.linear expects [out,in]
  return x.linear(W, B)
@onnx_op("MatMul")
def _matmul(node, ins, a, i):
  if not isinstance(ins[0], Tensor): raise NotImplementedError("ONNX MatMul: a constant first operand is not supported")
  b = ins[1]
  return ins[0] @ (np.asarray(b) if not isinstance(b, Tensor) else b)
@onnx_op("BatchNormalization")
def _bnorm(node, ins, a, i):
  x, s, bb, mean, var = ins[0], np.asarray(ins[1]), np.asarray(ins[2]), np.asarray(ins[3]), np.asarray(ins[4])
  return _bn(x, s, bb, mean, var, eps=float(a.get("epsilon", 1e-5)))
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
  if axes is None: raise NotImplementedError("ONNX Squeeze: squeeze-all (no axes) not supported")
  return ins[0].squeeze(tuple(axes))
@onnx_op("Unsqueeze")
def _unsqueeze(node, ins, a, i):
  axes = a.get("axes") or [int(v) for v in np.asarray(ins[1])]
  return ins[0].expand_dims(tuple(axes))
@onnx_op("Concat")
def _concat_h(node, ins, a, i): return _concat(list(ins), axis=int(a["axis"]))
@onnx_op("Softmax")
def _softmax(node, ins, a, i): return ins[0].softmax(int(a.get("axis", -1)))
@onnx_op("Constant")
def _const(node, ins, a, i):
  from onnx import numpy_helper
  return numpy_helper.to_array(a["value"])         # returns np.ndarray (folded by consumers)
@onnx_op("Identity")
def _identity(node, ins, a, i): return ins[0]      # pass-through (Tensor or array)
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
    raise NotImplementedError(f"ONNX Resize linear: coordinate_transformation_mode={ctm!r} not supported (ANE matches 'asymmetric'/'align_corners' only)")
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
@onnx_op("Gather")
def _gather_h(node, ins, a, i):
  """Static-index gather along `axis`; constant scalar/1-D integer indices only (data-dependent indices have no ANE path)."""
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
@onnx_op("TopK")
def _topk_h(node, ins, a, i):
  """TopK -> ANE topk (2D [C,W], last axis only); returns VALUES only - ONNX's indices output is unsupported."""
  x = ins[0]
  if len(x.shape) != 2: raise NotImplementedError("ONNX TopK: only 2D inputs supported on the ANE")
  if int(a.get("axis", -1)) not in (-1, 1): raise NotImplementedError("ONNX TopK: only last-axis (width) topk supported")
  if isinstance(ins[1], Tensor): raise NotImplementedError("ONNX TopK: data-dependent k (non-constant) not supported")
  k = int(np.asarray(ins[1]).ravel()[0])
  return _topk(x, k, largest=bool(int(a.get("largest", 1))))
