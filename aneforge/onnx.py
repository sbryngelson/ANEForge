"""Import an ONNX model and run it on the ANE. CNN-classifier op subset; see
docs. Public: load_onnx / onnx_to_tensor."""
from __future__ import annotations
from typing import Callable
from math import prod
import numpy as np
from .graph import Tensor, input as _input, conv as _conv, batch_norm as _bn, concat as _concat
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
  out = vals[g.output[0].name]
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
@onnx_op("Add")
def _add(node, ins, a, i): return ins[0] + ins[1]
@onnx_op("Sub")
def _sub(node, ins, a, i): return ins[0] - ins[1]
@onnx_op("Mul")
def _mul(node, ins, a, i): return ins[0] * ins[1]
@onnx_op("Div")
def _div(node, ins, a, i): return ins[0] / ins[1]
@onnx_op("Clip")
def _clip(node, ins, a, i):
  """Clip; opset<11 reads min/max attrs, opset>=11 reads inputs 2/3. (0,6)->relu6, (0,inf)->relu."""
  lo = a.get("min", -3.4e38); hi = a.get("max", 3.4e38)
  if len(ins) >= 2 and ins[1] is not None: lo = float(np.asarray(ins[1]))
  if len(ins) >= 3 and ins[2] is not None: hi = float(np.asarray(ins[2]))
  if lo == 0.0 and hi == 6.0: return ins[0].relu6()
  if lo == 0.0 and hi >= 3.4e38: return ins[0].relu()
  return ins[0].clip(float(lo), float(hi))

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
  x = ins[0]; w = np.asarray(ins[1]); b = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  return _conv(x, w, stride=_uniform(a.get("strides"), "Conv"), pad=_sympad(a.get("pads"), "Conv"),
               dilation=_uniform(a.get("dilations"), "Conv"), groups=int(a.get("group", 1)), bias=b)

@onnx_op("MaxPool")
def _maxpool(node, ins, a, i):
  return ins[0].max_pool(_uniform(a.get("kernel_shape"), "MaxPool"),
                         _uniform(a.get("strides"), "MaxPool"), _sympad(a.get("pads"), "MaxPool"))
@onnx_op("AveragePool")
def _avgpool(node, ins, a, i):
  return ins[0].avg_pool(_uniform(a.get("kernel_shape"), "AveragePool"),
                         _uniform(a.get("strides"), "AveragePool"), _sympad(a.get("pads"), "AveragePool"))
@onnx_op("GlobalAveragePool")
def _gap(node, ins, a, i): return ins[0].mean((2, 3))     # keepdims -> [N,C,1,1]

@onnx_op("Gemm")
def _gemm(node, ins, a, i):
  x = ins[0]; W = np.asarray(ins[1]); B = np.asarray(ins[2]) if len(ins) > 2 and ins[2] is not None else None
  if not int(a.get("transB", 0)): W = W.T            # x.linear expects [out,in]
  return x.linear(W, B)
@onnx_op("MatMul")
def _matmul(node, ins, a, i):
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
  if not axes: axes = [int(v) for v in np.asarray(ins[1])]
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
