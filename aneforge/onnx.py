"""Import an ONNX model and run it on the ANE. CNN-classifier op subset; see
docs. Public: load_onnx / onnx_to_tensor."""
from __future__ import annotations
from typing import Callable
from .graph import Tensor, input as _input
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
