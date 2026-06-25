import numpy as np, onnx  # noqa: F401  (harness imports for later op tests)
from onnx import helper, TensorProto
import aneforge as af
from _helpers import requires_ane  # noqa: F401  (on-device gate for later op tests)

def _model(nodes, inputs, outputs, inits=()):
  g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
  m.ir_version = 9
  return m

def _vi(name, shape): return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

def test_onnx_to_tensor_relu_builds():
  m = _model([helper.make_node("Relu", ["x"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  ins, out = af.onnx_to_tensor(m)
  assert out.shape == (1, 4) and out.op == "relu"
