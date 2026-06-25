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

def test_add_builds():
  m = _model([helper.make_node("Add", ["a", "b"], ["y"])], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "add"

def test_sigmoid_builds():
  m = _model([helper.make_node("Sigmoid", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "sigmoid"

def test_clip_relu6_builds():
  n = helper.make_node("Clip", ["x"], ["y"], min=0.0, max=6.0)
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "relu6"
