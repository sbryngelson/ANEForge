"""Build-level assertions that ONNX negative axis attributes are normalized in the graph.

Separate from `test_onnx.py` on purpose: that module is `pytestmark = requires_ane`, because every
test there compiles and dispatches. These are pure builder checks with no dispatch, so they run in
CI (`pytest -m "not requires_ane"`) and lock the invariant on machines without an ANE.
"""
import numpy as np, onnx, pytest  # noqa: F401
from onnx import helper, TensorProto
import aneforge as af


def _model(nodes, inputs, outputs, inits=(), opset=13):
  g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
  m.ir_version = 9
  return m

def _vi(name, shape): return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_negative_axis_normalized_in_graph_attrs():
  """The built graph must carry a normalized (non-negative) axis attribute.

  Output equality alone cannot catch every regression here: MIL tolerates a negative `axis`, and
  for `concat` a raw -1 indexes the same list slot as rank-1, so a dropped `% rank` would still
  produce correct numbers while leaking -1 into the op attrs. Assert on the attribute.
  """
  sh = [2, 3, 4, 5]
  m = _model([helper.make_node("Concat", ["a", "b"], ["y"], axis=-1)],
             [_vi("a", sh), _vi("b", sh)], [_vi("y", [2, 3, 4, 10])])
  _, out = af.onnx_to_tensor(m)
  assert out.op == "concat" and out.attrs["axis"] == 3, f"concat axis attr = {out.attrs.get('axis')}"
  m = _model([helper.make_node("Softmax", ["x"], ["y"], axis=-2)], [_vi("x", sh)], [_vi("y", sh)])
  _, out = af.onnx_to_tensor(m)
  assert out.attrs["axis"] == 2, f"softmax axis attr = {out.attrs.get('axis')}"


def test_negative_axis_argmax_argmin_match_positive():
  """ArgMax/ArgMin are 2-D only and shipped shape-validated, so check the built graph, not values."""
  for op in ("ArgMax", "ArgMin"):
    shapes = []
    for ax in (-1, 1):
      m = _model([helper.make_node(op, ["x"], ["y"], axis=ax, keepdims=1)], [_vi("x", [3, 5])],
                 [helper.make_tensor_value_info("y", TensorProto.INT64, [3, 1])])
      _, out = af.onnx_to_tensor(m)
      shapes.append(out.shape)
    assert shapes[0] == shapes[1], f"{op}: axis -1 -> {shapes[0]}, axis 1 -> {shapes[1]}"
