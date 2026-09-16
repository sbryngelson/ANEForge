"""coverage_report and the ONNX import error message. Build-level only: no ANE, no compile."""
import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto

import aneforge as af


def _model(nodes, inputs, outputs, inits=(), opset=13):
  g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
  m.ir_version = 9
  return m


def _vi(name, shape): return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_coverage_report_counts_unsupported():
  nodes = [helper.make_node("Relu", ["x"], ["a"]),
           helper.make_node("Foo", ["a"], ["b"]),
           helper.make_node("Foo", ["b"], ["c"]),
           helper.make_node("Bar", ["c"], ["y"])]
  m = _model(nodes, [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  assert af.coverage_report(m) == {"Foo": 2, "Bar": 1}


def test_coverage_report_empty_for_supported_model():
  m = _model([helper.make_node("Relu", ["x"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  assert af.coverage_report(m) == {}


def test_coverage_report_walks_if_branches():
  then_g = helper.make_graph([helper.make_node("Foo", ["x"], ["ty"])], "t", [], [_vi("ty", [1, 4])])
  else_g = helper.make_graph([helper.make_node("Bar", ["x"], ["ey"])], "e", [], [_vi("ey", [1, 4])])
  ci = onnx.numpy_helper.from_array(np.array(True), "c")
  n = helper.make_node("If", ["c"], ["y"], then_branch=then_g, else_branch=else_g)
  m = _model([n], [_vi("x", [1, 4])], [_vi("y", [1, 4])], inits=[ci])
  assert af.coverage_report(m) == {"Foo": 1, "Bar": 1}


def test_coverage_report_walks_loop_body():
  body = helper.make_graph(
    [helper.make_node("Identity", ["cin"], ["cout"]),
     helper.make_node("Add", ["acc", "x"], ["acc_out"]),
     helper.make_node("Foo", ["acc_out"], ["scan"])],
    "body",
    [helper.make_tensor_value_info("it", TensorProto.INT64, []),
     helper.make_tensor_value_info("cin", TensorProto.BOOL, []), _vi("acc", [1, 4])],
    [helper.make_tensor_value_info("cout", TensorProto.BOOL, []), _vi("acc_out", [1, 4]), _vi("scan", [1, 4])])
  mi = onnx.numpy_helper.from_array(np.array(3, np.int64), "M")
  ci = onnx.numpy_helper.from_array(np.array(True), "c0")
  a0 = onnx.numpy_helper.from_array(np.zeros((1, 4), np.float32), "acc0")
  n = helper.make_node("Loop", ["M", "c0", "acc0"], ["acc_final"], body=body)
  m = _model([n], [_vi("x", [1, 4])], [_vi("acc_final", [1, 4])], inits=[mi, ci, a0])
  assert af.coverage_report(m) == {"Foo": 1}


def test_unsupported_op_error_lists_all_missing():
  nodes = [helper.make_node("Relu", ["x"], ["a"]),
           helper.make_node("Foo", ["a"], ["b"]),
           helper.make_node("Foo", ["b"], ["c"]),
           helper.make_node("Bar", ["c"], ["y"])]
  m = _model(nodes, [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  with pytest.raises(NotImplementedError) as excinfo:
    af.onnx_to_tensor(m)
  msg = str(excinfo.value)
  assert "Foo" in msg and "Bar" in msg
  assert "2 ops" in msg and "Foo (2 nodes)" in msg
