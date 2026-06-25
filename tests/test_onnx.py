import numpy as np, onnx, pytest  # noqa: F401  (harness imports for later op tests)
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

def _init(arr, name): return onnx.numpy_helper.from_array(arr.astype(np.float32), name)

def test_conv_builds():
  w = _init(np.zeros((8, 3, 3, 3)), "W"); b = _init(np.zeros(8), "B")
  n = helper.make_node("Conv", ["x", "W", "B"], ["y"], strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1])
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 32, 32])], inits=[w, b])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 32, 32) and out.op == "conv"

def test_maxpool_builds():
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2])
  m = _model([n], [_vi("x", [1, 8, 32, 32])], [_vi("y", [1, 8, 16, 16])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 16, 16) and out.op == "max_pool"

def test_global_average_pool_builds():
  n = helper.make_node("GlobalAveragePool", ["x"], ["y"])
  m = _model([n], [_vi("x", [1, 8, 32, 32])], [_vi("y", [1, 8, 1, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 1, 1) and out.op == "reduce_mean"

def test_conv_default_strides_dilations():  # omitted strides/dilations -> ONNX default 1 (not 0)
  w = _init(np.zeros((4, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 1, 1, 1])  # no strides, no dilations
  m = _model([n], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 4, 8, 8])], inits=[w])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 8, 8) and out.op == "conv"

def test_maxpool_default_strides():  # omitted strides -> ONNX default 1 (not k)
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2])  # no strides
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 7, 7])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 7, 7) and out.op == "max_pool"

def test_gemm_transb_builds():  # transB=1 -> W stays [out,in]; x[1,16] linear W[10,16] -> (1,10)
  w = _init(np.zeros((10, 16)), "W"); b = _init(np.zeros(10), "B")
  n = helper.make_node("Gemm", ["x", "W", "B"], ["y"], transB=1)
  m = _model([n], [_vi("x", [1, 16])], [_vi("y", [1, 10])], inits=[w, b])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 10) and out.op == "matmul"

def test_flatten_builds():
  n = helper.make_node("Flatten", ["x"], ["y"], axis=1)
  m = _model([n], [_vi("x", [1, 8, 2, 2])], [_vi("y", [1, 32])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 32) and out.op == "flatten2d"

def test_batchnorm_builds():  # shape unchanged
  s = _init(np.ones(4), "s"); bb = _init(np.zeros(4), "bb")
  mean = _init(np.zeros(4), "mean"); var = _init(np.ones(4), "var")
  n = helper.make_node("BatchNormalization", ["x", "s", "bb", "mean", "var"], ["y"])
  m = _model([n], [_vi("x", [1, 4, 5, 5])], [_vi("y", [1, 4, 5, 5])], inits=[s, bb, mean, var])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 5, 5) and out.op == "batch_norm"

def test_concat_builds():
  n = helper.make_node("Concat", ["a", "b"], ["y"], axis=1)
  m = _model([n], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 6])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 6) and out.op == "concat"

def test_softmax_builds():
  n = helper.make_node("Softmax", ["x"], ["y"], axis=-1)
  m = _model([n], [_vi("x", [1, 10])], [_vi("y", [1, 10])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 10) and out.op == "softmax"

def test_reshape_infer_builds():  # -1 infers the flattened dim
  shp = _init(np.array([1, -1]), "shp")
  n = helper.make_node("Reshape", ["x", "shp"], ["y"])
  m = _model([n], [_vi("x", [1, 8, 2, 2])], [_vi("y", [1, 32])], inits=[shp])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 32) and out.op == "reshape"

def test_reshape_zero_keep_builds():  # 0 = copy input dim, resolved BEFORE -1 (else divide-by-zero)
  shp = onnx.numpy_helper.from_array(np.array([0, -1], dtype=np.int64), "shp")
  n = helper.make_node("Reshape", ["x", "shp"], ["y"])
  m = _model([n], [_vi("x", [2, 3, 4])], [_vi("y", [2, 12])], inits=[shp])
  _, out = af.onnx_to_tensor(m); assert out.shape == (2, 12) and out.op == "reshape"

def test_transpose_builds():
  n = helper.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 1])
  m = _model([n], [_vi("x", [1, 3, 4])], [_vi("y", [1, 4, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 3) and out.op == "transpose"

def test_squeeze_builds():  # opset>=13 reads axes from input initializer
  ax = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
  n = helper.make_node("Squeeze", ["x", "ax"], ["y"])
  m = _model([n], [_vi("x", [1, 1, 4])], [_vi("y", [1, 4])], inits=[ax])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4) and out.op == "squeeze"

def test_unsqueeze_builds():  # opset>=13 reads axes from input initializer
  ax = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
  n = helper.make_node("Unsqueeze", ["x", "ax"], ["y"])
  m = _model([n], [_vi("x", [1, 4])], [_vi("y", [1, 1, 4])], inits=[ax])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 1, 4) and out.op == "expand_dims"

def test_identity_builds():  # Identity passes its input straight through
  n = helper.make_node("Identity", ["x"], ["y"])
  m = _model([n], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4) and out.op == "input"

def test_sub_builds():
  m = _model([helper.make_node("Sub", ["a", "b"], ["y"])], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "sub"

def test_mul_builds():
  m = _model([helper.make_node("Mul", ["a", "b"], ["y"])], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "mul"

def test_div_builds():  # aneforge lowers division to real_div
  m = _model([helper.make_node("Div", ["a", "b"], ["y"])], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "real_div"

def test_tanh_builds():
  m = _model([helper.make_node("Tanh", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "tanh"

def test_clip_opset11_input_form_builds():  # opset>=11 reads min/max from inputs 2/3
  lo = onnx.numpy_helper.from_array(np.array(0.0, dtype=np.float32), "lo")
  hi = onnx.numpy_helper.from_array(np.array(6.0, dtype=np.float32), "hi")
  n = helper.make_node("Clip", ["x", "lo", "hi"], ["y"])
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])], inits=[lo, hi])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "relu6"

def test_dynamic_dim_raises():  # symbolic dim_param -> static-shapes-only error
  m = _model([helper.make_node("Relu", ["x"], ["y"])],
             [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 4])], [_vi("y", [1, 4])])
  with pytest.raises(ValueError): af.onnx_to_tensor(m)

def test_unsupported_op_raises():  # unregistered op type fails loudly
  m = _model([helper.make_node("Erf", ["x"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_conv_non_uniform_strides_raises():
  w = _init(np.zeros((8, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], strides=[1, 2], pads=[1, 1, 1, 1])
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 16, 32])], inits=[w])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_conv_asymmetric_pads_raises():
  w = _init(np.zeros((8, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 0, 1, 0])
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 32, 31])], inits=[w])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_conv_auto_pad_raises():
  w = _init(np.zeros((8, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], auto_pad="SAME_UPPER")
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 32, 32])], inits=[w])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_gemm_alpha_raises():
  w = _init(np.zeros((10, 16)), "W"); b = _init(np.zeros(10), "B")
  n = helper.make_node("Gemm", ["x", "W", "B"], ["y"], transB=1, alpha=2.0)
  m = _model([n], [_vi("x", [1, 16])], [_vi("y", [1, 10])], inits=[w, b])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_add_constant_operand_raises():  # elementwise is tensor-tensor only
  c = _init(np.zeros((1, 3)), "c")
  n = helper.make_node("Add", ["x", "c"], ["y"])
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])], inits=[c])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_squeeze_no_axes_raises():  # squeeze-all has no static lowering
  n = helper.make_node("Squeeze", ["x"], ["y"])
  m = _model([n], [_vi("x", [1, 1, 4])], [_vi("y", [4])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_maxpool_ceil_mode_raises():
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
  m = _model([n], [_vi("x", [1, 8, 31, 31])], [_vi("y", [1, 8, 16, 16])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

@requires_ane
def test_resnet18_onnx_matches_onnxruntime(tmp_path):
  """End-to-end: import a torch-exported ResNet-18 ONNX, run on the ANE, match onnxruntime (fp16 cosine)."""
  import torch, torchvision, onnxruntime
  net_t = torchvision.models.resnet18(weights=None).eval()
  x = torch.randn(1, 3, 224, 224)
  path = str(tmp_path / "resnet18.onnx")
  torch.onnx.export(net_t, (x,), path, opset_version=13, do_constant_folding=True,
                    input_names=["x"], output_names=["y"], dynamo=False)  # legacy exporter; numerics identical
  net = af.load_onnx(path)
  x_fp16 = x.numpy().astype(np.float16)
  got = np.asarray(net(x_fp16)).astype(np.float32).ravel()
  ref = np.asarray(onnxruntime.InferenceSession(path).run(None, {"x": x.numpy()})[0]).astype(np.float32).ravel()
  cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
  assert cos > 0.999, f"resnet18 ANE vs onnxruntime cosine={cos}"
