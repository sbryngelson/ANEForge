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

def test_tier3_unary_ops_build():  # cheap unary ops over existing graph ops
  for op, want in [("Abs", "abs"), ("Sign", "sign"), ("Sin", "sin"), ("Cos", "cos"), ("Softplus", "softplus"),
                   ("Floor", "floor"), ("Ceil", "ceil"), ("Round", "round"), ("Reciprocal", "inverse"), ("Neg", "muls")]:
    m = _model([helper.make_node(op, ["x"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
    _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4) and out.op == want, f"{op} -> {out.op}"

def test_tier3_max_min_tile_where_build():
  for op, want in [("Max", "maximum"), ("Min", "minimum")]:
    m = _model([helper.make_node(op, ["a", "b"], ["y"])], [_vi("a", [1, 4]), _vi("b", [1, 4])], [_vi("y", [1, 4])])
    _, out = af.onnx_to_tensor(m); assert out.op == want
  reps = onnx.numpy_helper.from_array(np.array([1, 2], np.int64), "r")
  m = _model([helper.make_node("Tile", ["x", "r"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 8])], inits=[reps])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8) and out.op == "tile"

def test_resize_half_pixel_needs_approx():  # default guards half_pixel; approx_resize=True maps it to bilinear
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], np.int64), "sz")
  n = helper.make_node("Resize", ["x", "roi", "sc", "sz"], ["y"], mode="linear", coordinate_transformation_mode="half_pixel")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])], inits=[_empty_f32("roi"), _empty_f32("sc"), sizes])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)
  _, out = af.onnx_to_tensor(m, approx_resize=True); assert out.shape == (1, 4, 16, 16) and out.op == "resize_bilinear"

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

def test_conv_asymmetric_pad_builds():  # ONNX pads [top,left,bottom,right] -> per-side MIL conv pad
  w = _init(np.zeros((4, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], pads=[2, 1, 0, 0], strides=[1, 1])  # top=2,left=1
  m = _model([n], [_vi("x", [1, 3, 10, 10])], [_vi("y", [1, 4, 10, 9])], inits=[w])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 10, 9) and out.op == "conv"

def test_maxpool_asymmetric_pad_builds():  # TF SAME emits pool pads like [0,0,1,1] -> explicit pad, then valid pool
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 1, 1])
  m = _model([n], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 3, 4, 4])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3, 4, 4) and out.op == "max_pool"

@requires_ane
def test_maxpool_asymmetric_pad_matches_onnxruntime():  # numerically matches ORT (pad cells never win the max)
  import onnxruntime as ort, tempfile, os
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 1, 1])
  m = _model([n], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 3, 4, 4])])
  p = os.path.join(tempfile.mkdtemp(), "m.onnx"); onnx.save(m, p)
  x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)   # negative values -> pad must be -inf-ish
  ref = np.asarray(ort.InferenceSession(p).run(None, {"x": x})[0])
  got = np.asarray(af.load_onnx(p)(x.astype(np.float16))).astype(np.float32)
  assert got.shape == ref.shape and np.abs(got - ref).max() < 1e-2

def _conv_bn_relu_model(seed=0):
  rng = np.random.default_rng(seed)
  W = _init(rng.standard_normal((4, 3, 3, 3)).astype(np.float32), "W")
  gg = _init((np.abs(rng.standard_normal(4)) + 0.5).astype(np.float32), "g"); bb = _init(rng.standard_normal(4).astype(np.float32), "b")
  mm = _init(rng.standard_normal(4).astype(np.float32), "m"); vv = _init((np.abs(rng.standard_normal(4)) + 0.5).astype(np.float32), "v")
  nodes = [helper.make_node("Conv", ["x", "W"], ["c"], pads=[1, 1, 1, 1]),
           helper.make_node("BatchNormalization", ["c", "g", "b", "m", "v"], ["bnout"], epsilon=1e-5),
           helper.make_node("Relu", ["bnout"], ["y"])]
  return _model(nodes, [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 4, 8, 8])], inits=[W, gg, bb, mm, vv])

def test_conv_bn_fold_removes_bn():  # Conv->BN folds into the Conv (exact); removes the standalone fp16 BN op
  from aneforge.onnx import _fold_conv_bn
  import copy
  fm = copy.deepcopy(_conv_bn_relu_model())
  n = _fold_conv_bn(fm)
  assert n == 1 and all(nd.op_type != "BatchNormalization" for nd in fm.graph.node)
  _, out = af.onnx_to_tensor(_conv_bn_relu_model()); assert out.shape == (1, 4, 8, 8)   # imports (folds in _load)

@requires_ane
def test_conv_bn_fold_matches_onnxruntime():  # folded model on the ANE matches ORT on the original (fold is exact)
  import onnxruntime as ort, tempfile, os
  m = _conv_bn_relu_model(1)
  p = os.path.join(tempfile.mkdtemp(), "m.onnx"); onnx.save(m, p)
  x = np.random.default_rng(2).standard_normal((1, 3, 8, 8)).astype(np.float32)
  ref = np.asarray(ort.InferenceSession(p).run(None, {"x": x})[0])
  got = np.asarray(af.load_onnx(p)(x.astype(np.float16))).astype(np.float32)
  assert got.shape == ref.shape and np.abs(got - ref).max() < 5e-2

def test_dequantize_weight_conv_builds():  # int8 weight DequantizeLinear -> fp32 const fed to conv
  w8 = onnx.numpy_helper.from_array(np.ones((4, 3, 3, 3), np.int8), "w8")
  sc = onnx.numpy_helper.from_array(np.full(4, 0.1, np.float32), "sc")
  zp = onnx.numpy_helper.from_array(np.zeros(4, np.int8), "zp")
  nodes = [helper.make_node("DequantizeLinear", ["w8", "sc", "zp"], ["w"], axis=0),
           helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])]
  m = _model(nodes, [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 4, 8, 8])], inits=[w8, sc, zp])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 8, 8) and out.op == "conv"

def test_quantize_activation_folds_relu():  # QuantizeLinear with zp pinning qmin to 0 -> a clip (the folded relu)
  sc = onnx.numpy_helper.from_array(np.array(0.05, np.float32), "sc"); zp = onnx.numpy_helper.from_array(np.array(-128, np.int8), "zp")
  n = helper.make_node("QuantizeLinear", ["x", "sc", "zp"], ["y"])
  m = _model([n], [_vi("x", [1, 8])], [_vi("y", [1, 8])], inits=[sc, zp])
  _, out = af.onnx_to_tensor(m); assert out.op == "clip" and out.attrs["lo"] == 0.0

def test_shape_subgraph_folds_to_reshape():  # Shape->Gather->Unsqueeze->Concat folds to a static Reshape target
  nodes = [helper.make_node("Shape", ["x"], ["s"]),
           helper.make_node("Gather", ["s", "i0"], ["d0"], axis=0),
           helper.make_node("Unsqueeze", ["d0", "ax0"], ["d0u"]),     # opset>=13: axes is an input
           helper.make_node("Concat", ["d0u", "neg1"], ["tgt"], axis=0),
           helper.make_node("Reshape", ["x", "tgt"], ["y"])]
  inits = [onnx.numpy_helper.from_array(np.array(0, np.int64), "i0"),
           onnx.numpy_helper.from_array(np.array([0], np.int64), "ax0"),
           onnx.numpy_helper.from_array(np.array([-1], np.int64), "neg1")]
  m = _model(nodes, [_vi("x", [2, 3, 4])], [_vi("y", [2, 12])], inits=inits)
  _, out = af.onnx_to_tensor(m); assert out.shape == (2, 12) and out.op == "reshape"

def test_slice_channel_split_builds():  # ONNX Slice on an activation -> static slice_by_size
  starts = onnx.numpy_helper.from_array(np.array([0], np.int64), "st")
  ends = onnx.numpy_helper.from_array(np.array([4], np.int64), "en")
  axes = onnx.numpy_helper.from_array(np.array([1], np.int64), "ax")
  n = helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 4, 4, 4])], inits=[starts, ends, axes])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 4, 4) and out.op == "slice_by_size"

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

def test_onnx_to_features_peels_classifier():  # features = the input to the final linear layer
  w = _init(np.zeros((10, 16)), "W"); b = _init(np.zeros(10), "B")
  nodes = [helper.make_node("Flatten", ["x"], ["f"], axis=1),
           helper.make_node("Gemm", ["f", "W", "B"], ["y"], transB=1)]
  m = _model(nodes, [_vi("x", [1, 16, 1, 1])], [_vi("y", [1, 10])], inits=[w, b])
  ins, feats = af.onnx_to_features(m)
  assert feats.shape == (1, 16) and feats.op == "flatten2d"    # the Gemm's input, not the logits

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

def test_layernorm_builds():  # transformer LayerNorm over the last axis -> reshape back to input shape
  s = _init(np.ones(8), "s"); bb = _init(np.zeros(8), "bb")
  n = helper.make_node("LayerNormalization", ["x", "s", "bb"], ["y"], axis=-1, epsilon=1e-5)
  m = _model([n], [_vi("x", [1, 16, 8])], [_vi("y", [1, 16, 8])], inits=[s, bb])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 16, 8) and out.op == "reshape"

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
  m = _model([helper.make_node("ScatterND", ["x", "i", "u"], ["y"])], [_vi("x", [1, 4])], [_vi("y", [1, 4])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_conv_non_uniform_strides_raises():
  w = _init(np.zeros((8, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], strides=[1, 2], pads=[1, 1, 1, 1])
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 16, 32])], inits=[w])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_conv_asymmetric_pads_builds():  # asymmetric pads map to the per-side MIL conv pad
  w = _init(np.zeros((8, 3, 3, 3)), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 0, 1, 0])  # H pad 1+1 -> 32; W pad 0+0 -> 30
  m = _model([n], [_vi("x", [1, 3, 32, 32])], [_vi("y", [1, 8, 32, 30])], inits=[w])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 32, 30) and out.op == "conv"

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

def test_add_constant_operand_builds():  # a constant operand bakes in as a fused const_array
  c = _init(np.zeros((1, 3)), "c")
  n = helper.make_node("Add", ["x", "c"], ["y"])
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])], inits=[c])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "add"

def test_hardsigmoid_builds():
  m = _model([helper.make_node("HardSigmoid", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "clip"

def test_squeeze_no_axes_raises():  # squeeze-all has no static lowering
  n = helper.make_node("Squeeze", ["x"], ["y"])
  m = _model([n], [_vi("x", [1, 1, 4])], [_vi("y", [4])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_maxpool_ceil_mode_builds():  # ceil_mode rounds the output size up (native MIL ceil_mode)
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
  m = _model([n], [_vi("x", [1, 8, 31, 31])], [_vi("y", [1, 8, 16, 16])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 16, 16) and out.op == "max_pool"

def test_avgpool_count_include_pad_builds():  # count_include_pad maps to MIL exclude_padding_from_average
  n = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1)
  m = _model([n], [_vi("x", [1, 8, 8, 8])], [_vi("y", [1, 8, 8, 8])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 8, 8) and out.op == "avg_pool"

def test_erf_builds():
  m = _model([helper.make_node("Erf", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "erf"

def test_exp_builds():
  m = _model([helper.make_node("Exp", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "exp"

def test_log_builds():
  m = _model([helper.make_node("Log", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "log"

def test_sqrt_builds():
  m = _model([helper.make_node("Sqrt", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "sqrt"

def test_elu_builds():
  m = _model([helper.make_node("Elu", ["x"], ["y"], alpha=0.5)], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "elu"

def test_leaky_relu_builds():
  m = _model([helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.2)], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "leaky_relu"

def test_gelu_builds():  # default approximate="none" -> exact erf-gelu
  m = _model([helper.make_node("Gelu", ["x"], ["y"])], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "gelu"

def test_gelu_tanh_raises():  # tanh approximation is not implemented
  n = helper.make_node("Gelu", ["x"], ["y"], approximate="tanh")
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_prelu_builds():  # slope is the 2nd input ([C,1,1] flattened to [C])
  slope = _init(np.ones((4, 1, 1)), "slope")
  n = helper.make_node("PRelu", ["x", "slope"], ["y"])
  m = _model([n], [_vi("x", [1, 4, 2, 2])], [_vi("y", [1, 4, 2, 2])], inits=[slope])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 2, 2) and out.op == "prelu"

def test_pow_builds():  # tensor-tensor exponent
  m = _model([helper.make_node("Pow", ["a", "b"], ["y"])], [_vi("a", [1, 3]), _vi("b", [1, 3])], [_vi("y", [1, 3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "pow"

def test_pow_constant_operand_builds():  # a constant exponent bakes in (Pow(x, 2) etc.)
  c = _init(np.full((1, 3), 2.0), "c")
  n = helper.make_node("Pow", ["x", "c"], ["y"])
  m = _model([n], [_vi("x", [1, 3])], [_vi("y", [1, 3])], inits=[c])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 3) and out.op == "pow"

def test_instance_norm_builds():  # shape unchanged
  s = _init(np.ones(4), "s"); b = _init(np.zeros(4), "b")
  n = helper.make_node("InstanceNormalization", ["x", "s", "b"], ["y"])
  m = _model([n], [_vi("x", [1, 4, 5, 5])], [_vi("y", [1, 4, 5, 5])], inits=[s, b])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 5, 5) and out.op == "instance_norm"

def test_space_to_depth_builds():  # [1,3,8,8] bs=2 -> [1,12,4,4]
  n = helper.make_node("SpaceToDepth", ["x"], ["y"], blocksize=2)
  m = _model([n], [_vi("x", [1, 3, 8, 8])], [_vi("y", [1, 12, 4, 4])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 12, 4, 4) and out.op == "space_to_depth"

def _empty_f32(name): return onnx.numpy_helper.from_array(np.array([], dtype=np.float32), name)
def _cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def test_lrn_builds():  # shape unchanged
  n = helper.make_node("LRN", ["x"], ["y"], size=5, alpha=1e-4, beta=0.75, bias=1.0)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 4, 4])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 4, 4) and out.op == "local_response_norm"

def test_depth_to_space_builds():  # [1,8,2,2] bs=2 -> [1,2,4,4]
  n = helper.make_node("DepthToSpace", ["x"], ["y"], blocksize=2)  # default mode=DCR
  m = _model([n], [_vi("x", [1, 8, 2, 2])], [_vi("y", [1, 2, 4, 4])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 2, 4, 4) and out.op == "depth_to_space"

def test_depth_to_space_crd_raises():  # ANE matches DCR only
  n = helper.make_node("DepthToSpace", ["x"], ["y"], blocksize=2, mode="CRD")
  m = _model([n], [_vi("x", [1, 8, 2, 2])], [_vi("y", [1, 2, 4, 4])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_resize_nearest_builds():  # sizes input -> [1,4,16,16]
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="nearest",
                       coordinate_transformation_mode="asymmetric", nearest_mode="floor")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 16, 16) and out.op == "resize_nearest_neighbor"

def test_resize_linear_builds():  # sizes input, linear+asymmetric -> bilinear
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="linear",
                       coordinate_transformation_mode="asymmetric")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 16, 16) and out.op == "resize_bilinear"

def test_resize_scales_builds():  # scales input (no sizes) -> round(2*8)=16
  scales = onnx.numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales")
  n = helper.make_node("Resize", ["x", "roi", "scales"], ["y"], mode="nearest",
                       coordinate_transformation_mode="asymmetric", nearest_mode="floor")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), scales])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 4, 16, 16) and out.op == "resize_nearest_neighbor"

def test_resize_cubic_raises():
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="cubic")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_resize_linear_half_pixel_raises():  # ANE bilinear does not match half_pixel
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="linear",
                       coordinate_transformation_mode="half_pixel")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_resize_nearest_half_pixel_raises():  # ANE nearest matches asymmetric only
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="nearest",
                       coordinate_transformation_mode="half_pixel")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_resize_nearest_default_mode_raises():  # ONNX default nearest_mode is round_prefer_floor; ANE samples floor
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="nearest",
                       coordinate_transformation_mode="asymmetric")   # nearest_mode omitted -> round_prefer_floor
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_resize_linear_pytorch_half_pixel_raises():  # torch exports this; ANE bilinear does not match it
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="linear",
                       coordinate_transformation_mode="pytorch_half_pixel")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
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

def _run_vs_ort(m, x):
  """Compile m on the ANE (fp16) and onnxruntime (fp32), return their flattened outputs."""
  net = af.load_onnx(m)
  got = np.asarray(net(x.astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnx_run(m, x)).astype(np.float32).ravel()
  return got, ref

def onnx_run(m, x):  # run an in-memory model through onnxruntime
  import onnxruntime
  return onnxruntime.InferenceSession(m.SerializeToString()).run(None, {"x": x})[0]

@requires_ane
def test_lrn_onnx_matches_onnxruntime():  # alpha=1 (sum-of-squares dominates) so the alpha mapping is exercised
  rng = np.random.default_rng(0); x = (rng.standard_normal((1, 8, 4, 4)).astype(np.float32)) * 2.0
  n = helper.make_node("LRN", ["x"], ["y"], size=5, alpha=1.0, beta=0.75, bias=1.0)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 4, 4])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"LRN ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_depth_to_space_onnx_matches_onnxruntime():  # DCR is a pure permutation -> near-exact
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 8, 2, 2)).astype(np.float32)
  n = helper.make_node("DepthToSpace", ["x"], ["y"], blocksize=2, mode="DCR")
  m = _model([n], [_vi("x", [1, 8, 2, 2])], [_vi("y", [1, 2, 4, 4])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.999, f"DepthToSpace ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_resize_nearest_onnx_matches_onnxruntime():  # nearest+asymmetric is the ANE-matched config
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="nearest",
                       coordinate_transformation_mode="asymmetric", nearest_mode="floor")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"Resize nearest ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_resize_linear_onnx_matches_onnxruntime():  # linear+asymmetric is the ANE-matched config
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], dtype=np.int64), "sizes")
  n = helper.make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="linear",
                       coordinate_transformation_mode="asymmetric")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])],
             inits=[_empty_f32("roi"), _empty_f32("scales"), sizes])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"Resize linear ANE vs onnxruntime cosine={cos}"

def _model18(nodes, inputs, outputs, inits=()):  # opset 18 graph (axes-as-input reduce ops)
  g = helper.make_graph(nodes, "g", inputs, outputs, list(inits))
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]); m.ir_version = 9
  return m

# -- reductions ------------------------------------------------------------- #
def test_reduce_max_builds():  # axes attr (opset<18), keepdims=1
  n = helper.make_node("ReduceMax", ["x"], ["y"], axes=[2, 3], keepdims=1)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 1, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 1, 1) and out.op == "reduce_max"

def test_reduce_min_builds():
  n = helper.make_node("ReduceMin", ["x"], ["y"], axes=[1], keepdims=1)
  m = _model([n], [_vi("x", [2, 5])], [_vi("y", [2, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (2, 1) and out.op == "reduce_min"

def test_reduce_mean_builds():
  n = helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=1)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 1, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 1, 1) and out.op == "reduce_mean"

def test_reduce_max_keepdims0_builds():  # keepdims=0 squeezes the reduced axes
  n = helper.make_node("ReduceMax", ["x"], ["y"], axes=[2, 3], keepdims=0)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8) and out.op == "squeeze"

def test_reduce_sum_axes_input_builds():  # opset>=13 ReduceSum reads axes from an input initializer
  ax = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
  n = helper.make_node("ReduceSum", ["x", "ax"], ["y"], keepdims=1)
  m = _model([n], [_vi("x", [2, 5])], [_vi("y", [2, 1])], inits=[ax])
  _, out = af.onnx_to_tensor(m); assert out.shape == (2, 1) and out.op == "reduce_sum"

def test_reduce_max_axes_input_opset18_builds():  # opset>=18 ReduceMax also moves axes to an input
  ax = onnx.numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "ax")
  n = helper.make_node("ReduceMax", ["x", "ax"], ["y"], keepdims=1)
  m = _model18([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 1, 1])], inits=[ax])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 8, 1, 1) and out.op == "reduce_max"

def test_reduce_all_axes_builds():  # absent axes -> reduce over every axis (keepdims)
  n = helper.make_node("ReduceMean", ["x"], ["y"], keepdims=1)
  m = _model([n], [_vi("x", [2, 3, 4])], [_vi("y", [1, 1, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (1, 1, 1) and out.op == "reduce_mean"

def test_reduce_noop_empty_axes_raises():  # noop_with_empty_axes=1 + empty axes (identity) is unsupported
  n = helper.make_node("ReduceSum", ["x"], ["y"], keepdims=1, noop_with_empty_axes=1)
  m = _model([n], [_vi("x", [2, 5])], [_vi("y", [2, 5])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_reduce_nonconstant_axes_raises():  # data-dependent (Tensor) axes have no static lowering
  n = helper.make_node("ReduceSum", ["x", "ax"], ["y"], keepdims=1)
  m = _model([n], [_vi("x", [2, 5]), helper.make_tensor_value_info("ax", TensorProto.INT64, [1])],
             [_vi("y", [2, 1])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

# -- gather / argmax / topk ------------------------------------------------- #
def test_gather_builds():  # 1-D index along axis 0 -> [len(idx), W]; lowers to slice+concat
  idx = onnx.numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), "idx")
  n = helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)
  m = _model([n], [_vi("x", [4, 8])], [_vi("y", [3, 8])], inits=[idx])
  _, out = af.onnx_to_tensor(m); assert out.shape == (3, 8) and out.op == "concat"

def test_gather_scalar_index_builds():  # scalar index drops the gathered axis (ONNX rank rule)
  idx = onnx.numpy_helper.from_array(np.array(2, dtype=np.int64), "idx")
  n = helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)
  m = _model([n], [_vi("x", [4, 8])], [_vi("y", [8])], inits=[idx])
  _, out = af.onnx_to_tensor(m); assert out.shape == (8,) and out.op == "squeeze"

def test_gather_nonconstant_index_raises():  # data-dependent (Tensor) indices have no ANE path
  n = helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)
  m = _model([n], [_vi("x", [4, 8]), helper.make_tensor_value_info("idx", TensorProto.INT64, [2])],
             [_vi("y", [2, 8])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_gather_2d_index_raises():  # >1-D indices change output rank in a way the 1-D gather can't replicate
  idx = onnx.numpy_helper.from_array(np.array([[0, 1], [2, 3]], dtype=np.int64), "idx")
  n = helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)
  m = _model([n], [_vi("x", [4, 8])], [_vi("y", [2, 2, 8])], inits=[idx])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_argmax_builds():  # 2D [C,W], axis=1 keepdims -> [C,1]
  n = helper.make_node("ArgMax", ["x"], ["y"], axis=1, keepdims=1)
  m = _model([n], [_vi("x", [3, 5])], [helper.make_tensor_value_info("y", TensorProto.INT64, [3, 1])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (3, 1) and out.op == "argmax"

def test_argmax_keepdims0_builds():  # keepdims=0 squeezes the reduced axis
  n = helper.make_node("ArgMax", ["x"], ["y"], axis=1, keepdims=0)
  m = _model([n], [_vi("x", [3, 5])], [helper.make_tensor_value_info("y", TensorProto.INT64, [3])])
  _, out = af.onnx_to_tensor(m); assert out.shape == (3,) and out.op == "squeeze"

def test_argmax_rank_not_2_raises():  # ANE argmax is 2D-only
  n = helper.make_node("ArgMax", ["x"], ["y"], axis=1)
  m = _model([n], [_vi("x", [1, 3, 5])], [helper.make_tensor_value_info("y", TensorProto.INT64, [1, 1, 5])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_argmax_select_last_index_raises():
  n = helper.make_node("ArgMax", ["x"], ["y"], axis=1, select_last_index=1)
  m = _model([n], [_vi("x", [3, 5])], [helper.make_tensor_value_info("y", TensorProto.INT64, [3, 1])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_topk_builds():  # 2D [C,W], k from input -> values [C,k]; k=2 avoids the arch-gated k in {3,4}
  k = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "k")
  n = helper.make_node("TopK", ["x", "k"], ["vals", "idx"], axis=-1)
  m = _model([n], [_vi("x", [3, 5])],
             [_vi("vals", [3, 2]), helper.make_tensor_value_info("idx", TensorProto.INT64, [3, 2])], inits=[k])
  _, out = af.onnx_to_tensor(m); assert out.shape == (3, 2) and out.op == "topk"

def test_topk_rank_not_2_raises():  # ANE topk is 2D-only
  k = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "k")
  n = helper.make_node("TopK", ["x", "k"], ["vals", "idx"], axis=-1)
  m = _model([n], [_vi("x", [1, 3, 5])],
             [_vi("vals", [1, 3, 2]), helper.make_tensor_value_info("idx", TensorProto.INT64, [1, 3, 2])], inits=[k])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_topk_axis_raises():  # only the last axis (width) is supported
  k = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "k")
  n = helper.make_node("TopK", ["x", "k"], ["vals", "idx"], axis=0)
  m = _model([n], [_vi("x", [3, 5])],
             [_vi("vals", [2, 5]), helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 5])], inits=[k])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_topk_nonconstant_k_raises():  # data-dependent (Tensor) k has no static lowering
  n = helper.make_node("TopK", ["x", "k"], ["vals", "idx"], axis=-1)
  m = _model([n], [_vi("x", [3, 5]), helper.make_tensor_value_info("k", TensorProto.INT64, [1])],
             [_vi("vals", [3, 2]), helper.make_tensor_value_info("idx", TensorProto.INT64, [3, 2])])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

def test_matmul_constant_first_operand_raises():  # ndarray @ Tensor has no ANE path
  a_w = _init(np.zeros((4, 6)), "A")
  n = helper.make_node("MatMul", ["A", "B"], ["y"])
  m = _model([n], [_vi("B", [6, 3])], [_vi("y", [4, 3])], inits=[a_w])
  with pytest.raises(NotImplementedError): af.onnx_to_tensor(m)

@requires_ane
def test_reduce_mean_onnx_matches_onnxruntime():  # global-avg-pool-style reduction
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
  n = helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=1)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 1, 1])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"ReduceMean ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_reduce_max_onnx_matches_onnxruntime():
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
  n = helper.make_node("ReduceMax", ["x"], ["y"], axes=[2, 3], keepdims=1)
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 8, 1, 1])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"ReduceMax ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_gather_onnx_matches_onnxruntime():  # static-index gather along axis 0
  rng = np.random.default_rng(0); x = rng.standard_normal((4, 8)).astype(np.float32)
  idx = onnx.numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), "idx")
  n = helper.make_node("Gather", ["x", "idx"], ["y"], axis=0)
  m = _model([n], [_vi("x", [4, 8])], [_vi("y", [3, 8])], inits=[idx])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.999, f"Gather ANE vs onnxruntime cosine={cos}"

# -- constant-operand elementwise (baked as a fused const_array, no graph cut) ---- #
@requires_ane
def test_mul_constant_onnx_matches_onnxruntime():  # per-channel constant scale
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  c = _init((rng.standard_normal((1, 4, 1, 1)) * 0.5).astype(np.float32), "c")
  m = _model([helper.make_node("Mul", ["x", "c"], ["y"])], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])], inits=[c])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"Mul-const ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_pow2_constant_onnx_matches_onnxruntime():  # Pow(x, 2) with a constant exponent
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  e = _init(np.array(2.0, np.float32), "e")
  m = _model([helper.make_node("Pow", ["x", "e"], ["y"])], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])], inits=[e])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"Pow2-const ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_hardsigmoid_onnx_matches_onnxruntime():  # clip(0.2x + 0.5, 0, 1)
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  m = _model([helper.make_node("HardSigmoid", ["x"], ["y"])], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"HardSigmoid ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_hardswish_onnx_matches_onnxruntime():  # x * relu6(x+3)/6; HardSwish is opset 14
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  g = helper.make_graph([helper.make_node("HardSwish", ["x"], ["y"])], "g",
                        [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])])
  m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)]); m.ir_version = 9
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"HardSwish ANE vs onnxruntime cosine={cos}"

# -- pooling attrs via native MIL params (ceil_mode, exclude_padding_from_average) -- #
@requires_ane
def test_maxpool_ceil_mode_onnx_matches_onnxruntime():  # ceil sizing must match onnxruntime's extra edge window
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  n = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1)
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 4, 4])])
  got, ref = _run_vs_ort(m, x)
  assert af.load_onnx(m)(x.astype(np.float16)).shape == (1, 4, 4, 4)  # ceil(8-3 /2)+1 = 4, not floor 3
  cos = _cos(got, ref); assert cos > 0.99, f"MaxPool ceil_mode ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_avgpool_exclude_pad_onnx_matches_onnxruntime():  # count_include_pad=0 -> exclude_padding_from_average
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  n = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=0)
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"AveragePool count_include_pad=0 ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_avgpool_include_pad_onnx_matches_onnxruntime():  # count_include_pad=1 -> ANE-native (divide by full kernel area)
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
  n = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1)
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 8, 8])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"AveragePool count_include_pad=1 ANE vs onnxruntime cosine={cos}"

# -- asymmetric conv pad + shape-subgraph folding (Inception, ShuffleNet) ---------- #
@requires_ane
def test_conv_asymmetric_pad_onnx_matches_onnxruntime():  # per-side MIL conv pad order [top,bottom,left,right]
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 3, 10, 10)).astype(np.float32)
  w = _init((rng.standard_normal((4, 3, 3, 3)) * 0.3).astype(np.float32), "W")
  n = helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 0, 3, 0], strides=[1, 1])  # asymmetric: top=1,bottom=3
  m = _model([n], [_vi("x", [1, 3, 10, 10])], [_vi("y", [1, 4, 12, 8])], inits=[w])
  got, ref = _run_vs_ort(m, x)
  assert af.load_onnx(m)(x.astype(np.float16)).shape == (1, 4, 12, 8)
  cos = _cos(got, ref); assert cos > 0.99, f"Conv asymmetric pad ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_slice_channel_split_onnx_matches_onnxruntime():  # activation Slice -> static slice_by_size
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
  st = onnx.numpy_helper.from_array(np.array([4], np.int64), "st")
  en = onnx.numpy_helper.from_array(np.array([8], np.int64), "en")
  ax = onnx.numpy_helper.from_array(np.array([1], np.int64), "ax")
  n = helper.make_node("Slice", ["x", "st", "en", "ax"], ["y"])
  m = _model([n], [_vi("x", [1, 8, 4, 4])], [_vi("y", [1, 4, 4, 4])], inits=[st, en, ax])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.999, f"Slice channel-split ANE vs onnxruntime cosine={cos}"

# -- transformers: LayerNorm + attention block on-device ---------------------------- #
@requires_ane
def test_layernorm_onnx_matches_onnxruntime():
  rng = np.random.default_rng(0); x = rng.standard_normal((1, 16, 8)).astype(np.float32)
  s = _init((rng.standard_normal(8) * 0.5 + 1).astype(np.float32), "s"); bb = _init((rng.standard_normal(8) * 0.1).astype(np.float32), "bb")
  n = helper.make_node("LayerNormalization", ["x", "s", "bb"], ["y"], axis=-1, epsilon=1e-5)
  m = _model([n], [_vi("x", [1, 16, 8])], [_vi("y", [1, 16, 8])], inits=[s, bb])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"LayerNorm ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_attention_block_matches_onnxruntime():  # full transformer attention: LN + QKV + batched attn + softmax + proj
  import torch, torch.nn as nn
  class Attn(nn.Module):
    def __init__(self, d=64, h=4):
      super().__init__(); self.h = h; self.d = d
      self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.ln = nn.LayerNorm(d)
    def forward(self, x):
      b, t, _ = x.shape; x = self.ln(x); qkv = self.qkv(x).reshape(b, t, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]; a = ((q @ k.transpose(-2, -1)) * (1.0 / (self.d // self.h) ** 0.5)).softmax(-1)
      return self.proj((a @ v).transpose(1, 2).reshape(b, t, self.d))
  x = torch.randn(1, 16, 64); p = "/tmp/_aneforge_attn.onnx"
  torch.onnx.export(Attn().eval(), (x,), p, opset_version=17, do_constant_folding=True, input_names=["x"], dynamo=False)
  net = af.load_onnx(p); got = np.asarray(net(x.numpy().astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnxruntime_run(p, x.numpy())).astype(np.float32).ravel()
  cos = _cos(got, ref); assert cos > 0.99, f"attention block ANE vs onnxruntime cosine={cos}"

def onnxruntime_run(path, x):
  import onnxruntime
  return onnxruntime.InferenceSession(path).run(None, {"x": x})[0]

# -- fuse_attention: route softmax(QKᵀ·s[+causal])@V onto the native SDPA layer ----- #
def _export_attn(causal, T=16, d=64, h=4):
  import torch, torch.nn as nn
  class A(nn.Module):
    def __init__(self):
      super().__init__(); self.h = h; self.d = d; self.causal = causal
      self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.ln = nn.LayerNorm(d)
      if causal: self.register_buffer("mask", torch.triu(torch.full((T, T), float("-inf")), 1))
    def forward(self, x):
      b, t, _ = x.shape; x = self.ln(x); qkv = self.qkv(x).reshape(b, t, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]; sc = (q @ k.transpose(-2, -1)) * (1.0 / (self.d // self.h) ** 0.5)
      if causal: sc = sc + self.mask
      return self.proj((sc.softmax(-1) @ v).transpose(1, 2).reshape(b, t, self.d))
  import torch as _t; x = _t.randn(1, T, d); p = f"/tmp/_aneforge_attn_{causal}.onnx"
  _t.onnx.export(A().eval(), (x,), p, opset_version=17, do_constant_folding=True, input_names=["x"], dynamo=False)
  return p, x.numpy()

@requires_ane
def test_fuse_attention_causal_uses_native_sdpa():  # causal mask -> native fused-attention layer (a graph cut)
  p, x = _export_attn(causal=True)
  net = af.load_onnx(p, fuse_attention=True)
  assert getattr(net, "n_netplist", 0) >= 1, "causal attention should route to the native SDPA bridge"
  got = np.asarray(net(x.astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnxruntime_run(p, x)).astype(np.float32).ravel()
  cos = _cos(got, ref); assert cos > 0.99, f"causal fused-attention ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_fuse_attention_noncausal_matches():  # non-causal fuses to the decomposed path, still exact
  p, x = _export_attn(causal=False)
  got = np.asarray(af.load_onnx(p, fuse_attention=True)(x.astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnxruntime_run(p, x)).astype(np.float32).ravel()
  cos = _cos(got, ref); assert cos > 0.99, f"non-causal fused-attention ANE vs onnxruntime cosine={cos}"

# -- quantized (int8) ONNX: QDQ weights dequantize, activation Q/DQ clips (relu fold) -- #
@requires_ane
def test_quantized_qdq_matches_onnxruntime(tmp_path):
  import torch, torch.nn as nn, onnxruntime
  from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
  net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)).eval()
  fp32 = str(tmp_path / "m.onnx"); qdq = str(tmp_path / "q.onnx")
  torch.onnx.export(net, (torch.randn(1, 3, 32, 32),), fp32, opset_version=13, input_names=["x"], dynamo=False)
  class DR(CalibrationDataReader):
    def __init__(self): self.it = iter([{"x": np.random.randn(1, 3, 32, 32).astype(np.float32)} for _ in range(8)])
    def get_next(self): return next(self.it, None)  # type: ignore[override]  # base stub omits the None sentinel
  quantize_static(fp32, qdq, DR(), quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, per_channel=True)
  x = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype(np.float32)
  got = np.asarray(af.load_onnx(qdq)(x.astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnxruntime.InferenceSession(qdq).run(None, {"x": x})[0]).astype(np.float32).ravel()
  cos = _cos(got, ref); assert cos > 0.99, f"quantized QDQ ANE vs onnxruntime cosine={cos}"

# -- Tier 3 cheap ops + Tier 5 half-pixel resize approximation -------------------- #
@requires_ane
def test_tier3_softplus_matches_onnxruntime():
  rng = np.random.default_rng(0); x = (rng.standard_normal((1, 8)) * 2).astype(np.float32)
  m = _model([helper.make_node("Softplus", ["x"], ["y"])], [_vi("x", [1, 8])], [_vi("y", [1, 8])])
  got, ref = _run_vs_ort(m, x)
  cos = _cos(got, ref); assert cos > 0.99, f"Softplus ANE vs onnxruntime cosine={cos}"

@requires_ane
def test_resize_half_pixel_approx_close():  # the opt-in approximation is ~0.99 on a smooth upsample
  rng = np.random.default_rng(0)
  x = np.repeat(np.repeat(rng.standard_normal((1, 4, 4, 4)).astype(np.float32), 2, 2), 2, 3)  # smooth 8x8
  sizes = onnx.numpy_helper.from_array(np.array([1, 4, 16, 16], np.int64), "sz")
  n = helper.make_node("Resize", ["x", "roi", "sc", "sz"], ["y"], mode="linear", coordinate_transformation_mode="half_pixel")
  m = _model([n], [_vi("x", [1, 4, 8, 8])], [_vi("y", [1, 4, 16, 16])], inits=[_empty_f32("roi"), _empty_f32("sc"), sizes])
  got = np.asarray(af.load_onnx(m, approx_resize=True)(x.astype(np.float16))).astype(np.float32).ravel()
  ref = np.asarray(onnx_run(m, x)).astype(np.float32).ravel()
  cos = _cos(got, ref); assert cos > 0.95, f"half-pixel approx Resize ANE vs onnxruntime cosine={cos}"
