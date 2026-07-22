# ONNX import

Load an ONNX model and run it on the Apple Neural Engine. The importer targets the
CNN-classifier op subset (the ops a torch-exported ResNet/VGG/MobileNet emits), builds
an ANEForge graph from the ONNX node list, and compiles it through the usual e5rt path.

## API

Two entry points, both top-level (`import aneforge as af`):

```python
import aneforge as af
import numpy as np

net = af.load_onnx("resnet18.onnx")          # import + compile -> runnable Model
y = net(np.zeros((1, 3, 224, 224), np.float16))
```

- `af.load_onnx(path, fuse_attention=False, **compile_kwargs)` - import the model and
  compile it to a runnable `Model`. `fuse_attention=True` rewrites the
  `softmax(Q@Kᵀ·scale [+ causal mask])@V` pattern onto `af.sdpa`; a **causal** mask routes
  to the native fused-attention layer (a graph cut), so GPT-style decoder attention runs
  on the dedicated SDPA hardware. Extra keyword arguments pass straight to `af.compile`
  (`target=`, `opt=`, `compress=`, ...). `path` is a filename or an in-memory `onnx.ModelProto`.
- `af.onnx_to_tensor(path)` - import only, returning `(graph_inputs, output)` ANEForge
  tensors so you can inspect or splice the graph before compiling it yourself.
- `af.onnx_to_features(path)` - import a classifier and return `(inputs, features)` where
  `features` is the input to the final linear layer. Compile it for a frozen feature
  extractor and train a fresh head on the ANE for transfer learning (see
  [`examples/onnx_finetune.py`](https://github.com/sbryngelson/ANEForge/blob/main/examples/onnx_finetune.py)).

A runnable end-to-end example (export a torchvision model, import it, validate against
onnxruntime) is in [`examples/onnx_import.py`](https://github.com/sbryngelson/ANEForge/blob/main/examples/onnx_import.py):
`python3 examples/onnx_import.py [model.onnx]`.

## Supported operators

The importer raises `NotImplementedError("ONNX op 'X' not supported")` for anything
outside this set, so an unsupported model fails loudly with the offending op named.

| Category | ONNX ops |
| --- | --- |
| Activations | `Relu`, `Sigmoid`, `Tanh`, `Clip` (`(0,6)`->relu6, `(0,inf)`->relu), `Elu`, `Selu`, `Celu`, `Mish`, `Softsign`, `ThresholdedRelu`, `LeakyRelu`, `PRelu`, `Gelu` (exact/erf only), `Erf`, `HardSigmoid`, `HardSwish` |
| Elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Exp`, `Log`, `Sqrt` (constant operands supported), `Abs`, `Neg`, `Sign`, `Reciprocal`, `Sin`, `Cos`, `Atan`, `Softplus`, `Floor`, `Ceil`, `Round`, `Min`, `Max`, `Sum`, `Mean` (variadic), `Where`, `Tile`, `Shrink` |
| Comparison / logic | `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Not` |
| Convolution / pooling | `Conv`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `GlobalMaxPool` |
| Linear | `Gemm` (full `alpha`/`beta`/`transA`/`transB`), `MatMul`, `Einsum` (matmul-reducible equations) |
| Normalization | `BatchNormalization`, `InstanceNormalization`, `LayerNormalization`, `GroupNormalization`, `RMSNormalization`, `LpNormalization` (p=1/2) |
| Shape / layout | `Reshape`, `Flatten`, `Transpose`, `Squeeze` (incl. squeeze-all), `Unsqueeze`, `Concat`, `Split`, `Expand`, `SpaceToDepth`, `DepthToSpace`, `Slice` (step 1; step -1 as a full-axis flip), `Shape`, `Trilu`, `Pad` (constant/edge/reflect/wrap) |
| Normalization (cross-channel) | `LRN` |
| Resampling | `Resize` (nearest / linear) |
| Reduction | `ReduceMax`, `ReduceMin`, `ReduceSum`, `ReduceMean`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare`, `CumSum` |
| Indexing | `Gather` (static indices), `ArgMax`, `ArgMin`, `TopK` (2D, values only) |
| Quantization | `DequantizeLinear`, `QuantizeLinear` (QDQ int8) |
| Misc | `Softmax`, `LogSoftmax`, `Constant`, `ConstantOfShape`, `Range`, `EyeLike`, `Identity`, `Dropout` (inference no-op), `Cast` (import-level), `OneHot` (constant depth/values) |

Export at `opset_version=13` with constant folding on (the default), which resolves the
`Shape`/`Gather`/dynamic-`Reshape` plumbing into static initializers before import.

## Caveats

- **fp16 compute.** ANEForge casts weights to fp16 at compile and computes in fp16, so a
  match against onnxruntime (fp32) is a **cosine-similarity** check, not bit-exact. A
  torch-exported ResNet-18 matches onnxruntime at cosine ~0.99999 end to end.
- **fp16 input.** `af.input` is fp16, so feed an fp16 array (cast your input with
  `x.astype(np.float16)`).
- **Static shapes only.** Every program compiles for a concrete shape; a dynamic or
  symbolic dim raises at import. Data-dependent value ops (on-engine gather / index by
  tensor data) have no ANE path - see [capabilities.md](capabilities.md).
- **NCHW.** Channels-first layout, matching ONNX's convolution convention.
- **Uniform/symmetric conv params.** Per-axis strides, dilations, and pads must be
  uniform and symmetric; a non-uniform value raises rather than silently mis-lowering.

## Limitations

These attribute forms fall outside the supported subset and raise rather than
mis-lower:

- **Constant weights/params only.** Weight/parameter inputs (Conv/Gemm/MatMul/
  BatchNormalization/InstanceNormalization/PRelu weights, reduce `axes`, TopK `k`,
  Gather indices) must be constant initializers; a data-dependent (computed) weight or
  param has no ANE path and raises `NotImplementedError`.
- **Gemm:** only `alpha=1`, `beta=1`, `transA=0` (`transB` is honored).
- **Conv:** explicit `pads` only (`auto_pad` raises); asymmetric per-side pads are
  supported (mapped to the native per-side MIL conv `pad`).
- **Shape subgraph.** `Shape` and the dynamic-reshape plumbing (`Slice`/`Gather`/`Concat`/
  arithmetic over shape vectors) constant-fold at import on the static input shape, so
  patterns like ShuffleNet's channel shuffle (`Reshape`->`Transpose`->`Reshape`) import as
  static ops. `Slice` on an activation lowers to a static `slice_by_size` (step 1 only).
- **Pooling:** `ceil_mode` (both modes) and `AveragePool` `count_include_pad` (0/1) map
  to the native MIL `ceil_mode` / `exclude_padding_from_average` params; `MaxPool` rejects
  `dilations != 1`.
- **Elementwise.** `Add`/`Sub`/`Mul`/`Div`/`Pow` accept either two tensors or a tensor
  and a constant: a constant operand (scalar, per-channel, or broadcastable) bakes in as
  a fused `const_array` (a MIL `const`, folded to the ANE's gain-offset epilogue for
  affine ops), so `Mul(x, c)`, `Add(x, c)`, `Pow(x, 2)`, and the like need no graph cut.
  This is what makes `HardSigmoid`/`HardSwish` and the scale/shift ops in MobileNetV3 and
  ConvNeXt importable.
- **Quantized (QDQ int8).** A statically-quantized ONNX model imports: an int8 weight
  `DequantizeLinear` folds to its dequantized constant, and an activation
  `QuantizeLinear`/`DequantizeLinear` pair becomes a fp16 `clip` to the quant range (which
  carries any relu/saturation the quantizer fused in). Weights run at fp16 by default;
  pass `compress="int8"` to keep them on the ANE int8 weight datapath. Matches onnxruntime
  on-device (cosine ~1.0).
- **Gelu:** exact erf-gelu only; `approximate="tanh"` raises.
- **Boolean algebra has no ANE path.** `And`/`Or`/`Xor` cannot be implemented: MIL's
  `logical_and`/`logical_or`/`logical_xor` (and general `cast`, which would allow a
  bool->fp16 workaround) do not compile for the ANE backend (see the on-device MIL
  vocabulary sweep). `IsNaN` is also out: fp16 NaN payloads do not survive the
  datapath, so `x != x` reads false on-device. Comparisons and `Not` are the
  supported boolean surface.
- **Cast** folds constants and treats float->float on an activation as identity (the
  engine computes fp16 regardless); float->int truncates toward zero as
  `sign(x)*floor(|x|)`, exact within fp16 integer range.
- **PRelu:** slope is a per-channel initializer (`[C]`, `[C,1,1]`, or scalar, flattened
  to `[C]`); input must be rank>=3 `[N,C,...]`.
- **InstanceNormalization:** `[N,C,H,W]` input with `scale`/`B` initializers `[C]`.
- **LRN:** the ANE `local_response_norm` folds the `alpha/size` scaling internally, so
  ONNX `alpha` maps straight through and `bias` maps to `k` (validated on-device against
  onnxruntime, cosine ~1.0). `size`/`beta` pass through unchanged.
- **DepthToSpace:** `DCR` mode only (the ONNX default, which the ANE op matches exactly);
  `CRD` channel ordering raises.
- **Resize:** `[N,C,H,W]`, opset 11+, target from `sizes` (preferred) or `scales`. Only
  the coordinate conventions the ANE actually matches are accepted, each validated
  on-device (cosine ~1.0); every other config raises rather than silently mis-resize:
    - `mode="nearest"` requires `coordinate_transformation_mode="asymmetric"` and
      `nearest_mode="floor"` (the ANE samples with `floor`; the ONNX default
      `round_prefer_floor` and the other round/ceil modes raise).
    - `mode="linear"` requires `asymmetric` (-> half-pixel-off bilinear) or
      `align_corners`.
    - `half_pixel`/`pytorch_half_pixel` sampling and `mode="cubic"` are **not** matched by
      the ANE resamplers and raise. Note the ONNX default `coordinate_transformation_mode`
      is `half_pixel`, so a `Resize` must set `asymmetric`/`align_corners` explicitly — or
      pass `load_onnx(path, approx_resize=True)` to map a half-pixel `Resize` to the closest
      ANE bilinear (an opt-in approximation, ~0.99 cosine on smooth maps). This is what lets
      segmentation models (FCN, DeepLabV3) — whose final upsample is half-pixel — import.
- **Reductions** (`ReduceMax`/`ReduceMin`/`ReduceSum`/`ReduceMean`): `axes` may be an
  attribute (opset < 18; ReduceSum < 13) or an input initializer (later opsets); an
  absent/empty `axes` reduces over every axis. `keepdims=1` (the default) keeps reduced
  dims as size 1; `keepdims=0` squeezes them afterward. `noop_with_empty_axes=1` with no
  axes (the identity edge) raises. `ReduceMean` and `ReduceMax` are validated on-device
  against onnxruntime (cosine ~1.0).
- **Gather:** static (constant-initializer) integer indices only - a data-dependent
  (tensor) index raises, since on-engine gather-by-data has no ANE path. Indices must be
  scalar or 1-D; a scalar index drops the gathered axis (ONNX rank rule) and a >1-D index
  array raises. Lowers to `slice_by_size`+`concat`; validated on-device (cosine ~1.0).
- **ArgMax:** 2-D `[C,W]` inputs only (the ANE `GlobalArgMinMax` bridge), `keepdims=1`
  default (`keepdims=0` squeezes the axis); `select_last_index=1` is unsupported. Indices
  are fp16-encoded and the bridge cuts the graph, so it is shipped shape-validated only.
- **TopK:** 2-D `[C,W]` inputs, last axis (`axis` in `{-1, 1}`) only. Returns the **values**
  output **only** - ONNX's second (indices) output is unsupported, so a model consuming the
  indices fails when that name is later looked up. `k` is read from the (opset 10+) input;
  `k` in `{3, 4}` is rejected by the ANE itself. Shipped shape-validated only.
