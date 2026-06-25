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

- `af.load_onnx(path, **compile_kwargs)` - import the model and compile it to a runnable
  `Model`. Extra keyword arguments pass straight to `af.compile` (`target=`, `opt=`,
  `compress=`, ...). `path` is a filename or an in-memory `onnx.ModelProto`.
- `af.onnx_to_tensor(path)` - import only, returning `(graph_inputs, output)` ANEForge
  tensors so you can inspect or splice the graph before compiling it yourself.

## Supported operators

The importer raises `NotImplementedError("ONNX op 'X' not supported")` for anything
outside this set, so an unsupported model fails loudly with the offending op named.

| Category | ONNX ops |
| --- | --- |
| Activations | `Relu`, `Sigmoid`, `Tanh`, `Clip` (`(0,6)`->relu6, `(0,inf)`->relu), `Elu`, `LeakyRelu`, `PRelu`, `Gelu` (exact/erf only), `Erf` |
| Elementwise | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Exp`, `Log`, `Sqrt` |
| Convolution / pooling | `Conv`, `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Linear | `Gemm`, `MatMul` |
| Normalization | `BatchNormalization`, `InstanceNormalization` |
| Shape / layout | `Reshape`, `Flatten`, `Transpose`, `Squeeze`, `Unsqueeze`, `Concat`, `SpaceToDepth`, `DepthToSpace` |
| Normalization (cross-channel) | `LRN` |
| Resampling | `Resize` (nearest / linear) |
| Reduction | `ReduceMax`, `ReduceMin`, `ReduceSum`, `ReduceMean` |
| Indexing | `Gather` (static indices), `ArgMax`, `TopK` (2D, values only) |
| Misc | `Softmax`, `Constant`, `Identity` |

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

- **Gemm:** only `alpha=1`, `beta=1`, `transA=0` (`transB` is honored).
- **Conv:** explicit `pads` only - `auto_pad` (`SAME_UPPER`/`SAME_LOWER`/`VALID`) raises.
- **Pooling:** `ceil_mode=0` (floor) only; `AveragePool` rejects `count_include_pad=1`
  and `MaxPool` rejects `dilations != 1`.
- **Elementwise.** `Add`/`Sub`/`Mul`/`Div`/`Pow` are tensor-tensor only; a constant
  operand raises (exporters usually fold these into the adjacent
  `Conv`/`BatchNormalization`). A constant `Pow` exponent (`Pow(x, 2)`) raises.
- **Gelu:** exact erf-gelu only; `approximate="tanh"` raises.
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
      is `half_pixel`, so a `Resize` must set `asymmetric`/`align_corners` explicitly.
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
