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
| Activations | `Relu`, `Sigmoid`, `Tanh`, `Clip` (`(0,6)`->relu6, `(0,inf)`->relu) |
| Elementwise | `Add`, `Sub`, `Mul`, `Div` |
| Convolution / pooling | `Conv`, `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Linear | `Gemm`, `MatMul` |
| Normalization | `BatchNormalization` |
| Shape / layout | `Reshape`, `Flatten`, `Transpose`, `Squeeze`, `Unsqueeze`, `Concat` |
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
- **Elementwise.** `Add`/`Sub`/`Mul`/`Div` are tensor-tensor only; a constant operand
  raises (exporters usually fold these into the adjacent `Conv`/`BatchNormalization`).
