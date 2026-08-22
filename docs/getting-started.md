# Getting started

From a fresh checkout to a model running on the Apple Neural Engine in a few
minutes.

## Prerequisites

- Apple Silicon Mac (verified on M5 Pro and M1 Max; other Apple Silicon should work).
- macOS 14+ (verified on 26.5).
- Xcode command-line tools (`xcode-select --install`).
- Python 3.10+ with `numpy`.

The `aneforge` package imports only `numpy`. `torch` / `torchvision` /
`transformers` (and `soundfile` for Whisper) are lazy imports used only by the
pretrained loaders (`af.load`, `af.load_resnet`, `af.load_vit`, `af.load_clip`,
`af.load_gpt2`, `af.load_llm`, `af.load_whisper`, `af.load_onnx`) and the
diffusion examples; install them with the `models` extra.

## Install

```sh
python3.12 -m venv .venv                       # .venv/ is gitignored
.venv/bin/python -m pip install -e .           # core: numpy only
.venv/bin/python -m pip install -e ".[models]" # optional: torch/torchvision/transformers
.venv/bin/python -m pip install -e ".[dev]"    # optional: ruff + pytest
```

## Build the dispatch shim

The on-device runtime loads one small native dylib, built once with clang++:

```sh
sh aneforge/_lib/build.sh    # -> aneforge/_lib/libane_e5rt_dispatch.dylib
```

It compiles against Apple frameworks, so it must run on the Mac. The artifact is
git-ignored; re-run it after a clone and after any pull that touches
`aneforge/_lib/ane_e5rt_dispatch.mm`. The dylib loads lazily, so `import aneforge`
and the off-device tooling work without it; compiling or dispatching to the ANE
raises a build hint until it is built. See [`development.md`](development.md) for
the full build and tests.

## First program

`aneforge` is a graph -> compile -> run API: build a small tensor graph,
`compile` it into ONE fused ANE program (weights packed automatically), and call
the result.

```python
import numpy as np
import aneforge as af

x = af.input((1, 3, 32, 32))            # graph input placeholder
h = af.conv(x, W1, pad=1).relu()        # build the graph from real weights
h = af.conv(h, W2, pad=1).relu()
y = h.mean((2, 3)).reshape(1, C) @ Wfc  # @ streams a weight matrix

net = af.compile(y, compress="int8")    # one fused ANE program
out = net(image)                        # run on the ANE -> np.float32
net.release()
```

Inputs are fed in the order they were created with `af.input`; arrays are cast
to fp16 in, fp32 out. `compile(out, compress="int8")` streams matmul/linear weights
as per-channel int8; `compile(out, compress=...)` selects the weight encoding
(`int8`, `int4`, `sparse`, `blockwise`, or family-aware `auto`) - see
the [API reference](aneforge-api.md#weight-compression).

The op surface covers conv/conv_transpose, matmul/`@`/linear/bmm, activations
(relu/silu/gelu/...), elementwise arithmetic, reductions and norms
(softmax, rms_norm/layer_norm/group_norm/batch_norm, l2_norm), pooling/upsample/
concat/reshape/transpose/pixel_(un)shuffle, and nn helpers (mha,
cross_attention, geglu). A second family of netplist-bridge ops (sdpa,
topk/sort/argmax, point-cloud and rearrange layers) run as native ANE
sub-programs and cut the graph into a `SegmentedModel`.

Pretrained loaders ship too:

```python
embed = af.load("sentence-transformers/all-MiniLM-L6-v2")  # sentence encoder
clf   = af.load_resnet18()                                 # ImageNet classifier
vit   = af.load_vit("google/vit-base-patch16-224")         # vision transformer
llm   = af.load_llm("Qwen/Qwen3-0.6B")                     # decoder LLM (prefill + KV-cache decode)
asr   = af.load_whisper("openai/whisper-base.en")          # speech-to-text, both towers on the ANE
```

Vision (`load_resnet`/`load_vit`/`load_clip`), encoders and rerankers (`load`/`CrossEncoder`),
decoder LLMs (`load_llm`/`load_gpt2`), Whisper (`load_whisper`), and ONNX (`load_onnx`) all load
the same way.

A few more things reachable from the same API:

```python
# Feed raw 8-bit pixels - uint8->fp16 dequant runs on the engine.
x = af.image_input((1, 3, 224, 224))         # scale=1/255, bias=0 by default

# Smaller weights: 4-bit LUT, accuracy-gated; or 'auto' (family-aware).
net = af.compile(y, compress="int4")

# Compile / gate for another ANE family from this host.
net = af.compile(y, target="h16s")           # lower + gate ops for M4

# Measurement-free latency + peak projection for any chip.
us  = af.estimate(y, target="h13")           # microseconds on M1
pk  = af.project_peak("h17s")                # M5 fp16 peak {tflops, ...}
```

Worked demos live in [`examples/`](https://github.com/sbryngelson/ANEForge/tree/main/examples) - start with
`examples/quickstart.py` (a CNN and a transformer encoder block), then
`resnet18.py`, `sentence_embeddings.py`, `sdpa.py`, `native_ranking.py`, and
`pointcloud.py`. For pretrained models see `vit.py`, `clip_zero_shot.py`, `gpt2.py`,
`llm_chat.py`, `whisper.py`, and `rag_chat.py`. The chapter-aligned mechanism demos are in `examples/demos/`.

## MIL input format

The frontend emits a small subset of Apple's MIL (Model Intermediate Language),
the same textual IR `coremltools.convert(convert_to='mlprogram')` produces. A
minimal program:

```mil
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"},
                                    {"coremlc-version", "3520.5.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 4]> x) {
        tensor<fp16, [1, 4]> relu = relu(x = x)[name = string("relu")];
    } -> (relu);
}
```

You rarely write MIL by hand - the frontend generates it - but the
[MIL primer](mil-primer.md) covers it.

## Next

- [`aneforge-api.md`](aneforge-api.md) - the full `aneforge` frontend reference
- [`training.md`](training.md) - train a model on the ANE
- [`cross-chip.md`](cross-chip.md) - compile / gate for another ANE family
- [`dispatch.md`](dispatch.md) - which dispatch path to use when
- [`capabilities.md`](capabilities.md) - operator coverage
