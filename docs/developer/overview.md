# Architecture overview

ANEForge is a clean `graph -> compile -> run` frontend for the Apple Neural Engine (ANE). You build a small tensor graph, `compile` it, and run it on the ANE — wrapping only the unentitled Espresso `e5rt` runtime, with no CoreML and no entitlement required.

## The model

```python
import aneforge as af

x = af.input((1, 3, 32, 32))
h = af.conv(x, W1, pad=1).relu()
h = af.conv(h, W2, pad=1).relu()
y = h.mean((2, 3)).reshape(1, C) @ Wfc
net = af.compile(y, int8=True)        # one fused ANE program
out = net(image)                      # run on the ANE
```

`graph.py` records a lazy `Tensor` whose methods and operators capture structure (op + sources + attrs) but never touch the device. `compile` (`_compile.py`) lowers that graph to one program; `_blob.py` packs weights; `autograd.py` provides on-ANE training; `models.py` ships pretrained loaders.

### Why fusing is the point

The ANE penalizes many tiny dispatches, so a whole subgraph is compiled into a **single** fused `e5rt` program. Weights pack automatically into one BLOBFILE — fp16, or per-channel int8 *streamed* (dequantized during the tile DMA) when `int8=True`.

## The two op routes

Every op takes one of two routes through the compiler.

| Route | What it is | Graph effect |
|---|---|---|
| **Fused e5rt-MIL** | Most ops. They lower to MIL and fuse into one program. | No graph cut. |
| **Netplist-bridge** | Native Path-A hardware layers Apple's MIL frontend never emits. | Each bridge node **cuts** the graph. |

For a fused-MIL op, lowering is the whole story: the op becomes part of the single program. A netplist-bridge op is different — Apple's user-space MIL compiler never emits these hardware layers at all, so ANEForge hand-authors a native ANECIR netplist for them. Each bridge op cuts the graph: surrounding regions run as `e5rt` programs, the bridge node runs as a separate native sub-program (sub-millisecond via the A2 persistent worker), and `compile` returns a `SegmentedModel`.

The bridge family includes `sdpa`, `argmax`/`topk`/`sort`, `cross_product`/`cross_correlation`/`cost_volume`, `fps`/`radius_search`, `minmax_norm`/`lrn`, the space/channel/batch rearranges, and `flatten`/`input_view`/`dynamic_slice`/`scaled_elementwise`. See [Native bridges and segmentation](bridges.md) for the full model and [Per-op ANE quirks](op-quirks.md) for the hardware limits.

!!! note "No CoreML, no entitlement"
    The dispatch path wraps the unentitled Espresso `e5rt` runtime only. The bridge invokers link nothing but system frameworks, so a plain `pip install` works on any Apple Silicon Mac with the Xcode command-line tools.

## Op surface

- **Linear algebra:** `conv`, `conv_transpose`; matmul/linear via `@`; `bmm`.
- **`dynamic_conv`:** conv with a *runtime-tensor* weight — the kernel is a graph value, not a baked constant. Lowers to the ANE's native dynamic-kernel path (`CreateDynamicKernel`/DynamicGOC), enabling hypernetworks and per-sample (per-image) kernels — a capability no other ANE frontend exposes, since Apple's MIL/CoreML conv bakes the weight. **Batch must be 1**; for batched convolution use `af.conv` or the im2col-based trainable `conv2d`.
- **Activations:** `relu`/`silu`/`gelu`/`sigmoid`/`tanh`/`exp`/`log`/`sqrt`/`rsqrt`/`abs`/`square`/`sin`/`cos`/`erf`/`softplus`/`relu6`/`elu`/`leaky_relu`/`clip`.
- **Arithmetic:** `add`/`sub`/`mul`/`div`(`/`)/`maximum`/`minimum`/`pow`.
- **Reductions/norms:** `mean`/`sum`/`amax`/`amin`, `softmax`, `l2_norm`, `rms_norm`/`layer_norm`/`group_norm`/`batch_norm`.
- **Spatial/shape:** `max_pool`/`avg_pool`, `upsample`, `concat`, `reshape`/`transpose`, `pixel_shuffle`/`pixel_unshuffle`.
- **NN helpers:** `mha`, `cross_attention`, `geglu`.

### Compositions (no native op)

Some ops have no ANE hardware layer and are built from primitives, made exact by the wide accumulator:

- **`cumsum`** (last axis): the ANE has no native cumsum, but a last-axis cumsum is exactly `x @ triu_ones` — a matmul with a baked upper-triangular-ones weight. For other axes, transpose the target axis to last first.
- **`gather`** (static indices): no native gather, but a constant-index gather is exact via `slice_by_size` + `concat`. Dynamic (data-dependent) indices are not reachable on the ANE.
- **`geglu`:** split the `[2*Dff, D]` projection into value/gate halves at build time (no slice op), `out = value * gelu(gate)`.

## Image input

`af.image_input(shape, scale=1/255, bias=0.0)` declares a uint8 input port and dequantizes it on the engine, so raw camera or decoded-video bytes feed the model directly and the host skips the float-convert/repack. The dequant is `cast(uint8->fp16) -> mul(scale) -> add(bias)`; identity add/mul are dropped, so the common `scale=1/255, bias=0` case is a cast plus one mul. `scale`/`bias` are scalar or per-channel (length-C, broadcast as `[1,C,1,1]` over NCHW).

## Pretrained loaders

- `af.load(".../all-MiniLM-L6-v2")` — sentence encoder (`CrossEncoder` for rerankers).
- `af.load_resnet18()` / `af.load_vit(...)` / `af.load_clip(...)` — vision classifiers and CLIP.
- `af.load_gpt2(...)` / `af.load_llm(...)` — decoder LLMs (prefill + resident-KV-cache decode).
- `af.load_whisper(...)` — Whisper speech-to-text, both towers on the ANE.
- `af.load_onnx(...)` — import an ONNX model.

## Numerics

Compute is **fp16 only** (fp32/int32/bf16 are rejected — not implemented on the backend). Reductions and matmuls use a **wide (fp32-class) accumulator** fed by radix-4 fp16-rounded input tiles, so representable sums are near-exact: a sum/dot of 16384 ones is bit-exact (naive fp16 stalls at ~2048), and a `+1` survives next to a 16000 partial that an fp16 running sum would swallow. The fp16 limit sits at the products and the I/O cast, not the running sum, so cancellation-heavy reductions still lose precision.

## Compression knobs

`compress=` chooses the weight encoding:

| Value | Encoding |
|---|---|
| `None` (default) | fp16 |
| `'int8'` | per-channel int8 (streamed at half the bytes; `int8=True` is the alias) |
| `'int4'` | 4-bit LUT palettization, per-tensor, accuracy-gated fallback to int8/fp16 via `compress_atol` |
| `'sparse'` | unstructured bitmask, emitted when the weight is >=50% zeros, else fp16 |
| `'auto'` | per-weight: sparse if sparse, else int4 if accurate, else int8, else fp16 |

## Training on the engine

`autograd.py` provides a tiny reverse-mode autograd: `af.parameter` / `af.backward` / `af.mse` / `af.SGD` / `af.Trainer` train a small model with both forward and backward passes compiled and run on the ANE. For classification, `af.softmax_cross_entropy` (analytic fp16-stable on-ANE gradient) plus `af.Adam` train a 784->128->10 MLP on MNIST to ~97% test accuracy.

With `Trainer(..., device_optimizer=True)` the **optimizer step also runs on the ANE** (SGD/Adam update as graph ops), so all training tensor-math is on the engine; the host only computes the scalar `lr_t` and shuttles state/grads. See `examples/train_mnist_mlp.py`.

### Depth-independent training

A monolithic compile fuses the whole forward, backward, and optimizer step into one program, so compile time grows superlinearly with depth. When layers are structurally identical (a transformer stack, a deep MLP), `CheckpointedStack` compiles the per-layer forward and per-layer backward **once** and reuses them for every layer, so compile cost does not grow with depth. The backward is the standard gradient-checkpointing trick — store only each layer's input activation and recompute its forward inside its backward — and the result is bit-identical to a monolithic `backward`. The optimizer runs host-side over the streamed gradients.
