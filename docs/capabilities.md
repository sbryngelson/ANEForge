# Capabilities

What ANEForge can and can't do, by category. For the complete per-op, per-chip
table see [op-catalog.md](op-catalog.md).

## Coverage summary

ANEForge classifies the full **187-op MIL vocabulary** against a machine-checkable
registry (`aneforge/_capabilities.py`, serialized to `capabilities.json` and
CI-gated). Each op falls into one of a few practical buckets:

| status | meaning |
| --- | --- |
| `fused` | Frontend primitive - lowers to ANE-MIL and fuses into one program |
| `bridge` | Native sub-program reached through a graph cut |
| `reachable` | Compiles and runs on the e5rt path, not yet a named frontend primitive |
| `walled` | A genuine native-op or codegen gap (design around it) |
| `not-authorable` | Control-flow / list / state ops with no feed-forward form |

Query reachability programmatically with `af.op_info(op)`,
`af.device_status(op, chip)`, `af.is_native(op, chip)`, `af.ops_on(chip, status)`,
`af.min_native_family(op)`, and `af.walled_everywhere()`.

## Datatypes

| dtype | I/O accepted | Compute | Notes |
| --- | --- | --- | --- |
| `fp16` | Y | Y | Native ANE dataplane |
| `int8` | Y | N (cast required) | Quantized weights via `constexpr_affine_dequantize` |
| `uint8` | Y | N | Pixel / index input lanes |
| `int16` | Y | N | Slow compile (~35 ms) |
| `uint16` | Y | N | |
| `fp32` | N | Y (intermediate) | Use `cast(x, fp32)` -> compute -> `cast(result, fp16)` |
| `int32` | N | N | Rejected at MIL parse |
| `bf16` | N | N | No path |
| `bool` | N at I/O | Y (computed) | Produced by comparison ops, consumed by `select` |

The hard precision limits to design around: I/O is fp16/int8/uint8/int16/uint16;
fp32 is available only as a compute intermediate (cast in, cast out); int32, bf16,
and boolean tensor I/O have no path.

## Dimension bounds

Tensor dimension caps are **per-family and per-op-class** - the limit rides the
op's lowering, not the tensor, so a graph that fits an M5 may overflow an M1.
Practical caps:

- **Spatial / contraction** extents (flat W/H, matmul-K, conv spatial): 16384
  through A15, 65536 at A16.
- **Channel** axis: 65536 on both A14 and A16.
- **Transpose** extent: very large (offset-field width), effectively unbounded.
- **Conv kernel width**: `kW <= 13` on M1, `<= 15` on M5 (route wider kernels via
  `space_to_depth`).

`tg.limit("max_tensor_dim" | "channel_extent" | "transpose_extent", family)`
(where `from aneforge import _targets as tg`) returns the measured caps, and
`tg.preflight` classifies each op's axes (including internal reshape extents like
group_norm's `[1, G, C/G, H*W]`) so you catch an overflow before the compile. Caps
are generation-monotone (A13 == A14 <= A16). See [cross-chip.md](cross-chip.md).

## Shapes (static only)

Every program compiles for a **concrete** shape. Variable-shape (symbolic) programs
are not reachable through the e5rt path ANEForge uses: a symbolic dim
parses but fails to compile. For variable-length (for example LLM sequence)
inference, either pad to a fixed maximum and compile once, or bucket a small set of
lengths and dispatch the nearest (the compile cache makes repeat lengths free).

This is shape polymorphism only - a separate axis from data-dependent *values*,
which remain unreachable (no on-engine gather / index-by-tensor-data), so a
dynamic-slice KV-cache is likewise off the e5rt path.

## Operator surface

For the complete native MIL op x device (M1-M5) table, see
[op-catalog.md](op-catalog.md). The notes below cover the practical details that
table does not carry: per-family floors, the quantized / compressed / dynamic-kernel
paths, group-norm, and the attention routes.

### Per-family op floors

A handful of operators are native only at or above a given capability family;
`tg.op_status(op, family)` (with `from aneforge import _targets as tg`) returns
`native` / `decompose` / `reject`, and `compile(target=...)` substitutes or rejects
accordingly. The family-gated ops are:

- `sin` / `cos` - A15+; decompose to a portable fp16 polynomial on M1 via
  `aneforge/special.py`.
- texture-engine ops `crop_resize` / `resample` / `affine` / `gather_hw` - A14+.
- `sq_after_reduce` - A14+.
- `dropout` / `random` - A15+.
- `global_argmax` / `global_argmin` - A15+.
- `topk` / `sort` / `dynamic_slice` - reject at codegen on M1 (family 2).

Everything else (conv, matmul, pooling, elementwise, activations, softmax, norms,
reductions, sqrt/rsqrt/erf/exp2/log2, SDPA, resize, tile, space<->channel, atan) is
native on M1 and up. See [cross-chip.md](cross-chip.md).

### Linear (quantized variants)

Beyond `matmul` / `linear` / `inner_product` / `batch_matmul`, the quantized macros
are `quantized_linear` (`dynamic_quantize -> matmul -> dynamic_dequantize`),
`dynamic_quantize_only`, and `dynamic_dequantize_only`. The working int8 substitute
is `matmul(constexpr_affine_dequantize(int8_blob, scale))`; both per-tensor and
per-channel scale work.

### Convolution (fused + quantized + dynamic-kernel)

The base `conv` (1D/2D/3D) and `conv_transpose`/`deconv` are in op-catalog.md;
ANEForge adds fused composites (`conv1x1`, `conv1x1_chain`, `conv_relu`,
`conv_gelu`, `conv1x1_batch_norm_relu`, `conv1x1_batch_norm_silu`,
`conv1x1_add_relu`, `conv1x1_project_add`) and a quantized chain
`conv1x1_int8_chain`.

**Dynamic-kernel conv** (`af.dynamic_conv`): a conv whose weight is a runtime
**input tensor** rather than a baked constant, enabling hypernetwork /
weight-generating inference. Reachable and correct at **batch 1** only;
**batch >= 2** does not compile, so `af.dynamic_conv` rejects `B >= 2`
at build time. A constant-weight conv at any batch is unaffected. (This is why the
trainable conv uses an im2col path rather than a native dynamic-weight conv.)

### Normalization

`layer_norm` / `instance_norm` / `l2_norm` are in op-catalog.md; ANEForge exposes
`group_norm` and `rms_norm` as composites. The channel-axis layer-norm is supported
as a fused "transpose -> instancenorm_1d -> transpose" route.

`group_norm` lowers at rank 4 (`[1, G, C/groups, H*W]`) with a chained reduce, so
every axis stays under the per-axis cap. This removes the former large-feature-map
wall: the Stable Diffusion 1.5 wall shapes (640ch@64, 512ch@128) now compile and
run (relerr ~ 0.002 vs fp32), and the unblock is family-wide.

### Attention

Three routes to attention, with different tradeoffs:

| Path | Use when |
| --- | --- |
| `af.sdpa(q, k, v)` (native fused SDPA) | Any shape; runs fused attention on the ANE, including `is_causal=True` and the decode shape (`seq_q != seq_kv`) |
| MIL `scaled_dot_product_attention` | Apple's lowering; ANE-routes only above `heads >= 64, seq >= ~496, d = 64` |
| Manual `matmul -> softmax -> matmul` | Masked attention; always works at any shape |

`af.sdpa` is the recommended route. Causal masking is native end-to-end
(`is_causal=True`), validated at cos 1.0 versus a masked-softmax reference. It is
reliable for sequence length `S <= 2048` (`SDPA_NATIVE_MAX_SEQ`); above that the op
decomposes (which carries no mask). Because it also handles the decode shape, a full
autoregressive GPT/LLaMA generation loop (causal-SDPA prefill, then per-step
decode-shape SDPA) runs on the engine token-for-token matching numpy
(`examples/gpt_generate_ane.py`).

**Resident KV-cache decode.** The decode KV-cache can stay resident on the engine
across steps so it never round-trips to the host: the masked positional write runs
in the graph, `compile_multi` emits the hidden state plus every cache output, and
`Program.share_buffer` aliases each cache output onto its own input. Works for a
single layer (`TinyDecoderANE.generate_resident`) and for a full L-layer decoder
with all `2L` caches resident (`examples/gpt_multilayer_resident.py`). On this path
the decode attention is decomposed (cheap at `seq_q=1`) since `compile_multi` cannot
take the native-SDPA graph cut; the resident-cache bandwidth saving is the goal.

For masked attention generally, use the manual decomposition with a causal mask via
comparison + `select` (`ane.causal_softmax`).

### Softmax

`softmax` is native; ANEForge adds the `softmax_nd`, `log_softmax`,
`causal_softmax`, and `threshold_softmax` composites.

### Layout (gotchas)

Layout ops (`reshape`, `transpose`, `flatten`, `unflatten`, `tile`, `pad`, `slice`,
`gather`, `concat`, `split`, `squeeze`, `expand_dims`) are in op-catalog.md with
per-chip status. Two gotchas to know: `flatten` is NCHW-only, and `concat` expects
exactly 2 inputs (more than 2 need decomposition).

### Activation modes

ReLU, Tanh, Sigmoid, LeakyReLU, ELU, ReLU6, GELU (exact), and `clip(alpha, beta)`
are all available as native activations.

### Specialized fused routes

`fused_qkv` (QKV projection in one call), `fused_qkv_norm_proj`,
`dynamic_matmul_packed`, `dynamic_matmul_packed_relu`, `goc_scalar_scale_bias`, and
the `dynamic_goc_*` family.

## Image input

`af.image_input(uint8)` accepts a byte image and dequantizes it on the engine,
byte-identical to a host-side convert and saving the host conversion latency
(~2 ms/frame at 1080p). This is the terminal image-input form.

Direct 4CC interchange input (a pixel buffer fed straight from a camera or video
surface with no host RGB convert) is a **no-go on the e5rt path** - it needs
the entitled CoreML route. Use `af.image_input(uint8)` instead.

## Compressed weight streaming

`af.compile(out, compress=None | "int8" | "int4" | "sparse" | "blockwise" | "auto")`
emits compressed weights that **stream** (dequant-during-DMA) rather than fold to
dense fp16, so a weight-bandwidth-bound op gets a real eval-latency win, not just a
smaller file. `compress=None` (the default) is byte-identical to fp16. int4-LUT and
sparse are accuracy-gated (int4 falls back int4->int8->fp16 within `compress_atol`).
All `constexpr_*` quant forms, including blockwise-affine, are reachable on the e5rt path.

Which formats stream natively is **per-family** (`tg.native_streams(family)`, with
`from aneforge import _targets as tg`):

| family / chip | int4-LUT | int8-affine | sparse | blockwise |
| --- | --- | --- | --- | --- |
| A13 (M1) | stream (2.37x) | fold | stream | fold |
| A14 (M2) | stream | stream | stream | fold |
| A15+ (M3, M4, M5) | stream | stream | stream | stream |

So `compress="auto"` is family-aware: on M1 it considers int4-LUT and sparse (the
native streams there) and skips int8/blockwise, while a budget-rejected int4 falls
back to fp16 rather than a folding encoding (which costs accuracy for zero bandwidth
win). Explicit single-mode knobs are never filtered. End to end, compression is
primarily a **footprint/capacity** lever (~4x smaller weights); the per-matmul win
dilutes through norms, attention, and dispatch in full models.

## Training (on-engine autograd)

ANEForge trains on the ANE: `aneforge/autograd.py` runs forward, backward, and the
optimizer update as ANE graph ops through the same e5rt path.
`af.parameter` makes a weight a graph *input* (no recompile per step); a VJP
registry builds the backward graph. `Trainer` (with `device_optimizer`,
`resident_state`) and `UnrolledTrainer` (K steps in one fused program, resident by
default) drive training; full MNIST reaches 97.79% with optimizer state resident
across steps. The gradient vocabulary covers MLPs, CNNs (trainable conv built from
primitives, since native conv needs a baked weight), and transformers. fp16
optimizer state suffices; paired-fp16 is not needed.

One codegen wall to know: `mul(reduce_output, 0.0)` (a reduce result times exactly
zero) fails to compile and is sidestepped with a subtract-based zero. See
[training.md](training.md).

## Cost estimation (measurement-free)

`af.estimate(out, target='hXX')` returns an analytic per-chip latency from the
compiler's own `cycles -> roofline -> wall-time` model
(`latency ~ overhead + max(compute, memory)`), and `af.project_peak(arch)` gives a
generational fp16-peak. Both cover all targets from one extraction (no on-device
measurement) and are validated on the two measured chips (M1: 3.25 TFLOP/s, 9 GB/s,
220 us floor; M5: 8.9 TFLOP/s, 57 GB/s, 110 us floor). The estimator backs the
lossless `opt='routes'` pass. `project_peak`'s ~5x M5/M1 is a peak-compute ceiling;
measured typical latency speedup is 2.3-3.3x.

## Operators with no hardware backing

Some operators have no ANE hardware support and either reject outright or require
algebraic decomposition over supported atoms:

| Op | Status | Notes |
| --- | --- | --- |
| `non_zero` | Hardware-blocked | No decomposition |
| `sliding_windows` | Hardware-blocked | No decomposition |
| `reduce_prod` | Hardware-blocked | No decomposition |
| `cumsum` | Hardware-blocked | No decomposition |
| `bitwise_and/or/xor/not` | Rejected | Bit-level ops not in fp16 surface |
| Generic boolean tensor I/O | Rejected | Bool tensors are compute-only |
| trig/hyperbolic `acos`/`asin`/`sinh`/... | Rejected | No hardware support |
| logical `and`/`or`/`xor` | Rejected | No hardware support |
| `nan_to_num` | Broken | ANE does not propagate NaN |

`non_maximum_suppression` is reachable as a MIL op (presence-only) but offloads to
CPU/GPU rather than running on the ANE, so do not count it as an ANE-native layer.

The image-pipeline ops have nuanced placement: `affine`, `resize_bilinear`, and
`upsample_bilinear` ship as fused, on-engine ops; `crop_resize` stays
not-implemented-on-ANE.

## Quick verification reference

- Direct dispatch reaches ANE silicon (~1.4 W on the ANE rail under powermetrics),
  not a silent CPU fallback.
- Operator outputs are bit-equal (or within fp16 noise) to numpy references.
- Per-call eval is ~80-110 us at the C-API level once compiled.
- The e5rt event graph provides true happens-before serialization across streams.

To confirm ANE placement rather than a silent CPU fallback, watch the ANE power rail
under `powermetrics` while a compiled model runs: a genuine dispatch draws ~1.4 W on
that rail (a CPU fallback does not), which is the same check the verification suite
uses.
