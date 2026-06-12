# aneforge - frontend API reference

`aneforge` is a clean **graph -> compile -> run** frontend for the Apple Neural
Engine. You build a small tensor graph from Python, `compile` it into ONE fused
e5rt program, and call the result on the ANE. Fusing is the point: the ANE
penalises many tiny dispatches, so a whole subgraph becomes a single program.

This page is the API reference. For the underlying hardware op surface and dtype
limits see [`capabilities.md`](capabilities.md).

```python
import numpy as np
import aneforge as af

x = af.input((1, 3, 32, 32))            # graph input placeholder
h = af.conv(x, W1, pad=1).relu()        # build the graph
h = af.conv(h, W2, pad=1).relu()
y = h.mean((2, 3)).reshape(1, C) @ Wfc  # @ streams a weight matrix

net = af.compile(y, int8=True)          # one fused ANE program
out = net(image)                        # run on the ANE -> np.float32
net.release()
```

- Inputs are fed to the compiled model **in the order they were created** with
  `af.input`. Arrays are cast to fp16 on the way in; outputs come back fp32.
- Weights (for `conv`, `@`, `linear`, norms, ...) are NumPy arrays passed at build
  time, packed into one weight blob - float dtype only (fp16/fp32/fp64 accepted;
  int32/bf16 rejected, the ANE dataplane is fp16).
- `compile(out, int8=True)` streams matmul/linear weights as per-channel int8
  (dequantised during the tile DMA - half the bytes, same fused program). `int8=True`
  is the alias for `compress='int8'`; see [weight compression](#weight-compression)
  for the full set of encodings (`int4`, `sparse`, `blockwise`, `auto`).

## Image input

`af.image_input(shape, scale=1/255, bias=0.0)` declares a **uint8** input port and
dequantises it on the engine: `cast(uint8->fp16) -> mul(scale) -> add(bias)` run as
in-graph ANE ops, so raw camera / decoded-video bytes feed the model directly and
the host skips the float-convert + repack. `scale` / `bias` are scalars (the usual
`x/255` normalisation) or length-`C` sequences for per-channel NCHW normalisation
(broadcast as `[1,C,1,1]`). The result is a normal fp16 `Tensor` for the rest of the
graph; it is byte-identical to converting on the host.

## Two op routes

Ops come in two families, decided automatically at `compile` time:

1. **Fused e5rt-MIL ops** lower to MIL and fuse into a single ANE program - no
   graph cut. These are the default and cover most networks.
2. **Netplist-bridge ops** run as native Path-A sub-programs that Apple's MIL
   frontend never emits (e.g. fused SDPA, TopK, point-cloud layers). The graph
   is **cut** around each one: the surrounding fused regions run as e5rt
   programs and the bridge node runs as a separate native-ANE sub-program, with
   tensors threaded between stages as host fp16 arrays. The presence of any
   bridge op turns `compile` into a `SegmentedModel`.

---

## Fused e5rt-MIL ops

These build/fuse into one program. Tensor-method ops are called as `t.op(...)`;
free functions are `af.fn(...)`.

### Linear algebra

| Op | Signature | Notes |
| --- | --- | --- |
| matmul | `x @ W` | `W` a 2D weight array (streamed), or another `Tensor` for activationxactivation (-> `bmm`). |
| linear | `x.linear(W, bias=None)` | `x @ W.T (+ bias)`, `W` is `[out, in]` (PyTorch layout). |
| bmm | `x @ Wt` | batched/activation matmul when the right operand is a `Tensor`. |
| conv | `af.conv(x, weight, stride=1, pad=0, dilation=1, groups=1, bias=None)` | 2D conv; `x`:[N,Cin,H,W], `weight`:[Cout,Cin/groups,kH,kW]. |
| conv_transpose | `af.conv_transpose(x, weight, stride=1, pad=0, dilation=1, groups=1, bias=None)` | 2D deconv; `weight`:[Cin,Cout,kH,kW] (PyTorch `ConvTranspose2d`). |

### Elementwise - unary

Parameter-free: `relu`, `silu`, `gelu`, `sigmoid`, `tanh`, `exp`, `sqrt`,
`rsqrt`, `abs`, `square`, `sin`, `cos`, `erf`, `softplus`, `relu6` - called as
`t.relu()`, `t.gelu()`, ...

With a parameter: `t.log(eps=0.0)`, `t.rsqrt(eps=0.0)`, `t.elu(alpha=1.0)`,
`t.leaky_relu(alpha=0.01)`, `t.clip(lo, hi)`.

### Elementwise - binary

`a + b`, `a - b`, `a * b`, `a / b` (operators), plus `a * scalar`, and the free
functions `af.maximum(a, b)`, `af.minimum(a, b)`. (`pow` is reachable through
the emitter.) Binary ops broadcast NumPy-style and take two graph `Tensor`s - use weights via `@` / `linear`, not `*`.

### Reductions & normalisation

| Op | Signature | Notes |
| --- | --- | --- |
| mean / sum / amax / amin | `t.mean(axes)` etc. | keepdims; `axes` int or tuple. |
| softmax | `t.softmax(axis=-1)` | |
| l2_norm | `t.l2_norm(axis=-1, eps=1e-12)` | `x / sqrt(sum(x**2, axis) + eps)`; fused (`reduce_l2_norm` + `real_div`). |
| rms_norm | `t.rms_norm(gamma, eps=1e-5)` | RMSNorm over last dim; `gamma`:[D]. |
| layer_norm | `t.layer_norm(gamma, beta, eps=1e-5)` | over last dim, 2D `[M,D]` inputs. |
| group_norm | `t.group_norm(gamma, beta, num_groups, eps=1e-5)` | over `[1,C,H,W]`, `C % groups == 0`. Rank-4 tiled lowering removes the former large-feature-map cliff: SD-1.5 wall shapes (e.g. 640@64, 512@128) compile and run family-wide. |
| batch_norm | `af.batch_norm(x, gamma, beta, mean, var, eps=1e-5)` | inference-mode, per-channel from running stats; rank >= 3 `[1,C,...]`. |

### Spatial & shape

| Op | Signature | Notes |
| --- | --- | --- |
| max_pool | `t.max_pool(k, stride=None, pad=0)` | `[N,C,H,W]`. |
| avg_pool | `t.avg_pool(k, stride=None, pad=0)` | `[N,C,H,W]`. |
| upsample | `t.upsample(scale=2)` | nearest-neighbour `[N,C,H,W]`. |
| concat | `af.concat(tensors, axis=1)` | e.g. UNet skip connections. |
| reshape | `t.reshape(*shape)` | |
| transpose | `t.transpose(perm)` | |
| pixel_shuffle | `af.pixel_shuffle(x, r)` | depth->space `[N,C*r*r,H,W] -> [N,C,H*r,W*r]`. |
| pixel_unshuffle | `af.pixel_unshuffle(x, r)` | space->depth (inverse). |

### NN helpers

| Op | Signature | Notes |
| --- | --- | --- |
| mha | `af.mha(x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, n_heads)` | multi-head self-attention on `x`:[S,D]; built from fused graph ops (split -> SDPA -> concat -> out-proj). |
| cross_attention | `af.cross_attention(x, context, Wq,Wk,Wv,Wo, n_heads, bq=..., bk=..., bv=..., bo=...)` | queries from `x`, keys/values from `context` (SD UNet text conditioning). |
| geglu | `af.geglu(x, W, b)` | GEGLU FFN gate `value * gelu(gate)`, weight split at build. |

---

## Netplist-bridge ops (native Path-A sub-programs)

Each of these **cuts the graph**: it runs as its own native-ANE sub-program,
accelerated by the persistent Path-A worker (see below). Get exact shape constraints
and arch-gated rejections from `aneforge/graph.py`; the highlights:

| Op | Signature | Notes / constraints |
| --- | --- | --- |
| sdpa | `af.sdpa(q, k, v, scale=None, is_causal=False, attn_mask=None)` | native fused attention; `q/k/v`:[1,heads,seq,d], batch 1. `is_causal=True` is **native** (causal mask on the SDPA layer's 5th bottom; cos 1.0 vs masked-softmax), as is a runtime additive `attn_mask` (a single shared `[1,1,Sq,Skv]` plane). The limit is sequence length: above `SDPA_NATIVE_MAX_SEQ` (S <= 2048) the op decomposes, and the decomposition carries no mask. |
| argmax | `t.argmax(axis=-1)` | 2D `[C,W]` only, over Width (axis 1) or Channel (axis 0); fp16-encoded index (exact for index < 2048). |
| topk | `af.topk(x, k, largest=True)` | 2D `[C,W]`, per-row. `k in {3,4}` is arch-gated (rejected). |
| sort | `af.sort(x, descending=False, return_indices=False)` | 2D `[C,W]`, per-row along Width. |
| cross_product | `af.cross_product(a, b)` | 3-vector cross product; both inputs 3 elements -> `(3,)`. |
| cross_correlation | `af.cross_correlation(x, template)` | valid (no-flip) 2D correlation; `x`:[H,W], `template`:[Th,Tw]. |
| cost_volume | `af.cost_volume(aux, ref, disparity_range=1)` | L1 stereo/flow cost; `ref` width >= `aux` width + range -> `(R+1, Wa)`. |
| fps | `af.fps(points, k)` | furthest-point sampling; `points`:[N,3] -> `[k,3]`. **L2-only**; `k <= 1024`, `N <= 8192`. |
| radius_search | `af.radius_search(points, centroids, radius)` | L2 ball-query membership; `[N,3]`,`[Nc,3]` -> `[N,Nc]` 0/1. |
| minmax_norm | `af.minmax_norm(x, dimension="Width", eps=1e-4)` | `[1,C,H,W]`; `dimension in {Width, Height}` (`Channel` arch-gated). |
| lrn | `af.lrn(x, alpha=1.0, beta=0.75, k=1.0)` | cross-channel local response norm over all C; `[1,C,H,W]`. |
| space_to_channel | `af.space_to_channel(x, r)` | TF `space_to_depth` (block-major); `[N,C,H*r,W*r] -> [N,C*r*r,H,W]`. |
| channel_to_space | `af.channel_to_space(x, r)` | TF `depth_to_space` (inverse). |
| space_to_batch | `af.space_to_batch(x, bh, bw)` | `[N,C,H,W] -> [N*bh*bw,C,H/bh,W/bw]`; grows the batch dim. |
| batch_to_space | `af.batch_to_space(x, bh, bw)` | inverse; input batch must be divisible by `bh*bw` (arch-gated). |
| flatten | `af.flatten(x)` | native NCHW flatten; 3D `[C,H,W]` input -> 1-D. |
| input_view | `af.input_view(x, offset, size)` | contiguous 1-D window `x[offset:offset+size]`. |
| dynamic_slice | `af.dynamic_slice(x, start, size=2)` | runtime-parametric slice; verified variant fixes length-4 input, `size==2`. |
| scaled_elementwise | `af.scaled_elementwise(x, z, op="Add", scale=1.0)` | `scale * (x OP z)`, `op in {Add, Mult, Sub, Min, Max}`; equal-size 1-D. |

---

## Pretrained loaders

```python
embed = af.load("sentence-transformers/all-MiniLM-L6-v2")  # -> Encoder
vecs  = embed(["hello world", "the cat sat"])              # [2, D], L2-normalised

clf    = af.load_resnet18()                                # -> Vision
logits = clf(image)                                        # [1,3,224,224] -> [1,1000]
```

- `af.load(name, int8=False) -> Encoder` - a BERT-family sentence encoder. The
  transformer layers run on the ANE as fused programs (cached per sequence
  length); tokenisation, embedding lookup, and mean-pooling run on the host.
- `af.load_resnet18(int8=False) -> Vision` - torchvision ResNet-18 (ImageNet).
  BatchNorm is folded into the preceding conv at load, so the ANE graph is pure
  conv/relu/pool/add/fc.

`transformers` / `torchvision` are imported lazily, only when these loaders are
used.

---

## Classes

| Class | Role |
| --- | --- |
| `Tensor` | a node in the compute graph; methods/operators record structure (no device work). |
| `Model` | a compiled single fused ANE program. Call with input array(s); `.n_ops` = fused graph ops; `.release()`. |
| `SegmentedModel` | a compiled plan of e5rt regions interleaved with native bridge sub-programs. `.n_ops`, `.n_netplist`, `.release()`. |
| `Encoder` | sentence-embedding model from `af.load`. |
| `Vision` | ResNet-18 classifier from `af.load_resnet18`. |

`compile(out, int8=False, build_dir=None)` returns a `Model`, or a
`SegmentedModel` if the graph contains any netplist-bridge op.

---

## `compile()` signature

```python
af.compile(out, int8=False, build_dir=None, opt="routes",
           compress=None, compress_atol=0.05, block_size=32,
           validate=False, target=None)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `int8` | `False` | Alias for `compress='int8'` (per-channel int8 weight streaming). |
| `opt` | `'routes'` | Graph optimizer. `'routes'` is the **lossless** default route pass (cost-model-driven, never changes numerics, no on-device measurement). `0` is the byte-identical historical path. `1` adds the cost-model variant pick. `2` / `'max'` autotunes by on-device measurement and validates each variant against the `opt=0` baseline. |
| `compress` | `None` | Weight encoding - see [weight compression](#weight-compression). |
| `compress_atol` | `0.05` | Relative-L2 fallback budget for the accuracy-gated `int4` / `blockwise` modes. |
| `block_size` | `32` | Inner-dim block width for `compress='blockwise'`. |
| `validate` | `False` | Raise (rather than warn) on a flagged fp16-precision risk. |
| `target` | `None` | Compile / gate for another ANE family - see [cross-chip deployment](#cross-chip-deployment). |

`compress=` always takes the byte-identical (`opt=0`) lowering, so passing
`compress` with an explicit `opt>=1` is rejected.

---

## Weight compression

`compress=` selects how matmul/linear weights are packed into the blob. All forms
**stream** (dequantise during the tile DMA) inside the same fused program.

| `compress` | Encoding | Notes |
| --- | --- | --- |
| `None` | fp16 (default) | Byte-identical at `opt=0`. |
| `'int8'` | per-channel int8 | `constexpr_affine_dequantize`; half the weight bytes. `int8=True` is the alias. |
| `'int4'` | 4-bit LUT palettization | Per-tensor; accuracy-gated with an automatic fallback to int8 -> fp16 controlled by `compress_atol`. |
| `'sparse'` | unstructured bitmask | Emitted when the weight is >=50% zeros, else fp16. |
| `'blockwise'` | per-inner-block int8 | `constexpr_blockwise_shift_scale`, `block_size` columns per scale; accuracy-gated -> int8 -> fp16. |
| `'auto'` | per-weight best | Sparse if sparse, else int4 if accurate, else int8, else fp16. |

`compress='auto'` is **family-aware**: only encodings that stream *natively* on the
target family (the host-detected family when `target=None`) are candidates
(`tg.native_streams(family)`). On **h13 / M1** the natively-streaming forms are
int4-LUT and sparse, so `auto` skips int8 and blockwise there (they would fold to
dense fp16 - accuracy cost, no bandwidth win) and a rejected int4 falls back to
fp16, not to a folding encoding. Explicit single-mode knobs (`compress='int8'`,
etc.) are never filtered. See [`cross-chip.md`](cross-chip.md).

int8 trades memory, not single-stream throughput; the int4/sparse streams give a
measured on-device win on large bandwidth-bound matmuls. Conv and norm weights
stay fp16.

---

## Cross-chip deployment

ANEForge can compile and gate a graph for an ANE family other than the host's. There
are 28 compiler targets covering M1 through M5 (and the A-series equivalents). See
[`cross-chip.md`](cross-chip.md) for the full model.

| API | Role |
| --- | --- |
| `af.compile(out, target='h16s')` | Gate the graph's ops and shapes for that ANE family before lowering (e.g. trig floors, slice-saturation). |
| `cross_compile_check(out, target)` (in `aneforge._compile`) | Does the graph **compile** for another family, checked from this host? Returns `True` iff the e5rt compiler produces a library for that `TargetArchitecture`. Compile-level validation only; numeric correctness still needs the real silicon. Raises on an unknown arch name. |
| `detect_family()` (in `aneforge._targets`) | Best-effort target family for the host ANE: the `ANEFORGE_TARGET` env var if set, else the CPU brand (M1/M5 are silicon-measured anchors), else the conservative floor with a one-time warning. |

The **`ANEFORGE_TARGET`** environment variable (an arch string such as `h13` or
`h16s`) overrides host detection everywhere.

**`CrossChipFP16Warning`** is raised when a graph carries an fp16 pattern whose
result can diverge across chip families (warn-only, so a compile error still
surfaces). Silence it with
`warnings.filterwarnings('ignore', category=af.CrossChipFP16Warning)`.

---

## Cost estimation (measurement-free)

| API | Returns |
| --- | --- |
| `af.estimate(out, int8=False, target=None)` | Estimated compiled latency in **microseconds**. With `target=None` it uses the M5-measured heuristic; pass an arch string (`target='h13'`) to switch to the measurement-free analytic per-chip roofline, anchored to the silicon-measured M1 and M5 points and scaled to any of the 28 chips. |
| `af.project_peak(arch)` | A per-chip fp16 peak-throughput projection `{tflops, rel_m1, cores, ghz}`, anchored to the measured M1 point. The generational table needs no silicon beyond M1/M5. |
| `af.precision_risk(out, verbose=False)` | Heuristic fp16-cancellation risk (`graph_error`, flagged `nodes`, `hotspots`). |

Both `estimate` and `project_peak` are structural - they touch no device. Real
variant selection in the autotuner is always by on-device measurement.

---

## Compile-failure backoff guard

As a defensive backstop for the autotuner's burst of variant compiles, ANEForge
**paces the next compile after a failure** by a short interval, keeping consecutive
failures apart.

- `af.CompileBackoffError` - raised (in strict mode) when a compile is attempted
  inside the backoff window after a recent failure.
- `af.reset_compile_breaker()` - clears the backoff state.
- `ANEFORGE_COMPILE_BACKOFF` - seconds to pace (default `15.0`; `0` disables pacing).
- `ANEFORGE_COMPILE_BREAKER_STRICT=1` - raise `CompileBackoffError` instead of sleeping.
- `ANEFORGE_DISABLE_COMPILE_BREAKER=1` - turn the guard off entirely.

A successful compile clears the backoff.

## Training on the ANE

ANEForge has a small reverse-mode autograd (`aneforge/autograd.py`): the forward
*and* backward passes compile and run on the engine. `af.parameter` declares a
trainable graph input (a mutable weight, no recompile per step); `af.backward` /
`af.softmax_cross_entropy` / `af.mse` build the on-ANE gradient graph; `af.SGD` /
`af.Adam` / `af.Trainer` drive a training loop. `Trainer(device_optimizer=True)`
runs the optimizer update itself as graph ops, and `Trainer(resident_state=True)`
keeps optimizer state on-engine across steps. See [`training.md`](training.md) and
`examples/train_mnist_mlp.py`.

## The persistent netplist worker

Netplist-bridge stages (sdpa, topk, ...) default to a **persistent Path-A worker**
(load-once, eval-many) built lazily on first call and reused for the model's
lifetime, then freed in `release()`. Sub-program dispatch is sub-millisecond.
Set `ANEFORGE_NETPLIST_WORKER=0` to force the A1 (subprocess-per-call) fallback;
the runtime also falls back automatically when no worker route exists for an op.

---

## See also

- [`capabilities.md`](capabilities.md) - the hardware op surface and dtype matrix.
- [`cross-chip.md`](cross-chip.md) - compiling and gating for other ANE families.
- [`training.md`](training.md) - on-ANE autograd and the `Trainer` loop.
- [`getting-started.md`](getting-started.md) - install + first program.
- Worked examples: `examples/quickstart.py` (CNN + encoder block),
  `examples/resnet18.py`, `examples/sentence_embeddings.py`, `examples/sdpa.py`,
  `examples/native_ranking.py, examples/native_norms.py`, `examples/pointcloud.py`.
