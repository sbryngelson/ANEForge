# Glossary

Terms a tool user is likely to meet across the ANEForge documentation, in alphabetical
order.

## A - D

**ANE** - Apple Neural Engine. A dedicated neural-network accelerator on
Apple Silicon. fp16-dataplane, ~38 TFLOPS peak on M5 Pro (the marketing
spec); the measured fp16 roofline anchor used throughout these docs is the
lower sustained figure (~8.9 TFLOP/s, see `capabilities.md`). Programmable
only through private framework APIs.

**ANECIR** - ANE Compiler Intermediate Representation. The third IR in
Apple's compile pipeline (after MIL and MLIR, before HWX). Surfaces as
the `net.plist` XML format consumed by `_ANEInMemoryModelDescriptor
modelWithNetworkDescription:`. Accepts ops MIL rejects, e.g., `Rsqrt`,
`Inv`, `Sqrt`, `Log2`, `Exp2`.

**ANEF** - Apple Neural Engine Format. Sometimes synonymous with HWX,
sometimes refers to the broader ANE deployment format including
weight blobs and metadata. The `kANEFModel*` constants in
`AppleNeuralEngine.framework` use this prefix.

**aned** - `/usr/libexec/aned`. The root-privileged XPC daemon that does
MIL -> HWX compile work. Receives requests via XPC, runs the four-IR
compile pipeline, returns a compiled program handle. Maintains a
per-PID HWX program cache.

**aneuserd** - `/usr/libexec/aneuserd`. The per-user broker that
arbitrates between processes and `aned`. Runs as `_neuralengine`.

**BLOBFILE** - A MIL reference form for external weight files:
`tensor<fp16, [...]>(BLOBFILE(@path="weights.bin", offset=0))`. Used by
the legacy subprocess Path A. Incompatible with e5rt's in-process
compile because there's no sibling weight file; e5rt requires
inlined fp16 hex literals.

**BNNS** - Basic Neural Network Subroutines. Apple's CPU-side neural-network
inference library (in Accelerate.framework). e5rt accepts `device_mask=0x1
= BNNS_MASK` to route a compiled program to BNNS on the CPU rather than
ANE.

**capability family** - An ANE generation's op-capability tier, gated in the
compiler by a `MinimumFamily<N>` trait on each op. M-series chips map to
H-targets by `M(n) = H(n + 12)` (M1 = H13 = family 2, M5 = H17 = family 5),
so capability is a property of the family, not of one chip. Walls (e.g. the
trig floor on older families) are family-wide. ANEForge resolves the running
family with `detect_family` and gates target names with `cross_compile_check`.
See [cross-chip.md](cross-chip.md).

**circuit breaker** - ANEForge's compile backoff rate-limiter
(`aneforge/_circuit.py`, wired into `Program.compile`). After a failed compile
the breaker paces the next compile by a short interval, a defensive backstop so
the autotuner's burst of variant compiles cannot pile up. Warn-and-sleep by
default; raises under `ANEFORGE_COMPILE_BREAKER_STRICT`. Exposes
`CompileBackoffError` and `reset_compile_breaker`.

## E - H

**Espresso** - `/System/Library/PrivateFrameworks/Espresso.framework/`.
Apple's internal ML runtime. CoreML, MPSGraph, and ANEForge's e5rt
backend all sit on top of Espresso. Exports 235+ `e5rt_*` C symbols.

**e5rt** - Espresso's runtime C API family. 235 symbols including
`e5rt_e5_compiler_*`, `e5rt_program_library_*`,
`e5rt_execution_stream_*`. ANEForge's fast backend uses this. See
[e5rt-dispatch-reference.md](e5rt-dispatch-reference.md).

**entitlement** - A code-signing capability Apple grants its own software
(for example CoreML) to use certain private interfaces. ANEForge reaches the
ANE through the `e5rt` path, which requires no entitlement, so it runs from an
ordinary user process. "Without an entitlement" or "no entitlement needed" in
these docs means exactly that. The one surface that does require an entitlement
is Path B.

**fp16** - IEEE 754 binary16 floating point. ANE's native compute type.
The dataplane is fp16-only; fp32 is acceptable as an intermediate but
not as an I/O type. ANE's fp16 has documented IEEE-754 deviations: NaN
propagation broken, round-half-away-from-zero, gelu(0.0) = -0.000754.
See [capabilities.md](capabilities.md#datatypes).

**H-target** - A compiler target architecture name of the form `hNN`
(`h13`-`h17` and variants such as `h17s`/`h17g`/`h17d`). The compiler holds
28 such targets; the `__TEXT` code is byte-identical across them and only
the per-target HAL data differs. M-series chips map by `M(n) = H(n + 12)`.
ANEForge selects one with `compile(target='hNN')` and `af.estimate(out,
target='hNN')`, validated statically without that silicon present. See
[cross-chip.md](cross-chip.md).

**HWX** - The compiled, ready-to-load binary format for ANE programs.
Mach-O variant with magic `0xBEEFFACE`, per-chip-generation
specialization. The kernel driver (`AppleH16ANEInterface`) enforces
code signatures on HWX binaries - you cannot synthesize and load your
own. ANEForge compiles via `aned`, which produces correctly-signed HWX.

## I - M

**IOSurface** - Apple's cross-process shared-memory buffer object.
Used as the input/output tensor surface for ANE workloads. Zero-copy
aliasable to Metal `MTLBuffer` via `EspressoANEIOSurface
metalBufferWithDevice:`. The basis for hybrid CPU/GPU/ANE pipelines.

**Path A** - `_ANEInMemoryModel.evaluateWithQoS:`. The canonical
unentitled ANE dispatch surface. Compiles MIL or ANECIR netplist
via `aned`, loads via `loadWithQoS:`, evaluates via `evaluateWithQoS:`.
Verified on ANE silicon via powermetrics (1.4 W on ANE rail).

**Path B** - The entitlement-gated streaming surface using
`_ANESharedEvents`, `_ANEChainingRequest`, and the always-nonzero
`intermediateBufferHandle`. Used by Apple's production frameworks for
streaming/chained execution. Blocked for third parties without
`com.apple.aned.private.allow`. ANEForge reproduces equivalent
behavior at the model level via paired `_in`/`_out` state tensors.

**MIL** - Model Intermediate Language. Apple's textual IR for ML
programs. Subset of CoreML's emitted MIL. ANEForge's primary input
format. Documented opset `ios18` is the current target. See
[mil-primer.md](mil-primer.md).

**MLComputePlan** - Public macOS 14.4+ API
(`MLComputePlan loadContentsOfURL:configuration:completionHandler:`).
Returns per-op `(preferred_device, supported_set, cost_weight)` for
any compiled `.mlmodelc`. Useful as a routing oracle for what Apple's
compiler *would* choose; does NOT reflect direct Path A behavior
(which bypasses the heuristic).

**MLIR** - Multi-Level Intermediate Representation. The second IR in
Apple's compile pipeline (between MIL and LLIR). Not directly
emittable from user-space; gated by daemon capability negotiation.

**MPSGraph** - Metal Performance Shaders Graph. Apple's GPU graph
framework, public. Routes mostly to Metal GPU; has private compilation
descriptor selectors that can route to ANE (`setPreferredDevice:`,
`setEnableANECValidationWorkflow:`, `setPrintANEPlacementAnalysis:`).
Not used by ANEForge as a dispatch backend but useful as a GPU
benchmark baseline.

## N - Z

**native (weight) streaming** - Reading compressed weights directly into the
multiply-accumulate datapath, dequantizing during DMA rather than expanding
them to a dense fp16 buffer first. int8, int4-LUT, and sparse weights stream
this way on the `e5rt` path, moving fewer bytes per dispatch and
buying a bandwidth win on weight-bound layers. Blockwise-affine instead
dequantizes in-program (a footprint lever, not a bandwidth one). Exposed via
`compile(compress=...)`. Whether a format streams is family-dependent (on
the bandwidth-starved families only int4-LUT streams).

**netplist** - The XML form of ANECIR programs. The plist root contains
a `Layers` array; each layer has `Type`, `Inputs`, `Outputs`, `Params`.
Loaded via `_ANEInMemoryModelDescriptor modelWithNetworkDescription:`.
Accepts hardware-native opcodes MIL rejects.

**powermetrics** - `/usr/bin/powermetrics`. macOS system tool that
samples power consumption per hardware rail (CPU, GPU, ANE, etc.).
Requires sudo. Used in ANEForge to confirm that Path A dispatches
draw power on the ANE rail (1.4 W sustained during hot loops; idle =
0 W).

**Program** - In `ane_e5rt.py`, the compiled-and-loaded e5rt
dispatch handle. Holds the compiled library, function, operation,
input/output ports, and execution stream. Designed for
compile-once-eval-many: `prog.eval(inputs)` per call has only
memcpy + execute_sync + memcpy overhead.

**programHandle** - A 64-bit handle returned by `_ANEInMemoryModel
programHandle` after compile + load. Identifies the model within
`aned`'s per-PID cache. Stable across submissions; lost across
process boundaries (per-PID).

**queueDepth** - The per-program in-flight cap, hardcoded at 127 via
`dispatch_semaphore_create(127)` inside `_ANEInMemoryModel`. The
foundation of 127-deep async pipelining for decoder loops.

**resident state** - Keeping tensor state (weights, optimizer moments)
resident on the ANE across `execute_sync` calls with no host round-trip, by
aliasing an op's output buffer onto its own input port (`share_buffer`).
ANEForge uses this for `Trainer(resident_state=True)` and the default
`UnrolledTrainer`, so during training the host supplies only the minibatch
and learning rate and reads weights back at checkpoints. Reachable
without an entitlement. See [training.md](training.md).

**SDPA** - Scaled Dot-Product Attention.
`softmax(Q @ K^T * scale) @ V`. The MIL op
`scaled_dot_product_attention` is fused at the MIL level; ANE-routing
kicks in above heads>=64, seq>=~496 at d=64. Exposed as `af.sdpa(...)`.

**TFLOPS** - Tera-floating-point operations per second. ANE's peak
fp16 throughput on M5 Pro is ~38 TFLOPS as a marketing spec; the measured
fp16 roofline anchor (`capabilities.md`) is the lower sustained ~8.9 TFLOP/s,
and Path A's measured sustained throughput at `conv3x3 [1, 256, 32, 32]` is
12 TFLOPS (well past CPU AMX/SME ceiling of ~5 TFLOPS).

**VJP** - Vector-Jacobian product, the backward rule for an op in ANEForge's
autograd (`aneforge/autograd.py`). The `VJP` registry maps each forward op
to a rule that builds its gradient as ordinary ANE Tensor ops, so the
backward pass runs on the engine. Coverage spans the structural and
linear-algebra ops (matmul/bmm, conv, pooling, reductions, shape ops), the
common activations (relu/silu/gelu/tanh/sigmoid/softmax and variants), the
math ops (exp/log/sqrt/rsqrt/erf/...), and the normalization layers (`layer_norm`,
`rms_norm`, `group_norm`, `l2_norm`) - enough to train transformer, LLaMA-style,
diffusion-UNet, CNN, and MLP graphs end to end. See [training.md](training.md).
