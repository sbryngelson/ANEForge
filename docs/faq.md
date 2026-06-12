# FAQ

## "Will this work on my machine?"

Verified on M5 Pro and M1 Max (macOS 26.5). Should work on any Apple Silicon
Mac (M1, M2, M3, M4, M5 - all variants) running macOS 14+. ANEForge
maps each M-generation to a compiler family `M(n) = H(n+12)` (M1 = H13,
M5 = H17), and the three silicon-measured chips (M1, M2, M5) carry these ANE core counts:

| Chip | Family | ANE cores | Notes |
| --- | --- | --- | --- |
| M1 (incl. Max) | H13 | 4 | Silicon-measured anchor |
| M2 / M3 / M4 | H14 / H15 / H16 | - | Ground-truth capability families |
| M5 (incl. Pro) | H17 | 16 | Silicon-measured anchor; verified host |

Op capability is per *family*, not per rail: Pro / Max / Ultra variants of a
generation differ in core count and clock, not in which ops compile.
`detect_family()` (in `aneforge._targets`) reports the host family, and the
`ANEFORGE_TARGET` environment variable overrides it. `af.project_peak(arch)`
gives the per-chip fp16 peak projection; `af.estimate(out, target='hXX')` a
measurement-free latency estimate.

ANEForge always compiles **in-process**, so compiled-bundle portability across
chip generations only matters if you copy build artifacts between machines. To
compile a graph *for* another family from one host, see
[`cross-chip.md`](cross-chip.md).

## "Can I compile for a different chip than the one I'm on?"

Yes. `af.compile(out, target='h16s')` gates a graph's ops and shapes for another ANE
family before lowering, and `cross_compile_check(out, target)` (in `aneforge._compile`)
checks from this host whether the graph compiles for that family's `TargetArchitecture`.
There are 28 compiler targets spanning M1 through M5. This is compile-level validation - numeric correctness still requires the real silicon. `detect_family()` reports the
host's family, and the `ANEFORGE_TARGET` environment variable overrides detection. See
[cross-chip.md](cross-chip.md).

## "Why did a compile suddenly pause for ~15 seconds?"

ANEForge paces a compile when a recent compile **failed**, a defensive backstop for the
autotuner's burst of variant compiles. The backoff guard keeps consecutive failures a
short interval apart. It is tunable: `ANEFORGE_COMPILE_BACKOFF`
(seconds, default `15.0`; `0` disables pacing), `ANEFORGE_COMPILE_BREAKER_STRICT=1`
(raise `af.CompileBackoffError` instead of sleeping), `ANEFORGE_DISABLE_COMPILE_BREAKER=1`
(off). A successful compile clears it; `af.reset_compile_breaker()` clears it manually.

## "Can I feed raw camera / video bytes directly?"

Yes. `af.image_input(shape, scale=1/255, bias=0.0)` declares a uint8 input port and
runs the `cast -> scale -> bias` dequantisation on the engine, so the host skips the
float-convert + repack. It is byte-identical to converting on the host.

## "Can I train on the ANE?"

Yes, for small models. ANEForge has a reverse-mode autograd that runs the forward
*and* backward passes on the engine, with an optional on-engine optimizer step
(`Trainer(device_optimizer=True)`) and resident on-engine optimizer state
(`Trainer(resident_state=True)`). See [training.md](training.md).

## "Why not just use CoreML?"

CoreML is the public, sanctioned route. Use it if:

- You want broad device compatibility (intel Macs, iOS, etc.)
- Your model is large enough that Apple's routing heuristic puts it
  on ANE (conv3x3 above `[1, 256, 32, 32]`; otherwise CPU)
- You don't need < 1 ms latency per call
- You're okay with the multi-second compile and `.mlmodelc` packaging

ANEForge makes sense if:

- You want sub-millisecond per-call latency
- You need fine-grained control of dispatch (per-op shape, async pipeline)
- Your shapes are smaller than CoreML's ANE-routing threshold and you
  want them on ANE anyway
- You're doing research on the ANE itself

The two are complementary. ANEForge uses `MLComputePlan` (a CoreML API) as
an audit oracle even though it doesn't use CoreML for dispatch.

## "Can I use this in production?"

Probably not yet, with caveats:

- The codebase is research. The Python API is unstable; method signatures
  may change.
- Private framework symbols can change with any macOS update. We test on
  the current macOS; older versions may need adjustments.
- Error handling is partial; many failure modes surface as opaque errors
  from `aned`.
- No threadsafety guarantees. e5rt's `Program` is single-threaded.

For research, prototyping, and exploratory ML work - yes, it's used. The
streaming demos run reliably. The validator catches most invalid programs
before they hit the slow XPC compile.

If you're building a production system, prefer CoreML for now.

## "Why fp16?"

The ANE's dataplane is fp16-native - that's the hardware. fp16 is:

- Half the memory bandwidth of fp32 (matters for ANE's TFLOPS targets)
- Sufficient precision for most ML inference (after careful training)
- The same dtype Apple's CoreML uses for ANE-bound models

ANEForge accepts int8 / uint8 / int16 / uint16 as I/O dtypes (for
quantized weights, indices, pixel data), but computation is always fp16.
fp32 is allowed as an intermediate inside the program via `cast(x,
dtype="fp32") -> compute -> cast(result, dtype="fp16")`.

There's no bf16. There's no fp32 dataplane. ANE is fp16-only.

Weights can still be *stored* compressed and dequantised during the tile DMA:
`compile(out, compress=...)` supports per-channel int8, 4-bit LUT (`int4`),
unstructured `sparse`, `blockwise` int8, and a family-aware `auto`. See
[weight compression](aneforge-api.md#weight-compression).

## "What about NaN handling?"

ANE's fp16 hardware doesn't propagate NaN correctly. For 23 probed ops,
ANE returns `+inf`, `0`, or other op-specific finite values instead of
NaN. Some specific cases:

- `log(+inf) -> 0`
- `softplus(+inf) -> 0`
- `mul(+inf, x) -> -52416` (overflow wraps to a representable negative)
- `cos(+inf) -> 0`
- `gelu(0.0) -> -0.000754` (polynomial bias leak)

If your model produces NaN through any op, the result is not IEEE-754
compliant. Train without producing inf/NaN intermediates. If you must
detect overflow, do it on CPU before dispatching.

Full details in [capabilities.md](capabilities.md#datatypes).

## "Why is my first call slow, and how do I run a single op?"

Most of the cost in a one-shot call is the one-time compile, not the eval. The
ANE eval itself is ~80-110 us once the program is resident. Compile once and
reuse the returned `net` across calls:

```python
import aneforge as af
import numpy as np

x = af.input((1, 4))
y = x.relu()
net = af.compile(y)            # one-time compile (~750 ms)
out = net(np.zeros((1, 4), np.float16))   # ~80 us per call thereafter
```

A single op is just a one-op graph. The mistake that makes every call slow is
recompiling per call: hoist `af.compile` out of your hot loop, keep the `net`
handle, and feed it new inputs. Call `net.release()` when you are done. For the
per-call cost breakdown and the underlying dispatch paths, see
[dispatch.md](dispatch.md).

## "What macOS versions are supported?"

Verified on macOS 26.5 (M5 Pro). Likely works on macOS 14+ but exact API
shapes change between versions:

- Pre-14: `MLComputePlan` doesn't exist, so the routing-truth audit
  oracle won't work, but other features should.
- macOS 26: new `e5rt_execution_stream_submit_async` symbol (old
  `_async_submit` deprecated). The wrapper handles both.

Bug reports with `sw_vers` output appreciated.

## "What if Apple changes the private API?"

It happens. The framework symbols ANEForge calls are private and can be
renamed or reordered across macOS releases (the ABI usually stays the
same). When something breaks:

1. Run the test suite (`docs/development.md`). The probes hard-fail on
   missing symbols.
2. Check `nm /System/Library/PrivateFrameworks/<framework>/<binary> |
   grep <symbol>` to see if the symbol was renamed.
3. Run `otool -tV` on the new symbol to confirm the same intent.
4. Open an issue documenting the API change.

The project has no SLA for tracking Apple's changes. Pull requests
welcome.

## "Why isn't this in tinygrad / pytorch?"

The ANE is a private accelerator. tinygrad's `extra/accel/ane` is the
historical precedent that ANEForge picks up from. Generic ML frameworks
hesitate to depend on private APIs that can change without notice.

If you want to use ANEForge from a higher-level framework, the `import
aneforge as af` API (build a graph, `af.compile`, call the returned `net`) is
small and self-contained - easy to integrate as a custom backend.

## "How does this compare to ANE-LM / johnmai-dev?"

[`johnmai-dev/ANE-LM`](https://github.com/johnmai-dev/ANE-LM) is the
closest precedent - a small LLM runtime over `AppleNeuralEngine.framework`.
It uses the same `_ANEInMemoryModel` path A as ANEForge.

Differences:

- ANEForge has the e5rt backend (~80 us/call vs ANE-LM's ms-scale dispatch).
- ANEForge has a validator that catches invalid shapes before the compile.
- ANE-LM is focused on LLM inference; ANEForge is more general-purpose.

Both projects are research code; treat them as exploration aids, not
production stacks.

## "Why is the validator's 'Scale is expected to be constant' a single bit?"

Because Apple's compile pipeline tracks tensor provenance internally. The
`is_constant` bit on `ANECTensorDesc` flags tensors that were produced
by a `const(...)` op (or `constexpr_*` chain). When the SDPA validator
runs, it just checks this flag - it doesn't re-derive constness from
the producer chain.

In practice this means: from user-space, you set the flag manually on the
descriptor before calling the validator. The compiler trusts you. (The
kernel-side signature check on HWX still enforces signing, so this
trust isn't a security boundary.)

## "Why does compiling take 750 ms?"

That's `aned` running the full MIL -> MLIR -> LLIR -> ANECIR -> HWX pipeline,
plus weight encoding, plus the kernel signature pass. It's a one-time cost
per shape - the compiled program then runs at ~80 us per call.

For shapes you'll use many times, compile once at startup and reuse.
For shapes that change per call (dynamic batch, variable sequence
length), you pay the compile cost each time. The `e5rt_execution_stream_operation_reshape_operation`
symbol exists but ANEForge doesn't yet expose it; an open follow-up.

## "Does this work in a Docker container?"

No. Containers don't have access to `aned` or the kernel ANE driver.
ANEForge needs to run on the macOS host directly.

For Linux + Apple silicon (Asahi), ANE access is unavailable - the
hardware exists but no open driver.

## "Can I distribute compiled bundles?"

You can copy the on-disk compile artifacts, but they won't load in a
different process. `aned` keys its HWX cache by PID; a fresh process
sees only `model.anehash` (a content hash) on disk and refuses to
load the cached HWX. Either re-compile in the new process, or use
shared-process semantics (open question).

## "What's the licensing situation?"

The code in this repository is yours to use. The Apple Neural Engine is
Apple's private hardware; the framework symbols ANEForge calls are
private, undocumented, and may change at any time. Nothing in this
project constitutes an API contract from Apple.

If you ship something built on ANEForge and Apple changes the underlying
API in a future macOS, your software will break. Plan accordingly.
