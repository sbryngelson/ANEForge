# Roadmap

Forward-looking strategy for ANEForge: what is solved, what is genuinely
open, and what is permanently out of reach. A living document; every entry
has a specific reason for being here, not a wishlist.

ANEForge is a direct, CoreML-free Python frontend for the Apple Neural
Engine. It lowers an operator graph into a single fused Espresso `e5rt`
program and dispatches it to ANE silicon, with no entitlement and no
CoreML dependency. The capability surface is characterized in
[capabilities.md](capabilities.md); the cross-chip story in
[cross-chip.md](cross-chip.md); and training in [training.md](training.md).

## Where things stand

The infrastructure is mature. The frontend compiles whole graphs to one
ANE program; pretrained vision CNNs (ResNet, ViT), CLIP, transformer encoders and
rerankers, decoder LLMs (Llama/Qwen/Mistral/Gemma, MoE, hybrid DeltaNet, GPT-2)
with prefill and resident-KV-cache decode, Whisper speech-to-text (encoder +
decoder), and diffusion UNet blocks load and run correctly; weights stream at fp16, int8, int4-LUT,
sparse, or blockwise; a reverse-mode autograd trains real networks
forward, backward, and optimizer-step on the engine; a per-chip cost model
estimates latency for any ANE generation without that hardware in hand; and
the same graph cross-compiles statically across the M1-M5 family. The
correctness corpus (`tests/run_corpus.py`) is the standing gate and is green.

The remaining work is concentrated in three places: extending autograd
coverage so transformers train end-to-end, pushing fp16-sensitive models
(notably Stable Diffusion) past their numerical walls, and filling the
cross-chip story with the one silicon class (M3/A15) we have not measured.

## Recently completed

These were open directions. They have shipped and moved here so the open
frontiers below read honestly.

### Cross-chip deployment (M1-M5)

A single ANEForge graph compiles statically for any ANE generation. The
compiler binary is byte-identical across chips; all per-chip difference is
HAL data plus a `MinimumFamily<N>` op gate, so a target is selected by
name (`compile(target='hXX')`) and validated without that silicon present.
`cross_compile_check` gates target names (an unknown architecture silently
falls back on `e5rt`, so it is rejected up front), and family detection
(`detect_family`) resolves the running chip, including the M2/M3/M4 classes
via the verified `M(n) = H(n + 12)` board-type ladder. The op walls are
family-wide, not M5-specific; the trig floor (Sin/Cos) and a few
resampling ops are the only core-net-irrelevant gaps on the oldest
families. fp16 numerics are the one axis that genuinely needs each chip,
and `predict_fp16_divergence` + `CrossChipFP16Warning` flag where a graph
may diverge across families. `cross_compile_check` runs the same static
`preflight` family-cap gate as `compile(target=)` before invoking the
compiler, so a per-family cap violation (conv kernel width, max tensor
dimension, a below-floor op) is rejected statically rather than left to the
host cross-compiler, which does not reliably enforce a different family's
HAL-gated caps and would otherwise pass cross-chip CI and only fail on the
real silicon. See [cross-chip.md](cross-chip.md).

### Per-chip cost model

`af.estimate(out, target='hXX')` returns a measurement-free latency
estimate for any of the 28 ANE targets, derived from the compiler's own
analytic `cycles -> roofline -> wall-time` model. `af.project_peak(arch)`
gives the generational fp16-peak table. The model is anchored to
silicon-measured rooflines on both M1 and M5 (`_cost._ANCHORS`): the M5
loop-closure correction is in place (effective bandwidth scales with core
count, not clock), so `estimate(target='h17s')` now matches measured M5
convs within ~13%. The A14/h14 anchor is silicon-validated out of sample
(1.17x over nine fresh shapes). `target=None` keeps the precise M5-measured
heuristic, so the optimizer path is unchanged. `af.estimate_provenance(arch)`
reports whether a target's estimate is silicon-anchored (A13/h13, A14/h14,
A16/h17s, with the A16 tier folding H16/H17* onto the h17s point) or
extrapolated from the nearest measured anchor, so a caller knows which
estimates rest on measured silicon. `project_peak`'s generational multiple
is a peak-compute ceiling; typical measured speedups are lower.

### Weight-compression datapaths

`compile(compress=...)` streams compressed weights on the
`e5rt` path with an accuracy-gated fallback chain: `int4` (4-bit LUT
palettization), `sparse` (unstructured bitmask), `int8` (per-tensor or
per-channel affine), and `blockwise` (blockwise affine). All forms compile
and run; `compress=None` is byte-identical to fp16. int4-LUT, sparse, and
int8 stream weights (dequant during DMA); blockwise dequantizes in-program,
a footprint lever rather than a bandwidth one. The single-matmul streaming
win does not always carry to full models - end-to-end this is primarily a
footprint and capacity lever, with the bandwidth win realized on large
weight-bound layers. `compress="auto"` is family-aware: it filters its
candidate encodings through `native_streams(family)`, so on a
bandwidth-starved family it only auto-selects forms that stream natively and
falls back to fp16 rather than a folding encoding (which would cost accuracy
for no bandwidth win). On M1 (h13) that means int4-LUT only; A14+ stream all
four. The family is taken from `compile(target=)`, or the host chip when no
target is given; an explicit single-mode `compress=` is never filtered.

### On-engine training (autograd + Trainer)

`aneforge/autograd.py` is a reverse-mode autograd whose forward, backward,
and optimizer step all run as ANE graph ops. `Trainer` and
`UnrolledTrainer` train real networks: a full-MNIST MLP reaches ~98% with a
host optimizer and ~97.8% with the optimizer on the engine.
`Trainer(resident_state=True)` keeps weights and optimizer moments resident
on the ANE across steps via output-to-input buffer aliasing, so the host
supplies only the minibatch and learning rate. `UnrolledTrainer` runs K
steps in one dispatch and is resident by default. Op coverage now spans MLP
and CNN graphs (matmul, bmm, conv, pooling, relu, gelu, the elementwise and
reduction ops, softmax, slice/concat/reshape/transpose), so CNNs train; the
gap that blocks transformers is below. See [training.md](training.md).

### group_norm at diffusion shapes

`group_norm` is tiled over rank-4 feature maps, so the SD-1.5 wall shapes
(640 channels at 64x64, 512 at 128x128) compile and run family-wide. This
removes the group-norm wall that previously blocked large diffusion feature
maps; the remaining SD-1.5 blocker is numerical, not structural (see below).

### On-engine image input

`af.image_input(shape, scale=, bias=)` declares a uint8 input and
dequantizes on the engine, byte-identical to a host convert and saving the
host-side conversion per frame. This is the terminal form of direct image
input on the e5rt path; true 4CC (FourCC) input is not reachable there (below).

### Compile-failure backoff guard

`aneforge/_circuit.py` is a backoff rate-limiter wired into `Program.compile`.
After a compile failure the breaker paces the next compile by a short interval, a
defensive backstop so the autotuner's burst of variant compiles cannot pile up;
success clears it. Warn-and-sleep by default
(`ANEFORGE_COMPILE_BREAKER_STRICT` raises, `ANEFORGE_DISABLE_COMPILE_BREAKER`
disables), and exposes `af.CompileBackoffError` and `af.reset_compile_breaker`.
`af.image_input` uses the uint8 route.

### Decoder-LLM inference and pretrained-model loaders

The `llm.py` runner loads Hugging Face decoder LLMs and runs prefill plus
resident-KV-cache decode as fused ANE programs, matching HF logits: dense
Llama/Qwen/Mistral, Gemma (GeGLU + scaled embeddings + `(1+w)` RMSNorm), sparse
Qwen-MoE, and hybrid DeltaNet+attention (Qwen3.5), with automatic segmentation
past the ~2 GB program ceiling, int8/int4 weights, and exact speculative decoding
(~2.3x on Qwen3-8B). A config-driven `LayerSpec` plan keeps the runner
architecture-agnostic, so a new family is an adapter, not a new code path; an
unsupported architecture is rejected rather than mis-loaded. Alongside it, model
loaders bring GPT-2 (the first LayerNorm decoder, tiled tied lm_head), ViT-B/16,
ResNet-18/34/50/101, CLIP, BERT/RoBERTa/DistilBERT rerankers (`CrossEncoder`),
and Whisper speech-to-text (audio encoder + autoregressive decoder, both on the
ANE with resident-KV-cache decode) onto the engine. Reproducible benchmarking
landed too: the roofline suite (per-machine `ROOFLINES.md`, contributor-driven)
and the MLPerf ResNet-50 harness that clears the MLCommons submission checker at
reference accuracy. See [llm.md](llm.md) and
[developer/models.md](developer/models.md).

## Open frontiers

The honest list of what is not done and is worth doing.

### End-to-end LLM training at scale

The autograd registry now covers the normalization layers (`layer_norm`,
`rms_norm`, `group_norm`, `l2_norm`) and `silu` alongside the structural,
linear-algebra, activation, and math ops, so transformer and LLaMA-style
blocks, diffusion-UNet / GroupNorm CNNs, plain CNNs, and MLPs all train end to
end on the engine; the pre-norm transformer and LLaMA-block examples converge as
the proof. The remaining gradient gaps are low-value and do not block mainstream
training: the reduction-routing ops `amax` / `amin` / `cumsum` and a few
parametric activations (`atan`, `softplus`, clamped variants), each an additive
frontend follow-up. Normalization affine is now learnable too: passing parameter
Tensors for a norm's `gamma` / `beta` composes the normalize op with an explicit
trainable scale and shift, so the affine trains with the rest of the model (a
numpy `gamma` / `beta` still bakes a fixed affine). Scale is demonstrated on
three axes, all on the engine: a four-layer model that reconstructs its training
text (`examples/train_charlm.py`), a corpus-trained model that generalizes to a
held-out split (`examples/train_charlm_corpus.py`), and a sixteen-layer model
trained with a layer-streamed compile (`examples/train_charlm_deep.py`). The
compile-size ceiling on depth is removed: `aneforge.streaming.CheckpointedStack`
compiles a stack of identical layers' per-layer forward and backward once and
reuses them for every layer, so compile work is constant in depth (the
sixteen-layer model's programs compile in about half a second, where the
eight-layer monolith took about 162 seconds), with gradients bit-exact against
the monolith. What remains is engineering reach, not a capability gap: larger
corpora, longer context windows, and a resident-state or on-engine optimizer for
the streamed path.

The conv input-gradient on M1 is now slice-saturation-safe (the im2col backward
concatenates patches on a non-last axis so it never transits the h13 width-axis
crop-DMA saturation). The remaining M1-specific item is behavior under many
rapid compiles, which the compile backoff in `aneforge/_circuit.py` paces as a
defensive backstop, best verified on M1 hardware.

Separately, LLM decode throughput remains GPU territory: decode is
dispatch-bound and the fp16-product precision limits force the
cancellation-sensitive layers onto the CPU, so the GPU wins decode on both
speed and energy at every batch size. ANEForge runs decoder-LLM inference
correctly (prefill plus resident-KV decode, HF-matching logits), and
compute-bound prefill is a genuine ANE win; speculative decoding narrows the
decode gap (exact, ~2.3x) without closing it. The engine's niche stays
fp16-tolerant, compute-bound work - vision, encoders, on-device embedding, and
prompt prefill - not high-throughput autoregressive decode. See the bottlenecks
section.

### Stable Diffusion 1.5 end-to-end

Per-component validation passes (UNet and VAE steps run on the ANE within a
few percent), and the group-norm shape wall is gone. The remaining wall is
classifier-free guidance (CFG) cancellation: the conditional minus
unconditional difference is a small fraction of the signal magnitude and is
swamped by accumulated fp16 noise over the denoising trajectory. A
numerical-envelope limit, not a missing op. The path forward is paired-fp16
precision carried through the UNet tail (the subtraction itself is exact by
Sterbenz; the loss is input quantization), partially built in the math toolkit
and not yet wired through the full UNet.

### M2 / M3 silicon power anchors

The watt-complete characterization is measured on three physical chips (M1 /
A13, M2 Pro / A14, and M5 / A16-class): the M2 run is the full 16-class watt map
(`device_compare_wattcomplete_results_M2.json`), plus a 25-point latency grid that
drives the h14 cost anchor and its mid-utilization ramp and a power anchor
(per-compression-mode energy and idle/compute/conv rails). The M-series-to-H-target
ladder is verified, so M3/M4 remain ground-truth capability targets (H15/H16).
The remaining gap is A15/M3: its absolute power rail and full watt-complete map
have not been measured, and neither M1 nor the now-anchored A14 can supply those
points. Capability and relative cost are covered family-wide; per-rail watts for
the A15 generation need that hardware.

## Known hard limits

These are not roadmap items. They are the boundaries of the approach,
stated so nothing here overclaims.

### The two locks

1. No fp32 / int32 / bf16 compute. The ANE compute dataplane is fp16
   (with a wide, at-least-fp32 accumulator). fp32, int32, and bf16 are
   accepted by the MIL parser but are "not implemented on any backend" for
   the ANE - even a bare cast is rejected. This is silicon, not a path
   limit. Every cancellation-sensitive workload that needs exact products
   (notably long-contraction matmuls under cancellation, and the
   transformer down-projection) has no fp16-tolerable form.

2. The entitlement boundary. Custom signed HWX cannot be loaded (the
   kernel driver verifies code signatures), and the fully autonomous,
   zero-host-call dispatch loop is entitlement-gated. Measurement shows the
   e5rt surface is functionally complete for the workloads here: bounded
   multi-step host-free dispatch is reachable without an entitlement, and
   per-step host dispatch was never the bottleneck, so the entitlement adds no
   capability these models need.

### True 4CC (FourCC) image input

Declaring a true interchange image format (e.g. `&BGA`) as an input is
not reachable on the e5rt path. The format grammar is solved and the MIL
parses and type-checks, but the final lowering of `pixel_buffer_to_tensor`
does not complete without the entitled CoreML `Input4CCFormat` + IOSurface
route. `af.image_input(uint8)` is the terminal form for direct image input.

### Control flow and recurrence

`cf_if`/`else`/`loop`, `phi_virtual`, and `rnn_arch` (LSTM/GRU) are a
genuine single-procedure wall on the authorable netplist surface, and Apple
itself routes recurrence off the ANE. Static-iteration unrolling fits;
data-dependent control flow does not.

## Known bottlenecks

Measured, not speculative.

- Per-call dispatch floor. Each `e5rt` dispatch has a fixed launch cost
  (tens of microseconds; higher on bandwidth-rich chips). Tiny op chains
  are floor-bound, where BNNS or numpy can win; ANE territory begins around
  ~100 MFLOPs per call. Fusing the whole graph into one program - the
  frontend's default - amortizes this.

- LLM decode is dispatch-bound and GPU-favored. The fp16-product
  precision limits force the precision-sensitive layers onto the CPU, which
  burns more power than running the whole model on the GPU; with the
  per-token dispatch overhead, the GPU wins LLM decode on speed and energy at
  every batch size. Batching is the dominant decode-throughput lever, but ANE
  batched serving plateaus far below the GPU because it stays CPU-dominated.

- Per-PID program cap. `aned` enforces a hard cap of 128 simultaneously
  loaded programs per process, with no LRU eviction (program 129 fails to
  compile). `Program.release()` frees a slot immediately. Budget shape
  specializations against this cap, or fuse more aggressively.

- Very large or rapid back-to-back compiles. Very large unrolled graphs
  or many programs compiled back-to-back in one process can fail to compile.
  This is environmental, not a code bug; reduce program size or re-run. The
  compile backoff in `aneforge/_circuit.py` paces repeated failures as a
  defensive backstop.

## Limits of the capability map

The op census is exhaustive over the surface Apple exports, not over the
silicon's full op set. Three structural blind spots remain:

1. Internal-only layers. The crackability predictor keys on exported
   validators; at least one real hardware layer (`RCAS`) has an internal
   validator and no exported symbol, found only by an opcode cross-check.
   Others may exist that the export census structurally cannot see.

2. Composite reachability without a single-layer validator. Several ops
   run on silicon with no `*Layer` validator (`conv_transpose`,
   `group_norm`, `rms_norm`, and others). Reachability is empirical
   (probe-confirmed), not derivable from the validator list, and the
   predictor is necessary-leaning but not sufficient - HWX codegen is the
   real gate.

3. Temporal and single-process scope. The whole map is one OS build and
   one compiler version, measured single-process and steady-state.
   Contention with other ANE clients, QoS preemption, cache eviction under
   memory pressure, and per-op thermal behavior are unmapped. Negative
   labels are characterized at specific configs, not swept boundaries, and
   several decompose through a second authoring path (so "hard limit" can
   over-count genuine silicon walls). A second OS build or chip converts
   these from inferred to measured.

The largest source of unknown-unknowns is what Apple's own models exercise
(Path B, streaming state, dynamic shapes, control flow) that we would not think
to author. Sweeping the on-system Apple `espresso.net` corpus is the cheapest
way to surface those.
