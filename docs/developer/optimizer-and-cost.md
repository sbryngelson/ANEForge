# The optimizer and the cost model

ANEForge picks how to lower a graph using two cooperating pieces: an **optimizer/tuner** that chooses among proven-equivalent lowerings, and an **analytic cost model** that orders those choices so the tuner can prune what it measures. This page covers both, plus the dispatch-floor reasoning, the roofline, the per-chip scaling rules, and the precision-risk model.

## The optimizer and `opt` levels

`compile(out, opt=...)` selects the graph optimizer. The design separates **lossless** route selection from **lossy** variants, and on-device measurement from measurement-free cost estimation.

### opt levels

- **`opt='routes'` (default)** — the *lossless route pass*. Cost-model-driven, per shape, it chooses for each route-bearing bridge node (`sdpa`, `minmax_norm`, `flatten`, `lrn`) between the native bridge (a graph cut) and the proven-equivalent fused decomposition (cut removed), with **no on-device measurement**. The route registry is lossless (cos 1.0), so this never changes numerics; a cut-free graph compiles to exactly the `opt=0` program (no regression). For `sdpa`, the cost model keeps native where it wins (long sequences), so the default never blindly removes a cut that helps.
- **`opt=0`** — no optimization, the historical byte-identical path (`int8` honored as given). Use for byte-identity tests.
- **`opt=1`** — cost-model pick over the full variant set (route swaps *and* the lossy whole-graph int8 variant), without measuring on-device.
- **`opt=2` / `opt='max'`** — autotune. Measure the legal proven-safe variants on the ANE, validate each against the `opt=0` baseline, and return the fastest correct one (cached; instant on a cache hit).

!!! note "Lossless vs. lossy gating"
    Route swaps are *lossless* — selected purely on predicted/measured speed. The whole-graph int8 variant is *lossy*, so it only enters at `opt>=1`, and at `opt=2` every measured variant is validated against the `opt=0` baseline before it can be chosen. The `reduce_sum→@ones` rewrite sits on a separate *precision* axis (`tune_precision`), not the bridge-alt axis.

### The equivalence-route registry

Every surfaced capability is either **route-selectable** (the autotuner picks its cheapest equivalent lowering by measured cost) or **single-route** (one lowering, no alternative). The `_BRIDGE_ROUTES` table closes this over the bridge ops: each `NETPLIST_OPS` op records either an `alt` (a fused decomposition that removes the graph cut, tagged with loss-class + on-device evidence) or `single_route=True` with the reason it has none. Only on-device-validated alts appear — the builder lives in `_rewrite._BRIDGE_DECOMPOSERS`, keyed identically, reconciled by `tests/test_routes.py`. A candidate that failed validation is marked single-route with its failure evidence.

Loss class **"lossless"** means mathematically identical (bit-identical or within fp16 op-noise), so it's chosen purely on measured speed. Fused ops are single-route by construction and not listed; `_routes_for()` reports any absent op as single-route ("only lowering").

Selectable alts (each lossless, removing the cut):

| Bridge op | Fused decomposition | Evidence |
|-----------|---------------------|----------|
| `sdpa` | `((q@kᵀ)*scale).softmax(-1)@v` | bit-identical (`fuzz_metamorphic.py` mha_vs_sdpa) |
| `minmax_norm` | `(x-amin)/(amax-amin+eps)` | relerr ≤1.5e-3 == fp16 op-noise |
| `flatten` | reshape to the same 1-D shape | relerr 0.0 |
| `lrn` | `local_response_norm(size=C, alpha=alpha*C, beta, k)` | cos 1.0 / relerr 0.0; ~1250× cut removal on conv→lrn→relu; no C<16 cap |

The notable single-route reasons (the rest are listed on the [capabilities page](capabilities-and-targets.md)):

- **`space_to_channel` / `channel_to_space` / `space_to_batch` / `batch_to_space`** — the reshape+transpose decomposition needs a rank-6 intermediate; ANECCompile rejects reshape rank>5 ("Rank of the shape parameter must be between 0 and 5"). The **rank-5 wall**, confirmed on-device.
- **`argmax` / `topk` / `sort`** — no fused reduction/selection/sort lowering (gather/argmax arch-gated; no in-graph sort); the native bridge is the only route.
- **`input_view` / `dynamic_slice`** — no fused slice op surfaced; native bridge only.
- **`fps` / `radius_search`** — data-dependent/iterative; inexpressible as a fused feed-forward graph.

## The cost model: a pruner, not ground truth

`_cost.py` is an **op-agnostic structural cost model** — the optimizer's *pruner*. `estimate(out) -> microseconds` gives a fast, structural roofline estimate of a compiled program's latency **without touching the device**.

!!! note "Why a pruner and not a predictor"
    The cost model's job is to *order* variants so the autotuner can skip measuring ones predicted far worse than the current best. Real selection is always by on-device measurement (`_optimize.measure`). Calibrated constants load from the bundled `aneforge/ane_cost_model.json` when present; otherwise the documented defaults below apply.

### The roofline

Every node's cost is a roofline `max(floor, bytes/BW, flops/COMPUTE)`:

- **`bytes`** is derived generically from shapes: `(sum of input elems + output elems + weight elems) * 2` (fp16). No per-op byte table — this works for *any* op.
- **`flops`** is only known for the few ops with a closed form (matmul/linear/bmm/conv/conv_transpose); every other op contributes 0 flops and is therefore bytes- or floor-bound. That matches calibration: the cheap fused ops all sit at the dispatch floor. (matmul: `x[..,K] @ W[N,K]` stored transposed → `[..,N]`, i.e. `2*M*K*N`.)

The per-node roofline carries **no floor** — the analytic model charges the dispatch overhead once per program, additively, not per node.

### Composition mirrors segmentation

Composition mirrors `_compile`'s segmentation exactly. The graph is one fused program unless it contains a netplist-bridge op, in which case it's cut into fused regions interleaved with native sub-programs:

- A **fused region** costs `floor + sum(max(0, node_cost - floor))` — the *fusion discount*: one dispatch floor for the whole region, plus only the above-floor work of each node.
- Each **cut** adds a `cut_penalty` plus the bridge node's own roofline.
- For a segmented graph, the region count is approximated as the number of distinct fused segments (at most `n_cuts + 1`); the global above-floor sum is kept and `n_regions` floors plus the cuts are added.

### The int8 lever and tie-breaker

int8 is the only dtype lever the model needs: it scales streamed weight bytes by ~0.5 (per-channel int8 streams half the bytes; activations stay fp16). Everything else is structural. There's also an **int8 tie-breaker**: even when a graph is floor-bound (so the roofline ties int8 and fp16 at the floor), int8 always streams ≤ fp16 bytes. That's encoded as a tiny weight-byte-proportional discount so the model is *decisive and directionally correct* (int8 predicted ≤ fp16), scaled well below a floor's worth so it never reorders variants with a real cost difference.

### Calibrated default constants

Read off `ane_cost_model.json`'s calibration (defaults when absent):

| Constant | Meaning | Anchor |
|----------|---------|--------|
| `FLOOR_US` | per-call dispatch floor | smallest fused-op min latency |
| `CUT_US` | per-cut penalty (native sub-program + host round-trip) | composition probe ~150–300 µs; 200 documented mid |
| `BW_BPUS` | effective fp16 streaming bandwidth | matmul sweep: K=4096 streams 32 MiB in ~293 µs → ~114 GB/s ≈ 1.1e5 B/µs |
| `FLOPS_PUS` | effective fp16 compute | conv C=512: 4.83 GFLOP in ~334 µs → ~14.5 TFLOP/s ≈ 1.45e7 flop/µs |

BW and COMPUTE are derived from the matmul + conv scaling fits when present. The cut penalty uses the smallest bridge min as the marginal cut cost when present.

### Bridge ops have a measured cost model

The 19 `NETPLIST` bridge ops (sdpa, argmax, topk, sort, …) have no closed-form flops, so `node_cost()` would cost them at the generic dispatch floor. Instead, `ane_cost_model.json`'s `model.bridge_ops` holds **measured** per-config min latencies, keyed by a size string (e.g. `"sdpa H=8 S=128 D=64"`) and tagged with a `family`. `_bridge_model()` parses these into per-family points so `bridge_cost()` uses the measured value.

Key formats → size params: `bridge_sdpa "sdpa H=<H> S=<S> D=<D>"` → (H,S,D); `bridge_argmax "argmax [<C>,<W>]"` → (C,W); `bridge_topk "topk k=<k> [<C>,<W>]"` → (C,W); `bridge_sort "sort [<C>,<W>]"` → (C,W).

Interpolation is approximate (the grid is sparse, 2–6 points/family): pick a monotone **work scalar** `w(size)` per family, anchor to the nearest measured point by it (in log space, so ratios are symmetric), and scale `min_us` by the work ratio, clamped to the dispatch floor. This captures magnitude and ordering — all the pruner needs — not a per-shape fit. Defensible work scalars: `sdpa ~ H*S²*D` (attention MACs, QK^T + AV both scale this way); `argmax/topk/sort ~ C*W` (a full row-major pass). Families with no data fall back to the roofline.

!!! note "Bridge cost is mostly dispatch-floor dominated"
    15 of the 19 native-bridge ops run via the A1 subprocess-per-call path (no A2 persistent worker exists for them), so their measured per-call cost is **dominated by ~30–60 ms subprocess spawn + ANECCompile load** and is nearly flat in size. The work scalar still gives a sensible nearest-neighbour anchor + ordering; proportional scaling barely moves. `fps` is the exception — it scales ~N*k (seconds/call). An op missing from the table, or whose family has no rows, falls back to the roofline.

## The dispatch floor

The ANE per-call latency is a fixed dispatch + firmware round-trip (~0.2 ms on M1-class parts). `compile` warns once (`DispatchFloorWarning`) when a program's predicted cost is dominated by it, so a caller looping per-sample isn't silently paying the full fixed cost every time. Batching and op-chaining amortize the floor; **concurrency does not** — dispatch is single-in-flight, so threads don't help. To amortize, do more work per call: larger batch N (per-sample cost falls ~linearly toward the compute rate) or chain more ops into one program (`share_buffer` keeps state resident). The check is cheap and structural; it fires only when the prediction is floor-dominated, staying silent on already compute/bytes-bound programs where amortizing buys little.

## The analytic per-chip model (Direction A)

When `estimate(out, target=...)` is given an ANE arch string (e.g. `'h13'`/`'h17s'`), it switches to a **measurement-free analytic per-chip model** valid for all 28 chips. `target=None` (the default) keeps the precise M5-measured heuristic above.

The compiler carries its own analytic cycles→roofline→wall-time model (**ZinNEPerf**, non-SIP). It was decompiled and the per-chip HAL perf fields + freq/efficiency curves were walked live for all 28 targets into the bundled `costmodel_curves.json`. The model:

```
t = overhead + max( flops/peak , bytes/bw )      [per fused program]
```

It's anchored to silicon-measured chips (`_ANCHORS`) and scaled to any other chip from its family's anchor.

### Silicon anchors

Three measured anchors:

- **A13/h13 (M1)** — the latency-roofline fit reproduces the 5 measured M1 convs within ±17%: peak **3.25 TFLOP/s**, BW **9.0 GB/s**, dispatch overhead **0.22 ms**. The headline measured fp16 peak is **1.8 TFLOP/s** (the absolute anchor for `project_peak()`).
- **A14/h14 (M2 Pro)** — measured. Carries a mid-utilization compute ramp `eff_peak = peak * min(1, (flops/F)^q)` (a *capped* power law, F=1.8e10 FLOP, q=0.38). Effective compute throughput is far below peak for mid-size ops (a 768³ GEMM sustains ~1.5 of 7.24 TFLOP/s) and ramps with per-op FLOPs; the cap matters so large ops (a 2048³ GEMM already at peak) aren't slowed. Cuts the h14 25-point grid's mean error 1.61× → 1.16×.
- **A16/h17s (M5)** — the 2026-06-05 loop-closure re-fit: BW **57 GB/s**, floor **~110 µs**, peak **8.9 TFLOP/s** (= `project_peak('h17s')`, validated by the re-fit landing the quoted convs within ~13%).

A15 (no M3 silicon yet) uses the A14 anchor as the nearest below it.

### Scaling rules

`_analytic_constants` takes the nearest measured anchor and scales `{floor_us, cut_us, bw_bytes_per_us, flops_per_us}`:

- **BW scales with CORE COUNT, not clock.** This is the verified M5 loop-closure mechanism. The earlier single-anchor model scaled M1's BW by *clock* and over-predicted M5 ~2× (mean |err| 99%); effective BW tracks cores (16/4 → 5.5×), because a faster clock doesn't widen the DMA path — more cores do.
- **floor scales with operating clock** — setup runs at the engine clock, and overhead shrinks with clock.
- **compute peak scales with `cores * eff_freq`** — the cross-chip peak-throughput scaler.
- **`cut_us` is chip-independent** — a host round-trip for a netplist bridge.

Cross-chip throughput scales by `cores * eff_freq(0.8*fmax)` relative to M1. The operating clock is `0.8 * fmax` (`_CLOCK_FRACTION`); `eff_freq` is the second column of `eff_map_0x7a8` — the engine's effective frequency, already derated for the high-clock MAC-rate falloff on A14+ (M1 = 1.0·f; A14+ derates ~0.84). `project_peak()` gives a measurement-free fp16 peak projection for any target anchored to the measured M1 point, yielding a generational table (M5 ~5.5×, H17d ~22×, M11 ~0.1×) with no silicon beyond M1.

!!! note "Known M1 BW caveat"
    A 2026-06-09 estimate-vs-measured sweep found the M1 fit BW over-predicts *extreme* bandwidth-bound shapes ~4–5× (m1 1×4096×4096: pred ~3700 µs vs measured ~800 µs → ~42 GB/s effective). But it's jointly calibrated with peak/overhead into the ±17% 5-conv fit (`test_cost_model_analytic` pins it), so it can't be bumped alone without regressing that fit — a proper re-anchor needs a joint roofline re-fit over a broader measured set.

### Provenance and fallbacks

`estimate_provenance` reports whether a per-chip estimate is silicon-anchored or extrapolated: a target whose capability family *owns* one of the three anchors is silicon-measured (the A16 tier folds H16/H17* into the h17s point); every other target is extrapolated from the nearest anchor by its `{cores, clock, efficiency}` curve. An arch missing its own curve folds to its family's representative die (matching `_targets._ARCH_FAMILY` tiers).

## The precision risk model

A companion to `estimate()`: a cheap structural pattern-matcher (not an error bound) over three characterized fp16 failure modes (`fp16_envelope.py`). Each flagged node gets an order-of-magnitude error proxy in [0,1]; the per-graph signal is the max over nodes. `_precision_signal` runs this as a static pass (no device work) and warns — or, with `strict`/`validate=True`, raises — so an unstable computation never runs silently.

- **(a) Narrow reduce-sum** — `reduce_sum` over signed terms with a long contraction: the narrow fp16 reduce accumulator re-injects per-add rounding the wide matmul avoids. Error grows ~`sqrt(K) * fp16_eps`, capped at 1.0. Fixable by the `reduce_sum→matmul` rewrite (≥ accuracy). A tensor is treated as possibly-signed unless it's a square/abs/relu/sigmoid/exp/softplus output (non-negative → a sum over it doesn't cancel).
- **(b) Cancellation subtract** — subtract of two large, structurally-small-result quantities (CFG-style). Not detectable structurally (data-dependent), so it's a *candidate* flag only, fixable only if operands carry sub-ulp bits (paired-fp16).
- **(c) Compile cliffs** — group_norm / large-feature-map cliffs. The rank-4 tiled lowering reduces over `[1,G,C/groups,H*W]`, so the cliff is `max(C/groups, H*W) > 65536` (aligned to `af.group_norm`'s guard), **not** the flattened product — the tiling keeps 640@64 and 512@128 under the cap.

!!! note "cancel_sub is informational-only"
    Default hotspots are only the *reliable, structurally-determinable* signals: a narrow `reduce_sum` over the fp16-clean floor, and the group_norm per-axis wall. `cancel_sub` is speculative — most subtracts (residuals, losses, differences) are benign and unconfirmable without data, and flagging every one trained users to ignore the warning. It stays in `nodes` (surfaced by `precision_risk(verbose=True)`) but does not raise the default warning. The tuner's vs-fp32 gate confirms any real gain.

Calibration constants: `_FP16_CLEAN = 1e-3` (a clean fused fp16 op's relative error, the corpus median — fp16 has ~3–4 decimal digits); `_NARROW_SUM_FLOOR = 256` (contraction length above which a signed `reduce_sum` starts losing digits to the narrow accumulator). These are order-of-magnitude proxies, deliberately coarse.
