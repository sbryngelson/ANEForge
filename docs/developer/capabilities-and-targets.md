# Capabilities and targets

ANEForge keeps a single, closed, machine-checkable census of *what ANE capability it surfaces* — the capability registry in `_capabilities.py` — and a per-family target table covering 28 ANE chips. This page covers the census and its classification discipline, the cracked layers and arch-gated walls, the target families and cross-compile gating, and the cross-chip fp16-divergence heads-up.

## The capability census

The authoritative runtime op sets are `_EMIT` (the fused e5rt-MIL emitters) and `NETPLIST_OPS` (the native Path-A bridge ops), both in `_compile.py`. But the *discovery evidence* behind them — which layers were proven on silicon, which are genuine hardware walls, which are only predicted — was scattered across the reverse-engineering corpus, and nothing reconciled the two. A new op could appear unclassified, or a "surfaced" op could silently stop compiling unnoticed.

`build_registry()` closes the loop. It introspects the live `_EMIT` / `NETPLIST_OPS` and merges them with curated, citation-carrying classification data for everything that can't be introspected (cracked-but-not-promoted layers, the arch-gated negatives, the predicted-but-unbuilt frontier). The result is a **closed classification of every ANE capability ANEForge knows about**. `tests/test_routes.py` fails CI if the live runtime drifts from this registry either way.

!!! note "This is a publication artifact"
    Every entry carries evidence. `cracked` entries cite their bridge in the reverse-engineering corpus; every negative cites the finding that established the wall; every `reachable`/`not-authorable` cites the sweep matrix.

### Classification discipline

Every entry is exactly one of:

| Status | Meaning |
|--------|---------|
| `fused` | In `_EMIT`; lowers to e5rt-MIL, fuses into one program. |
| `bridge` | In `NETPLIST_OPS`; runs as a native Path-A sub-program (a graph cut). |
| `cracked` | Reverse-engineered native layer **proven on silicon** in the corpus, but not yet promoted to the frontend. |
| `reachable` | The native MIL op compiles + runs correct on ANEForge's own e5rt path (proven by the full-vocabulary sweep), but is not promoted to a frontend `_EMIT` emitter or `af.` method. Reachable through the MIL door, not direct `Type=` netplist authoring — so not a cracked native layer. |
| `arch-gated-negative` | Known **not** to work, carrying the specific wall (an authored op/config the ANE backend rejects or fails to lower). |
| `not-authorable` | Control-flow / list / state ops (cond, while_loop, make_list, read_state, …) with no single-op MIL form on the single-procedure feed-forward surface — the recurrence/scan wall. Recorded, not defeated. |
| `predicted` | Predicted crackable by the callable-`_ANECValidate*` heuristic, but not built and not verified. |

`fused`/`bridge`/`cracked`/`reachable` all carry `verified="silicon"` — the distinction between them is **promotion, not evidence**. `predicted` is explicitly `verified="no"`.

The registry is exhaustive over the full **166-op MIL vocabulary** swept in `full_mil_vocabulary_sweep.json` (116/166 reachable), not just the 50 `_ANECValidate*Layer` validators. A sweep layer (D) synthesizes an entry for every swept op not already represented, classifying from its sweep `klass`: `reachable+correct`/`reachable` → `reachable`; `not-implemented-on-ANE` → `arch-gated-negative`; `compile-walled` → `not-authorable` for control-flow/list/state, else `arch-gated-negative`. Curated entries always win over the sweep (exact-name match skipped), preserving the hand-written walls and evidence.

### The cracked-layer manifest (the "26 cracked")

A **cracked native hardware layer** is a distinct native ANE layer reached by the *netplist-authoring* method (authoring `Type=<Layer>` directly in the netplist / Path-A) which Apple's user-space MIL frontend does **not** emit. In the registry these are exactly the entries with status in `{cracked, bridge}`: `bridge` ops are Path-A netplist sub-programs authored directly; `cracked` ops are unpromoted direct-netplist cracks.

```
distinct cracked layers = (#cracked) + (#bridge) − 1
```

The one dedupe: `slice_by_index` (cracked) is the same hardware capability as `input_view` (bridge) — both `Type=InputView` — folded in so the 26 are distinct layers, not 27 with an alias.

!!! note "Promoted ops are not retroactively counted as cracks"
    Some `fused` ops (e.g. PixelShuffle, L2Norm) were originally reached by netplist authoring before being promoted to a MIL `@op` emitter. They're excluded from the manifest: their *current* route is the MIL emitter, not direct `Type=` authoring. The manifest counts the current direct-netplist route only.

## Cracked-but-unpromoted native layers

Native layers proven on silicon (a self-verifying `*_fused.py` in the corpus) but not promoted into `_EMIT` / `NETPLIST_OPS`:

- **Dropout** (`Type=Dropout`) — inference identity at rate 0; not promoted (no training-time use on the inference engine).
- **Broadcast** (`Type=Broadcast`, BroadcastInfo replicates a length-1 axis) — MIL add/sub/mul already broadcast in the fused path; standalone layer unneeded.
- **Resample** (`Type=Resample`, NearestNeighbor mode) — the BILINEAR mode is now promoted to fused (`resize_bilinear`/`upsample_bilinear`); NearestNeighbor stays cracked (fused `resize_nearest_neighbor` covers it).
- **RandomGenerator** (Normal/Bernoulli/Categorical/Gaussian) — on-device RNG compiles, but seeding/streaming semantics are unsettled; host RNG used today.
- **Shape** — static shapes are known at compile time, so no frontend need yet.
- **DynamicGOC** (gain-offset-clamp, `out=idx*(data+upd)`) — compiles+loads; it is **not** a scatter (Mode/Axis are inert). Niche.
- **Tile** — `Params` is a dimension-NAME-keyed factor dict (`{'Width':fx,'Height':fy,'Channel':fc}`, *not* FactorX/Y/Z). Cracked 2026-05-30 (was `predicted`). A clean bridge-promotion candidate (B=1 static).
- **slice_by_index** (`Type=InputView`) — bit-exact contiguous slice, the same capability as the `input_view` bridge op. Promotable as a frontend `slice`, not a new layer (not double-counted).

## Arch-gated walls (negatives)

A **negative** means: rejected at compile, or the named configuration fails ANECCompile/HWX codegen even though the validator accepts the schema. Each carries its specific wall; the `probe` field (when present) is a tiny ANEForge call CI runs to confirm it still fails.

### Native-op gaps reachable by decomposition

A key distinction (2026-05-31 calibration): the native **op** is unimplemented on the ANE backend ("Not implemented … not supported on any backend"), but the **capability** is reachable on the same e5rt path via a decomposition. These are native-op gaps, not hardware walls:

- **reduce_prod** → `exp(reduce_sum(log(x)))` (relerr ~1e-3; positive inputs; sign/zero need extra ops).
- **gather** → for trace-time-constant (STATIC) indices, a one-hot @ matmul (bit-exact). Only **dynamic** large-vocab embedding lookup stays host-side (one-hot is O(vocab) and needs runtime indices).
- **cumsum** → an upper-triangular ones matmul `x @ U` (relerr ~3e-4; O(N²) in the matmul) — **not** a "no parallel-prefix silicon" wall.

### True walls (no in-graph route)

- **scan / iir_recurrence** — no data-dependent in-graph loop: sequential recurrence is inexpressible on the feed-forward engine. Only static FIR-unroll fits.
- **scatter** — no scatter-write path; DynamicGOC is gain-offset-clamp only.

### dtype walls

- **fp32_compute** — the ANE computes in fp16 only. ANEForge accepts fp32/fp64 weights but silently casts them to fp16 (no frontend raise — the wall is a compute-precision fact).
- **int32_compute** — rejected at MIL parse ("not implemented"). probe: `int32_weight`.
- **bf16_compute** — no path on the backend (cleanly rejected; numpy has no native bf16, so no clean frontend probe).

### Layer / codegen walls (validator accepts schema, M5 HWX codegen fails)

- **conv_kernel_ge16** — conv (and the CrossCorrelation bridge) is capped at kernel K≤15 on M5: a 1×K kernel with K≥16 fails ANECCompile. (`aneforge/dsp.py` routes wider kernels to FFT.)
- **group_norm_large_featuremap** — **relaxed** by rank-4 tiling: group_norm lowers to `[1,G,C/groups,H*W]`, so the bound is `max(C/groups, H*W)` vs. the per-axis cap 65536, not the flattened product. SD-UNet's 640ch@64 (81920) and 512ch@128 (262144) now compile in fp16; `af.group_norm` only raises when a single tiled axis > 65536.
- **square_opcode_tiling** — the native `square` opcode fused with a following nonlinearity fails ANECCompile when `(Width % 128) > 64`. ANEForge emits `mul(x,x)` instead (identical math, compiles everywhere), so it never surfaces this.
- **rank5_transpose_matmul** — transpose+matmul fails ANECCompile at rank≥5 (the rank-4 cap): a fully-split tensor carries at most 3 factor groups, forcing matmul-FFT to 3-stage rather than full log-depth radix-2.
- **batch_to_space_indivisible** — BatchToSpace with input batch `N % (FactorX*FactorY) != 0` fails HWX codegen. (`af.batch_to_space` guards at construction.) probe: `batch_to_space_indivisible`.
- **topk_k3_k4** — TopK with K ∈ {3,4} is a fixed forbidden band (ANECCompile fails); K=1,2 and K≥5 work. (`af.topk` guards.) probe: `topk_k3`.
- **minmax_norm_channel_axis** — MinMaxNormalization over the Channel axis fails HWX codegen; Width/Height work. (`af.minmax_norm` rejects Channel.) probe: `minmax_norm_channel`.
- **lrn_channels_ge16** — LocalResponseNormalization with KernelChannel=C fails ANECCompile for C≥16; C≤15 works. (`af.lrn` guards.) probe: `lrn_c16`.
- **scaled_elementwise_sub** — ScaledElementWise op='Sub' is rejected; op='Mult' silently ignores `scale`. (`af.scaled_elementwise` guards both.) probe: `scaled_elementwise_sub`.
- **Unflatten** — fails HWX codegen for all variants (composite-lowering specific); bare Reshape/Transpose compile fine.
- **KernelRasterizer** — im2col unfold only compiles in the degenerate identity config (OutputChannels=1, output dims = input dims); real im2col always fails ANECCompile.
- **RingBufferWriter** — a multi-procedure streaming-state primitive; fails from any single-procedure netplist regardless of Params.
- **CropResize** — texture-family codegen gate: direct-netplist CropResize fails ANECCompile on M5 (plane-equation coefficient constraint).

### Internal-optimizer-gated and architectural walls

- **MATDECOMP_MATMULT_0x19** — internal-optimizer-gated (not schema-gated): the fused matrix-decomp-matmul composite (0x19) is emitted only by `ZinIrOpt::BatchMatDecompMatMul` on a batched topology the netplist can't express; 7 variants all lower to 0x60.
- **PEFUSED_GOC_0x5C** — internal-optimizer-gated: nested-container grammar recovered but no single-0x5C HWX emerged.
- **RCAS** — internal-optimizer-gated: opcodes 0x44/0x65 have a `ZinRCASLayer` + internal validate but no exported `_ANECValidateRCASLayer` and no .tbd entry — not netplist-reachable. Same class as MATDECOMP_MATMULT_0x19.
- **MatrixDecomposition_factorization** — the native MatrixDecomposition (NonQRGivens) is not currently callable (carriers compile, the factorization doesn't). The LAPACK wall is architectural, not numerical.
- **on_device_rng_seeded** — RandomGenerator compiles but seeding/streaming semantics are unsettled; RNG kept host-side.
- **custom_hwx_load** — hits the kernel-signature wall (unentitled).
- **nan_to_num** — the ANE does not propagate NaN, so NaN-handling ops are broken (no fix).
- **bitwise_logical** — bitwise/logical ops (and/or/xor/not) rejected (not in the fp16 surface); trig/hyperbolic inverses (acos/asin/sinh) are netplist dead-ends too.
- **Reverse** — the native netplist Reverse *layer* is schema-unreachable (no exported `_ANECValidateReverseLayer`, no atlas opcode — only an internal MIL-frontend op that lowers away). The `predicted` classification was over-optimistic. The MIL `reverse` *op*, however, is reachable+correct on the e5rt path, so the reversal capability is available through the MIL door.

!!! note "Walls reconciled to reachable (2026-06-02)"
    Several earlier "walls" were over-cautious *direct-netplist* probes scoped to the netplist door; the equivalent **MIL ops** compile + run on the e5rt path. **AffineTransform** → `affine` (genuine warp, relerr 2.3e-3) and **Resample_Bilinear** → `resize_bilinear`/`upsample_bilinear` (relerr 2e-4) now ship as **fused** _EMIT ops. **instance_norm** → fused `x.instance_norm(...)` (the decomposition route remains too). **NMS** → `non_maximum_suppression` is now `reachable` (presence-only, not numerically validated — selection-ordering output). **No quantization walls remain**: the former "blockwise_int8"/"int4_weight"/"sparse" negatives were malformed empty-paren probes (the named-arg form compiles, relerr ~3e-4); int8-affine, int4-LUT, sparse, and blockwise all ship via `compress=`. The remaining texture-family negative is CropResize.

## The predicted frontier (Pad)

- **Pad** — predicted netplist (`ZinPadLayer` + `ZinPadValidator`; the `kANECNetUnitPaddingInfo` dict schema is unrecovered). The validator accepts the schema (status=1) but ~13 binary-derived PaddingInfo spellings all fail ANECCompile; spatial-only, constant amount_before/after. Next step: recover the exact dict from a CoreML-emitted Pad bundle. As with Reverse, the MIL `pad` *op* is reachable+correct, so the padding capability is available through the MIL door.

## Targets: the 28 ANE families

ANEForge models 28 ANE targets across the M1–M5 generations (`_targets._ARCH_FAMILY` tiers). Each target carries per-family capability caps — conv kernel width, max tensor dim, and the per-op `MinimumFamily` floor — that are **HAL-data gated**. The cost model's per-chip curves (`costmodel_curves.json`) cover the distinct cost profiles; an arch missing its own entry (h14g, h16c, h17a, h18, …) folds to its family's representative die.

Three families are **silicon-measured** (own a cost anchor): A13/h13 (M1), A14/h14 (M2 Pro), A16/h17s (M5). The A16 tier folds H16/H17* into the h17s point. Every other target is extrapolated by its `{cores, clock, efficiency}` curve — see the [analytic per-chip model](optimizer-and-cost.md#the-analytic-per-chip-model-direction-a).

### Per-family gating

`_retarget_for` gates a graph for a target family before lowering: it raises on the H13+ hard floor (ANEForge requires an A13+ ANE — MIL is only supported for H13+), on an unreachable op, or on a tensor exceeding the family's limits, and substitutes below-floor ops that have an in-graph decomposition (`sin`/`cos` → `aneforge.special`). On the host family with an all-native graph it's a no-op fast path.

## Cross-compile validation

`cross_compile_check` asks: does the graph compile for *another* ANE family, checked from this host? It returns True iff the e5rt compiler produces a library for that `TargetArchitecture`. This is compile-level validation only — a different-family host cannot execute there. It's the **keystone of cross-chip CI**: validate that the op corpus compiles for every chip family on one box. Numeric correctness still needs real silicon.

!!! warning "Three traps the gate must avoid"
    1. **Silent host fallback.** The e5rt compiler silently falls back to the host target on an unrecognized `TargetArchitecture` string (measured: `'zzz'` compiles fine), turning a typo into a false cross-target pass. Gate on the known-target table first.
    2. **Caps not enforced cross-family.** A target's per-family caps (conv kernel width, max tensor dim, below-`MinimumFamily` ops) are HAL-data gated, and the host cross-compiler does *not* reliably enforce a different family's caps when emitting for its `TargetArchitecture` — so a cap violation would compile here and only fail on real silicon (a false CI pass). `preflight()` answers the same question statically and family-aware (it's what `compile(target=)` gates on), so reject up front and skip the compiler entirely.
    3. **fp16 divergence.** Annotate cross-chip fp16 divergence risk (Direction B) before compiling — warn-only, so a typo'd/unreachable graph still surfaces its compile error.

## Cross-chip fp16 divergence (Direction B)

`CrossChipFP16Warning` fires (via `cross_compile_check`) when a graph compiles for a different-family target but carries an op whose **fp16 value** can diverge from the host's on that chip (per the Direction B HAL-field predictor). The compile is still valid — this is a numeric heads-up, not a rejection. `_warn_fp16_cross_chip` groups by risk class (one line per op×class) and never rejects.

`_fp16_risk_kind` maps a node to the predictor's op-kind: `'slice'`, `'reduce_square'` (a reduce feeding a square/mul = variance/L2/RMS — the 0x494 fuse axis, flagged only when a source is a reduction, not every elementwise square), `'reduce'` (any reduction/softmax/norm), or `''`. The risk classes:

- **saturation** — finite→inf; `|value| > 4094` on A13's ×16 crop-DMA slice.
- **round1** — ~1 fp16 round; the 0x494 reduce→square fusion differs A13↔A14+.
- **ulp1** — ≤1 ULP; the 0x3f0 reduction-route threshold differs.

### The pre-A16 slice saturation quirk

A silicon-pinned quirk worth its own note: nonzero-width (last-axis) `slice_by_size` offsets on a **pre-A16** ANE lower through a Q.4 fixed-point crop-DMA (an implied ×16 scale), with two distinct failure modes:

1. **Wrong elements** when multiple width-offset slices are concatenated — the gather axis-1 bug, confirmed on A14 and A13 at any magnitude. A single width-offset slice selects correctly (a 180-config bare-slice sweep is numpy-exact on A14), so this mode is specific to the concat-of-width-slices pattern.
2. **Saturation** — a sliced element with `|value| > 4094` (= 65504/16) clamps to ±inf, magnitude-driven, on a single width-offset slice. Silicon-confirmed on A13 (conv weight-grad) and A14 (4094 finite → 4100 inf, bit-exact); A15 pending M3 data.

A zero last-axis begin, or an offset on any non-last axis, avoids the path and is exact; A16/M5 is unaffected. `gather` and the conv im2col backward already route off the last axis, so `_warn_h13_slice_saturation` is a safety net for any remaining hand-written occurrence.
