# Cross-chip deployment

ANEForge compiles a tensor graph for any Apple Neural Engine generation from one host,
and predicts how that graph will behave on the target chip without owning it. Apple's
ANE compiler is the same code across chips; everything that differs between an M1 and an
M5 is per-chip target data plus the `MinimumFamily<N>` op-capability floors. Cross-chip
support is mostly a lookup problem: ANEForge ships a per-family capability table, a
measurement-free per-chip cost model, and a static fp16 divergence predictor.

## Where these live

`af.compile`, `af.estimate`, and `af.project_peak` are top-level (`import aneforge as af`).
The rest of the cross-chip machinery is in two submodules, imported directly:

```python
from aneforge import _targets as tg        # Family, detect_family, family_of_arch,
                                           # arch_for_family, op_status, limit, preflight,
                                           # predict_fp16_divergence, native_streams
from aneforge._compile import cross_compile_check
```

The examples below use these import names.

## The family model

ANEForge groups the ANE into five capability families, exposed as `tg.Family`:

| Family | Tier | Chips |
| --- | --- | --- |
| `OLDER` (1) | A11/A12 | below the MIL floor - cannot run ANEForge |
| `A13` (2) | A13 | M1 |
| `A14` (3) | A14 | M2 |
| `A15` (4) | A15 | M3 (also M11) |
| `A16` (5) | A16 | M4, M5, and every A17/A18 die |

The e5rt/MIL path ANEForge dispatches through carries a hard assert, "MIL is only
supported for H13+ ANE architectures", so family 2 (`tg.MIN_FAMILY`) is the absolute floor
regardless of per-op traits. `tg.Family.OLDER` chips are unreachable.

### The M(n) = H(n+12) ladder

The mapping from M-series generation to ANE architecture is `M(n) == H(n+12)`:

| Chip | Arch | Family |
| --- | --- | --- |
| M1 | H13 | A13 |
| M2 | H14 | A14 |
| M3 | H15 | A15 |
| M4 | H16 | A16 |
| M5 | H17s | A16 |

M2/M3/M4 resolve to ground-truth capability families this way (only each chip's
absolute power/watt rail still needs its own silicon).

### Runtime arch vs compiler target

There are two distinct namespaces, and ANEForge keeps them separate:

- Runtime arch - what the OS reports for the host ANE: `h13`, `h13g`, `h14g`, and so
  on (the `g`/`c`/`s`/`d` suffix is a die variant: more NE cores, same capability).
- Compiler target - the `TargetArchitecture` string the compiler accepts: `h13`,
  `h16s`, `h17s`, `h17d`, etc.

The compiler enumerates 28 valid `TargetArchitecture` strings across the five families.
A17 (the H17* targets) and A18 (H18) add no op capabilities over A16 - they
scale the NE-core count only (the suffix `g`/`s`/`c`/`d` is a core-count variant; M5 ==
H17s). The A16 tier is therefore the capability ceiling, and ANEForge folds every A17/A18
arch into `Family.A16`.

`tg.arch_for_family()` returns one representative compiler arch per family (`h13`, `h14`,
`h15`, `h16s`), which keeps the cross-compile matrix to one column per capability tier.

## Compiling for another chip

`af.compile(out, target=...)` lowers and compiles a graph for a specific family.
`target` may be a compiler arch string (`'h13'`), a `Family` int, or `None` (the default,
which auto-detects the host). Before lowering, the graph is gated for that family
(`_retarget_for`), which can do one of four things per op:

- native - emit it directly (the bulk of the F0/F2 vocabulary: conv, matmul, pooling,
  elementwise, activations, softmax, norms, reductions, sqrt/rsqrt/erf/exp2/log2, SDPA,
  resize, tile, space<->channel - all native on A13+).
- decompose - substitute an in-graph equivalent. `sin`/`cos` are A15+ on the silicon,
  so below that floor they are rewritten through `aneforge.special` (a Horner expansion);
  `dropout`/`random` route to host-side RNG.
- reject - a clear compile-time error. Texture-engine ops (`crop_resize`, `resample`,
  `affine`, A14+) and bridge ops that pass validation but reject at codegen on M1
  (`topk`, `sort`, `dynamic_slice`, A14+) hard-reject below their floor rather than crash
  at dispatch.
- oversize - a tensor (or a known internal-reshape extent, e.g. `group_norm`'s rank-4
  tiled lowering) exceeds the family's extent cap for that op class and must be tiled.
  The caps are per-op-class (A14/A16-measured): spatial/contraction 16384 through A15 ->
  65536 at A16; channel 65536 on both; transpose 2^2^3-1 (A14) -> >=2^2^4-1 (A16). preflight
  classifies axes accordingly (channel cap on axis 1 of rank-4, wide extent for
  transposes, spatial elsewhere). The A13 row is measured on the live M1 and equals
  A14 exactly (16385 W/H, 65537 C both reject), so the caps are generation-monotone (no
  inversion). Conv kernel width is a further monotone per-family cap (A13 <= 13, A16 <= 15;
  16+ -> `space_to_depth`), enforced family-aware in preflight.

`tg.preflight(out, family)` runs this walk as pure static analysis and returns the lists
(native / decompose / reject / oversize) plus an `.ok` flag, with no compile and no
hardware. It is the fast pre-check before a cross-compile.

```python
import aneforge as af
from aneforge import _targets as tg

x = af.input((1, 3, 32, 32))
y = af.conv(x, W, pad=1).relu()             # any aneforge graph

rep = tg.preflight(y, tg.Family.A13)        # will it run on an M1?
print(rep.ok, [r.op for r in rep.reject], [r.op for r in rep.oversize])

af.compile(y, target="h13")                 # gate + lower for M1 from any host
```

### The h13 slice-saturation warning

On the A13/M1 family only, a `slice_by_size` with a nonzero last-axis begin-offset
routes through a Q.4 fixed-point crop-DMA with an implied x16 scale, which silently
clamps any sliced element with `|value| > 4094` (= 65504/16) to +/-inf. A zero last-axis
begin, or an offset on any other axis, takes a clean route. The value threshold is a
runtime property, so `compile(target='h13')` warns about such a slice rather than
blocking it. A14+ takes the clean route and no warning fires.

### e5rt TargetArchitecture and the silent-fallback gotcha

Cross-compilation is separable from device-load: the e5rt compiler honors
`TargetArchitecture=<arch>` via its custom ANE-compiler options, so one box can compile a
library for any chip. The catch: e5rt silently falls back to the host target on an
unrecognized `TargetArchitecture` string (a typo like `'zzz'` compiles cleanly against
the host, turning a mistake into a false cross-target pass).

`cross_compile_check(out, target)` (from `aneforge._compile`) guards against this. It first
rejects any arch not in the known-target table (so a typo raises instead of passing), then
lowers the fused graph and asks the e5rt compiler to produce a library for that
`TargetArchitecture`, returning `True` iff it succeeds. This is the keystone of cross-chip
CI: validate that the op corpus compiles for every family from one machine. It is
compile-level validation only; numeric correctness still needs the real silicon, and
bridge/segmented graphs are out of scope (it raises). `cross_compile_check` also emits the
fp16 divergence warning below before compiling.

```python
import aneforge as af
from aneforge._compile import cross_compile_check

x = af.input((1, 3, 32, 32))
y = af.conv(x, W, pad=1).relu()
cross_compile_check(y, "h13")               # True iff the e5rt compiler builds it for M1
```

### Host detection and the env override

`tg.detect_family()` resolves the host ANE family in order:

1. the `ANEFORGE_TARGET` environment variable (an arch string, e.g. `ANEFORGE_TARGET=h16s`) - an explicit override for CI, for forcing a higher tier on an under-detected
   chip, or for host-independent tests.
2. the CPU brand string (`Apple M5 Pro`), mapped through the M-series ladder. The model
   identifier is a trap (`MacBookPro17,1` is an M1, `Mac17,8` an M5), so detection reads
   the clean CPU brand; the Pro/Max/Ultra variant changes core count, not family.
3. `MIN_FAMILY` (family 2 / H13) as a conservative floor for any unmeasured future chip,
   with a one-time warning. A family-2 program runs on every H13+ chip (higher families
   are strict supersets), so under-claiming stays correct.

## fp16 portability

The MAC accumulator width and the compiler `__TEXT` are uniform across chips, so cross-chip
fp16 value divergence can only come from HAL-data-selected codegen routes that reorder or
saturate fp16 ops - making it statically predictable.
`tg.predict_fp16_divergence(kind, shape, target_a, target_b, ...)` compares the relevant
per-family HAL fields and returns one of four verdicts, strongest first:

- `saturation` - a slice with a nonzero last-axis begin-offset where one target is
  A13: A13 routes the offset through the Q.4 x16 crop-DMA that clamps `|value| > 4094`
  (= 65504/16) to +/-inf, while A14+ takes a clean route. This is the only finite->inf
  axis. It is magnitude-gated: a finite `max_abs <= 4094` downgrades it.
- `round1` - a reduce immediately followed by a square/mul (variance, L2-norm,
  RMSNorm) where the `0x494` reduce->square fusion bit differs (A13 = 0, A14+ = 1): one
  target keeps a rounding step the other fuses away, ~ 1 fp16 round of difference.
- `ulp1` - a reduction/softmax/norm whose `0x3f0` route threshold differs (192 on
  A13/A14, 384 on A15+): a partial-sum reorder, <= 1 ULP. Empirically a no-op (block
  accumulation absorbs it), but a differing field still flags the risk.
- `none` - no HAL field selects a differing route.

`cross_compile_check` surfaces any non-`none` verdict as a `CrossChipFP16Warning` (a
numeric heads-up, never a rejection; silence it with
`warnings.filterwarnings('ignore', category=aneforge.CrossChipFP16Warning)`).

### The A13 conv weight-grad guard

Training a conv on M1 builds im2col from width-offset slices, so the loss-scaled backward
activations pass through the same A13 Q.4 x16 crop-DMA. `Trainer` warns (warn-only) when
the target is A13, the graph trains a conv weight with `kW > 1`, and `loss_scale >= 512`,
since `loss_scale x |backward activation|` could then exceed 4094. This is deliberately
not an auto-cap: a real normalized CNN trains identically at loss_scale 128, 1024, and
65536 on M1 - the saturation needs unusually large backward magnitudes a real net never
reaches. A14/A16 have no such route.

### Measured end-to-end training parity

The same seeded MNIST-CNN, trained 300 steps, reaches 0.9080 on M1, 0.9080 on
M2/A14 (deterministic x3), and 0.9070 on M5 - the A16 generation differs by
exactly one test sample out of 1000 (0.1%), each run internally deterministic. A14 lands
exactly on M1's number, so the fp16 drift boundary is the A15/A16 generation, not per-chip
noise. Training is chip-portable, cross-chip fp16 drift accumulating to a negligible
<= 1-ULP-per-step level.

## Per-chip performance

ANEForge ships a measurement-free per-chip cost model (a bundled `costmodel_curves.json`
covering all 28 targets). Two entry points expose it:

- `af.estimate(out, target='hXX')` - projected latency in microseconds for a compiled
  program on a given chip, via an analytic roofline `overhead + max(flops/peak,
  bytes/bw)` per fused program. With `target=None` (default) it uses the precise
  M5-measured heuristic instead, so the optimizer is unchanged.
- `af.project_peak('hXX')` - the fp16 peak-throughput projection, returning
  `{tflops, rel_m1, cores, ghz}`, the generational-scaling table from one anchor.

The model is anchored to silicon-measured chips (M1, M2, M5; only A15/M3 is unmeasured)
and validated out-of-sample: `estimate(target='h17s')` reproduces the M5 loop-closure
convs with mean |error| ~12%.

Honest caveat on `project_peak`. It reports a peak ceiling: M5 projects to ~5.5x
M1, but the measured M5/M1 end-to-end latency speedup is 2.3-3.3x (the dispatch floor
caps real workloads below both the bandwidth ratio and the compute-peak ratio). Use
`project_peak` for the generational shape, not as a promised speedup, and measure
on-device when the number matters.

## Per-family compressed-weight streaming

Compressed weights are only a bandwidth win where the per-family lowering keeps them as a
native streaming kernel rather than folding them to a dense fp16 const. Which encodings
stream is family-specific, exposed as `tg.native_streams(family)`:

| Family | Native-streaming encodings | Measured win |
| --- | --- | --- |
| A13 (M1) | int4-LUT, sparse | int4 2.37x |
| A14 (M2) | int4-LUT, int8, sparse (blockwise folds) | 1.6-1.8x |
| A15+ (M3, M4, M5) | int4-LUT, int8, sparse, blockwise | 1.6-1.8x (M5-measured) |

On M1, `int4` (the `constexpr_lut_to_dense` palette) and `sparse`
(`constexpr_sparse_to_dense`, on weights >= 50% zeros) stream natively; int4 streams
better than on M5 (2.37x vs 1.6-1.8x), because M1's lower effective bandwidth makes
quartering the weight bytes pay off more. int8 and blockwise compile and stay
correct on M1, but the compiler folds them to dense fp16 (an accuracy cost for zero
bandwidth win). A14/M2 adds int8 streaming but still folds blockwise; from A15 up the
full set streams.

ANEForge wires this into `compile(compress='auto', target=...)`: `auto` is
family-aware, considering only encodings that stream natively on the target family
(host-detected when `target=None`). On M1 that is int4-LUT and sparse, so `auto` skips
int8/blockwise there and a rejected int4 falls back to fp16 rather than a folding
encoding; on A14+ it picks from the wider native set. Explicit single-mode knobs
(`compress='int8'`, etc.) are never filtered - they always compile, correctness
preserved even where streaming does not help.
