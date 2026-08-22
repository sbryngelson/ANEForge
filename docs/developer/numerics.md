# Numerics

ANEForge computes in fp16 only, fed through a wide (fp32-class) accumulator. This
page is the model for reasoning about accuracy on the ANE: where the fp16 limit
actually bites, why every reduction is a matmul, and the compensated-arithmetic
and accuracy-gated-compression machinery built on top.

## The fp16-only compute model

The ANE computes in fp16 and nothing else. fp32, int32, and bf16 are not
implemented on the backend, so the graph builder rejects them at construction
rather than failing cryptically at compile time. Every value in flight is a
16-bit float with roughly 3-4 significant decimal digits.

That single dtype constraint is the root of everything below. The headroom is real
but narrow:

- The fp16 range ceiling is 65504. Any intermediate that crosses it saturates
  to infinity, silently corrupting the result. Several modules pre-scale to keep
  intermediates inside this envelope (see the dynamic-range walls below).
- The fp16 precision floor is ~1e-3 relative error. A polynomial that is exact
  in fp64 is capped at ~1e-3..1e-4 relerr once its operands and products round to
  fp16.

## The WIDE accumulator and where fp16 actually bites

The crucial asymmetry: reductions and matmuls accumulate in a wide,
fp32-class accumulator fed by radix-4 fp16-rounded input tiles. The inputs and
products round to fp16, but the running sum does not.

So representable sums are near-exact where a naive fp16 running sum would have
stalled long ago:

- A sum (or dot) of 16384 ones is bit-exact, where naive fp16 saturation of the
  running sum stalls at ~2048.
- A `+1` survives next to a 16000 partial that an fp16 running sum would simply
  swallow.

So the fp16 limit lives at the products and the I/O cast, not the running sum:

| Source of error | Affected by the wide accumulator? |
| --- | --- |
| Rounding of operands to fp16 (storage / I/O cast) | No - irreducible |
| Rounding of each product to fp16 | No - irreducible |
| Accumulating the products into the sum | **Yes - near-exact** |

Cancellation-heavy reductions still lose precision, because the loss happens in the
fp16-rounded operands feeding the sum, not in the summation itself. A tiny result
formed by subtracting large, nearly-equal quantities is the wall - what the
compensated-arithmetic path (below) exists to climb.

### The reduce_sum trap

One catch every numerically careful kernel must respect:

> On this ANE, `matmul` accumulates **wide** but `reduce_sum` accumulates
> **narrow** (fp16). They are not interchangeable for accuracy.

This is the single most important numerics rule in the codebase. Every dot product
and accumulation in the math modules is written as `(u * v) @ ones` - a matmul -
never `(u * v).sum()`. A narrow `reduce_sum` re-injects exactly the rounding error
a careful kernel is trying to avoid.

### reduce_sum-as-matmul

The optimizer makes this rule mechanical. The `reduce_sum_to_matmul` rewrite
rebuilds a single-axis `reduce_sum` node as a contraction against a ones-vector:

```
x[..., K].sum(-1, keepdims) == x @ ones[K, 1]
```

This is mathematically identical (`sum_k x_k == x @ 1`) and strictly >= accuracy
under cancellation, because it routes the sum through the wide accumulator. It is
therefore classed lossless-or-better: always safe, and the precision-aware tuner
offers it as the flagship cheap accuracy win (one matmul vs one reduce). Only the
single last-axis case is rewritten directly; a non-last axis is transposed to the
end first, and multi-axis sums are left alone.

The same identity powers `cumsum`: no native cumsum, but a last-axis cumulative sum
is exactly `x @ triu_ones` - a matmul against a baked upper-triangular ones weight,
made exact by the wide accumulator.

## Compensated / paired-fp16 arithmetic

When cancellation is the wall - a tiny result from large nearly-equal operands,
where the fp16 rounding of the operands and products (not the accumulation) swamps
the signal - ANEForge offers paired-fp16 ("double-fp16") extended precision, with
no fp32 anywhere in the compute path.

A value is carried as an unevaluated pair `(hi, lo)` with `hi = fp16(x)` and
`lo = fp16(x - hi)`, so the pair represents `hi + lo` to roughly twice the fp16
significand (~22 effective bits). Both limbs are ordinary fp16 tensors, so every
paired operation compiles to a pure-fp16 graph that runs on the ANE.

The arithmetic is the classic error-free transforms, every intermediate rounded
to fp16:

- TwoSum (Knuth) - add/sub: `a + b = s + e` exactly, `s = fl(a + b)`.
- TwoProduct (Dekker) - mul: `a * b = p + e` exactly via a Veltkamp split (the
  split constant is `2^ceil(11/2)+1` for fp16's 11-bit significand). The
  `hi*lo + lo*hi` cross terms are captured; `lo*lo` is dropped (below fp16 ulp).
- Compensated dot - TwoProduct each element, then accumulate the product and
  the error streams. Crucially, the accumulation is again via `@ ones` (the wide
  matmul accumulator), never `reduce_sum` - a narrow sum would re-inject the very
  error the compensation just removed.

The `lo` terms capture exactly the operand/product rounding that plain fp16 loses,
so carrying the pair recovers it. Paired-fp16 recovers the most when its inputs
already carry sub-ulp information (a residual, or a value produced upstream in
higher working precision) - the "regime B" recovering case. A genuine fp16 input
has no sub-ulp bits, so there the win comes from the compensated ops capturing
each operation's own rounding.

Approximate op cost per element:

| Operation | fp16 ops/elem |
| --- | --- |
| TwoSum / compensated add-sub | ~6 |
| TwoProduct / compensated mul | ~17 |
| Compensated dot of length K | ~17 + 2 matmul-accumulates |

Because it is markedly more expensive than plain fp16, the optimizer gates it
behind the error budget. The single-subtract entry point `paired_subtract` carries
one `a - b` through a compensated TwoSum and returns a plain fp16 tensor (the best-
fp16 result), dropping straight into an existing graph at a cancellation hotspot.

> **Compiler caveat.** An aggressive algebraic simplifier could in principle
> collapse `hi + lo` back to `hi` (since `lo` is "just the rounding of `hi`"),
> defeating the trick. On this ANE / e5rt it does not happen - the transforms are
> opaque fp16 add/sub/mul chains with no fp32 island for the compiler to see
> through, and `to_tensor()` writes the result as an explicit `hi + lo` add so a
> dead-code pass cannot drop the `lo` computation. The tell that a future toolchain
> has fused them away is the on-device relerr jumping back to the plain-fp16 value;
> the demo asserts against that.

## Numeric graph rewrites

The optimizer distinguishes lossless rewrites (always-on canonicalization) from
numeric ones (accuracy-affecting, tuner-gated):

- `reduce_sum_to_matmul` (above) - lossless-or-better, the safe accuracy lever.
- Scalar-chain folding - `muls(b) after muls(a) -> muls(a*b)`, likewise for `adds`.
  The fold is done in fp16 to match device semantics, but because the fp16 NumPy
  kernels are never assumed bit-identical to the engine, const-folding is gated as
  a numeric (not lossless) rewrite.
- SDPA / bridge decompositions - `sdpa -> ((q @ k^T) * scale).softmax(-1) @ v`,
  and the bridge-elimination family (`minmax_norm`, `flatten`, `lrn`). These are
  metamorphic-proven bit-identical (or within fp16 op-noise), so they are lossless
  and chosen purely by speed.

## FFT: complex as real pairs

The ANE has no complex dtype - compute is fp16 real only - but it is a matmul
machine, and the DFT is a matmul against a twiddle matrix. The FFT modules exploit
the wide accumulator: the naive O(N^2) DFT-as-twiddle-matmul is fp16-clean to
N=2048+.

Every value is carried as a `(re, im)` tensor pair. A complex matmul `C = A @ B`
becomes four real matmuls:

```
Cre = Are@Bre - Aim@Bim
Cim = Are@Bim + Aim@Bre
```

This is the straight 4-matmul form, not Karatsuba: on the ANE matmul is cheap,
and the wide accumulator keeps the straight form cleanest - Karatsuba's `(a+b)(c+d)`
sums would lose a bit of fp16 headroom for no real op savings. The twiddle matrices
ride in as small fp16 constants folded into the graph.

The staged Cooley-Tukey FFT keeps the "every stage is a matmul" property while
cutting MACs to sub-quadratic. Its accuracy is ~5e-4..8e-4, flat in N (the wide
accumulator means per-stage sums don't compound) - about 3x the naive single-DFT
floor (~2.5e-4) from the extra cross-twiddle multiplies, but still fp16-clean to
N=2048+. Staged is not more accurate than the dense DFT, just far cheaper at the
same precision class.

The DFT butterfly contains exact-by-construction subtracts that trip the generic
cancellation precision heuristic; these kernels are numerically verified against
`np.fft` and skip the check (`_check_precision=False`).

### The dynamic-range walls

Several places must actively defend the 65504 ceiling, because the accumulator
holds the unscaled sum before any normalization:

- 2-D FFT folds the `1/(M*N)` normalization into the twiddles (`1/N` on the
  row pass, `1/M` on the column pass), never as one scale at the end. A real
  spectrum is O(M*N) at the dominant modes, so an unscaled first-axis transform
  would push intermediates past 65504 and shred precision.
- iFFT for convolution spectra pre-scales `Y` so the unscaled inverse-DFT sum
  stays well within range (`s = 30000 / peak`), runs the linear iFFT, then undoes
  the scale on the host. Products of two transforms can otherwise peak past 65504
  and saturate.
- The max usable FFT length is bounded by this fp16 dynamic range (normalize by
  `1/sqrt(N)` for power spectra), not by stage error.

The same crop-DMA saturation surfaces in direct linear-algebra factorizations: a
width-offset slice rides the A13/A14 x16 fixed-point crop-DMA, which saturates any
sliced `|value| > 4094` (= 65504/16) to +/-infinity. Element-recurrence kernels
(Cholesky / LU) route the accessor off the width axis to keep the usable entry range
at the full fp16 span.

## The fp16 envelope in linear algebra

The wide accumulator makes a single `A @ x` clean even at cond ~1e4 - but iterates
and residuals are stored and re-fed as fp16, and the residual `b - A x` is a
catastrophic-cancellation subtract. That subtract, not the matmul, is the wall:

- Iterative refinement recovers about one order of magnitude of accuracy for
  moderate conditioning. By cond ~1e3 the fp16 approximate solve barely converges
  and refinement only nibbles.
- Conjugate gradient uses symmetric Jacobi (diagonal) preconditioning to keep
  the fp16 dot products in range: scaling `A` so its entries are ~1 stops raw
  cond ~1e2 GEMV products from blowing past 65504. Without a convergence test to
  stop early, the unrolled iterates can overflow fp16 at high cond - reported as a
  finite large relerr rather than a NaN.
- Least squares via normal equations squares the condition number; refinement
  buys back roughly the order of magnitude the squaring cost.
- LSQR / GMRES Krylov envelopes: ~1e-3 at cond <= 1e1, ~1e-2 at cond <= 1e2 for
  overdetermined least squares.
- Power iteration (dominant SVD) normalizes between the `A` and `A^T`
  applications: `A^T(A v)` has magnitude ~sigma^2, which overflows fp16 for sigma > ~250.
  Likewise, extra `A^T A` power steps hurt trailing singular values in fp16 - each
  step squares the spectrum and crushes the small values toward the dominant one.

Every dot/norm in these kernels is `(z*z) @ ones`, never `reduce_sum` - the same
wide-accumulator rule.

## Special functions: fp16 shapes the math

Special functions are long, dependent chains of fused fp16 mul/add (Horner /
Clenshaw evaluation of minimax or rational approximations). Two fp16 constraints
shape every one:

1. fp16 compute (~3-4 digits). Where a function's output leaves the fp16 range
   (gamma > 65504 past x~8, I0 past x~12) that is a hard wall, not a coefficient
   problem.
2. No scalar-add op and no in-graph branch. A constant `c` is added with a
   `exp(0) == 1` ones tensor scaled by `c` (built on `exp`, an F0 native unary on
   every ANE family including M1, rather than `cos`, which is A15+). Only fixed,
   smooth range reduction is used - never a data-dependent step count.

Several functions exist because the naive fp16 form cancels or centers badly:

- erfc is evaluated directly (A&S 7.1.26), not as `1 - erf`: in fp16,
  `1 - erf(x)` cancels to 0 for x > ~2 (`fp16(1 - erf(3)) == 0`, but
  `erfc(3) = 2.2e-5`).
- erf is the mirror case, and gets its own polynomial for the same reason:
  `x * P(x^2)`, a deg-5 minimax of `erf(x)/x` on [0, 2], not `1 - erfc`. Near 0
  `erf` is small while `erfc -> 1`, so the subtraction cancels exactly where `erf`
  is wanted: on `geomspace(1e-3, 1)` the naive form is off by 187% against
  0.10% for the direct one, and it returns `-0.000977` at `x = 0` - a negative
  value for a function that is odd through the origin. Past |x| ~ 2 the two swap
  roles and `1 - erfc(|x|)` is the accurate path, since there `erfc` is small.
- lgamma / gamma evaluate in a centered variable (`x - 4.5`, `x - 1.5`). A raw
  Horner in `x` has terms up to ~1e7 with alternating large coefficients that cancel
  catastrophically (abserr ~8); centering keeps the powers small and the chain
  fp16-clean (abserr ~3e-3).
- sin / cos use half-period [-pi/2, pi/2] minimax polynomials. Over the full
  [-pi, pi] the alternating terms reach ~5 in magnitude, a cancellation that loses
  ~2 fp16 digits (abserr ~0.2 at +/-pi); on the half-period the terms stay O(1) and the
  error is fp16-rounding-limited (~1e-3). These polynomial forms also give a portable
  path on M1/H13, where native trig is unavailable.

A negative result: `exp_wide` / `log_wide` range-reduction recipes do not reliably
beat the native fp16 `exp` / `log` here - the re-rounding from repeated squaring/sqrt
usually costs more than the small-argument benefit. They are documented recipes, not
the default.

## Accuracy-gated compression and int8

Weight compression is a deliberate accuracy/bandwidth trade, and the optimizer never
makes it silently. Weights can stream as fp16 (default), per-channel int8, per-tensor
int4 LUT palettization, or unstructured sparse bitmask - each halving or better the
DRAM bytes during the tile DMA.

The numeric contract:

- Default tune is accuracy-preserving. The default tolerance is fp16 noise
  (`_ACCURACY_TOL = 5e-3`). A lossy rewrite like int8 (~1-2% quantization error,
  well above fp16 noise) is rejected at this default, so `tune()` returns the
  lossless baseline. int8 is opt-in: `tune(out, atol=0.1)` admits it within a stated
  budget, and even then a lossy variant must beat the baseline by at least 1.10x to
  be chosen, so a measurement-noise "win" never costs accuracy.
- `compress='auto'` chooses per weight: sparse if the weight is >=50% zeros,
  else int4 if it stays within `compress_atol`, else int8, else fp16. int4 carries an
  accuracy-gated fallback to int8/fp16.
- An encoding only helps if the per-family lowering streams it natively. Encodings
  the device folds back to dense fp16 (e.g. blockwise on A14) pay an accuracy cost
  for zero bandwidth win - `native_streams` records which formats stream per family.

### Precision-aware tuning

Beyond preserving accuracy, the precision-aware path can improve it. Given an
explicit `target_error=E`, the tuner selects the numerics-aware rewrite set
(`reduce_sum -> matmul`, paired-fp16, +/-int8) that meets the budget at minimum
predicted cost - measuring accuracy against an fp32-faithful reference where one can
be emulated (products fp16-rounded, accumulation wide, so the reference is the
achievable target, not an unreachable fp64 ideal), else against the fp16 baseline's
own output. `reduce_sum -> matmul` is auto-detected (structurally visible); the
CFG-style paired-fp16 cancellation fix stays opt-in, because near-equal cancellation
is data-dependent and can't be confirmed at graph-build time.

## Cross-chip fp16 divergence

The MAC accumulator width and the compiler `__TEXT` are uniform across chips, so a
graph's fp16 values are largely chip-independent. The remaining divergence comes
only from HAL-data-selected codegen routes that reorder or saturate fp16 ops; the
target table predicts it statically into one of:

- `saturation` - a width-offset slice on A13 (family 2) routes through the Q.4
  x16 crop-DMA that clamps `|value| > 4094` to +/-infinity, while A14+ takes a clean
  route. Magnitude-gated: only flagged when values can exceed 4094.
- `ulp1` - a reduction/softmax/norm whose route threshold differs (192 on
  A13/A14 vs 384 on A15+): a partial-sum reorder, <= 1 ULP.
- `round1` - a reduce-then-square fusion difference; currently never returned
  (A13/A14/A16 silicon all compute `fp16(sum)^2`, a measured no-op).
- `none` - no HAL field selects a differing route.

The native fused-attention layer carries a numeric reliability envelope: reliable
to ~3e-3 up to S=2048, but garbage (~1.0) at S=4096 and once both query and key
axes exceed 512. Above those bounds `af.sdpa` emits the accurate decomposed form,
where the wide matmul accumulator holds (~5e-3).
