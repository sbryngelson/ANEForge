"""Per-chip ANE target capabilities - the host-independent core of aneforge's
cross-chip support.

Apple's ANE compiler (ANECompiler 9.509) is byte-identical across chips; everything
that varies between an M1 and an M5 is the per-chip HAL/target DATA the identical code
reads, plus the `MinimumFamily<N>` op-trait floors. So portability is a data problem:
this module holds the measured per-family capability table and answers, for a given
target family, whether each op is native / must be decomposed / is unreachable, plus the
numeric limits a graph must respect.

The compiler enumerates 28 HAL targets in 5 capability families. A17 (H17*) and A18
(H18) targets exist but add no op capabilities over A16 - they scale NE-core count only
(suffix decoder: base=4, g=8, s=16, c=32, d=64) - so the A16 tier is the capability
ceiling:

    family 1  older   H11, H12, M9, T0                       A11 / A12  (no MIL path)
    family 2  A13     H13, H13g, T1                          A13 (M1)
    family 3  A14     H14, H14g, H14c                        A14
    family 4  A15     H15, H15g, H15c                        A15  (also ~= M11, see below)
    family 5  A16     H16, H16g, H16c, H16s,                 A16
                      H17, H17a, H17g, H17s, H17c, H17d,     A17 (M5 == H17s)
                      H18                                    A18

(U1/U2/U3 are compiler-internal reference targets, not silicon; M11 is a 1-core
efficiency ANE with A16 features but A13-sized dims - capability-wise the A15 tier.)

Physically measured anchors, both via the compiled bundle's directory name: M1 == H13
(`H13C.bundle`) and M5 == H17s (`H17S.bundle`). M-series == H(generation+12) follows
from the two anchors; intermediate chips are still detected at runtime, never assumed.

Sources (ANEForge reverse-engineering): op floors from the per-family code-generation
analysis; numeric limits from the measured `hal_extraction/named_hal.json`; the "MIL is
only supported for H13+ ANE architectures" hard floor from `ANE_constraint_strings.txt`.
"""
import os
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache

__all__ = ["Family", "MIN_FAMILY", "supports_mil", "family_of_arch", "arch_for_family",
           "op_status", "has_texture_engine", "limit", "OpReport", "Preflight",
           "preflight", "detect_family", "predict_fp16_divergence", "FP16_SLICE_SAT",
           "native_streams"]


class Family(IntEnum):
  OLDER = 1   # A11/A12/M9/T0 - below the MIL floor; cannot run aneforge at all
  A13 = 2     # M1
  A14 = 3
  A15 = 4
  A16 = 5     # M5


# The e5rt/MIL path aneforge dispatches through carries the hard assert
# "MIL is only supported for H13+ ANE architectures" - family >= 2 required
# regardless of any per-op floor.
MIN_FAMILY = int(Family.A13)


def supports_mil(family: int) -> bool:
  """Whether the e5rt/MIL path runs at all on this family (the H13+ hard floor)."""
  return int(family) >= MIN_FAMILY


# Compiler arch string -> capability family index. All 28 compiler HAL targets are
# valid TargetArchitecture strings (verified on-device 2026-06-05); the generation digit
# gives the capability family, with H17*/H18 folding into the A16 tier (core-count
# scaling only).
_ARCH_FAMILY = {
  "h11": Family.OLDER, "h12": Family.OLDER, "m9": Family.OLDER, "t0": Family.OLDER,
  "h13": Family.A13, "h13g": Family.A13, "t1": Family.A13,
  "h14": Family.A14, "h14g": Family.A14, "h14c": Family.A14,
  "h15": Family.A15, "h15g": Family.A15, "h15c": Family.A15,
  "h16": Family.A16, "h16g": Family.A16, "h16c": Family.A16, "h16s": Family.A16,
  "h17": Family.A16, "h17a": Family.A16, "h17g": Family.A16, "h17s": Family.A16,
  "h17c": Family.A16, "h17d": Family.A16,
  "h18": Family.A16,
  # M11: A16 op features but A13-sized dims (16384) -- the A15 tier is the exact
  # capability match (every A15 op native, 16384 dims). U1/U2/U3 are reference
  # targets, not silicon, deliberately not mapped.
  "m11": Family.A15,
}


def family_of_arch(arch: str) -> int:
  """Resolve an ANE arch string (OS or compiler-internal, e.g. 'h13' or 'h17s') to a
    compiler family index. Raises KeyError on an unrecognized arch."""
  return int(_ARCH_FAMILY[arch.strip().lower()])


# A representative arch string per capability family, for the TargetArchitecture
# compile option. h16s stands in for the whole A16 tier: h17*/h18 compile too but are
# capability-identical, so one representative keeps the cross-compile matrix small.
_FAMILY_ARCH = {Family.A13: "h13", Family.A14: "h14", Family.A15: "h15", Family.A16: "h16s"}


def arch_for_family(family: int) -> str:
  """A compiler `TargetArchitecture` string for the given family (h13/h14/h15/h16s)."""
  return _FAMILY_ARCH[Family(int(family))]


# --- per-family native weight-streaming formats ------------------------------------------
# Which compressed-weight encodings the per-family lowering keeps as a native streaming
# kernel (DRAM bandwidth win) vs folds to a dense fp16 const (no win). The kernel-streaming
# master gate (HAL +0x48f) turns on at A13, but the per-format gates (the +0x520-0x539
# cluster) are 0 on h13: only the palette/LUT path streams there (its +0x529 gate is
# A13-on). Silicon-measured endpoints: A13/M1 = int4-LUT (2.37x) AND sparse (~1.6x); A14/M2
# = int4 + int8 + sparse all stream, blockwise folds (measured 0.985x, no win); A16/M5 = the
# broad set streams. (The earlier "A14 streams all four" was an inferred HAL guess, refuted
# by M2 silicon - blockwise never gets a native stream.) papers M2_SILICON_FINDINGS.md +
# per-family compressed-weight streaming.
#
# Sparse on A13/M1 is a round-9 correction: compress="sparse" lowers to MIL
# `constexpr_sparse_to_dense` (a 1-bit mask + packed-fp16 DMA that decompresses on-chip),
# a different path from the HAL kernel-format sparse gate (the +0x520-0x539 cluster that is
# 0 on h13). On genuinely-sparse weights (>=50% zeros) it streams: live-measured ~1.55-1.64x
# (0.43x dense bytes, cos=1.0) on a conv1x1 - the native-stream fingerprint. The old "no win"
# reading came from running sparse mode on dense-random weights (all-ones mask, 0 bytes saved).
# See the reverse-engineering corpus
_ALL_FORMATS = frozenset({"int4", "int8", "sparse", "blockwise"})
_NATIVE_STREAMS = {
  int(Family.A13): frozenset({"int4", "sparse"}),           # M1-measured: int4-LUT + sparse stream
  int(Family.A14): frozenset({"int4", "int8", "sparse"}),   # M2-measured; blockwise folds
}


def native_streams(family: int) -> frozenset:
  """The compressed-weight encodings that stream natively (a bandwidth win) on
    `family`. Encodings outside the set still compile and stay correct, but the on-device
    compiler folds them to dense fp16 - an accuracy cost for zero win."""
  return _NATIVE_STREAMS.get(int(family), _ALL_FORMATS)


# --- runtime host-chip detection -------------------------------------------------------
# The chip -> ANE-family map. M1 (H13) and M5 (H17s) are physically measured anchors (via
# their H13C/H17S bundles); M2/M3/M4 are resolved by the verified M-series == H(gen+12)
# rule (Apple's own +[_ANEDeviceInfo aneArchitectureType] boardType ladder, disassembled
# and live-validated on M1 Max = h13g; VideoProcessing's cnn_frame_enhancer ships exactly
# {H13..H17} = M1..M5). So M2=H14/A14, M3=H15/A15, M4=H16/A16 are ground-truth capability
# families (only absolute power/watt still needs each chip's rail). The model identifier
# stays a trap (MacBookPro17,1 is an M1, Mac17,8 an M5), so we match the M-generation off
# the clean CPU brand string ("Apple M5 Pro"); the Pro/Max/Ultra ('g') variant changes core
# count, not capability family. Chips beyond this map (a future M6+) fall back to MIN_FAMILY
# - a family-2 program runs on every H13+ chip (higher families are strict op/shape/fp16
# supersets), so under-claiming stays correct.
_BRAND_FAMILY = {
  1: Family.A13,   # M1  - measured H13
  2: Family.A14,   # M2  - verified H14 (M-series ladder)
  3: Family.A15,   # M3  - verified H15 (M-series ladder)
  4: Family.A16,   # M4  - verified H16 (M-series ladder; H16 == A16 tier)
  5: Family.A16,   # M5  - measured H17s (A16-equivalent capability tier)
}


@lru_cache(maxsize=1)
def _cpu_brand() -> str:
  try:
    return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
  except Exception: return ""


def _family_from_brand(brand: str) -> int:
  """Map a CPU brand string to a compiler family. M1-M5 resolve exactly (M1/M5 measured,
    M2/M3/M4 by the verified M-series ladder); a chip beyond the map (future M6+) falls back
    to the conservative MIN_FAMILY."""
  m = re.search(r"Apple M(\d+)", brand)
  if m:
    gen = int(m.group(1))
    if gen in _BRAND_FAMILY: return int(_BRAND_FAMILY[gen])
  return MIN_FAMILY


def detect_family() -> int:
  """Best-effort target family for the host ANE. Resolution order:

    1. `ANEFORGE_TARGET` env var (an arch string, e.g. 'h13') - explicit override.
    2. the CPU brand string, for the measured anchors (M1, M5).
    3. MIN_FAMILY (the safe floor) for any unmeasured chip, with a one-time warning.
    """
  override = os.environ.get("ANEFORGE_TARGET")
  if override: return family_of_arch(override)
  brand = _cpu_brand()
  fam = _family_from_brand(brand)
  m = re.search(r"Apple M(\d+)", brand)
  if not (m and int(m.group(1)) in _BRAND_FAMILY):
    warnings.warn(
      f"aneforge: ANE family not measured for {brand or 'this chip'!r}; assuming "
      f"the conservative floor (family {MIN_FAMILY}/H13). Set ANEFORGE_TARGET "
      f"(e.g. 'h16s') to target this chip's real capabilities.",
      stacklevel=2)
  return fam


# --- op floors -------------------------------------------------------------------------
# Minimum native family per op. Anything not listed defaults to family 2 (the F0/F2
# vocabulary: conv/matmul/pooling/elementwise/activations/softmax/norms/reductions/
# sqrt/rsqrt/erf/exp2/log2/SDPA/resize/tile/space<->channel/atan - all native on M1+).
_OP_FLOOR = {
  # F4 trig (A15+) - the one genuinely family-gated trig pair in this MIL vocabulary.
  "sin": Family.A15, "cos": Family.A15,
  # texture-engine ops (A14+, HAL 0x81d) - no HW path before A14.
  "crop_resize": Family.A14, "resample": Family.A14, "affine": Family.A14,
  "gather_hw": Family.A14,
  # square-after-reduce (0x494) - A14+.
  "sq_after_reduce": Family.A14,
  # Dropout / Random (0x4a9) - A15+; unsupported on M1 AND A14.
  "dropout": Family.A15, "random": Family.A15,
  # GlobalArgMinMax (0x4f2) - A15+ (carries a floor-vs-flag conflict; verify on silicon).
  "global_argmax": Family.A15, "global_argmin": Family.A15,
  # bridge/native ops that PASS validation but REJECT at codegen on M1 (family 2).
  "topk": Family.A14, "sort": Family.A14, "dynamic_slice": Family.A14,
}

# Below-floor ops for which aneforge has a working substitution. sin/cos -> special.py
# Horner; dropout/random -> host-side RNG. The texture-engine and bridge-codegen ops have
# no substitution wired, so below their floor they hard-reject (a clear compile-time
# error) instead of crashing at dispatch.
_DECOMPOSABLE = {"sin", "cos", "dropout", "random"}


def op_status(op: str, family: int) -> str:
  """Return 'native', 'decompose', or 'reject' for `op` on the given target family."""
  floor = int(_OP_FLOOR.get(op, MIN_FAMILY))
  if int(family) >= floor: return "native"
  if op in _DECOMPOSABLE: return "decompose"
  return "reject"


# --- numeric HAL limits (measured per family from the live compiler) -------------------
# The texture engine (HAL 0x81d) is 0 through M1 and 1 only on A14+.
def has_texture_engine(family: int) -> bool:
  return int(family) >= int(Family.A14)


# Per-family numeric limits. Values keyed by the lowest family at which they take effect;
# _limit_for() picks the value for the highest threshold <= the queried family.
_LIMITS = {
  # Per-op-class extents (A14 measured exact by binary search + run-at-cap, M2 measurement
  # A14_MAXDIM_CAPS.md; A16 row measured the same way on the M5 dev box). Three distinct
  # caps, riding the op's LOWERING, not the tensor:
  #   spatial/contraction (H, W, matmul-K, conv spatial)  A14 16384  ->  A16 65536
  #   channel (C; elementwise and conv Cin)               A14 65536  ==  A16 65536
  #   transpose extent (an offset-field width)            A14 2^23-1 ->  A16 >=2^24-1
  # A13 row now measured on the live M1: W/H 16384 (16385 rejects), C 65536 (65537
  # rejects) -- exactly A14. So the caps are generation-monotone (A13 == A14 <= A16); the
  # old "M1 ran a 262144-wide relu" was a red herring (a direct compile probe rejects
  # 16385). The conservative A13 = A14 pin was correct; no inversion.
  "max_tensor_dim": {Family.A13: 16384, Family.A16: 65536},     # spatial/contraction (compat name)
  "channel_extent": {Family.A13: 65536},
  "transpose_extent": {Family.A13: 8388607, Family.A16: 16777215},
  # conv kernel WIDTH (fp16): A13 measured kW<=13 (14 rejects); A16 measured kW<=15 (16
  # rejects -> route via space_to_depth). Monotone A13<=A16. A14/A15 inherit 13 (unmeasured;
  # conservative). kH is not a fixed cap (rejects only when taller than the input).
  # graph.conv() also hard-guards the family ceiling kW>15 at build.
  "conv_kw_max": {Family.A13: 13, Family.A16: 15},
  # reduction->transpose threshold: 192 on A13/A14, raised to 384 at A15.
  "reduction_transpose_extent": {Family.A13: 192, Family.A15: 384},
}


def limit(name: str, family: int) -> int:
  """Measured numeric limit for `name` on the given target family."""
  table = _LIMITS[name]
  val = None
  for thresh in sorted(table):
    if int(family) >= int(thresh): val = table[thresh]
  if val is None: raise ValueError(f"family {family} below the floor for limit {name!r}")
  return val


# --- cross-chip fp16 divergence predictor (Direction B) --------------------------------
# The MAC accumulator width and the compiler __TEXT are uniform across chips, so cross-chip
# fp16 VALUE divergence can only come from HAL-data-selected codegen routes that reorder (or
# saturate) fp16 ops - predictable by comparing a small set of per-family HAL fields. Each
# field below is keyed by the lowest family at which it takes its value; _field_for()
# resolves the value for a queried family (highest threshold <= family). Source: ANEForge
# reverse-engineering (field offsets cited per row).

# fp16 max / 16 = 65504/16: the slice-x16 Q.4 crop-DMA saturation threshold (finite->inf).
FP16_SLICE_SAT = 4094.0

_HAL_FIELDS = {
  # 0x494 bit0: reduce->square fusion. Silicon-measured no-op: A13, A14 (M2), and A16 (M5)
  # all compute fp16(sum)^2 (the "unfused" value) -- a non-tied reduce->square probe gives
  # 62624 on all three, not the fused 62656. Consistent with the fp16 reduce OUTPUT (the
  # reduce result is cast to fp16 before the square, so a fuse has no extra precision to
  # preserve). The earlier HAL-inferred flip {A13:0, A14:1} was wrong; kept uniform 0 so the
  # predictor never flags a (nonexistent) reduce->square divergence.
  "reduce_square_fuse": {Family.A13: 0},
  # 0x3f0: reduction->transpose route threshold (= reduction_transpose_extent). 192 on
  # A13/A14, 384 on A15+. Empirically a no-op (block accumulation absorbs it), but a
  # differing value still flags a <=1-ULP reorder risk for the reduced extent.
  "reduce_route_thresh": {Family.A13: 192, Family.A15: 384},
}


def _field_for(name: str, family: int) -> int:
  table = _HAL_FIELDS[name]
  val = None
  for thresh in sorted(table):
    if int(family) >= int(thresh): val = table[thresh]
  return val


# Op-kind classes the predictor recognizes, mapped to the divergence axis they ride.
# A graph node's `op` is matched against these (substring/exact) by the compile-side
# wiring; the predictor takes the resolved kind string.
_REDUCE_OPS = ("reduce", "softmax", "norm", "mean", "sum", "variance", "rms")
_SLICE_OPS = ("slice_by_size", "slice", "crop")


def predict_fp16_divergence(kind: str, shape, target_a: int, target_b: int,
                            begin=None, max_abs: float | None = None) -> str:
  """Statically predict whether an op's fp16 VALUE can diverge between two target ANE
    families, from the HAL fields that select its codegen route. Returns one verdict:

      `"saturation"` - a slice with a nonzero last-axis/width begin-offset, where one
          target is A13 (family 2): A13 routes the offset through a Q.4 x16 crop-DMA that
          clamps any |value| > 4094 (=fp16max/16) to +/-inf, while A14+ takes a clean
          route. Magnitude-gated: flagged only when values can exceed 4094 (`max_abs`
          None = unknown, treated as possible; a finite `max_abs` <= 4094 downgrades it).
      `"round1"` - a reduce immediately followed by a square/mul (variance, L2-norm,
          RMSNorm) where the 0x494 reduce->square fusion bit differs. Currently never
          returned: A13/A14/A16 silicon all compute fp16(sum)^2 (the field is uniform, a
          measured no-op given the fp16 reduce output); kept for completeness.
      `"ulp1"` - a reduction / softmax / norm whose 0x3f0 route threshold differs (192
          A13/A14 vs 384 A15+) for the reduced extent: a partial-sum reorder, <=1 ULP.
      `"none"` - no HAL field selects a differing route for this op/shape pair.

    `kind` is an op-kind string (the node `op`, or a coarse class like 'slice' /
    'reduce' / 'reduce_square'); `shape` is the op's output shape; `begin` is the slice
    begin-offset tuple (for slice kinds); `max_abs` bounds the op's value magnitude.
    The strongest verdict wins (saturation > round1 > ulp1 > none)."""
  fa, fb = int(target_a), int(target_b)
  k = kind.lower()

  # 1. slice with nonzero last-axis/width begin-offset -> Q.4 x16 crop-DMA saturation.
  # The quirk is present on A13 AND A14 (M2 silicon: 4094 finite -> 4100 inf, bit-exact,
  # same as M1), absent on A16; A15 pending M3 data. Flag a divergence when exactly one
  # target is affected (family <= A14) and the other is not -- both-affected and both-clean
  # pairs match.
  if any(s in k for s in _SLICE_OPS):
    last_off = bool(begin) and len(begin) > 0 and int(begin[-1]) > 0
    sat_a, sat_b = fa <= int(Family.A14), fb <= int(Family.A14)
    if last_off and sat_a != sat_b:
      if max_abs is None or float(max_abs) > FP16_SLICE_SAT: return "saturation"

  # 2. reduce -> square/mul (variance / L2 / RMSNorm): 0x494 reduce->square fuse bit.
  if "square" in k or "reduce_square" in k or "rms" in k or "variance" in k or "l2" in k:
    if _field_for("reduce_square_fuse", fa) != _field_for("reduce_square_fuse", fb): return "round1"

  # 3. reduction / softmax / norm: 0x3f0 route threshold differs -> <=1-ULP reorder.
  if any(s in k for s in _REDUCE_OPS):
    if _field_for("reduce_route_thresh", fa) != _field_for("reduce_route_thresh", fb): return "ulp1"

  return "none"


# --- preflight: static "will this graph run on chip X?" report -------------------------
@dataclass
class OpReport:
  """One graph node's status on a target family."""
  op: str
  shape: tuple
  status: str          # 'native' | 'decompose' | 'reject'
  oversize: bool = False   # a dim exceeds the family's max tensor extent (needs tiling)


@dataclass
class Preflight:
  """Result of walking a graph for a target family. `ok` iff nothing hard-blocks
    compilation: no rejected ops and no oversize tensors. `decompose` ops are
    recoverable (route through the host/special.py substitution) and do not clear ok."""
  family: int
  native: list = field(default_factory=list)
  decompose: list = field(default_factory=list)
  reject: list = field(default_factory=list)
  oversize: list = field(default_factory=list)

  @property
  def ok(self) -> bool: return not self.reject and not self.oversize


def _internal_axis_oversize(t, max_dim: int) -> bool:
  """Some ops reshape to a larger per-axis extent INSIDE their MIL lowering than any
    node shape shows. group_norm's rank-4 tiled lowering reshapes to [1,G,C/groups,H*W],
    so its largest internal axis is max(C/groups, H*W); this can exceed the family's max
    dimension even when [1,C,H,W] does not (e.g. C512@128 -> H*W=16384, at the A13 cap).
    Catch these so preflight predicts the compiler instead of missing the overflow."""
  if t.op == "group_norm" and len(t.shape) == 4:
    _, c, h, w = t.shape
    g = int(t.attrs.get("groups", 1)) or 1
    return max(c // g, h * w) > max_dim
  return False


def preflight(out, family: int) -> Preflight:
  """Walk the graph feeding `out` and report, for the given target family, which ops
    are native / need decomposition / are unreachable, plus any tensor whose dimensions
    (or known internal-reshape extents) exceed the family's limits. Pure static analysis -
    no compile, no hardware. `out` is an aneforge `Tensor` (anything with
    `.op`/`.shape`/`.srcs`)."""
  rep = Preflight(family=int(family))
  seen: set = set()

  if not supports_mil(family):
    # Below the H13+ MIL floor (Family.OLDER): the e5rt/MIL path carries the hard
    # "MIL is only supported for H13+ ANE architectures" assert, so NOTHING runs.
    # Report every op as rejected (not ok) without querying the per-family limits,
    # which are only defined at or above the floor — limit() raises below it.
    walk = [out]
    while walk:
      t = walk.pop()
      if id(t) in seen: continue
      seen.add(id(t))
      walk.extend(t.srcs)
      rep.reject.append(OpReport(op=t.op, shape=tuple(t.shape), status="reject"))
    return rep

  max_dim = limit("max_tensor_dim", family)          # spatial/contraction extent
  chan_dim = limit("channel_extent", family)
  tr_dim = limit("transpose_extent", family)
  # nodes whose extent rides the WIDE transpose lowering: transposes and their direct
  # inputs (the measured (N,2)->(2,N) silicon cap covers both sides of the op).
  nodes, walk = [], [out]
  tr_wide: set = set()
  while walk:
    t = walk.pop()
    if id(t) in seen: continue
    seen.add(id(t))
    nodes.append(t)
    walk.extend(t.srcs)
    if t.op == "transpose":
      tr_wide.add(id(t))
      tr_wide.update(id(s) for s in t.srcs if s.op == "input")
  for t in nodes:
    # per-axis caps ride the op's lowering (A14_MAXDIM_CAPS.md): transpose ops take
    # the wide offset-field extent on every axis; rank-4 tensors get the channel cap on
    # axis 1 and the spatial cap elsewhere; other ranks are spatial-capped.
    if id(t) in tr_wide:
      over = any(int(d) > tr_dim for d in t.shape)
    elif len(t.shape) == 4:
      over = int(t.shape[1]) > chan_dim or any(int(d) > max_dim for i, d in enumerate(t.shape) if i != 1)
    else:
      over = any(int(d) > max_dim for d in t.shape)
    over = over or _internal_axis_oversize(t, max_dim)
    # conv kernel width is a per-family cap (A13<=13, A16<=15) below the build-time
    # ceiling guard (kW>15) - catch the family-specific case statically. conv_transpose
    # shares the cap; its weight is [Cin,Cout,kH,kW], so kW is the last axis for both
    # layouts.
    if t.op in ("conv", "conv_transpose") and "weight" in getattr(t, "attrs", {}):
      over = over or int(t.attrs["weight"].shape[-1]) > limit("conv_kw_max", family)
    status = op_status(t.op, family)
    r = OpReport(op=t.op, shape=tuple(t.shape), status=status, oversize=over)
    {"native": rep.native, "decompose": rep.decompose,
     "reject": rep.reject}[status].append(r)
    if over: rep.oversize.append(r)
  return rep
