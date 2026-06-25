"""Per-chip ANE target capabilities - the host-independent core of cross-chip support."""
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


# The e5rt/MIL path requires family >= 2 ("MIL is only supported for H13+ ANE architectures").
MIN_FAMILY = int(Family.A13)


def supports_mil(family: int) -> bool:
  """Whether the e5rt/MIL path runs at all on this family (the H13+ hard floor)."""
  return int(family) >= MIN_FAMILY


# Compiler arch string -> capability family (H17*/H18 fold into the A16 tier).
_ARCH_FAMILY = {
  "h11": Family.OLDER, "h12": Family.OLDER, "m9": Family.OLDER, "t0": Family.OLDER,
  "h13": Family.A13, "h13g": Family.A13, "t1": Family.A13,
  "h14": Family.A14, "h14g": Family.A14, "h14c": Family.A14,
  "h15": Family.A15, "h15g": Family.A15, "h15c": Family.A15,
  "h16": Family.A16, "h16g": Family.A16, "h16c": Family.A16, "h16s": Family.A16,
  "h17": Family.A16, "h17a": Family.A16, "h17g": Family.A16, "h17s": Family.A16,
  "h17c": Family.A16, "h17d": Family.A16,
  "h18": Family.A16,
  # M11: A16 op features but A13-sized dims -> the A15 tier is the exact match.
  # U1/U2/U3 are reference targets, not silicon, deliberately not mapped.
  "m11": Family.A15,
}


def family_of_arch(arch: str) -> int:
  """Resolve an ANE arch string ('h13'/'h17s') to a compiler family. KeyError if unknown."""
  return int(_ARCH_FAMILY[arch.strip().lower()])


# A representative arch string per family for TargetArchitecture (h16s stands in for the A16 tier).
_FAMILY_ARCH = {Family.A13: "h13", Family.A14: "h14", Family.A15: "h15", Family.A16: "h16s"}


def arch_for_family(family: int) -> str:
  """A compiler `TargetArchitecture` string for the given family (h13/h14/h15/h16s)."""
  return _FAMILY_ARCH[Family(int(family))]


# --- per-family native weight-streaming formats ------------------------------------------
# Which compressed-weight encodings stream natively (bandwidth win) vs fold to dense fp16.
# Silicon-measured: A13 = int4+sparse; A14 = int4+int8+sparse; A16 = broad set.
_ALL_FORMATS = frozenset({"int4", "int8", "sparse", "blockwise"})
_NATIVE_STREAMS = {
  int(Family.A13): frozenset({"int4", "sparse"}),           # M1-measured
  int(Family.A14): frozenset({"int4", "int8", "sparse"}),   # M2-measured; blockwise folds
}


def native_streams(family: int) -> frozenset:
  """Compressed-weight encodings that stream natively on `family` (others fold to dense fp16)."""
  return _NATIVE_STREAMS.get(int(family), _ALL_FORMATS)


# --- runtime host-chip detection -------------------------------------------------------
# Chip -> ANE-family, matched off the CPU brand string. M1/M5 measured; M2/M3/M4 by the
# M-series == H(gen+12) rule. A future M6+ falls back to MIN_FAMILY (under-claiming is safe).
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
  """Map a CPU brand string to a compiler family (M1-M5 resolve exactly; beyond the map -> MIN_FAMILY)."""
  m = re.search(r"Apple M(\d+)", brand)
  if m:
    gen = int(m.group(1))
    if gen in _BRAND_FAMILY: return int(_BRAND_FAMILY[gen])
  return MIN_FAMILY


def detect_family() -> int:
  """Best-effort target family for the host ANE (ANEFORGE_TARGET env override, else CPU brand, else MIN_FAMILY)."""
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
# Minimum native family per op; unlisted ops default to family 2 (all native on M1+).
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

# Below-floor ops with a working substitution (sin/cos -> Horner; dropout/random -> host RNG).
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


# Per-family numeric limits, keyed by the lowest family a value takes effect; limit() picks
# the highest threshold <= the queried family. Generation-monotone, ride the op's LOWERING.
_LIMITS = {
  "max_tensor_dim": {Family.A13: 16384, Family.A16: 65536},     # spatial/contraction (H,W,matmul-K)
  "channel_extent": {Family.A13: 65536},                        # C (elementwise, conv Cin)
  "transpose_extent": {Family.A13: 8388607, Family.A16: 16777215},  # offset-field width
  # conv kernel WIDTH (fp16): A13 kW<=13, A16 kW<=15 (graph.conv() also guards kW>15 at build).
  "conv_kw_max": {Family.A13: 13, Family.A16: 15},
  "reduction_transpose_extent": {Family.A13: 192, Family.A15: 384},
}


def limit(name: str, family: int) -> int:
  """Measured numeric limit for `name` on the given target family."""
  table = _LIMITS[name]
  val = next((table[t] for t in sorted(table, reverse=True) if int(family) >= int(t)), None)
  if val is None: raise ValueError(f"family {family} below the floor for limit {name!r}")
  return val


# --- cross-chip fp16 divergence predictor (Direction B) --------------------------------
# Cross-chip fp16 VALUE divergence comes only from HAL-data-selected codegen routes that
# reorder/saturate fp16 ops, predictable from a few per-family HAL fields.

# fp16 max / 16 = 65504/16: the slice-x16 Q.4 crop-DMA saturation threshold (finite->inf).
FP16_SLICE_SAT = 4094.0

_HAL_FIELDS = {
  # 0x494 reduce->square fusion: silicon-measured no-op; kept uniform 0 so the predictor never flags it.
  "reduce_square_fuse": {Family.A13: 0},
  # 0x3f0 reduction->transpose route threshold (192 A13/A14, 384 A15+); a differing value flags a <=1-ULP reorder.
  "reduce_route_thresh": {Family.A13: 192, Family.A15: 384},
}


def _field_for(name: str, family: int) -> int | None:
  table = _HAL_FIELDS[name]
  return next((table[t] for t in sorted(table, reverse=True) if int(family) >= int(t)), None)


# Op-kind classes the predictor recognizes (matched substring/exact against the kind string).
_REDUCE_OPS = ("reduce", "softmax", "norm", "mean", "sum", "variance", "rms")
_SLICE_OPS = ("slice_by_size", "slice", "crop")


def predict_fp16_divergence(kind: str, shape, target_a: int, target_b: int,
                            begin=None, max_abs: float | None = None) -> str:
  """Statically predict whether an op's fp16 VALUE can diverge between two ANE families: 'saturation' | 'round1' | 'ulp1' | 'none' (strongest wins)."""
  fa, fb = int(target_a), int(target_b)
  k = kind.lower()

  # 1. slice with nonzero last-axis begin-offset -> Q.4 x16 crop-DMA saturation (A13/A14, not
  # A16). Flag when exactly one target is affected (family <= A14).
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
  """Result of walking a graph for a target family; `ok` iff no rejected ops and no oversize tensors."""
  family: int
  native: list = field(default_factory=list)
  decompose: list = field(default_factory=list)
  reject: list = field(default_factory=list)
  oversize: list = field(default_factory=list)

  @property
  def ok(self) -> bool: return not self.reject and not self.oversize


def _internal_axis_oversize(t, max_dim: int) -> bool:
  """True if an op's internal lowering exceeds the per-axis cap (group_norm's max(C/groups, H*W)) even when its node shape does not."""
  if t.op == "group_norm" and len(t.shape) == 4:
    _, c, h, w = t.shape
    g = int(t.attrs.get("groups", 1)) or 1
    return max(c // g, h * w) > max_dim
  return False


def preflight(out, family: int) -> Preflight:
  """Static walk of the graph feeding `out`, reporting per-family native/decompose/reject ops + oversize tensors."""
  rep = Preflight(family=int(family))
  seen: set = set()

  if not supports_mil(family):
    # below the H13+ MIL floor: report every op rejected (limit() is undefined here).
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
  # nodes whose extent rides the WIDE transpose lowering: transposes + their direct inputs.
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
    # per-axis caps ride the op's lowering: transposes use the wide offset-field extent;
    # rank-4 tensors get the channel cap on axis 1 and spatial elsewhere; else spatial-capped.
    if id(t) in tr_wide:
      over = any(int(d) > tr_dim for d in t.shape)
    elif len(t.shape) == 4:
      over = int(t.shape[1]) > chan_dim or any(int(d) > max_dim for i, d in enumerate(t.shape) if i != 1)
    else:
      over = any(int(d) > max_dim for d in t.shape)
    over = over or _internal_axis_oversize(t, max_dim)
    # conv kernel width is a per-family cap (A13<=13, A16<=15); kW is the last weight axis.
    if t.op in ("conv", "conv_transpose") and "weight" in getattr(t, "attrs", {}):
      over = over or int(t.attrs["weight"].shape[-1]) > limit("conv_kw_max", family)
    status = op_status(t.op, family)
    r = OpReport(op=t.op, shape=tuple(t.shape), status=status, oversize=over)
    {"native": rep.native, "decompose": rep.decompose,
     "reject": rep.reject}[status].append(r)
    if over: rep.oversize.append(r)
  return rep
