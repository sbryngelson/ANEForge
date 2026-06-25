"""Op-agnostic structural cost model - the optimizer's PRUNER (orders variants; does not select).

`estimate(out) -> microseconds` is a structural roofline (max(floor, bytes/BW, flops/COMPUTE)
per node, fusion-discounted per region) that never touches the device. Real selection is by
on-device measurement (_optimize.measure). See .superpowers/sdd/gold-core-heavy.md for the model.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._compile import NETPLIST_OPS, _topo

# --------------------------------------------------------------------------- #
# calibrated constants (load from the cost-model JSON; else defaults below)      #
# --------------------------------------------------------------------------- #
# Defaults from ane_cost_model.json's calibration (floor/cut/BW/FLOPS); see
# .superpowers/sdd/gold-core-heavy.md for how each was measured.
_DEFAULTS = {
  "floor_us": 70.0,
  "cut_us": 200.0,
  "bw_bytes_per_us": 1.1e5,
  "flops_per_us": 1.45e7,
}


@lru_cache(maxsize=1)
def _constants() -> dict:
  c = dict(_DEFAULTS)
  path = _cost_model_path()
  if path is not None:
    try:
      j = json.loads(path.read_text())
      model = j.get("model", {})
      floor = model.get("dispatch_floor_us")
      if floor: c["floor_us"] = float(floor)
      # derive BW / COMPUTE from the matmul + conv scaling fits when present (us/flop -> flop/us)
      mm = model.get("matmul", {})
      if mm.get("slope_us_per_unit"):
        c["flops_per_us"] = 1.0 / float(mm["slope_us_per_unit"])
      cv = model.get("conv_channels", {})
      if cv.get("slope_us_per_unit"):
        c["flops_per_us"] = max(c["flops_per_us"], 1.0 / float(cv["slope_us_per_unit"]))
      # cut penalty: the smallest measured bridge min as the marginal cut cost.
      bridge = model.get("bridge_ops", {})
      mins = [float(v["min_us"]) for v in bridge.values()
              if isinstance(v, dict) and v.get("min_us") and v["min_us"] < 1000]
      if mins: c["cut_us"] = min(mins)
    except Exception:
      pass
  return c


def _cost_model_path():
  bundled = Path(__file__).resolve().parent / "ane_cost_model.json"
  if bundled.exists(): return bundled
  for up in Path(__file__).resolve().parents:        # legacy out-of-tree calibration
    p = up / "ane_artifacts" / "runtime" / "ane_cost_model.json"
    if p.exists(): return p
  return None


# --------------------------------------------------------------------------- #
# the measurement-free ANALYTIC per-chip cost model (Direction A)              #
# --------------------------------------------------------------------------- #
# Decompiled from the compiler's own ZinNEPerf model; per-chip HAL fields + freq/eff
# curves -> bundled costmodel_curves.json. Model: t = overhead + max(flops/peak, bytes/bw)
# per fused program, anchored to measured chips (_ANCHORS) and scaled by {cores (BW),
# clock (floor), cores*eff (peak)}. See .superpowers/sdd/gold-core-heavy.md for the anchors.
_M1_FIT_PEAK_FLOPS = 3.25e12        # latency roofline compute ceiling (fit on M1 convs)
_M1_FIT_BW_BYTES = 9.0e9            # effective streaming BW (jointly fit; over-predicts extreme BW-bound shapes ~4-5x)
_M1_FIT_OVERHEAD_US = 220.0         # additive per-program dispatch overhead (0.22 ms)
_M1_MEASURED_PEAK_TFLOPS = 1.8      # headline fp16 peak (the project_peak absolute anchor)
_CLOCK_FRACTION = 0.8              # operating clock ~= 0.8 * fmax (DAT_2241e37f8)

# arch -> per-family fallback curve key for chips without their own costmodel_curves.json entry.
_FAMILY_CURVE = {2: "H13", 3: "H14", 4: "H15", 5: "H16s"}


def _curves_path() -> Path:
  return Path(__file__).resolve().parent / "costmodel_curves.json"


@lru_cache(maxsize=1)
def _curves() -> dict:
  """Per-chip cost curves (cores/divisor/clocks/eff), bundled costmodel_curves.json."""
  p = _curves_path()
  try:
    return json.loads(p.read_text())
  except (OSError, ValueError) as e:
    raise RuntimeError(
      f"cannot load the bundled per-chip cost curves at {p}: {e}. "
      "costmodel_curves.json ships inside the aneforge package, so a missing or "
      "unparseable copy means this installation is broken - reinstall aneforge.") from e


def _curve_for_arch(arch: str) -> dict:
  """Resolve an arch string to its cost curve, case-insensitively, with a per-family
    fallback for chips that share another die's profile. Raises KeyError on a name that
    is neither a known curve nor a known arch."""
  curves = _curves()
  key = arch.strip()
  # exact, then case-insensitive (JSON uses 'H13'/'H17s'; arch is 'h13'/'h17s').
  if key in curves: return curves[key]
  low = {k.lower(): k for k in curves}
  if key.lower() in low: return curves[low[key.lower()]]
  # fall back to the arch's capability family's representative die.
  from . import _targets as _TG
  fam = _TG.family_of_arch(key)                  # raises on a truly unknown name
  fk = _FAMILY_CURVE.get(int(fam))
  if fk and fk in curves: return curves[fk]
  raise KeyError(f"no cost curve for arch {arch!r}")


def _interp(x: float, xs, ys) -> float:
  """Piecewise-linear interp of ys at x over sorted xs (clamped at the ends)."""
  if x <= xs[0]: return ys[0]
  if x >= xs[-1]: return ys[-1]
  for i in range(1, len(xs)):
    if x <= xs[i]:
      t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
  return ys[-1]


def _eff_freq_at_op(curve: dict) -> tuple[float, float]:
  """(operating_freq, effective_freq) at f = 0.8*fmax. effective_freq = eff_map_0x7a8 col 2
    interpolated at f (the engine's derated throughput freq; M1=1.0*f, A14+ ~0.84)."""
  freqs = curve["freq_0x760"]
  f = _CLOCK_FRACTION * max(freqs)
  em = curve["eff_map_0x7a8"]                    # [[freq, eff_freq], ...]
  xs = [p[0] for p in em]; ys = [p[1] for p in em]
  return f, _interp(f, xs, ys)


def _compute_scale(arch: str) -> float:
  """Compute-throughput multiplier of `arch` vs M1 (H13): ratio of cores*effective_freq."""
  c = _curve_for_arch(arch)
  _, ce = _eff_freq_at_op(c)
  m1 = _curve_for_arch("h13")
  _, m1e = _eff_freq_at_op(m1)
  return (c["cores_0x238"] * ce) / (m1["cores_0x238"] * m1e)


def project_peak(arch: str) -> dict:
  """Measurement-free fp16 peak projection for any ANE target, anchored to measured M1
    (1.8 TFLOP/s). Returns {tflops, rel_m1, cores, ghz}; needs no silicon beyond M1."""
  scale = _compute_scale(arch)
  c = _curve_for_arch(arch)
  return {
    "tflops": _M1_MEASURED_PEAK_TFLOPS * scale,
    "rel_m1": scale,
    "cores": int(c["cores_0x238"]),
    "ghz": _CLOCK_FRACTION * max(c["freq_0x760"]) / 1e9,
  }


# Silicon-measured roofline anchors, one per measured chip (h13=M1, h14=M2 Pro, h17s=M5).
# Effective BW tracks CORE COUNT, not clock (clock-scaling over-predicted M5 ~2x); `util` is
# h14's mid-utilization compute ramp. See .superpowers/sdd/gold-core-heavy.md for the fits.
_ANCHORS = {
  "h13": {"bw": _M1_FIT_BW_BYTES, "floor_us": _M1_FIT_OVERHEAD_US, "peak": _M1_FIT_PEAK_FLOPS},
  # eff_peak = peak * min(1, (flops/F)^q): a CAPPED power law (large ops return to full peak).
  "h14": {"bw": 48.0e9, "floor_us": 100.0, "peak": 7.3e12, "util": (1.80e10, 0.38)},
  "h17s": {"bw": 57.0e9, "floor_us": 110.0, "peak": 8.9e12},
}


def _anchor_for_arch(arch: str) -> str:
  """The measured anchor for `arch`: its own when silicon-measured, else the nearest
    measured generation (anchors: A13/h13, A14/h14, A16/h17s; A15 -> A14)."""
  key = arch.strip().lower()
  if key in _ANCHORS: return key
  from . import _targets as _TG
  fam = int(_TG.family_of_arch(key))
  if fam >= 5: return "h17s"
  if fam >= 3: return "h14"                 # A14 (measured) and A15 (nearest measured below A16)
  return "h13"


def estimate_provenance(target: str) -> dict:
  """Is `estimate(out, target=...)` silicon-anchored or extrapolated for `target`?
    Returns {target, anchor, measured: bool, basis} (basis = 'silicon' or
    'extrapolated-from-<anchor>'). Raises ValueError on an unknown arch."""
  from . import _targets as _TG
  key = target.strip().lower()
  if key not in _TG._ARCH_FAMILY:
    raise ValueError(f"unknown ANE target arch {target!r}; known: "
                     f"{sorted(_TG._ARCH_FAMILY)}")
  anchor = _anchor_for_arch(key)
  measured_families = {int(_TG.family_of_arch(a)) for a in _ANCHORS}
  measured = int(_TG.family_of_arch(key)) in measured_families
  return {
    "target": key,
    "anchor": anchor,
    "measured": measured,
    "basis": "silicon" if measured else f"extrapolated-from-{anchor}",
  }


def _analytic_constants(arch: str) -> dict:
  """The {floor_us, cut_us, bw_bytes_per_us, flops_per_us} for `arch`'s analytic roofline,
    scaled from the nearest anchor: BW by cores, floor by clock, peak by cores*eff_freq;
    cut_us is host-side (chip-independent)."""
  a_key = _anchor_for_arch(arch)
  a = _ANCHORS[a_key]
  c, ac = _curve_for_arch(arch), _curve_for_arch(a_key)
  f, _ = _eff_freq_at_op(c); fa, _ = _eff_freq_at_op(ac)
  clk = f / fa
  cores = c["cores_0x238"] / ac["cores_0x238"]
  scale = _compute_scale(arch) / _compute_scale(a_key)
  out = {
    "floor_us": a["floor_us"] / clk,                        # overhead shrinks with clock
    "cut_us": _DEFAULTS["cut_us"],                          # host round-trip, chip-agnostic
    "bw_bytes_per_us": (a["bw"] * cores) / 1e6,             # BW scales with CORES, not clock
    "flops_per_us": (a["peak"] * scale) / 1e6,
  }
  if "util" in a:                                           # mid-utilization compute ramp (h14)
    uf, uq = a["util"]                                      # eff_peak = peak*min(1,(flops/F)^q)
    out["util_k"], out["util_p"] = uf * scale, uq           # F (sat flops) scales with compute
  return out


# --------------------------------------------------------------------------- #
# measured per-bridge-op cost model                                            #
# --------------------------------------------------------------------------- #
# Bridge ops have no closed-form flops; ane_cost_model.json's `model.bridge_ops` holds
# measured per-config min latencies (keyed by a size string + a `family`). bridge_cost()
# nearest-neighbour anchors by a per-family work scalar (_BRIDGE_WORK) and scales by the
# work ratio, clamped to the floor. Approximate ordering, not a per-shape fit; families with
# no data fall back to the roofline. See .superpowers/sdd/gold-core-heavy.md for key formats.
import re

# op name -> bridge family in the JSON (all 19 bridge ops are measured by the cost sweeps).
_OP_TO_FAMILY = {
  "sdpa": "bridge_sdpa",
  "argmax": "bridge_argmax",
  "topk": "bridge_topk",
  "sort": "bridge_sort",
  "cross_product": "bridge_cross_product",
  "cross_correlation": "bridge_cross_correlation",
  "cost_volume": "bridge_cost_volume",
  "fps": "bridge_fps",
  "radius_search": "bridge_radius_search",
  "minmax_norm": "bridge_minmax_norm",
  "lrn": "bridge_lrn",
  "space_to_channel": "bridge_space_to_channel",
  "channel_to_space": "bridge_channel_to_space",
  "space_to_batch": "bridge_space_to_batch",
  "batch_to_space": "bridge_batch_to_space",
  "flatten": "bridge_flatten",
  "input_view": "bridge_input_view",
  "dynamic_slice": "bridge_dynamic_slice",
  "scaled_elementwise": "bridge_scaled_elementwise",
}

# per-family "work scalar": a monotone proxy for per-call cost (nearest-neighbour key +
# scaling ratio). e.g. sdpa ~ H*S^2*D (attention MACs); argmax/topk/sort ~ C*W (scan).
# The 15 subprocess-per-call ops are dispatch-floor-dominated (~flat in size); the scalar
# still gives sensible ordering. fps is the exception (scales ~N*k).
_BRIDGE_WORK = {
  "bridge_sdpa":   lambda p: float(p[0]) * float(p[1]) ** 2 * float(p[2]),  # H*S^2*D
  "bridge_argmax": lambda p: float(p[0]) * float(p[1]),                     # C*W
  "bridge_topk":   lambda p: float(p[0]) * float(p[1]),                     # C*W
  "bridge_sort":   lambda p: float(p[0]) * float(p[1]),                     # C*W
  "bridge_cross_product":     lambda p: 1.0,                                # fixed length-3
  "bridge_cross_correlation": lambda p: float(p[0]) * float(p[1]) * float(p[2]) * float(p[3]),  # H*W*Th*Tw
  "bridge_cost_volume":       lambda p: (float(p[1]) + 1.0) * float(p[0]),  # (R+1)*Wa
  "bridge_fps":               lambda p: float(p[0]) * float(p[1]),          # N*k (greedy)
  "bridge_radius_search":     lambda p: float(p[0]) * float(p[1]),          # N*Nc (all-pairs)
  "bridge_minmax_norm":       lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_lrn":               lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_space_to_channel":  lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_channel_to_space":  lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_space_to_batch":    lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_batch_to_space":    lambda p: float(p[0]) * float(p[1]) * float(p[2]) * float(p[3]),  # B*C*H*W
  "bridge_flatten":           lambda p: float(p[0]) * float(p[1]) * float(p[2]),  # C*H*W
  "bridge_input_view":        lambda p: float(p[1]),                        # window size
  "bridge_dynamic_slice":     lambda p: 1.0,                                # fixed W=4,size=2
  "bridge_scaled_elementwise": lambda p: float(p[0]),                       # W elems
}

# families whose key carries no numeric size (single anchor point); their work
# scalar is constant, so bridge_cost returns the anchor's measured min directly.
_SIZELESS_FAMILIES = ("bridge_cross_product", "bridge_dynamic_slice")


def _parse_bridge_key(family: str, key: str):
  """Parse a bridge_ops key's size string into a tuple of numbers, per family.
    Returns None if the key doesn't match the expected format for `family`."""
  if family == "bridge_sdpa":
    m = re.search(r"H=(\d+)\s+S=(\d+)\s+D=(\d+)", key)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
  if family in ("bridge_argmax", "bridge_topk", "bridge_sort"):
    m = re.search(r"\[(\d+)\s*,\s*(\d+)\]", key)
    return (int(m.group(1)), int(m.group(2))) if m else None
  # --- the 15 native-bridge families (bridge_cost_sweep.py key formats) -------
  if family in _SIZELESS_FAMILIES:
    return ()  # no size axis; single anchor point
  if family == "bridge_cross_correlation":  # "... [H,W] t=ThxTw"
    m = re.search(r"\[(\d+)\s*,\s*(\d+)\]\s*t=(\d+)x(\d+)", key)
    return tuple(int(m.group(i)) for i in (1, 2, 3, 4)) if m else None
  if family == "bridge_cost_volume":  # "cost_volume Wa=<Wa> R=<R>"
    m = re.search(r"Wa=(\d+)\s+R=(\d+)", key)
    return (int(m.group(1)), int(m.group(2))) if m else None
  if family == "bridge_fps":  # "fps N=<N> k=<k>"
    m = re.search(r"N=(\d+)\s+k=(\d+)", key)
    return (int(m.group(1)), int(m.group(2))) if m else None
  if family == "bridge_radius_search":  # "radius_search N=<N> Nc=<Nc>"
    m = re.search(r"N=(\d+)\s+Nc=(\d+)", key)
    return (int(m.group(1)), int(m.group(2))) if m else None
  if family in ("bridge_minmax_norm", "bridge_lrn", "bridge_space_to_channel",
                "bridge_channel_to_space", "bridge_space_to_batch", "bridge_flatten"):
    # "... [C,H,W] ..." -> (C, H, W)  (first 3-number bracket)
    m = re.search(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", key)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
  if family == "bridge_batch_to_space":  # "... [B,C,H,W] ..." -> (B, C, H, W)
    m = re.search(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", key)
    return tuple(int(m.group(i)) for i in (1, 2, 3, 4)) if m else None
  if family == "bridge_input_view":  # "input_view W=<W> size=<size>"
    m = re.search(r"W=(\d+)\s+size=(\d+)", key)
    return (int(m.group(1)), int(m.group(2))) if m else None
  if family == "bridge_scaled_elementwise":  # "scaled_elementwise W=<W> op=..."
    m = re.search(r"W=(\d+)", key)
    return (int(m.group(1)),) if m else None
  return None


@lru_cache(maxsize=1)
def _bridge_model() -> dict:
  """Parse model.bridge_ops into {family: [(size_tuple, min_us), ...]} (cached)."""
  out: dict = {}
  path = _cost_model_path()
  if path is None: return out
  try:
    j = json.loads(path.read_text())
    bridge = j.get("model", {}).get("bridge_ops", {})
  except Exception:
    return out
  for key, v in bridge.items():
    if not isinstance(v, dict): continue
    fam = v.get("family"); mn = v.get("min_us")
    if not fam or not mn: continue
    size = _parse_bridge_key(fam, key)
    if size is None: continue
    out.setdefault(fam, []).append((size, float(mn)))
  return out


def _node_size(fam: str, t):
  """Extract a bridge node's size tuple in the SAME convention _parse_bridge_key
    produces for `fam` (so the work scalar matches the measured-key parse).
    Returns None if the node doesn't have the shape this family expects."""
  s = t.srcs
  if fam == "bridge_sdpa":  # srcs[0] = [1, H, S, D]
    if not s or len(s[0].shape) != 4: return None
    _, H, S, D = s[0].shape
    return (int(H), int(S), int(D))
  if fam in ("bridge_argmax", "bridge_topk", "bridge_sort"):  # srcs[0] = [C, W]
    if not s or len(s[0].shape) != 2: return None
    C, W = s[0].shape
    return (int(C), int(W))
  if fam in _SIZELESS_FAMILIES:  # cross_product / dynamic_slice: fixed
    return ()
  if fam == "bridge_cross_correlation":  # srcs[0]=[H,W], srcs[1]=[Th,Tw] -> (H,W,Th,Tw)
    if len(s) < 2 or len(s[0].shape) != 2 or len(s[1].shape) != 2: return None
    H, W = s[0].shape; Th, Tw = s[1].shape
    return (int(H), int(W), int(Th), int(Tw))
  if fam == "bridge_cost_volume":  # out shape = (R+1, Wa) -> (Wa, R)
    if len(t.shape) != 2: return None
    R = int(t.shape[0]) - 1
    return (int(t.shape[1]), R)
  if fam == "bridge_fps":  # srcs[0]=[N,3], out=(k,3) -> (N, k)
    if not s or len(s[0].shape) != 2: return None
    return (int(s[0].shape[0]), int(t.shape[0]))
  if fam == "bridge_radius_search":  # out = [N, Nc]
    if len(t.shape) != 2: return None
    return (int(t.shape[0]), int(t.shape[1]))
  if fam in ("bridge_minmax_norm", "bridge_lrn", "bridge_space_to_channel",
             "bridge_channel_to_space", "bridge_space_to_batch"):  # src [1,C,H,W] -> (C,H,W)
    if not s or len(s[0].shape) != 4: return None
    _, C, H, W = s[0].shape
    return (int(C), int(H), int(W))
  if fam == "bridge_batch_to_space":  # src [B,C,H,W] -> (B,C,H,W)
    if not s or len(s[0].shape) != 4: return None
    B, C, H, W = s[0].shape
    return (int(B), int(C), int(H), int(W))
  if fam == "bridge_flatten":  # src [C,H,W] -> (C,H,W)
    if not s or len(s[0].shape) != 3: return None
    C, H, W = s[0].shape
    return (int(C), int(H), int(W))
  if fam == "bridge_input_view":  # src flattened to W, out=(size,) -> (W, size)
    if not s: return None
    return (_elems(s[0].shape), int(t.shape[0]))
  if fam == "bridge_scaled_elementwise":  # src flattened to W -> (W,)
    if not s: return None
    return (_elems(s[0].shape),)
  return None


def bridge_cost(t):
  """Measured per-call cost (us) for a bridge node, or None if its family has no data
    (caller falls back to roofline node_cost). Nearest-neighbour by work scalar, scaled."""
  fam = _OP_TO_FAMILY.get(t.op)
  if fam is None: return None
  size = _node_size(fam, t)
  if size is None: return None
  work_fn = _BRIDGE_WORK[fam]
  q = work_fn(size)
  floor = _constants()["floor_us"]
  pts = _bridge_model().get(fam)
  if not pts:
    # no measured points (e.g. sdpa): analytic roofline on the work scalar (~2 flops/MAC).
    return max(floor, 2.0 * q / _constants()["flops_per_us"])
  # nearest measured point by work scalar (in log space so ratios are symmetric)
  log_q = np.log(max(q, 1e-9))
  (psize, pmin) = min(pts, key=lambda pt: abs(log_q - np.log(max(work_fn(pt[0]), 1e-9))))
  pw = work_fn(psize)
  # proportional scaling along the dominant work axis, clamped at the floor.
  scaled = pmin * (q / pw) if pw > 0 else pmin
  return max(floor, scaled)


# --------------------------------------------------------------------------- #
# per-node structural roofline                                                 #
# --------------------------------------------------------------------------- #
def _elems(shape) -> int:
  n = 1
  for d in shape: n *= int(d)
  return n


def _weight_elems(t) -> int:
  """Total constant-weight elements a node carries (generic over attrs): any attr
    value that is a numpy array counts as streamed weight bytes."""
  return sum(v.size for v in t.attrs.values() if isinstance(v, np.ndarray))


def _node_flops(t) -> float:
  """Closed-form flops for the few ops that have one; 0 otherwise (bytes/floor-bound)."""
  op = t.op
  if op in ("matmul",):
    # x[..,K] @ W[N,K] (stored transposed) -> [..,N]; 2*M*K*N
    M = _elems(t.shape) // t.shape[-1]; N = t.shape[-1]
    W = t.attrs.get("wt")
    K = int(W.shape[1]) if W is not None else (t.srcs[0].shape[-1] if t.srcs else 1)
    return 2.0 * M * K * N
  if op == "bmm":
    # [..,M,K] @ [..,K,N] -> [..,M,N]
    a, b = t.srcs
    M, K = a.shape[-2], a.shape[-1]; N = b.shape[-1]
    batch = _elems(a.shape[:-2])
    return 2.0 * batch * M * K * N
  if op in ("conv", "conv_transpose"):
    w = t.attrs.get("weight")
    if w is None: return 0.0
    Cout = w.shape[0] if op == "conv" else w.shape[1]
    kH, kW = w.shape[2], w.shape[3]
    Cin_g = w.shape[1] if op == "conv" else w.shape[0]
    out_spatial = _elems(t.shape[2:]); N = t.shape[0]
    return 2.0 * N * Cout * out_spatial * Cin_g * kH * kW
  return 0.0


def node_cost(t, c: dict | None = None) -> float:
  """Roofline microseconds for one graph node: max(floor, bytes/BW, flops/COMPUTE)."""
  c = c if c is not None else _constants()
  in_elems = sum(_elems(s.shape) for s in t.srcs)
  bytes_moved = (in_elems + _elems(t.shape) + _weight_elems(t)) * 2.0  # fp16
  flops = _node_flops(t)
  return max(c["floor_us"], bytes_moved / c["bw_bytes_per_us"], flops / c["flops_per_us"])


def _node_roofline_us(t, c: dict, int8: bool) -> float:
  """max(compute, memory) microseconds for one node (NO floor) - the analytic model's
    additive form charges the dispatch overhead once per program, not per node."""
  in_elems = sum(_elems(s.shape) for s in t.srcs)
  if int8 and t.op == "matmul":
    bytes_moved = (in_elems + _elems(t.shape)) * 2.0 + _weight_elems(t) * 1.0
  else:
    bytes_moved = (in_elems + _elems(t.shape) + _weight_elems(t)) * 2.0
  flops = _node_flops(t)
  fpus = c["flops_per_us"]
  uf = c.get("util_k")                   # mid-utilization compute ramp (h14): eff peak < peak for
  if uf and flops > 0:                   # mid ops, CAPPED at peak so large ops are unaffected
    fpus *= min(1.0, (flops / uf) ** c["util_p"])
  return max(bytes_moved / c["bw_bytes_per_us"], flops / fpus)


def _estimate_analytic(out, arch: str, int8: bool) -> float:
  """Measurement-free per-chip latency (microseconds) via the extracted analytic
    roofline: `overhead + sum_nodes max(compute, memory)` per fused program, plus a
    host round-trip per netplist cut. Reproduces the measured M1 convs within +/-17%
    and projects to any of the 28 chips from costmodel_curves.json (Direction A)."""
  c = _analytic_constants(arch)
  order = _topo(out)
  nodes = [t for t in order if t.op != "input"]
  cut_nodes = [t for t in nodes if t.op in NETPLIST_OPS]
  region_nodes = [t for t in nodes if t.op not in NETPLIST_OPS]
  region_work = sum(_node_roofline_us(t, c, int8) for t in region_nodes)
  if not cut_nodes: return c["floor_us"] + region_work
  n_cuts = len(cut_nodes)
  n_regions = n_cuts + 1 if region_nodes else 0
  cut_work = sum(_node_roofline_us(t, c, int8) for t in cut_nodes)
  return n_regions * c["floor_us"] + region_work + cut_work + n_cuts * c["cut_us"]


# --------------------------------------------------------------------------- #
# composition: replicate _compile's fused-region vs netplist-cut segmentation   #
# --------------------------------------------------------------------------- #
def estimate(out, int8: bool = False, target: str | None = None) -> float:
  """Estimate the compiled latency (us) of the graph rooted at `out`.

    int8 scales streamed weight bytes by ~0.5 (the only dtype lever; everything else is
    structural). `target` (an arch string) switches to the analytic per-chip model
    (Direction A); `target=None` uses the M5-measured heuristic below.
    """
  if target is not None: return _estimate_analytic(out, target, int8)
  c = _constants()
  order = _topo(out)
  nodes = [t for t in order if t.op != "input"]
  cut_nodes = [t for t in nodes if t.op in NETPLIST_OPS]
  region_nodes = [t for t in nodes if t.op not in NETPLIST_OPS]

  floor = c["floor_us"]

  def _ncost(t) -> float:
    if int8 and t.op in ("matmul",):
      # int8 halves the weight bytes for the (dominant) projection weights
      in_elems = sum(_elems(s.shape) for s in t.srcs)
      wbytes = _weight_elems(t) * 1.0   # int8 = 1 byte/elem instead of 2
      bytes_moved = (in_elems + _elems(t.shape)) * 2.0 + wbytes
      flops = _node_flops(t)
      return max(floor, bytes_moved / c["bw_bytes_per_us"], flops / c["flops_per_us"])
    return node_cost(t)

  # int8 tie-breaker: a tiny weight-byte-proportional discount so a floor-bound graph
  # predicts int8 <= fp16 (decisive, not an arbitrary tie); kept below a floor's worth.
  int8_discount = 0.0
  if int8:
    saved_bytes = sum(_weight_elems(t) for t in region_nodes if t.op == "matmul")
    int8_discount = min(0.49 * floor, saved_bytes / c["bw_bytes_per_us"] * 0.5)

  if not cut_nodes:
    # one fused program: one floor + each node's above-floor work
    region = sum(max(0.0, _ncost(t) - floor) for t in region_nodes)
    return floor + region - int8_discount

  # segmented: fused regions (each pays one floor) interleaved with cuts. Approximate
  # region count as (n_cuts + 1); keep the global above-floor sum, add the floors + cuts.
  n_cuts = len(cut_nodes)
  n_regions = (n_cuts + 1) if region_nodes else 0
  region_work = sum(max(0.0, _ncost(t) - floor) for t in region_nodes)
  # each bridge node's own compute cost: prefer the measured per-family lookup
  # (bridge_cost); fall back to the generic roofline for families with no data.
  def _bridge_or_roofline(t) -> float:
    bc = bridge_cost(t)
    return bc if bc is not None else _ncost(t)
  cut_work = sum(_bridge_or_roofline(t) for t in cut_nodes)
  return n_regions * floor + region_work + cut_work + n_cuts * c["cut_us"] - int8_discount


# --------------------------------------------------------------------------- #
# precision / fp16-cancellation risk MODEL  (the optimizer's numerics pruner)  #
# --------------------------------------------------------------------------- #
# Cheap structural pattern-matcher (not an error bound) over three fp16 failure modes:
# (a) signed long-contraction reduce_sum (narrow accumulator; fix reduce_sum->matmul),
# (b) sub of two live tensors (candidate CFG cancellation; fix paired-fp16),
# (c) group_norm per-axis cliff (avoid/flag). Per-graph signal = max node error proxy.
# See .superpowers/sdd/gold-core-heavy.md for the full failure-mode notes.

# order-of-magnitude error proxies, deliberately coarse (clean fp16 op ~1e-3 relerr).
_FP16_CLEAN = 1e-3            # a clean fp16 op's relative error (corpus-calibrated)
_NARROW_SUM_FLOOR = 256      # contraction length above which a signed reduce_sum
                             # starts to lose digits to the narrow accumulator


def _is_signed_producer(t) -> bool:
  """True if `t` can produce signed values (a sum over it can cancel); non-negative
    outputs (square/abs/relu/sigmoid/exp/softplus) are excluded, else conservatively signed."""
  return t.op not in ("square", "abs", "relu", "relu6", "sigmoid", "exp",
                      "softplus", "softmax")


def _reduce_len(t) -> int:
  """Number of elements summed per output element of a reduce node."""
  src = t.srcs[0] if t.srcs else None
  if src is None: return 1
  n = 1
  for ax in t.attrs.get("axes", ()):
    if 0 <= ax < len(src.shape): n *= int(src.shape[ax])
  return n


def precision_risk(out, verbose: bool = False) -> dict:
  """Heuristic fp16-cancellation risk for the graph rooted at `out`.

    Returns a dict::

        {"graph_error": float,        # est. worst-case relerr proxy in [0,1]
         "nodes": [ {idx, op, kind, est_error, fixable, reason}, ... ],
         "hotspots": [idx, ...]}      # node indices flagged above the clean floor

    `kind` in {narrow_sum, cancel_sub, groupnorm_cliff}. `fixable` names the
    numerics-aware rewrite that addresses it ("reduce_sum->matmul", "paired-fp16",
    or "" for an avoid/flag-only cliff). This is a HEURISTIC, not a bound - see the
    module note above for what it catches and (importantly) misses.
    """
  order = _topo(out)
  nodes = []
  for i, t in enumerate(order):
    # (a) narrow-accumulator signed reduce_sum
    if t.op == "reduce_sum":
      K = _reduce_len(t)
      signed = (not t.srcs) or _is_signed_producer(t.srcs[0])
      if signed and K >= _NARROW_SUM_FLOOR:
        # error grows ~ sqrt(K) under the narrow accumulator (proxy, not a bound), cap 1.0.
        est = min(1.0, _FP16_CLEAN * (K ** 0.5))
        nodes.append({"idx": i, "op": t.op, "kind": "narrow_sum",
                      "est_error": est, "fixable": "reduce_sum->matmul",
                      "reason": f"signed reduce_sum over K={K} (narrow fp16 accumulator)"})
      continue
    # (b) CFG-style subtract - candidate cancellation (data-dependent, can't
    #     confirm structurally). Flag sub of two non-trivial activations.
    if t.op == "sub" and len(t.srcs) == 2:
      big = _elems(t.shape) >= 64    # a vector/tensor sub (not a scalar bias)
      both_live = all(s.op not in ("muls",) for s in t.srcs)
      if big and both_live:
        nodes.append({"idx": i, "op": t.op, "kind": "cancel_sub",
                      "est_error": _FP16_CLEAN,  # only RISKS blowing up; unknown w/o data
                      "fixable": "paired-fp16",
                      "reason": "subtract of two live tensors (CANDIDATE catastrophic "
                                "cancellation - confirm with data; fix is upstream paired-fp16)"})
      continue
    # (c) group_norm per-axis wall: the rank-4 tiling makes the cliff max(C/groups, H*W) > 65536.
    if t.op == "group_norm" and len(t.shape) == 4:
      _, C, H, W = t.shape
      groups = int(t.attrs.get("groups", 1)) or 1
      if max(C // groups, H * W) > 65536:
        nodes.append({"idx": i, "op": t.op, "kind": "groupnorm_cliff",
                      "est_error": 0.0, "fixable": "",
                      "reason": f"group_norm tiled axis max(C/groups,H*W)>65536 at {H}x{W}: "
                                "AVOID - exceeds the ANE per-axis bound"})
      continue

  # Default hotspots = only the reliable structural signals (narrow reduce_sum over the
  # floor + the group_norm wall). cancel_sub is speculative/unconfirmable -> informational
  # only (in `nodes` for verbose, not a default warning).
  hotspots = [n["idx"] for n in nodes if n["est_error"] > _FP16_CLEAN
              or n["kind"] == "groupnorm_cliff"]
  graph_error = max([_FP16_CLEAN] + [n["est_error"] for n in nodes])
  if verbose:
    print(f"[precision] graph_error~{graph_error:.2e}, {len(nodes)} flagged node(s):")
    for n in nodes:
      print(f"  node {n['idx']:3d} {n['op']:12s} kind={n['kind']:16s} "
            f"est~{n['est_error']:.2e} fix={n['fixable'] or '(avoid)'}: {n['reason']}")
  return {"graph_error": graph_error, "nodes": nodes, "hotspots": hotspots}
