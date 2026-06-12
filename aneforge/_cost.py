"""Op-agnostic structural cost model for aneforge graphs — the optimizer's PRUNER.

`estimate(out) -> microseconds` gives a fast, structural roofline estimate of a
compiled program's latency WITHOUT touching the device. It is deliberately simple
and op-agnostic: every node's cost is a roofline of `max(floor, bytes/BW,
flops/COMPUTE)`, where

  - `bytes` is derived GENERICALLY from shapes — (sum of input elems + output
    elems + weight elems) * 2 (fp16). No per-op byte table; this works for ANY op.
  - `flops` is only known for the few ops with a closed-form (matmul/linear/bmm/
    conv/conv_transpose); every other op contributes 0 flops (it is then bytes- or
    floor-bound, which matches the calibration: the cheap fused ops all sit at the
    dispatch floor).

Composition mirrors `_compile`'s segmentation exactly: the graph is one fused
program unless it contains a netplist-bridge op (NETPLIST_OPS), in which case it is
cut into fused regions interleaved with native sub-programs. A fused region costs
`floor + sum(max(0, node_cost - floor))` (the fusion discount: one dispatch floor
for the whole region, plus only the above-floor work of each node). Each cut adds a
`cut_penalty` and the bridge node's own roofline.

A PRUNER, not ground truth. Its job is to ORDER variants so the autotuner can skip
measuring ones predicted far worse than the current best. Real selection is always
by on-device measurement (see _optimize.measure). Calibrated constants load from the
bundled aneforge/ane_cost_model.json when present; else the documented defaults below.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._compile import NETPLIST_OPS, _topo

# --------------------------------------------------------------------------- #
# calibrated constants (load from the cost-model JSON; else documented defaults) #
# --------------------------------------------------------------------------- #
# Defaults are read off ane_cost_model.json's calibration:
#   - FLOOR_US  ~ the per-call dispatch floor (smallest fused-op min latency).
#   - CUT_US    ~ the per-cut penalty (a native sub-program + host round-trip; the
#                 composition probe measured ~150-300us; 200 is the documented mid).
#   - BW_BPUS   ~ effective fp16 streaming bandwidth, bytes/us. From the matmul
#                 sweep: K=4096 streams 32 MiB in ~293us -> ~114 GB/s ~ 1.1e5 B/us.
#   - FLOPS_PUS ~ effective fp16 compute, flops/us. From conv C=512: 4.83 GFLOP in
#                 ~334us -> ~14.5 TFLOP/s ~ 1.45e7 flop/us.
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
            if floor:
                c["floor_us"] = float(floor)
            # derive BW / COMPUTE from the matmul + conv scaling fits when present
            mm = model.get("matmul", {})
            if mm.get("slope_us_per_unit"):
                # us per flop -> flop per us. (matmul has bytes==flops here, but the
                # fit captures the compute-or-stream slope either way.)
                c["flops_per_us"] = 1.0 / float(mm["slope_us_per_unit"])
            cv = model.get("conv_channels", {})
            if cv.get("slope_us_per_unit"):
                c["flops_per_us"] = max(c["flops_per_us"], 1.0 / float(cv["slope_us_per_unit"]))
            # cut penalty: a native bridge sub-program's floor sits ~90-110us above
            # nothing; use the smallest bridge min as the marginal cut cost when present.
            bridge = model.get("bridge_ops", {})
            mins = [v.get("min_us") for v in bridge.values()
                    if isinstance(v, dict) and v.get("min_us") and v["min_us"] < 1000]
            if mins:
                c["cut_us"] = float(min(mins))
        except Exception:
            pass
    return c


def _cost_model_path():
    bundled = Path(__file__).resolve().parent / "ane_cost_model.json"
    if bundled.exists():
        return bundled
    for up in Path(__file__).resolve().parents:        # legacy out-of-tree calibration
        p = up / "ane_artifacts" / "runtime" / "ane_cost_model.json"
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# the measurement-free ANALYTIC per-chip cost model (Direction A)              #
# --------------------------------------------------------------------------- #
# The compiler carries its own analytic cycles->roofline->wall-time model
# (ZinNEPerf, non-SIP). It was decompiled and the per-chip HAL perf fields +
# freq/efficiency curves walked live for all 28 targets -> the bundled
# costmodel_curves.json (ANEForge reverse-engineering). The model is
#   t = overhead + max( flops/peak , bytes/bw )          [per fused program]
# anchored to silicon-measured chips (_ANCHORS: M1/h13 + M5/h17s) and scaled to any
# other chip from its family's anchor by {cores (BW), clock (floor), cores*eff (peak)}.
# The M1 anchors, both from measurement:
#   * latency-roofline FIT (reproduces the 5 measured M1 convs within +/-17%):
#       peak 3.25 TFLOP/s, BW 9.0 GB/s, dispatch overhead 0.22 ms.
#   * headline fp16 PEAK (measured): 1.8 TFLOP/s -> project_peak().
# The M5 anchor is the 2026-06-05 loop-closure re-fit (BW 57 GB/s, floor 110 us,
# peak 8.9 TFLOP/s) — see _ANCHORS below for why BW is core-scaled, not clock-scaled.
# Cross-chip throughput scales by cores*eff_freq(0.8*fmax) relative to M1 (the
# eff_freq is the second column of eff_map_0x7a8: the engine's effective frequency,
# already derated for the high-clock MAC-rate falloff on A14+).
_M1_FIT_PEAK_FLOPS = 3.25e12        # latency roofline compute ceiling (fit on M1 convs)
_M1_FIT_BW_BYTES = 9.0e9            # effective streaming BW (dispatch-bound; ~= measured 10 GB/s).
#   (2026-06-09): a broad estimate-vs-measured sweep found this over-predicts EXTREME
#   bandwidth-bound shapes ~4-5x (m1 1x4096x4096: pred ~3700us vs measured ~800us -> ~42 GB/s
#   effective). But it is jointly calibrated with peak/overhead into the +/-17% 5-conv fit
#   (test_cost_model_analytic pins it), so it can't be bumped alone without regressing that
#   fit -- a proper re-anchor needs a joint roofline re-fit over a broader measured set.
_M1_FIT_OVERHEAD_US = 220.0         # additive per-program dispatch overhead (0.22 ms)
_M1_MEASURED_PEAK_TFLOPS = 1.8      # headline fp16 peak (the project_peak absolute anchor)
_CLOCK_FRACTION = 0.8              # operating clock ~= 0.8 * fmax (DAT_2241e37f8)

# arch string (aneforge lowercase, e.g. 'h13'/'h17s') -> a per-family fallback curve
# key for chips without their own entry in costmodel_curves.json. The curves cover the
# distinct cost profiles; an arch missing one (h14g, h16c, h17a, h18, ...) folds to its
# family's representative die (matching _targets._ARCH_FAMILY tiers).
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
            "unparseable copy means this installation is broken — reinstall aneforge.") from e


def _curve_for_arch(arch: str) -> dict:
    """Resolve an arch string to its cost curve, case-insensitively, with a per-family
    fallback for chips that share another die's profile. Raises KeyError on a name that
    is neither a known curve nor a known arch."""
    curves = _curves()
    key = arch.strip()
    # exact, then case-insensitive (JSON uses 'H13'/'H17s'; arch is 'h13'/'h17s').
    if key in curves:
        return curves[key]
    low = {k.lower(): k for k in curves}
    if key.lower() in low:
        return curves[low[key.lower()]]
    # fall back to the arch's capability family's representative die.
    from . import _targets as _TG
    fam = _TG.family_of_arch(key)                  # raises on a truly unknown name
    fk = _FAMILY_CURVE.get(int(fam))
    if fk and fk in curves:
        return curves[fk]
    raise KeyError(f"no cost curve for arch {arch!r}")


def _interp(x: float, xs, ys) -> float:
    """Piecewise-linear interp of ys at x over sorted xs (clamped at the ends)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _eff_freq_at_op(curve: dict) -> tuple[float, float]:
    """(operating_freq, effective_freq) at the operating clock f = 0.8*fmax for a chip.
    effective_freq is eff_map_0x7a8's second column interpolated at f — the engine's
    derated throughput frequency (M1=1.0*f; A14+ derates ~0.84)."""
    freqs = curve["freq_0x760"]
    f = _CLOCK_FRACTION * max(freqs)
    em = curve["eff_map_0x7a8"]                    # [[freq, eff_freq], ...]
    xs = [p[0] for p in em]
    ys = [p[1] for p in em]
    return f, _interp(f, xs, ys)


def _compute_scale(arch: str) -> float:
    """Compute-throughput multiplier of `arch` relative to M1 (H13): the ratio of
    cores * effective_freq(0.8*fmax). This is the cross-chip peak-throughput scaler."""
    c = _curve_for_arch(arch)
    _, ce = _eff_freq_at_op(c)
    m1 = _curve_for_arch("h13")
    _, m1e = _eff_freq_at_op(m1)
    return (c["cores_0x238"] * ce) / (m1["cores_0x238"] * m1e)


def project_peak(arch: str) -> dict:
    """Measurement-free fp16 peak-throughput projection for any ANE target, anchored to
    the measured M1 point (1.8 TFLOP/s). Returns {tflops, rel_m1, cores, ghz} — the
    generational-scaling table (M5 ~5.5x, H17d ~22x, M11 ~0.1x) needs no silicon beyond M1."""
    scale = _compute_scale(arch)
    c = _curve_for_arch(arch)
    return {
        "tflops": _M1_MEASURED_PEAK_TFLOPS * scale,
        "rel_m1": scale,
        "cores": int(c["cores_0x238"]),
        "ghz": _CLOCK_FRACTION * max(c["freq_0x760"]) / 1e9,
    }


# Silicon-measured roofline anchors, one per measured chip. h13 is the M1 latency fit
# (the +/-17% 5-conv validation); h17s is the M5 LOOP-CLOSURE re-fit (m5_bw_floor.py /
# m5_weight_stream.py): BW 57 GB/s measured, dispatch floor ~110 us, peak 8.9 TFLOP/s
# (= project_peak('h17s'), validated by the re-fit landing the quoted convs within ~13%).
# The earlier single-anchor model scaled M1's BW by CLOCK and over-predicted M5 ~2x
# (mean |err| 99%): effective BW tracks CORE COUNT (16/4 -> 5.5x), not clock (x1.66) —
# a faster clock does not widen the DMA path, more cores do.
_ANCHORS = {
    "h13": {"bw": _M1_FIT_BW_BYTES, "floor_us": _M1_FIT_OVERHEAD_US, "peak": _M1_FIT_PEAK_FLOPS},
    # M2 Pro silicon (A14), measured. `util` = a mid-utilization compute ramp fit to the 25-point
    # h14 grid (papers H14_CALIBRATION_GRID.md): effective compute throughput is far below peak for
    # mid-size ops (a 768^3 GEMM sustains ~1.5 of the 7.24 TFLOP/s), ramping with per-op FLOPs.
    # eff_peak = peak * min(1, (flops/F)^q): a CAPPED power law that returns to full peak for large
    # ops (the cap matters — a 2048^3 GEMM already hits peak and must not be slowed). Fit (F=1.8e10
    # FLOP, q=0.38) cuts the grid's mean error 1.61x -> 1.16x with the peak points recovered.
    "h14": {"bw": 48.0e9, "floor_us": 100.0, "peak": 7.3e12, "util": (1.80e10, 0.38)},
    "h17s": {"bw": 57.0e9, "floor_us": 110.0, "peak": 8.9e12},
}


def _anchor_for_arch(arch: str) -> str:
    """The measured anchor chip for `arch`: its own entry when silicon-measured, else the
    nearest measured generation. Three measured anchors now: A13/h13 (M1), A14/h14 (M2 Pro),
    A16/h17s (M5). A15 (no M3 silicon yet) uses the A14 anchor as the nearest below it."""
    key = arch.strip().lower()
    if key in _ANCHORS:
        return key
    from . import _targets as _TG
    fam = int(_TG.family_of_arch(key))
    if fam >= 5:
        return "h17s"
    if fam >= 3:                 # A14 (measured) and A15 (nearest measured below A16)
        return "h14"
    return "h13"


def estimate_provenance(target: str) -> dict:
    """Is `estimate(out, target=...)` silicon-anchored or extrapolated for `target`?

    Three chips were measured and fit a roofline anchor (`_ANCHORS`): A13/h13 (M1),
    A14/h14 (M2 Pro), A16/h17s (M5). A target whose capability family OWNS one of those
    anchors is silicon-measured (the A16 tier folds H16/H17* into the h17s point); every
    other target is extrapolated from the nearest measured anchor by its {cores, clock,
    efficiency} curve. Surfaces that distinction so a caller knows whether a per-chip
    estimate rests on measured silicon or a generational projection.

    Returns `{'target', 'anchor', 'measured': bool, 'basis': str}` where `anchor` is
    the silicon point the estimate is built on, `measured` is True iff `target`'s family
    has its own anchor, and `basis` is `'silicon'` or `'extrapolated-from-<anchor>'`.
    Raises `ValueError` on an unknown arch (mirrors `cross_compile_check`)."""
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
    """The {floor_us, cut_us, bw_bytes_per_us, flops_per_us} for `arch`'s ANALYTIC
    roofline, taken from the nearest silicon-measured anchor (_ANCHORS) and scaled:
    effective streaming BW by CORE-COUNT ratio (the verified M5 loop-closure mechanism,
    NOT clock), the dispatch floor by operating-clock ratio (setup runs at engine clock),
    and the compute peak by the relative cores*eff_freq scale. A measured chip gets its
    anchor exactly. cut_us (a host round-trip for a netplist bridge) is host-side, so
    chip-independent."""
    a_key = _anchor_for_arch(arch)
    a = _ANCHORS[a_key]
    c, ac = _curve_for_arch(arch), _curve_for_arch(a_key)
    f, _ = _eff_freq_at_op(c)
    fa, _ = _eff_freq_at_op(ac)
    clk = f / fa
    cores = c["cores_0x238"] / ac["cores_0x238"]
    scale = _compute_scale(arch) / _compute_scale(a_key)
    out = {
        "floor_us": a["floor_us"] / clk,                        # overhead shrinks with clock
        "cut_us": _DEFAULTS["cut_us"],                          # host round-trip, chip-agnostic
        "bw_bytes_per_us": (a["bw"] * cores) / 1e6,             # BW scales with CORES, not clock
        "flops_per_us": (a["peak"] * scale) / 1e6,
    }
    if "util" in a:                                             # mid-utilization compute ramp (h14)
        uf, uq = a["util"]                                      # eff_peak = peak*min(1,(flops/F)^q)
        out["util_k"], out["util_p"] = uf * scale, uq           # F (sat flops) scales with compute
    return out


# --------------------------------------------------------------------------- #
# measured per-bridge-op cost model                                            #
# --------------------------------------------------------------------------- #
# The 19 NETPLIST bridge ops (sdpa, argmax, topk, sort, ...) have no closed-form
# flops, so node_cost() would cost them at the generic dispatch floor. But
# ane_cost_model.json's `model.bridge_ops` holds measured per-config min latencies
# (keyed by a size string like "sdpa H=8 S=128 D=64", tagged with a `family`).
# _bridge_model() parses these into per-family points so bridge_cost() uses the
# measured value, not the floor.
#
# Key format -> size params (parsed by family):
#   bridge_sdpa    "sdpa H=<H> S=<S> D=<D>"   -> (H, S, D)
#   bridge_argmax  "argmax [<C>,<W>]"         -> (C, W)
#   bridge_topk    "topk k=<k> [<C>,<W>]"     -> (C, W)   (k fixed 5 in the sweep)
#   bridge_sort    "sort [<C>,<W>]"           -> (C, W)
#
# Interpolation (approximate): the grid is sparse (2-6 points/family). Per family
# we pick a work scalar w(size) (see _BRIDGE_WORK), anchor to the nearest measured
# point by it, and scale min_us by the work ratio, clamped to the dispatch floor.
# Captures the right magnitude and ordering (all the pruner needs), not a per-shape
# fit. Families with no data fall back to the roofline node_cost().
import re

# op name -> bridge family in the JSON. All 19 NETPLIST bridge ops are measured by
# the cost sweeps (sdpa/argmax/topk/sort by ane_cost_model_sweep.py's `bridge`
# group; the remaining 15 by bridge_cost_sweep.py), so every bridge node maps to a
# measured family here. (An op missing from this table, or whose family has no rows
# in the JSON, still falls back to the roofline in bridge_cost().)
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

# per-family "work scalar": a monotone proxy for the op's per-call cost, used as the
# nearest-neighbour key + the proportional-scaling ratio. Defensible choices:
#   sdpa  ~ attention MACs = H * S^2 * D  (QK^T + AV both scale this way)
#   argmax/topk/sort ~ elements scanned = C * W  (a full pass over the row-major map)
#
# The 15 native-bridge ops below run via the A1 subprocess-per-call path (no A2
# persistent worker exists for them — see _netplist_worker._WORKER_BUILDERS), so
# their measured per-call cost is DISPATCH-FLOOR DOMINATED (~30-60ms subprocess
# spawn + ANECCompile load), nearly flat in size. The work scalar is still the
# right element/MAC proxy so the nearest-neighbour anchor + ordering is sensible;
# the proportional scaling barely moves since the measured points are ~flat.
# fps is the exception — it scales ~N*k (seconds/call).
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
    if path is None:
        return out
    try:
        j = json.loads(path.read_text())
        bridge = j.get("model", {}).get("bridge_ops", {})
    except Exception:
        return out
    for key, v in bridge.items():
        if not isinstance(v, dict):
            continue
        fam = v.get("family")
        mn = v.get("min_us")
        if not fam or not mn:
            continue
        size = _parse_bridge_key(fam, key)
        if size is None:
            continue
        out.setdefault(fam, []).append((size, float(mn)))
    return out


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _node_size(fam: str, t):
    """Extract a bridge node's size tuple in the SAME convention _parse_bridge_key
    produces for `fam` (so the work scalar matches the measured-key parse).
    Returns None if the node doesn't have the shape this family expects."""
    s = t.srcs
    if fam == "bridge_sdpa":  # srcs[0] = [1, H, S, D]
        if not s or len(s[0].shape) != 4:
            return None
        _, H, S, D = s[0].shape
        return (int(H), int(S), int(D))
    if fam in ("bridge_argmax", "bridge_topk", "bridge_sort"):  # srcs[0] = [C, W]
        if not s or len(s[0].shape) != 2:
            return None
        C, W = s[0].shape
        return (int(C), int(W))
    if fam in _SIZELESS_FAMILIES:  # cross_product / dynamic_slice: fixed
        return ()
    if fam == "bridge_cross_correlation":  # srcs[0]=[H,W], srcs[1]=[Th,Tw] -> (H,W,Th,Tw)
        if len(s) < 2 or len(s[0].shape) != 2 or len(s[1].shape) != 2:
            return None
        H, W = s[0].shape; Th, Tw = s[1].shape
        return (int(H), int(W), int(Th), int(Tw))
    if fam == "bridge_cost_volume":  # out shape = (R+1, Wa) -> (Wa, R)
        if len(t.shape) != 2:
            return None
        R = int(t.shape[0]) - 1
        return (int(t.shape[1]), R)
    if fam == "bridge_fps":  # srcs[0]=[N,3], out=(k,3) -> (N, k)
        if not s or len(s[0].shape) != 2:
            return None
        return (int(s[0].shape[0]), int(t.shape[0]))
    if fam == "bridge_radius_search":  # out = [N, Nc]
        if len(t.shape) != 2:
            return None
        return (int(t.shape[0]), int(t.shape[1]))
    if fam in ("bridge_minmax_norm", "bridge_lrn", "bridge_space_to_channel",
               "bridge_channel_to_space", "bridge_space_to_batch"):  # src [1,C,H,W] -> (C,H,W)
        if not s or len(s[0].shape) != 4:
            return None
        _, C, H, W = s[0].shape
        return (int(C), int(H), int(W))
    if fam == "bridge_batch_to_space":  # src [B,C,H,W] -> (B,C,H,W)
        if not s or len(s[0].shape) != 4:
            return None
        B, C, H, W = s[0].shape
        return (int(B), int(C), int(H), int(W))
    if fam == "bridge_flatten":  # src [C,H,W] -> (C,H,W)
        if not s or len(s[0].shape) != 3:
            return None
        C, H, W = s[0].shape
        return (int(C), int(H), int(W))
    if fam == "bridge_input_view":  # src flattened to W, out=(size,) -> (W, size)
        if not s:
            return None
        return (_prod(s[0].shape), int(t.shape[0]))
    if fam == "bridge_scaled_elementwise":  # src flattened to W -> (W,)
        if not s:
            return None
        return (_prod(s[0].shape),)
    return None


def bridge_cost(t):
    """Measured per-call cost (microseconds) for a NETPLIST bridge node, or None if
    the node's family has no measured data (caller falls back to roofline node_cost).

    Maps the node's op -> family, extracts its size parameters from the graph node
    (sdpa: srcs[0]=[1,H,S,D]; argmax/topk/sort: srcs[0]=[C,W]), then nearest-neighbour
    anchors to the measured point with the closest work scalar and scales by the work
    ratio (see the module note above for the interpolation's approximate scope)."""
    fam = _OP_TO_FAMILY.get(t.op)
    if fam is None:
        return None
    # extract this node's size parameters from its shape/srcs/attrs, in the same
    # tuple convention _parse_bridge_key produces for this family.
    size = _node_size(fam, t)
    if size is None:
        return None
    work_fn = _BRIDGE_WORK[fam]
    q = work_fn(size)
    floor = _constants()["floor_us"]
    pts = _bridge_model().get(fam)
    if not pts:
        # No measured points for this family in the shipped cost model (e.g. sdpa is not in the
        # bridge sweep, so af.estimate used to return None for attention). Fall back to an
        # analytic roofline on the family's work scalar (~2 flops per MAC), clamped at the
        # dispatch floor — an order-of-magnitude estimate beats none.
        return max(floor, 2.0 * q / _constants()["flops_per_us"])
    # nearest measured point by work scalar (in log space so ratios are symmetric)
    def _key(pt):
        pw = work_fn(pt[0])
        return abs(np.log(max(q, 1e-9)) - np.log(max(pw, 1e-9)))
    (psize, pmin) = min(pts, key=_key)
    pw = work_fn(psize)
    # proportional scaling along the dominant work axis, clamped at the floor.
    scaled = pmin * (q / pw) if pw > 0 else pmin
    return max(floor, scaled)


# --------------------------------------------------------------------------- #
# per-node structural roofline                                                 #
# --------------------------------------------------------------------------- #
def _elems(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _weight_elems(t) -> int:
    """Total constant-weight elements a node carries (generic over attrs): any attr
    value that is a numpy array counts as streamed weight bytes."""
    n = 0
    for v in t.attrs.values():
        if isinstance(v, np.ndarray):
            n += v.size
    return n


def _node_flops(t) -> float:
    """Closed-form flops for the few ops that have one; 0 otherwise (bytes/floor-bound)."""
    op = t.op
    if op in ("matmul",):
        # x[..,K] @ W[N,K] (stored transposed) -> [..,N]; 2*M*K*N
        M = _elems(t.shape) // t.shape[-1]
        N = t.shape[-1]
        W = t.attrs.get("wt")
        K = int(W.shape[1]) if W is not None else (t.srcs[0].shape[-1] if t.srcs else 1)
        return 2.0 * M * K * N
    if op == "bmm":
        # [..,M,K] @ [..,K,N] -> [..,M,N]
        a, b = t.srcs
        M, K = a.shape[-2], a.shape[-1]
        N = b.shape[-1]
        batch = _elems(a.shape[:-2])
        return 2.0 * batch * M * K * N
    if op in ("conv", "conv_transpose"):
        w = t.attrs.get("weight")
        if w is None:
            return 0.0
        Cout = w.shape[0] if op == "conv" else w.shape[1]
        kH, kW = w.shape[2], w.shape[3]
        Cin_g = w.shape[1] if op == "conv" else w.shape[0]
        out_spatial = _elems(t.shape[2:])
        N = t.shape[0]
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
    """max(compute, memory) microseconds for one node (NO floor) — the analytic model's
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
    if not cut_nodes:
        return c["floor_us"] + region_work
    n_regions = (n_cuts := len(cut_nodes)) + 1 if region_nodes else 0
    cut_work = sum(_node_roofline_us(t, c, int8) for t in cut_nodes)
    return n_regions * c["floor_us"] + region_work + cut_work + n_cuts * c["cut_us"]


# --------------------------------------------------------------------------- #
# composition: replicate _compile's fused-region vs netplist-cut segmentation   #
# --------------------------------------------------------------------------- #
def estimate(out, int8: bool = False, target: str | None = None) -> float:
    """Estimate the compiled latency (microseconds) of the graph rooted at `out`.

    int8 scales streamed weight bytes by ~0.5 (per-channel int8 streams half the
    bytes; activations stay fp16). This is the only dtype lever the model needs to
    rank int8 vs fp16; everything else is structural.

    `target` (an ANE arch string, e.g. 'h13'/'h17s') switches to the measurement-free
    ANALYTIC per-chip model (Direction A) — a roofline taken from the nearest
    silicon-measured anchor (M1/h13 or M5/h17s) and scaled to that chip's {cores, clock,
    efficiency} curve, valid for all 28 chips with no on-device measurement (+/-17% on
    the measured M1 convs; the M5 anchor lands the loop-closure convs within ~15% on the
    quoted set). `target=None` (the default) uses the precise M5-measured heuristic below,
    unchanged.
    """
    if target is not None:
        return _estimate_analytic(out, target, int8)
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

    # int8 tie-breaker: even when a graph is floor-bound (so the roofline ties int8
    # and fp16 at the dispatch floor), int8 always streams <= fp16 bytes, never more.
    # Encode that as a tiny weight-byte-proportional discount so the model is DECISIVE
    # and directionally correct (int8 predicted <= fp16) instead of an arbitrary tie.
    # Scaled well below a floor's worth so it never reorders variants with a real cost
    # difference.
    int8_discount = 0.0
    if int8:
        saved_bytes = sum(_weight_elems(t) for t in region_nodes if t.op == "matmul")
        int8_discount = min(0.49 * floor, saved_bytes / c["bw_bytes_per_us"] * 0.5)

    if not cut_nodes:
        # one fused program: one floor + each node's above-floor work
        region = sum(max(0.0, _ncost(t) - floor) for t in region_nodes)
        return floor + region - int8_discount

    # segmented: fused regions (each pays one floor) interleaved with cuts.
    # Mirror _compile_segmented: a region is built per cut-source and for the final
    # output. Approximate region count as the number of distinct fused-program segments
    # — at most (n_cuts + 1) — and charge each a floor; the cheap nodes spread across
    # them, so keep the global above-floor sum and add (n_regions) floors plus the cuts.
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
# Companion to estimate(): a cheap structural pattern-matcher (not an error bound)
# over the three characterized fp16 failure modes (fp16_envelope.py):
#   (a) reduce_sum over signed terms, long contraction -> the narrow fp16 reduce
#       accumulator re-injects per-add rounding the wide matmul avoids. Fixable by
#       the reduce_sum->matmul rewrite (>= accuracy).
#   (b) subtract of two large, structurally-small-result quantities -> CFG-style
#       cancellation. Not detectable structurally (data-dependent), so every
#       elementwise sub/add-of-negation is flagged as a *candidate*. Fixable only
#       if operands carry sub-ulp bits (paired-fp16, regime B).
#   (c) group_norm / large-feature-map compile cliffs ((W%128)>64 squared, large
#       group_norm map): avoid/flag, not a rewrite target.
# Each flagged node gets an order-of-magnitude error proxy in [0,1]; the per-graph
# signal is the max over nodes. (b) is a candidate flag only; the tuner's vs-fp32
# gate confirms the real gain.

# fp16 has ~3-4 decimal digits; a clean fused op sits at ~1e-3 relerr (the corpus
# median). These are order-of-magnitude error proxies, deliberately coarse.
_FP16_CLEAN = 1e-3            # a clean fp16 op's relative error (corpus-calibrated)
_NARROW_SUM_FLOOR = 256      # contraction length above which a signed reduce_sum
                             # starts to lose digits to the narrow accumulator


def _is_signed_producer(t) -> bool:
    """True if `t` can produce signed values (so a sum over it can cancel). A
    square/abs/relu/sigmoid/exp/softplus output is non-negative -> a sum over it does
    NOT cancel; everything else is conservatively treated as possibly-signed."""
    return t.op not in ("square", "abs", "relu", "relu6", "sigmoid", "exp",
                        "softplus", "softmax")


def _reduce_len(t) -> int:
    """Number of elements summed per output element of a reduce node."""
    src = t.srcs[0] if t.srcs else None
    if src is None:
        return 1
    axes = t.attrs.get("axes", ())
    n = 1
    for ax in axes:
        if 0 <= ax < len(src.shape):
            n *= int(src.shape[ax])
    return n


def precision_risk(out, verbose: bool = False) -> dict:
    """Heuristic fp16-cancellation risk for the graph rooted at `out`.

    Returns a dict::

        {"graph_error": float,        # est. worst-case relerr proxy in [0,1]
         "nodes": [ {idx, op, kind, est_error, fixable, reason}, ... ],
         "hotspots": [idx, ...]}      # node indices flagged above the clean floor

    `kind` in {narrow_sum, cancel_sub, groupnorm_cliff}. `fixable` names the
    numerics-aware rewrite that addresses it ("reduce_sum->matmul", "paired-fp16",
    or "" for an avoid/flag-only cliff). This is a HEURISTIC, not a bound — see the
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
                # error grows ~ sqrt(K) * fp16_eps under the narrow accumulator;
                # cap at 1.0. This is an order-of-magnitude proxy, not a bound.
                est = min(1.0, _FP16_CLEAN * (K ** 0.5))
                nodes.append({"idx": i, "op": t.op, "kind": "narrow_sum",
                              "est_error": est, "fixable": "reduce_sum->matmul",
                              "reason": f"signed reduce_sum over K={K} (narrow fp16 accumulator)"})
            continue
        # (b) CFG-style subtract — candidate cancellation (data-dependent, can't
        #     confirm structurally). Flag sub of two non-trivial activations.
        if t.op == "sub" and len(t.srcs) == 2 and all(s.op != "input" or True for s in t.srcs):
            big = _elems(t.shape) >= 64    # a vector/tensor sub (not a scalar bias)
            both_live = all(s.op not in ("muls",) for s in t.srcs)
            if big and both_live:
                nodes.append({"idx": i, "op": t.op, "kind": "cancel_sub",
                              "est_error": _FP16_CLEAN,  # only RISKS blowing up; unknown w/o data
                              "fixable": "paired-fp16",
                              "reason": "subtract of two live tensors (CANDIDATE catastrophic "
                                        "cancellation — confirm with data; fix is upstream paired-fp16)"})
            continue
        # (c) group_norm at the per-axis wall. The rank-4 tiled lowering reduces over
        #     [1,G,C/groups,H*W], so the cliff is max(C/groups, H*W) > 65536 (aligned to
        #     af.group_norm's construction guard) — NOT the flattened (C/groups)*H*W,
        #     which the tiling now keeps under the cap (640@64, 512@128 run fine in fp16).
        if t.op == "group_norm" and len(t.shape) == 4:
            _, C, H, W = t.shape
            groups = int(t.attrs.get("groups", 1)) or 1
            if max(C // groups, H * W) > 65536:
                nodes.append({"idx": i, "op": t.op, "kind": "groupnorm_cliff",
                              "est_error": 0.0, "fixable": "",
                              "reason": f"group_norm tiled axis max(C/groups,H*W)>65536 at {H}x{W}: "
                                        "AVOID — exceeds the ANE per-axis bound"})
            continue

    # Default hotspots = only the RELIABLE, structurally-determinable signals: a narrow
    # reduce_sum whose estimated error exceeds the fp16-clean floor, and the group_norm
    # per-axis wall. cancel_sub is SPECULATIVE — a subtract of two live tensors is a
    # *candidate* for cancellation, but most (residuals, losses, differences) are benign
    # and unconfirmable without data. Flagging every such subtract trained users to ignore
    # the warning, so cancel_sub is now informational-only: it stays in `nodes` (surfaced
    # by precision_risk(verbose=True)) but does not raise the default warning.
    hotspots = [n["idx"] for n in nodes if n["est_error"] > _FP16_CLEAN
                or n["kind"] == "groupnorm_cliff"]
    graph_error = max([_FP16_CLEAN] + [n["est_error"] for n in nodes])
    if verbose:
        print(f"[precision] graph_error~{graph_error:.2e}, {len(nodes)} flagged node(s):")
        for n in nodes:
            print(f"  node {n['idx']:3d} {n['op']:12s} kind={n['kind']:16s} "
                  f"est~{n['est_error']:.2e} fix={n['fixable'] or '(avoid)'}: {n['reason']}")
    return {"graph_error": graph_error, "nodes": nodes, "hotspots": hotspots}
