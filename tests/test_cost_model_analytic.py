"""Measurement-free analytic per-chip cost model in aneforge/_cost.py."""
import numpy as np

import aneforge as af
from aneforge import _cost


# --- project_peak: the generational-scaling table, anchored to the M1 measurement ------
def test_project_peak_m1_is_the_measured_anchor():
  p = _cost.project_peak("h13")
  assert abs(p["tflops"] - 1.8) < 0.15        # measured M1 fp16 peak
  assert abs(p["rel_m1"] - 1.0) < 1e-6


def test_project_peak_m5_about_5x():
  p = _cost.project_peak("h17s")              # M5 == H17s
  assert 4.5 < p["rel_m1"] < 6.2
  assert 8.0 < p["tflops"] < 11.5


def test_project_peak_ultra_about_22x():
  p = _cost.project_peak("h17d")              # 64-core Ultra
  assert 19.0 < p["rel_m1"] < 25.0            # doc: ~22x


def test_project_peak_m11_about_tenth():
  p = _cost.project_peak("m11")               # 1-core, 500 MHz
  assert 0.05 < p["rel_m1"] < 0.18            # doc: ~0.1x


def test_project_peak_unknown_arch_falls_back_to_family():
  # h18 has no own cost curve; folds to the A16 tier
  p = _cost.project_peak("h18")
  assert p["tflops"] > 1.8                    # an A16-tier chip beats M1


# --- the latency roofline: reproduce the doc's +/-17% M1 conv validation ---------------
# (workload, kH, kW, Cin, Cout, spatial, measured_ms)
_M1_CONVS = [
  ("3x3 C256@28",  3, 3, 256,  256, 28, 0.507),
  ("1x1 C512@32",  1, 1, 512,  512, 32, 0.444),
  ("3x3 C512@14",  3, 3, 512,  512, 14, 0.832),
  ("1x1 C1024@16", 1, 1, 1024, 1024, 16, 0.686),
  ("1x1 C2048@8",  1, 1, 2048, 2048, 8,  1.047),
]


def _conv_graph(kH, kW, Cin, Cout, S):
  x = af.input((1, Cin, S, S))
  w = np.zeros((Cout, Cin, kH, kW), np.float32)
  return af.conv(x, w, pad=kH // 2)           # 'same' padding keeps output SxS


def test_m1_conv_latency_within_17pct():
  worst = 0.0
  for name, kH, kW, Cin, Cout, S, meas_ms in _M1_CONVS:
    g = _conv_graph(kH, kW, Cin, Cout, S)
    pred_ms = _cost.estimate(g, target="h13") / 1000.0     # us -> ms
    rel = abs(pred_ms - meas_ms) / meas_ms
    worst = max(worst, rel)
    assert rel <= 0.17 + 1e-6, f"{name}: pred {pred_ms:.3f} vs meas {meas_ms:.3f} ({rel:.0%})"
  assert worst <= 0.17 + 1e-6


def test_m1_classifies_deep_narrow_as_slower():
  # memory-bound 1x1 C2048@8 slower than compute-bound 3x3 C256@28
  deep_narrow = _cost.estimate(_conv_graph(1, 1, 2048, 2048, 8), target="h13")
  compute_bound = _cost.estimate(_conv_graph(3, 3, 256, 256, 28), target="h13")
  assert deep_narrow > compute_bound


# --- the M5 loop closure: per-chip measured {BW, floor, peak} anchors ------------------
_M5_CONVS = [
  # (workload, kH, kW, Cin, Cout, spatial, M5 measured ms)
  ("3x3 C256@28",  3, 3, 256,  256,  28, 0.216),
  ("1x1 C512@32",  1, 1, 512,  512,  32, 0.178),
  ("3x3 C512@14",  3, 3, 512,  512,  14, 0.192),
  ("1x1 C1024@16", 1, 1, 1024, 1024, 16, 0.248),
  ("1x1 C2048@8",  1, 1, 2048, 2048, 8,  0.235),
]


def test_m5_anchor_constants():
  c = _cost._analytic_constants("h17s")
  assert abs(c["bw_bytes_per_us"] - 57.0e3) < 1.0     # 57 GB/s measured, NOT clock-scaled 14.9
  assert abs(c["floor_us"] - 110.0) < 1e-6            # measured dispatch floor
  assert abs(c["flops_per_us"] - 8.9e6) < 1e3         # = project_peak('h17s') silicon-validated


def test_m1_anchor_unchanged():
  # h13 path must stay the exact M1 fit (the +/-17% validation depends on it)
  c = _cost._analytic_constants("h13")
  assert abs(c["bw_bytes_per_us"] - 9.0e3) < 1e-6
  assert abs(c["floor_us"] - 220.0) < 1e-6
  assert abs(c["flops_per_us"] - 3.25e6) < 1e-6


def test_m5_conv_latency_loop_closed():
  QUOTED = {"3x3 C256@28", "1x1 C512@32", "1x1 C2048@8"}      # writeup: 214/170/265 us
  errs = {}
  for name, kH, kW, Cin, Cout, S, meas_ms in _M5_CONVS:
    pred_ms = _cost.estimate(_conv_graph(kH, kW, Cin, Cout, S), target="h17s") / 1000.0
    errs[name] = abs(pred_ms - meas_ms) / meas_ms
  for name in QUOTED:
    assert errs[name] <= 0.15, f"{name}: {errs[name]:.0%}"
  assert max(errs.values()) <= 0.35, errs
  assert sum(errs.values()) / len(errs) <= 0.20, errs


def test_unmeasured_chip_bw_core_scaled():
  # h17d (64 cores) takes the h17s (16-core) anchor scaled by core ratio, not clock
  c5, cd = _cost._analytic_constants("h17s"), _cost._analytic_constants("h17d")
  assert abs(cd["bw_bytes_per_us"] / c5["bw_bytes_per_us"] - 4.0) < 0.01     # 64/16
  # h14 (M2 Pro / A14) is its own silicon-measured anchor (~48 GB/s)
  c1, c4 = _cost._analytic_constants("h13"), _cost._analytic_constants("h14")
  assert abs(c4["bw_bytes_per_us"] - 48.0e3) < 1.0           # 48 GB/s = 48e3 B/us
  assert c4["bw_bytes_per_us"] > 5 * c1["bw_bytes_per_us"]   # ~48 vs ~9 GB/s


# --- cross-chip monotonicity: a bigger engine is never slower on a compute-bound graph -
def test_bigger_chip_not_slower():
  g = _conv_graph(3, 3, 256, 256, 28)
  t_m1 = _cost.estimate(g, target="h13")
  t_m5 = _cost.estimate(g, target="h17s")
  t_ultra = _cost.estimate(g, target="h17d")
  assert t_m5 < t_m1                          # M5 faster than M1
  assert t_ultra <= t_m5                       # Ultra at least as fast


# --- backward compat: the default (no target) path is the M5-measured heuristic --------
def test_default_path_unchanged_without_target():
  g = _conv_graph(3, 3, 256, 256, 28)
  default = _cost.estimate(g)
  assert default > 0.0
  # target=None identical to omitting it
  assert _cost.estimate(g, target=None) == default


def test_h14_midband_compute_ramp():
  # h14 anchor's util ramp lifts the mid-band toward measured (768^3 GEMM measured 596us)
  W = np.zeros((768, 768), np.float32)
  gemm = af.input((768, 768)) @ W            # 768^3 GEMM, the worst mid-band point
  e_mid = _cost.estimate(gemm, target="h14")
  assert 350.0 < e_mid < 596.0               # ramped up from the ~224us no-ramp estimate
  # ramp is h14-only: h13/h17s constants carry no util_k
  assert "util_k" not in _cost._analytic_constants("h13")
  assert "util_k" not in _cost._analytic_constants("h17s")
  assert "util_k" in _cost._analytic_constants("h14g")


# --- estimate provenance: is a target's estimate silicon-anchored or extrapolated? ------
def test_estimate_provenance_marks_silicon_measured_families():
  # measured anchors: A13/h13 (M1), A14/h14 (M2 Pro), A16/h17s (M5); h16/h17* fold to A16 tier
  for arch in ("h13", "h14", "h17s", "h16", "h17d"):
    p = af.estimate_provenance(arch)
    assert p["measured"] is True
    assert p["basis"] == "silicon"


def test_estimate_provenance_marks_extrapolated_targets():
  # A15 has no M3 silicon -> extrapolated from h14; h11 extrapolates from h13
  p15 = af.estimate_provenance("h15")
  assert p15["measured"] is False
  assert p15["anchor"] == "h14"
  assert p15["basis"] == "extrapolated-from-h14"
  assert af.estimate_provenance("h11")["anchor"] == "h13"


def test_estimate_provenance_rejects_unknown_arch():
  import pytest
  with pytest.raises(ValueError, match="unknown"):
    af.estimate_provenance("zzz")
