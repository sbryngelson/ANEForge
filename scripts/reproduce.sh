#!/usr/bin/env bash
# =============================================================================
# aneforge — paper artifact reproduction driver
# =============================================================================
# Maps each major paper CLAIM to the ONE command that regenerates it. Every
# measurement script lives in bench/ and writes a committed JSON to
# bench/results/; see docs/reproducibility.md for the full claim->command table.
#
# HARDWARE / OS REQUIREMENTS (running, not installing):
#   - Apple Silicon Mac (verified: M5 Pro). The ANE numbers are host- and
#     thermal-dependent; they will differ on other chips/states.
#   - macOS 14+ (verified macOS 26.5), Xcode command-line tools.
#   - The e5rt dispatch dylib must be built first (see README "Install"):
#       sh aneforge/_lib/build.sh   ->   aneforge/_lib/libane_e5rt_dispatch.dylib
#   - Python 3.10+ (verified 3.14), `pip install -e .` (core dep: numpy).
#   - GPU-comparison + power steps need MLX (`pip install -e ".[bench]"`).
#   - POWER steps run `sudo powermetrics` and WILL prompt for your password.
#     powermetrics reports an *estimated* per-rail power, not a wall meter.
#
# This script is RE-RUNNABLE and FAIL-SOFT: it does not `set -e`. If a single
# device/power step fails (no sudo, thermal throttle, missing MLX), it echoes
# the failure and continues so the deterministic claims still reproduce.
#
# Usage:   bash scripts/reproduce.sh
# Run from the repo root.
# =============================================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# Mode: `gates` runs only the deterministic, no-sudo correctness gates and returns a
# real exit code (for CI); the default runs the full paper-artifact reproduction.
MODE="${1:-all}"

# Most tools want the repo root importable (aneforge tolerates the duplicate
# OpenMP runtime itself, on import).
export PYTHONPATH="$REPO"

PASS=0; FAIL=0
section() { printf '\n\033[1m========== %s ==========\033[0m\n' "$1"; }
claim()   { printf '\n\033[1m-- CLAIM: %s\033[0m\n   $ %s\n' "$1" "$2"; }
run() {
  # run "<claim>" "<shell command>"
  claim "$1" "$2"
  if bash -c "$2"; then
    PASS=$((PASS+1)); printf '\033[32m   [ok]\033[0m\n'
  else
    FAIL=$((FAIL+1)); printf '\033[31m   [FAILED — continuing]\033[0m\n'
  fi
}

# Preflight: dylib must exist or every ANE step fails for the same reason.
if [ ! -f "aneforge/_lib/libane_e5rt_dispatch.dylib" ]; then
  printf '\033[33mWARNING: aneforge/_lib/libane_e5rt_dispatch.dylib not found.\n'
  printf 'Build it first (`sh aneforge/_lib/build.sh`); ANE steps below will fail.\033[0m\n'
fi

# -----------------------------------------------------------------------------
section "§ Deterministic gates (no sudo, no GPU)"
# CLAIM: the dispatch-route registry is closed and matches docs/capabilities.json.
run "Route / capability gate" \
  "python3 tests/test_routes.py"
# CLAIM: the correctness corpus (the optimizer gate) is GREEN.
run "Correctness corpus" \
  "python3 tests/run_corpus.py"

# `gates` mode (CI): the deterministic correctness gates above are the whole job.
# Everything below measures power/throughput, needs `sudo powermetrics` + MLX, and is
# host/thermal-dependent — that runs on the bench machine, not in CI. Return a real exit
# code so a failed gate fails the build.
if [ "$MODE" = gates ]; then
  section "Summary (gates only)"
  printf 'Gates: %d passed, %d failed.\n' "$PASS" "$FAIL"
  [ "$FAIL" -eq 0 ]; exit
fi

# -----------------------------------------------------------------------------
section "§ Single-stream device map — ANE vs GPU vs CPU [needs sudo + MLX]"
# CLAIM (Table 'device map', Fig 1): latency, fp16 relerr, idle-subtracted
#        per-rail watts, perf/watt across the workload classes.
run "Device map (wattcomplete; powermetrics per-rail)" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/device_compare_wattcomplete.py --window 6"

# -----------------------------------------------------------------------------
section "§ Compute / bandwidth / serving sweeps [needs sudo + MLX]"
# CLAIM (Fig 2, compute peaks): saturation ceilings + large-square-GEMM falloff.
run "Saturation sweep" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/device_saturation_sweep.py"
# CLAIM (Fig 3, bandwidth ceiling): the two effective bandwidths + roofline.
run "Bandwidth roofline sweep" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/device_bandwidth_roofline.py --window 6"
# CLAIM (Fig 5, crossover): batched-serving throughput + throughput/watt.
run "Serving sweep (batched multi-stream)" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/device_serving_sweep.py --window 5"

# -----------------------------------------------------------------------------
section "§ Roofline synthesis (reads the saturation + bandwidth JSONs above)"
# CLAIM (Fig 4, roofline tables): per-device ridge points + archetype placement.
run "Roofline analysis" \
  "python3 bench/roofline_analysis.py"

# -----------------------------------------------------------------------------
section "§ Appendix claims: fused-GPU baseline, weight stream, decode [needs sudo]"
run "Fused-GPU baseline" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/fused_gpu_baseline.py"
run "Weight-stream (GEMV-K) sweep" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/gemv_bandwidth_sweep.py"
run "End-to-end LLM decode sweep" \
  "sudo -E env PYTHONPATH=$REPO python3 bench/decode_measurement.py"

# -----------------------------------------------------------------------------
section "§ Compressed-weight streaming (latency only, no sudo) [needs MLX]"
run "int4-LUT / sparse single-matmul speedup" \
  "python3 bench/compress_speedup_bench.py"
run "Cross-path compressed-matmul latency" \
  "python3 bench/cross_path_compress_bench.py"
run "int4 whole-model bench" \
  "python3 bench/model_int4_bench.py"
run "Encoder cross-path serving" \
  "python3 bench/encoder_serving_crosspath.py"

# -----------------------------------------------------------------------------
section "Summary"
printf 'Steps passed: %d   failed: %d\n' "$PASS" "$FAIL"
printf 'Committed result JSONs live in bench/results/*.json.\n'
printf 'See docs/reproducibility.md for the full claim->command table + caveats.\n'
