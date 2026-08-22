# Reproducibility

This document maps every measured claim in the characterization paper to the one
command that regenerates it and the committed result file it produces. A single
fail-soft driver, [`scripts/reproduce.sh`](https://github.com/sbryngelson/ANEForge/blob/main/scripts/reproduce.sh), runs the
whole set end to end; each row below is also a standalone claim.

## Environment

- Hardware: an Apple Silicon Mac. The absolute numbers were measured on an
  M5 Pro (ANE architecture `h17s`) and are host- and thermal-dependent; the
  portable claim is the qualitative device map, not the multipliers. The
  single-stream map also reproduces, verdict for verdict, on an M1 (A13)
  and an M2 (A14).
- OS / toolchain: macOS 14+ (verified macOS 26.5, ANECompiler 3520.4.1),
  Xcode command-line tools.
- Install:
  ```sh
  pip install -e .            # core (numpy only)
  pip install -e ".[bench]"   # adds MLX for the GPU comparison + power tooling
  sh aneforge/_lib/build.sh   # build the e5rt dispatch dylib (not tracked)
  ```
- Power steps run `sudo powermetrics`, which reports an estimated per-rail
  power (not a wall meter) and prompts for your password. The deterministic
  gates (routes, corpus) need no sudo.

All measurement scripts live in [`bench/`](https://github.com/sbryngelson/ANEForge/tree/main/bench) and write a committed JSON
to [`bench/results/`](https://github.com/sbryngelson/ANEForge/tree/main/bench/results). A fresh run overwrites the plain
`<script>_results.json` filename with your host's numbers; the host-suffixed
snapshots (`device_compare_wattcomplete_results_M{1,2,5}.json`) are the paper's
committed runs and are never overwritten. As committed, the plain
`device_compare_wattcomplete_results.json` is byte-identical to the `_M5`
snapshot (the paper's single-stream run, idle package 1196 mW).

Two derived metrics are computed inside a single sustained power window
(see the paper's methodology section): throughput = iterations completed during
the window / wall time of that loop, and perf/W divides it by the median
idle-subtracted package power of the same window. The `latency` columns come from
a separate minimum-latency probe that isolates each device's dispatch floor; at
sub-millisecond shapes the sustained per-iteration time exceeds the minimum
latency, so perf/W cannot be reconstructed from the latency column.

## Claim -> command -> result

| Paper element | Command | Result file |
| --- | --- | --- |
| Single-stream device map (table + Fig. 1) | `python3 bench/device_compare_wattcomplete.py --window 6` | `bench/results/device_compare_wattcomplete_results.json` |
| Three-generation reproduction (M1 / M2 / M5) | same script, run on each host | `bench/results/device_compare_wattcomplete_results_M{1,2,5}.json` |
| Compute peaks + GEMM falloff (Fig. 2) | `python3 bench/device_saturation_sweep.py` | `bench/results/device_saturation_sweep_results.json` |
| Bandwidth ceilings, two effective BWs (Fig. 3) | `python3 bench/device_bandwidth_roofline.py --window 6` | `bench/results/device_bandwidth_roofline_results.json` |
| Batched-serving crossovers (Fig. 5) | `python3 bench/device_serving_sweep.py --window 5` | `bench/results/device_serving_sweep_results.json` |
| Per-device roofline synthesis (Fig. 4, ridge/placement tables) | `python3 bench/roofline_analysis.py` | `bench/results/roofline_analysis_results.json` |
| Weight-stream effective bandwidth (App. A) | `python3 bench/gemv_bandwidth_sweep.py` | `bench/results/gemv_bandwidth_sweep_results.json` |
| Fused-GPU baseline (App. A) | `python3 bench/fused_gpu_baseline.py` | `bench/results/fused_gpu_baseline_results.json` |
| Compressed-weight single-matmul speedup (App. A) | `python3 bench/compress_speedup_bench.py` | `bench/results/compress_speedup_bench.json` |
| Cross-path compressed-matmul latency (App. A) | `python3 bench/cross_path_compress_bench.py` | `bench/results/cross_path_compress_bench.json` |
| int4 whole-model bench | `python3 bench/model_int4_bench.py` | `bench/results/model_int4_bench.json` |
| Encoder cross-path serving | `python3 bench/encoder_serving_crosspath.py` | `bench/results/encoder_serving_crosspath.json` |
| End-to-end LLM decode sweep | `python3 bench/decode_measurement.py` | `bench/results/decode_measurement_results.json` |
| Per-silicon numeric-cliff rooflines (matmul / slice / reduce) | `PYTHONPATH=. python3 bench/numeric_cliffs.py` | `bench/results/numeric_cliffs_results.json` |
| Cross-machine roofline suite (fingerprint + cliffs; `--perf` fast headline, `--perf-full` full battery) | `PYTHONPATH=. python3 bench/roofline_suite.py` | `bench/results/rooflines/roofline-*.json` -> `bench/results/ROOFLINES.md` |
| Route / capability gate (deterministic) | `python3 tests/test_routes.py` | `docs/capabilities.json` |
| Correctness corpus (deterministic) | `python3 tests/run_corpus.py` | - (pass/fail gate) |

`roofline_analysis.py` reads the saturation and bandwidth result JSONs rather
than re-measuring, so run those two first for a fresh synthesis.

## Supporting experiments

These back specific claims in the text and appendix but do not produce a numbered
figure or table. Each writes a committed JSON to `bench/results/`.

| Claim | Command | Result file |
| --- | --- | --- |
| Fusion AI-lever (a memory-bound block crosses the weight ridge when fused) | `python3 bench/below_ridge_fusion.py` | `bench/results/below_ridge_fusion.json` |
| int8 decode stays within tolerance (token agreement, top-5 overlap, logit relerr, softmax KL) | `python3 bench/decode_int8_accuracy.py` | `bench/results/decode_int8_accuracy.json` |
| fp16-vs-fp16 GPU/ANE energy for the real models (ResNet-18, ViT-B/16, MiniLM) | `python3 bench/real_models_fp16.py --window 6` | `bench/results/real_models_fp16_results.json` |

## Notes and caveats

- `powermetrics` is a modeled SoC estimator. The headline is idle-subtracted
  total-package active power, cross-validated against Apple's IOReport "Energy
  Model" counters (App. A); relative device comparisons are robust to a shared
  model bias since it cancels in ratios.
- The CPU is measured in fp32 (Accelerate/AMX has no fast fp16 GEMM) against
  fp16 ANE/GPU; this asymmetry is labeled wherever it matters and carried
  through the roofline arithmetic.
- Sixteen subprocess-bridge operators are excluded from the speed race as a
  dispatch artifact rather than silicon time; see the methodology section.
