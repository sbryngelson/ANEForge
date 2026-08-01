# `bench/` - measurement tooling

The scripts that produce every measured number in the characterization paper.
Each writes a committed JSON to [`results/`](results/). For the full
claim -> command -> result-file table and the environment requirements, see
[`docs/reproducibility.md`](../docs/reproducibility.md); to run everything end to
end use [`scripts/reproduce.sh`](../scripts/reproduce.sh).

All scripts ride on the public `aneforge` package (fp16 ANE dispatch) plus
[MLX](https://github.com/ml-explore/mlx) for the GPU baseline and numpy for the
CPU baseline (`pip install -e ".[bench]"`). The power-reading scripts call
`sudo powermetrics`; the latency-only ones do not.

| Script | Produces |
| --- | --- |
| `device_compare.py` | shared device-comparison harness (imported by the others) |
| `device_compare_wattcomplete.py` | single-stream device map (latency, relerr, per-rail watts) |
| `device_saturation_sweep.py` | compute ceilings + large-square-GEMM falloff |
| `device_bandwidth_roofline.py` | bandwidth ceilings + the two effective bandwidths |
| `device_serving_sweep.py` | batched-serving throughput + throughput/watt crossovers |
| `roofline_analysis.py` | per-device ridge points + archetype placement (reads the two sweep JSONs) |
| `gemv_bandwidth_sweep.py` | weight-stream effective bandwidth vs K |
| `fused_gpu_baseline.py` | fused-GPU comparison baseline |
| `compress_speedup_bench.py` | int4-LUT / sparse single-matmul speedup |
| `cross_path_compress_bench.py` | cross-path compressed-matmul latency |
| `model_int4_bench.py` | int4 whole-model bench |
| `encoder_serving_crosspath.py` | encoder cross-path serving |
| `decode_measurement.py` | end-to-end LLM decode sweep |
| `below_ridge_fusion.py` | fusion AI-lever demo: a memory-bound block crosses the weight ridge when fused |
| `decode_int8_accuracy.py` | int8 decode accuracy: token agreement, top-5 overlap, logit relerr, softmax KL |
| `real_models_fp16.py` | fp16-vs-fp16 GPU/ANE energy for ResNet-18, ViT-B/16, MiniLM |
| `numeric_cliffs.py` | per-silicon correctness rooflines (matmul saturation, slice-x16 crop-DMA, reduce exactness) |
| `roofline_suite.py` | one-shot suite: machine fingerprint + numeric cliffs; `--perf` = fast headline perf (a few min), `--perf-full` = the full battery above |
| `aggregate_rooflines.py` | merges `results/rooflines/*.json` into `results/ROOFLINES.md` (`--check` gates CI) |

## `results/rooflines/` - accumulated cross-machine submissions

Each contributor runs `roofline_suite.py` and PRs the fingerprinted JSON it drops
in [`results/rooflines/`](results/rooflines/); [`results/ROOFLINES.md`](results/ROOFLINES.md)
is the generated, GitHub-rendered table across every machine. See
[`results/rooflines/README.md`](results/rooflines/README.md) to submit yours.

## `mlperf/` - MLPerf-style measurement

[`mlperf/`](mlperf/) is a self-contained, LoadGen-shaped harness (SUT / QSL / SingleStream + Offline
scenarios) for running MLPerf Inference workloads on the ANE - the on-ramp to an official MLCommons
submission. First workload is ResNet-50. See [`mlperf/README.md`](mlperf/README.md).
