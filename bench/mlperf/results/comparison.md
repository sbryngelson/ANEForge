### MLPerf ResNet-50: the ANE next to published competitors

| System | Category | MLPerf round | SingleStream p90 (ms) | Offline (samples/s) | Power class |
| --- | --- | --- | --- | --- | --- |
| Apple Neural Engine (Apple M-series) *(unofficial -- self-measured)* | Edge | unofficial | 0.773 | 1,149 | single-digit W (SoC block) |
| NVIDIA Jetson AGX Orin | Edge | v3.1 | 0.640 | 6,424 | 15-60 W module |
| NVIDIA H100-SXM-80GB (1 GPU) | Datacenter | v4.0 | -- | 88,714 | 350-700 W board |

**How it lands (numbers only):**
- vs NVIDIA Jetson AGX Orin (v3.1): SingleStream 0.773 ms vs 0.640 ms (1.21x its latency); Offline 1,149 vs 6,424 samples/s (0.18x its throughput).
- vs NVIDIA H100-SXM-80GB (1 GPU) (v4.0, datacenter): Offline 1,149 vs 88,714 samples/s (77x the ANE) -- a different power and cost class.

**Reading this table**
- The ANE row is **unofficial** -- self-measured with `bench/mlperf` under the real MLCommons LoadGen (Result is: VALID), with no MLCommons audit trail. Competitor rows are **official** MLPerf Inference results from the public `mlcommons/inference_results_*` repos (source path per row in `competitors.csv`).
- **SingleStream** is the ANE's strong scenario (latency-bound at batch 1); **Offline** rewards batching, which the batch-1 ANE program does not do -- so it trails there, honestly.
- **Power class** is each part's device power ENVELOPE (SoC block / module TDP / board TDP), NOT a MLPerf Power measurement -- do not read it as a certified perf/watt number.
- **Rounds differ**: ResNet-50 *edge* submissions from the big vendors thinned out after ~v3.1 (NVIDIA moved Jetson to LLM / Stable Diffusion), so the most recent official ResNet-50 Orin result is v3.1. Numbers are compared across rounds for that reason; ResNet-50 itself is unchanged.

Sources: mlcommons/inference_results_v3.1 closed/NVIDIA/results/Orin_TRT/resnet50 (SingleStream + Offline); mlcommons/inference_results_v4.0 closed/NVIDIA/results/DGX-H100_H100-SXM-80GBx1_TRT/resnet50/Offline; self-measured via bench/mlperf under real MLCommons LoadGen (VALID); results/resnet50_loadgen_singlestream.json
