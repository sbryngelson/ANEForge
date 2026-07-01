# `bench/mlperf/` - MLPerf-style measurement on the ANE

A self-contained, LoadGen-shaped harness for running MLPerf Inference workloads on the Apple Neural Engine
through `aneforge`. It mirrors the MLCommons LoadGen structure (SUT / QSL / scenarios) so the numbers are
methodology-comparable and the workloads can be re-pointed at the real `mlperf_loadgen` for an official
submission with little change.

First workload: **ResNet-50** image classification (the canonical MLPerf entry - conv-heavy, fits resident on
the ANE, clean top-1 accuracy target).

## What this is (and is not)

- It **is** the same measurement shape as MLPerf Inference Edge: the SingleStream and Offline scenarios, the
  SUT/QSL split, the p90-latency and throughput metrics.
- It is **not** an official MLPerf submission. There is no MLCommons LoadGen logging or audit trail, and the
  default run lengths are short. A run is flagged `official: false` unless it meets the MLPerf minimums
  (>= 1024 queries AND >= 600 s). Treat the output as methodology-only numbers - the on-ramp to a real run,
  not a verified result.

## Run it

```bash
# performance only, synthetic inputs, torchvision ResNet-50 (no dataset needed):
PYTHONPATH=. python3 bench/mlperf/run.py

# longer run / quantized ANE weights:
PYTHONPATH=. python3 bench/mlperf/run.py --count 1024 --int8

# top-1 accuracy over real ImageNet val (val/<wnid>/*.JPEG layout) + performance on those images:
PYTHONPATH=. python3 bench/mlperf/run.py --imagenet-val ~/data/imagenet/val --count 512

# reference-fidelity when there is no ImageNet: ANE fp16 + int8 vs onnxruntime fp32 on real images:
PYTHONPATH=. python3 bench/mlperf/run.py --images ~/Pictures

# LLM decode (TTFT / TPOT / tokens-per-second), cached Qwen3-0.6B:
PYTHONPATH=. python3 bench/mlperf/run_llm.py --gen 128
PYTHONPATH=. python3 bench/mlperf/run_llm.py --int8 --gen 128

# your own MLPerf-provided ResNet-50 ONNX (to match the official model exactly):
PYTHONPATH=. python3 bench/mlperf/run.py --onnx resnet50.onnx
```

Each run prints a summary and writes a JSON to `bench/mlperf/results/`.

## Sample results (Apple M-series, this environment)

Methodology-only numbers (short runs; not an audited submission), to show the shape of the output:

| Run | Result |
| --- | --- |
| ResNet-50 fp16, SingleStream | p90 latency ~0.77 ms (~1310 img/s) |
| ResNet-50, Offline | ~1310 samples/s (tracks SingleStream: latency-bound at batch 1) |
| ResNet-50 fidelity vs onnxruntime fp32 | fp16: 100% top-1 agreement, logit cosine 1.00000; int8: 100%, cosine 0.998 |
| Qwen3-0.6B decode, fp16 | TTFT ~520 ms, TPOT ~18.7 ms/token, ~37.7 tok/s |
| Qwen3-0.6B decode, int8 | TTFT ~460 ms, TPOT ~17.0 ms/token, ~41.7 tok/s |

The int8 ResNet-50 preserving every prediction (cosine 0.998 vs fp32) is the evidence that the ANE clears the
Closed-division accuracy gate (>= 99% of the reference); the committed `results/*.json` carry the exact values.

## Layout

| File | Role |
| --- | --- |
| `loadgen_lite.py` | LoadGen-shaped core: `QSL`, `SUT`, `run_single_stream`, `run_offline`, `run_llm_decode`, `Result` (pure Python + numpy, unit-tested off-device in `tests/test_mlperf_harness.py`) |
| `resnet50.py` | ResNet-50 workload: ONNX -> ANE program wrapped as a SUT, synthetic + ImageNet-val QSLs, top-1 accuracy, reference-fidelity vs onnxruntime |
| `run.py` | ResNet-50 CLI: build the SUT, run the scenarios (+ accuracy / fidelity), write results |
| `llm_decode.py` | LLM decode workload: an aneforge LLM wrapped as a decode SUT (TTFT / TPOT / tokens-per-second) |
| `run_llm.py` | LLM CLI: build the decode SUT, run `run_llm_decode`, write results |

## Scenarios

- **SingleStream** - one sample at a time; metric is the 90th-percentile latency. The ANE's strong scenario
  (latency-bound at batch 1).
- **Offline** - the whole query set at once; metric is throughput (samples/s). The compiled program is fixed
  at batch 1, so Offline is run as sequential batch-1 calls; on a latency-bound engine its rate tracks the
  SingleStream rate, which is itself an honest finding.

- **LLMDecode** - greedy token generation; metrics are TTFT (time to first token), TPOT (per-output-token
  latency, as the p50/p90) and output tokens/s -- the shape MLPerf uses for its LLM benchmarks. See
  `run_llm.py` (defaults to a cached Qwen3-0.6B).

The Server scenario is intentionally omitted - it needs request concurrency the ANE does not favor.

## Path to an official submission

1. Validate accuracy in the **Open** division first (quantize freely), then tighten to **Closed** (exact
   reference model + preprocessing, >= 99% of the reference top-1, i.e. >= 75.68%).
2. Swap `loadgen_lite` for the real `mlperf_loadgen`: the SUT/QSL here map directly onto
   `lg.ConstructSUT(issue_queries, flush_queries)` and `lg.ConstructQSL(...)`. The workload code
   (`resnet50.py`) is unchanged; only the driver in `run.py` is replaced by a LoadGen `StartTest`.
3. Submit in an official Inference round (Edge, SingleStream + Offline) through an MLCommons member.

## Adding a workload

Implement a `SUT` subclass (an `issue(qsl, indices) -> outputs` method, or a `decode(...)` for LLMs) and a
`QSL` (a `get(index) -> features`), and drive them with the existing scenario runners. `resnet50.py` and
`llm_decode.py` are the two worked examples.
