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

# your own MLPerf-provided ResNet-50 ONNX (to match the official model exactly):
PYTHONPATH=. python3 bench/mlperf/run.py --onnx resnet50.onnx
```

Each run prints a per-scenario summary and writes a JSON to `bench/mlperf/results/`.

## Layout

| File | Role |
| --- | --- |
| `loadgen_lite.py` | LoadGen-shaped core: `QSL`, `SUT`, `run_single_stream`, `run_offline`, `Result` (pure Python + numpy, unit-tested off-device in `tests/test_mlperf_harness.py`) |
| `resnet50.py` | ResNet-50 workload: ONNX -> ANE program wrapped as a SUT, synthetic + ImageNet-val QSLs, top-1 accuracy |
| `run.py` | CLI: build the SUT, run the scenarios (+ optional accuracy), write results |

## Scenarios

- **SingleStream** - one sample at a time; metric is the 90th-percentile latency. The ANE's strong scenario
  (latency-bound at batch 1).
- **Offline** - the whole query set at once; metric is throughput (samples/s). The compiled program is fixed
  at batch 1, so Offline is run as sequential batch-1 calls; on a latency-bound engine its rate tracks the
  SingleStream rate, which is itself an honest finding.

The Server scenario is intentionally omitted - it needs request concurrency the ANE does not favor.

## Path to an official submission

1. Validate accuracy in the **Open** division first (quantize freely), then tighten to **Closed** (exact
   reference model + preprocessing, >= 99% of the reference top-1, i.e. >= 75.68%).
2. Swap `loadgen_lite` for the real `mlperf_loadgen`: the SUT/QSL here map directly onto
   `lg.ConstructSUT(issue_queries, flush_queries)` and `lg.ConstructQSL(...)`. The workload code
   (`resnet50.py`) is unchanged; only the driver in `run.py` is replaced by a LoadGen `StartTest`.
3. Submit in an official Inference round (Edge, SingleStream + Offline) through an MLCommons member.

## Adding a workload

Implement a `SUT` subclass (an `issue(qsl, indices) -> outputs` method) and a `QSL` (a `get(index) ->
features`), and drive them with the existing scenario runners. An LLM decode workload (tok/s) is the intended
next addition.
