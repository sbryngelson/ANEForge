# MLPerf on the Apple Neural Engine

The MLPerf Inference reference ResNet-50 runs on the ANE through aneforge -- no CoreML, no CPU/GPU fallback --
and passes the upstream MLCommons `submission_checker` (v5.1): all three edge scenarios VALID, at reference
accuracy. Believed to be the first MLPerf-shaped ResNet-50 result on the ANE. It is **unofficial**:
self-measured, no MLCommons audit trail.

## Reproduce in one command

```bash
./bench/mlperf/repro.sh
```

No dataset, ~1-2 min on any Apple Silicon Mac. Compiles the reference ResNet-50 onto the ANE; prints
SingleStream p90 latency and fp16/int8 vs onnxruntime fp32 fidelity. Uses the real MLCommons LoadGen if
installed, else the lite harness. Full 50k accuracy plus checker:

```bash
./bench/mlperf/repro.sh --full --imagenet-val ~/Models/mlperf/val --val-map val_map.txt
```

## On the ANE, not near it

aneforge compiles the ONNX graph to one fused e5rt program and runs it under the ANE device mask. There is no
CPU/GPU fallback: an unsupported op fails the compile. SingleStream p90 is 0.773 ms, 70x the same graph on CPU.

## Accuracy

The reference model (TensorFlow export, 1001-class, ArgMax-terminated, mean-subtracted inputs) lost ~10 points
on the ANE naively (fp16 67% vs fp32 77%). Not accumulation -- the ANE reduces in a wide fp32-class
accumulator, and the convs are bit-exact. The loss was the one standalone stem `BatchNormalization`, run in the
fp16 datapath on large values. Folding `Conv -> BatchNorm` into the conv (exact) removes it, and the ONNX
importer now does this automatically. Full 50k ILSVRC-2012 val, MLPerf preprocessing:

| Path | top-1 |
| --- | --- |
| onnxruntime fp32 (reference) | 76.45% |
| ANE fp16 | 76.44% (cosine 1.00000 vs fp32) |
| ANE int8 | 76.37% |

MLCommons' reference is 76.46%. ANE fp16 equals fp32 and clears the Closed gate (>= 75.70%). The fold is a
general importer feature for TF exports with a standalone Conv->BN.

## Where it lands

`bench/mlperf/compare/compare.py` places these next to official MLPerf results, each sourced to a
`mlcommons/inference_results_*` path.

| System | Category | Round | SingleStream p90 (ms) | Offline (samples/s) | Power class |
| --- | --- | --- | --- | --- | --- |
| Apple Neural Engine (M-series), unofficial | Edge | -- | 0.773 | 1,149 | single-digit W (SoC block) |
| NVIDIA Jetson AGX Orin | Edge | v3.1 | 0.640 | 6,424 | 15-60 W module |
| NVIDIA H100-SXM-80GB (1 GPU) | Datacenter | v4.0 | -- | 88,714 | 350-700 W board |

Within ~1.2x of the Jetson AGX Orin on SingleStream latency, at a fraction of the power. It trails on Offline
throughput: the program is batch-1 and compute-bound.

## Roofline

ResNet-50 at 224x224 is ~8.2 GFLOP/image; at 0.82 ms that is ~10 TFLOP/s. The measured M5 Pro ANE roofs are
18.8 TFLOP/s (conv) and 10.2 TFLOP/s (GEMM), so ResNet-50 sits at the GEMM roof. It is compute-bound, and three
levers do not move it:

- **Batching**: fitting `t = D + N*C` over batch 1..64 gives `C ~ 0.855 ms/sample` and `D ~ 0` -- no fixed cost
  to amortize.
- **int8/int4**: ~9% then flat. Near the plateau the weight stream is element-rate-bound, not byte-bound, so
  compression buys energy, not throughput; the int8 flag quantizes weights but leaves the MAC in fp16.
- **Latency-bound batching** (verify K tokens for the cost of one) applies only to dispatch-bound decode;
  ResNet-50 has no idle compute.

The gap to 18.8 is structural: that roof is a Winograd effect for dense 3x3 stride-1 convs, and ~half of
ResNet-50's FLOPs are Winograd-ineligible 1x1 (and strided) convs. Reaching it needs a different network. This
is the ceiling for ResNet-50 on the engine.

## Official vs not

Produced with the real MLCommons LoadGen; passes the upstream `submission_checker` (v5.1: "Closed Results = 3,
Systems = 1, SUMMARY: submission looks OK"), TEST01 + TEST04 PASS, accuracy logs truncated and hashed. No
MLCommons audit trail, not submitted in an official round -- a membership and process step.

## Full submission

```bash
PYTHONPATH=. python3 bench/mlperf/run_submission.py \
    --imagenet-val ~/Models/mlperf/val --val-map val_map.txt \
    --count 1024 --acc-count 0 --min-duration 600 --compliance-duration 600
```

Builds the complete edge/Closed tree (all three scenarios, real-LoadGen performance and accuracy, TEST01 +
TEST04, system description) under `bench/mlperf/submission/`; validate with the upstream checker from a clone of
`mlcommons/inference`. Harness: [`bench/mlperf/README.md`](https://github.com/sbryngelson/ANEForge/tree/main/bench/mlperf).
