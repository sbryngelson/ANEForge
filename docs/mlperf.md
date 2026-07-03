# MLPerf on the Apple Neural Engine

The MLPerf Inference reference ResNet-50 runs on the Apple Neural Engine through aneforge -- no CoreML, no
CPU/GPU fallback -- and passes the upstream MLCommons `submission_checker` (v5.1) with all three edge scenarios
VALID, at the reference accuracy. As far as we can tell this is the first MLPerf-shaped ResNet-50 result on the
ANE. It is an **unofficial** result (self-measured, no MLCommons audit trail); the numbers and the exact
reproduction are below.

## Reproduce it in one command

```bash
./bench/mlperf/repro.sh
```

No dataset, no arguments, ~1-2 minutes on any Apple Silicon Mac. It compiles the MLPerf reference ResNet-50
onto the ANE and prints the SingleStream p90 latency plus the fp16/int8 vs onnxruntime fp32 fidelity. It uses
the real MLCommons LoadGen if `mlcommons-loadgen` is installed, otherwise the bundled lite harness. For the
full 50,000-image accuracy run plus the upstream checker:

```bash
./bench/mlperf/repro.sh --full --imagenet-val ~/Models/mlperf/val --val-map val_map.txt
```

## What "on the ANE" means here

Every layer dispatches to the Neural Engine. aneforge compiles the ONNX graph to a single fused e5rt program
and runs it with the ANE device mask; there is no silent fall-back to the CPU or GPU -- an unsupported op fails
the compile instead of migrating. The measured latency (0.773 ms SingleStream p90) is 70x faster than the same
graph on the CPU, and the dispatch carries the ANE device mask, which is how we know where it ran.

## Accuracy: matching the reference

The MLPerf reference ResNet-50 is a TensorFlow export (1001-class, background at index 0, ArgMax-terminated)
with raw-scale mean-subtracted preprocessing. Naively it lost ~10 points on the ANE (fp16 67% vs fp32 77%).
That was not an accumulation limit -- the ANE reduces in a wide fp32-class accumulator, and the reference's
convs are bit-exact on the engine. The loss localized to the model's one standalone `BatchNormalization` (the
stem), which runs in the fp16 datapath on large pre-BN values and rounds; that rounding then amplifies through
the network.

The fix is to fold `Conv -> BatchNorm` into the convolution (exact, per-output-channel scale and bias), so the
stem computes as a single wide-accumulator conv with no standalone fp16 BN. aneforge's ONNX importer now does
this automatically. On the exact reference model with MLPerf preprocessing, over the full 50,000-image
ILSVRC-2012 validation set:

| Path | top-1 |
| --- | --- |
| onnxruntime fp32 (reference) | 76.45% |
| ANE fp16 | 76.44% (cosine 1.00000 vs fp32) |
| ANE int8 | 76.37% |

MLCommons' published reference is 76.46%. The ANE fp16 result equals fp32 to 0.01 point and clears the Closed
gate (>= 75.70%). The fold is a general importer feature: any TF-exported model with a standalone Conv->BN
benefits.

## Where it lands, vs published MLPerf

`bench/mlperf/compare/compare.py` places these numbers next to official MLPerf Inference results. Every
competitor number is sourced to a `mlcommons/inference_results_*` path in `compare/competitors.csv`.

| System | Category | MLPerf round | SingleStream p90 (ms) | Offline (samples/s) | Power class |
| --- | --- | --- | --- | --- | --- |
| Apple Neural Engine (M-series), unofficial | Edge | -- | 0.773 | 1,149 | single-digit W (SoC block) |
| NVIDIA Jetson AGX Orin | Edge | v3.1 | 0.640 | 6,424 | 15-60 W module |
| NVIDIA H100-SXM-80GB (1 GPU) | Datacenter | v4.0 | -- | 88,714 | 350-700 W board |

On SingleStream latency the ANE (0.773 ms) is within ~1.2x of the Jetson AGX Orin (0.640 ms) at a fraction of
the power. On Offline throughput it trails, because the ANE program is batch-1 and latency-bound. That is not a
tuning miss -- it is the shape of the engine, and the roofline says so.

## How close to the roofline (why it is not faster)

ResNet-50 at 224x224 is ~8.2 GFLOP/image; at 0.82 ms that is ~10 TFLOP/s achieved. Against our measured M5 Pro
ANE ceilings (from a separate ANE roofline characterization) the conv roof is 18.8 TFLOP/s and the GEMM roof is
10.2 TFLOP/s -- so ResNet-50 sits essentially *at* the GEMM roof. It is compute-bound, and three obvious levers
do not move it:

- **Batching** does nothing. Fitting `t = D + N*C` across batch 1..64 gives a per-sample cost `C ~ 0.855 ms`
  and a dispatch floor `D ~ 0`: there is no fixed cost to amortize.
- **int8/int4** give ~9% then flat. Near the compute plateau the ANE weight stream is element-rate-bound, not
  byte-bound, so compression buys energy (~20%), not throughput -- and aneforge's int8 flag quantizes the
  weights while leaving the MAC in fp16.
- **The latency-bound batch trick** (verify K tokens for the cost of one, which makes speculative decode nearly
  free on the ANE) applies only to dispatch-bound decode. ResNet-50 has no idle compute to reclaim.

The residual gap to the 18.8 conv roof is structural: that roof is a Winograd effect available only to dense
3x3 stride-1 convs, and roughly half of ResNet-50's FLOPs are in 1x1 pointwise (and strided) convs that are
Winograd-ineligible. Reaching 18.8 needs a different network, not a different backend setting. In short: this
is the ceiling for ResNet-50 on this engine.

## What is official, and what is not

This is an unofficial result. It is produced with the real MLCommons LoadGen and passes the upstream
`submission_checker` (v5.1: "Closed Results = 3, Systems = 1, SUMMARY: submission looks OK"), with TEST01 and
TEST04 compliance PASS and the accuracy logs truncated and hashed -- but it has no MLCommons audit trail and
was not submitted in an official round. Landing it in an official round is a membership and process step, not
an engineering one.

## Reproduce the full submission

```bash
PYTHONPATH=. python3 bench/mlperf/run_submission.py \
    --imagenet-val ~/Models/mlperf/val --val-map val_map.txt \
    --count 1024 --acc-count 0 --min-duration 600 --compliance-duration 600
```

This builds the complete edge/Closed ResNet-50 tree (all three scenarios, real-LoadGen performance and
accuracy, TEST01 + TEST04, system description) under `bench/mlperf/submission/`. Validate it with the upstream
checker from a clone of `mlcommons/inference`. See [`bench/mlperf/README.md`](https://github.com/sbryngelson/ANEForge/tree/main/bench/mlperf)
for the full harness, the accuracy path, and the layout.
