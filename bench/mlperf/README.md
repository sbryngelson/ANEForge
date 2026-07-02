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
- It is **not** an official MLPerf submission. The `loadgen_lite` runs have no MLCommons audit trail, and the
  default run lengths are short. A `loadgen_lite` run is flagged `official: false` unless it meets the MLPerf
  minimums (>= 1024 queries AND >= 600 s).
- A **real MLCommons LoadGen** driver (`loadgen_official.py`) runs the SAME workloads through the actual
  generator/logger (writes `mlperf_log_summary.txt`, uses LoadGen's early-stopping metric). Install it with
  `pip install mlcommons-loadgen`. On this hardware the two agree: real-LoadGen ResNet-50 SingleStream
  p90 = 0.773 ms vs lite p90 = 0.771 ms (ratio 0.998), so the lite numbers track LoadGen's.

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

# REAL MLCommons LoadGen (pip install mlcommons-loadgen) + differential vs the lite harness:
PYTHONPATH=. python3 bench/mlperf/run_official.py --count 2000 --compare
PYTHONPATH=. python3 bench/mlperf/run_official.py --count 1024 --min-duration 600   # official length
```

Each run prints a summary and writes a JSON to `bench/mlperf/results/`.

## Sample results (Apple M-series, this environment)

Methodology-only numbers (short runs; not an audited submission), to show the shape of the output:

| Run | Result |
| --- | --- |
| ResNet-50 fp16, SingleStream (official length: 785,720 queries / 600 s, `official: true`) | p90 latency 0.776 ms (~1310 img/s) |
| ResNet-50 fp16, SingleStream under REAL LoadGen | p90 0.773 ms, `Result is: VALID`; lite/loadgen ratio 0.998 |
| ResNet-50, Offline | ~1310 samples/s (tracks SingleStream: latency-bound at batch 1) |
| ResNet-50 fidelity vs onnxruntime fp32 | fp16: 100% top-1 agreement, logit cosine 1.00000; int8: 100%, cosine 0.998 |
| Qwen3-0.6B decode, fp16 | TTFT ~520 ms, TPOT ~18.7 ms/token, ~37.7 tok/s |
| Qwen3-0.6B decode, int8 | TTFT ~460 ms, TPOT ~17.0 ms/token, ~41.7 tok/s |

The int8 ResNet-50 preserving every prediction (cosine 0.998 vs fp32) is evidence the ANE clears the accuracy
gate *for a normalized-input model*; the committed `results/*.json` carry the exact values.

## Closed-division accuracy: the MLPerf reference model on the ANE

`run_accuracy.py` runs the actual **MLPerf reference ResNet-50** (a TF export from Zenodo; the trailing ArgMax
is stripped and the 1001-class background offset is auto-detected) with the **MLPerf preprocessing** (resize
256 / center-crop 224 / per-channel mean subtraction, no `/255`, no std). Fetch the model once:

```bash
curl -L -o ~/Models/mlperf/resnet50_v1_mlperf.onnx https://zenodo.org/record/2592612/files/resnet50_v1.onnx
PYTHONPATH=. python3 bench/mlperf/run_accuracy.py --preprocess mlperf \
  --imagenet-val ~/Models/mlperf/val --val-map val_map.txt      # full ILSVRC-2012 val
```

### The precision fix (Conv->BN fold)

Naively, the reference model lost ~10 points on the ANE (fp16 67.0% vs onnxruntime fp32 77.05% on val,
logit cosine 0.917). This is **not** an accumulation limit: per the ANE guide's numerics chapter the engine
reduces in a **wide fp32-class accumulator**, and the reference's convs are individually bit-exact on the ANE
(the stem conv reads back at cosine 1.0). The loss was localized to the **one standalone `BatchNormalization`**
(the stem; the other 52 layers already have BN folded into their conv). A standalone BN runs in the fp16
datapath on the large pre-BN values (~1000s), and that rounding then amplifies through the network.

The importer now **folds `Conv->BatchNorm` into the conv** (`aneforge/onnx.py`, exact) -- so the reference
model computes the stem as a single wide-accumulator conv, with no standalone fp16 BN. Effect on the exact
reference model with MLPerf preprocessing, over the **full 50,000-image ILSVRC-2012 val set**:

| Reference ResNet-50, MLPerf preprocessing, 50k val | before fold | **after Conv->BN fold** |
| --- | --- | --- |
| ANE fp16 top-1 | ~67% | **76.16%** (cosine 0.917 -> **0.99999** vs fp32) |
| ANE int8 top-1 | ~67% | **76.06%** |
| onnxruntime fp32 | 76.16% | 76.16% |

**ANE fp16 == fp32 == 76.16% over the full val set, on the exact MLPerf reference model -- clearing the
Closed-division gate** (>= 99% of the 76.46% reference = >= 75.70%). The ANE is exactly as accurate as full
precision here. The fold is a general importer feature -- any TF-exported model with a standalone Conv->BN
benefits. (Note: the ~0.3pt below the canonical 76.46% is a preprocessing detail -- this harness resizes with
PIL bilinear; the MLPerf reference uses cv2 INTER_AREA. It moves fp32 and ANE together and does not affect the
ANE-vs-fp32 equality; an exactly-official run would match the interpolation.)

### Open division (normalized-input model), full 50k val

A normalized-input model (torchvision ResNet-50 V2, `--preprocess torch`) is also fp32-faithful on the ANE,
over all 50,000 val images:

| Path | top-1 | fidelity vs fp32 |
| --- | --- | --- |
| onnxruntime fp32 | 80.32% | -- |
| **ANE fp16** | **80.33%** | 99.9% agree, cosine 0.99999 |
| ANE int8 | 80.22% | 98.2% agree, cosine 0.9962 |

### Full ImageNet-val run

The tables above use the 1000-image preview. For the full 50,000-image ILSVRC-2012 val set (not committed --
ImageNet is licensed): download `ILSVRC2012_img_val.tar`, extract to a flat dir, and build a val_map (each
`ILSVRC2012_val_*.JPEG` -> its 0-999 label). The labels come from the public per-image WNID list; sorted-WNID
order is the torchvision/PyTorch class index:

```bash
tar xf ILSVRC2012_img_val.tar -C ~/Models/mlperf/val
curl -L -o labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
python3 -c "w=[l.split()[0] for l in open('labels.txt')]; idx={x:i for i,x in enumerate(sorted(set(w)))}; open('val_map.txt','w').writelines(f'ILSVRC2012_val_{i:08d}.JPEG {idx[x]}\n' for i,x in enumerate(w,1))"
PYTHONPATH=. python3 bench/mlperf/run_accuracy.py --preprocess torch --imagenet-val ~/Models/mlperf/val --val-map val_map.txt
```

The runner streams (one decode per image, no 50k cache). Sanity: onnxruntime fp32 should read ~76% top-1.

Both submission paths now hold on the ANE: **Open division** (normalized-input model) and **Closed division**
(the exact MLPerf reference model + preprocessing, via the Conv->BN fold) both reach fp32-level top-1.

## Assembling a submission

`run_submission.py` produces the MLPerf submission directory (Closed, Edge, SingleStream) with the REAL LoadGen
for both runs -- a PerformanceOnly run (`mlperf_log_summary.txt`, `Result is: VALID`) and an AccuracyOnly run
(`mlperf_log_accuracy.json`, scored to `accuracy.txt`) -- plus `systems/<system>.json` (system_description),
`measurements/.../{mlperf.conf,user.conf}`, and a `code/` pointer:

```bash
PYTHONPATH=. python3 bench/mlperf/run_submission.py \
    --imagenet-val ~/Models/mlperf/val --val-map val_map.txt \
    --count 1024 --min-duration 600            # official-length: 1024 queries AND 600s perf, full 50k accuracy
```

The tree is written under `bench/mlperf/submission/` (gitignored -- regenerate it; it carries LoadGen logs).
Validate it with the upstream checker (needs the public `mlcommons/inference` repo, not auth-gated):

```bash
python3 inference/tools/submission/submission_checker.py --input bench/mlperf/submission
```

### submission-checker status

Run against the tree, the upstream `submission_checker` (v5.1) already passes the **system-description**,
**measurement**, and **RNG-seed** checks, with a **VALID** official-length SingleStream performance run and a
real `mlperf_log_accuracy.json`. What it still wants (all mechanical MLPerf-submission plumbing, not accuracy
or latency):

- **all three scenarios** for edge ResNet-50 -- add MultiStream + Offline runs (this harness has Offline; a
  batched MultiStream SUT is the one new piece);
- **accuracy-log truncation + hash** -- run the repo's `truncate_accuracy_log.py` so `accuracy.txt` carries the
  hash of `mlperf_log_accuracy.json`;
- a **`compliance/` dir** with TEST01 + TEST04 (re-run LoadGen under the audit configs from `mlcommons/inference`).

Then MLCommons membership + an official round -- process, not engineering. The accuracy (fp16 == fp32, clears
the gate), the VALID official-length latency, and the real LoadGen artifacts are all in hand.

## Layout

| File | Role |
| --- | --- |
| `loadgen_lite.py` | LoadGen-shaped core: `QSL`, `SUT`, `run_single_stream`, `run_offline`, `run_llm_decode`, `Result` (pure Python + numpy, unit-tested off-device in `tests/test_mlperf_harness.py`) |
| `resnet50.py` | ResNet-50 workload: ONNX -> ANE program wrapped as a SUT, synthetic + ImageNet-val QSLs, top-1 accuracy, reference-fidelity vs onnxruntime |
| `run.py` | ResNet-50 CLI: build the SUT, run the scenarios (+ accuracy / fidelity), write results |
| `llm_decode.py` | LLM decode workload: an aneforge LLM wrapped as a decode SUT (TTFT / TPOT / tokens-per-second) |
| `run_llm.py` | LLM CLI: build the decode SUT, run `run_llm_decode`, write results |
| `loadgen_official.py` | REAL MLCommons LoadGen driver behind the same SUT/QSL (import-gated on `mlperf_loadgen`) |
| `run_official.py` | ResNet-50 under real LoadGen + differential vs `loadgen_lite` |
| `run_accuracy.py` | ResNet-50 top-1 with the MLPerf reference model + preprocessing (Closed accuracy path) |
| `run_submission.py` | assemble an MLPerf submission tree (real LoadGen perf + AccuracyOnly, configs, system_description) |

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

1. Real `mlperf_loadgen` -- **done** (`loadgen_official.py`): the SUT/QSL map onto `lg.ConstructSUT` /
   `lg.ConstructQSL` and the run is driven by `StartTestWithLogSettings`; the workload code is unchanged.
2. Accuracy -- **done** (`run_accuracy.py`): the exact MLPerf reference model + preprocessing reach fp32-level
   top-1 on the ANE and clear the Closed >= 99% gate, via the Conv->BN fold (see the section above). Remaining:
   emit accuracy through LoadGen AccuracyOnly (`mlperf_log_accuracy.json`) rather than the direct top-1 scorer.
3. Add the submission package: `mlperf.conf` / `user.conf`, `system_description.json`, and pass the
   compliance tests (TEST01 / TEST04 / TEST05) and `submission-checker.py`.
4. Submit in an official Inference round (Edge, SingleStream + Offline) through an MLCommons member.

## Adding a workload

Implement a `SUT` subclass (an `issue(qsl, indices) -> outputs` method, or a `decode(...)` for LLMs) and a
`QSL` (a `get(index) -> features`), and drive them with the existing scenario runners. `resnet50.py` and
`llm_decode.py` are the two worked examples.
