#!/usr/bin/env bash
# One-command ANE MLPerf ResNet-50 repro.
#
#   ./bench/mlperf/repro.sh                 # no dataset: compile on the ANE, print p90 latency + fp16/int8 fidelity
#   ./bench/mlperf/repro.sh --full --imagenet-val <val_dir> --val-map <val_map.txt>
#                                           # full 50k top-1 accuracy (ANE fp16/int8 vs onnxruntime fp32)
#
# This is NOT an official MLPerf submission (see bench/mlperf/README.md). It reproduces, on any Apple Silicon
# Mac, that the MLPerf reference ResNet-50 runs on the Apple Neural Engine at fp32-level fidelity.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# --- platform preflight (aneforge only dispatches to the ANE on Apple Silicon macOS) ---
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "warning: the ANE is Apple Silicon macOS only (found $(uname -s)/$(uname -m)); dispatch will fail." >&2
fi

# --- dependency hint (the [mlperf] extra: onnx + onnxruntime reference + pillow) ---
if ! python3 -c "import onnxruntime, onnx" 2>/dev/null; then
  echo "error: missing deps. Install the harness extra:  pip install -e \".[mlperf]\"" >&2
  exit 1
fi

# --- MLPerf reference ResNet-50 (Zenodo, ~98 MB), fetched once ---
MODEL_DIR="$HOME/Models/mlperf"
MODEL="$MODEL_DIR/resnet50_v1_mlperf.onnx"
if [ ! -f "$MODEL" ]; then
  echo "fetching the MLPerf reference ResNet-50 (~98 MB) -> $MODEL ..."
  mkdir -p "$MODEL_DIR"
  curl -fL -o "$MODEL" https://zenodo.org/record/2592612/files/resnet50_v1.onnx
fi

# --- full accuracy path: escalate to run_accuracy.py over real ImageNet val ---
if [ "${1:-}" = "--full" ]; then
  shift
  echo "full 50k accuracy run (ANE fp16/int8 vs onnxruntime fp32, exact MLPerf preprocessing) ..."
  exec python3 bench/mlperf/run_accuracy.py --preprocess mlperf --model "$MODEL" "$@"
fi

# --- default: no-dataset demo (compile + latency + fidelity) ---
exec python3 bench/mlperf/repro.py --onnx "$MODEL" "$@"
