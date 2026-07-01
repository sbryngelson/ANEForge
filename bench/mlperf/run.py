#!/usr/bin/env python3
"""MLPerf-lite runner: ResNet-50 on the ANE, SingleStream + Offline (performance) and optional top-1 (accuracy).
Self-contained -- with no dataset it uses synthetic inputs for a performance-only run.

  PYTHONPATH=. python3 bench/mlperf/run.py                          # synthetic perf, torchvision ResNet-50
  PYTHONPATH=. python3 bench/mlperf/run.py --count 1024             # longer run
  PYTHONPATH=. python3 bench/mlperf/run.py --int8                   # int8 ANE weights
  PYTHONPATH=. python3 bench/mlperf/run.py --imagenet-val ~/data/imagenet/val --count 512   # + top-1 accuracy
  PYTHONPATH=. python3 bench/mlperf/run.py --onnx my_resnet50.onnx  # your own MLPerf ResNet-50

Writes a JSON to bench/mlperf/results/. This is NOT an official MLPerf submission (see README.md)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # loadgen_lite, resnet50

import loadgen_lite as lg      # noqa: E402
import resnet50 as rn          # noqa: E402

_REF_TOP1 = 0.7646             # MLPerf reference ResNet-50 top-1 (Closed needs >= 99% of it)


def main():
    ap = argparse.ArgumentParser(description="MLPerf-lite ResNet-50 on the ANE")
    ap.add_argument("--onnx", default=None, help="ResNet-50 ONNX path (default: export torchvision)")
    ap.add_argument("--scenario", default="both", choices=("single", "offline", "both"))
    ap.add_argument("--count", type=int, default=256, help="queries (SingleStream) / samples (Offline)")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--int8", action="store_true", help="int8 ANE weights")
    ap.add_argument("--int4", action="store_true", help="int4 ANE weights")
    ap.add_argument("--imagenet-val", default=None, help="ImageNet val dir (val/<wnid>/*.JPEG) -> top-1 accuracy")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    compress = "int8" if args.int8 else ("int4" if args.int4 else None)
    print(f"building ResNet-50 SUT ({compress or 'fp16'}, ANE) ...", flush=True)
    sut = rn.build_sut(model_path=args.onnx, compress=compress)

    results = []
    if args.imagenet_val:                              # accuracy mode needs the real dataset
        print(f"accuracy: preprocessing ImageNet val from {args.imagenet_val} ...", flush=True)
        qsl, labels = rn.imagenet_qsl(args.imagenet_val, count=args.count)
        top1, n = rn.accuracy(sut, qsl, labels)
        print(f"  top-1 = {top1 * 100:.2f}%  over {n} images   (ref {_REF_TOP1 * 100:.2f}%, "
              f"Closed needs >= {_REF_TOP1 * 0.99 * 100:.2f}%)")
        results.append({"mode": "accuracy", "top1": top1, "n": n, "sut": sut.name})
    else:
        qsl = rn.synthetic_qsl(count=max(args.count, 64))

    scenarios = ("single", "offline") if args.scenario == "both" else (args.scenario,)
    for sc in scenarios:
        r = (lg.run_single_stream if sc == "single" else lg.run_offline)(sut, qsl, count=args.count, warmup=args.warmup)
        print(r.summary(), flush=True)
        results.append(r.to_dict())

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", f"resnet50_{compress or 'fp16'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": "resnet50", "compress": compress or "fp16", "count": args.count, "results": results},
                  f, indent=2, allow_nan=False)          # fail loudly rather than write invalid JSON (NaN)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
