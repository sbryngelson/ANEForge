#!/usr/bin/env python3
"""MLPerf-lite accuracy runner: ResNet-50 top-1 on the ANE with the MLPerf reference model + MLPerf
preprocessing -- the Closed-division accuracy path. Reports ANE fp16 and int8 top-1 (and their fidelity vs
onnxruntime fp32) over a labeled image set.

  # 1000-image preview (imagenet-sample-images, one per class -> label = sorted class index):
  PYTHONPATH=. python3 bench/mlperf/run_accuracy.py --sample-images ~/Models/mlperf/imagenet-sample-images
  # full ILSVRC2012 val (flat dir of ILSVRC2012_val_*.JPEG + MLPerf val_map of "<file> <label>" lines):
  PYTHONPATH=. python3 bench/mlperf/run_accuracy.py --imagenet-val ~/data/imagenet/val --val-map val_map.txt

The default model is the MLPerf reference ResNet-50 (~/Models/mlperf/resnet50_v1_mlperf.onnx) if present, else
the torchvision export (Open-division). The trailing ArgMax is stripped so the ANE runs the backbone; the
1001-class background offset is handled automatically. Writes a JSON to bench/mlperf/results/."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # loadgen_lite, resnet50

import numpy as np       # noqa: E402
import resnet50 as rn    # noqa: E402

_REF = os.path.expanduser("~/Models/mlperf/resnet50_v1_mlperf.onnx")
_REF_TOP1 = 0.7646       # MLPerf reference full-val top-1; Closed needs >= 99% of it


def main():
    ap = argparse.ArgumentParser(description="MLPerf-lite ResNet-50 top-1 accuracy on the ANE")
    ap.add_argument("--model", default=None, help="ResNet-50 ONNX (default: MLPerf reference if present)")
    ap.add_argument("--sample-images", default=None, help="imagenet-sample-images dir (1/class; label=index)")
    ap.add_argument("--imagenet-val", default=None, help="flat ILSVRC2012 val dir")
    ap.add_argument("--val-map", default=None, help="MLPerf val_map ('<file> <label>' lines) for --imagenet-val")
    ap.add_argument("--count", type=int, default=None, help="cap the number of images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = args.model or (_REF if os.path.exists(_REF) else None)
    if args.imagenet_val:
        if not args.val_map:
            print("--imagenet-val needs --val-map"); return 1
        qsl, labels = rn.imagenet_val_qsl(args.imagenet_val, args.val_map, count=args.count)
    elif args.sample_images:
        qsl, labels = rn.sample_images_qsl(args.sample_images, count=args.count)
    else:
        print("pass --sample-images DIR or --imagenet-val DIR --val-map FILE"); return 1

    tag = os.path.basename(model) if model else "torchvision-resnet50"
    print(f"model {tag}; {qsl.count} images, MLPerf preprocessing ...", flush=True)

    # onnxruntime fp32 reference (same stripped logits model) -- the accuracy target + the fidelity baseline
    logits_path = rn.strip_to_logits(rn.resolve_onnx(model))
    ref_pred, ref_logits = rn.onnx_logits(logits_path, qsl)
    off = 1 if ref_logits.shape[1] == 1001 else 0

    rows = {}
    for compress in (None, "int8"):
        sut = rn.build_sut(model_path=model, compress=compress, strip_logits=True)
        top1, n = rn.accuracy(sut, qsl, labels)
        preds, logits = rn.net_logits(sut.net, qsl)                 # raw argmax + logits, for fidelity vs the reference
        agree, cos = rn.fidelity(ref_pred, ref_logits, preds, logits)
        rows[compress or "fp16"] = {"top1": top1, "n": n, "class_offset": sut.class_offset,
                                    "agreement_vs_fp32": agree, "logit_cosine_vs_fp32": cos}
        print(f"  ANE {sut.name:22s} top-1 = {top1 * 100:6.2f}%   (vs fp32: {agree * 100:5.1f}% agree, cosine {cos:.4f})")

    lab = np.asarray(labels[:len(ref_pred)])
    ref_top1 = float(((ref_pred - off) == lab).mean())
    print(f"  onnxruntime-fp32 reference     top-1 = {ref_top1 * 100:6.2f}%")
    print(f"  (MLPerf reference full-val {_REF_TOP1 * 100:.2f}%; Closed needs >= {_REF_TOP1 * 99:.2f}%)")
    rows["onnxruntime_fp32"] = {"top1": ref_top1, "n": int(len(ref_pred))}

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", "resnet50_accuracy.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": tag, "images": qsl.count, "preprocessing": "mlperf", "top1": rows}, f, indent=2, allow_nan=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
