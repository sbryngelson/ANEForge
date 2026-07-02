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
    ap.add_argument("--preprocess", default="mlperf", choices=("mlperf", "torch"),
                    help="mlperf: raw-scale mean-subtract (Closed, fp16-bound); torch: /255+std normalized (ANE=fp32)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # raw-scale MLPerf preprocessing pairs with the reference model; normalized preprocessing pairs with the
    # torchvision model (its weights expect normalized input). See the fp16-magnitude finding in the README.
    if args.preprocess == "torch":
        pre, model = rn._preprocess, args.model            # None -> torchvision export
    else:
        pre, model = rn.preprocess_mlperf, args.model or (_REF if os.path.exists(_REF) else None)
    if args.imagenet_val:
        if not args.val_map:
            print("--imagenet-val needs --val-map"); return 1
        qsl, labels = rn.imagenet_val_qsl(args.imagenet_val, args.val_map, count=args.count, pre=pre)
    elif args.sample_images:
        qsl, labels = rn.sample_images_qsl(args.sample_images, count=args.count, pre=pre)
    else:
        print("pass --sample-images DIR or --imagenet-val DIR --val-map FILE"); return 1

    tag = os.path.basename(model) if model else "torchvision-resnet50"
    print(f"model {tag}; {qsl.count} images, {args.preprocess} preprocessing ...", flush=True)

    import onnxruntime as ort
    fp16 = rn.build_sut(model_path=model, compress=None, strip_logits=True)
    int8 = rn.build_sut(model_path=model, compress="int8", strip_logits=True)
    sess = ort.InferenceSession(rn.strip_to_logits(rn.resolve_onnx(model)))
    inm = sess.get_inputs()[0].name
    off = fp16.class_offset

    # single streaming pass: decode each image once, run fp32 (ref) + fp16 + int8 together (no 50k cache)
    n = qsl.count
    c_ref = c_16 = c_8 = agree16 = agree8 = 0
    cos16 = cos8 = 0.0
    for i in range(n):
        x = np.asarray(qsl.get(i), np.float16)
        lr = np.asarray(sess.run(None, {inm: x.astype(np.float32)})[0]).astype(np.float32).ravel()
        l16 = np.asarray(fp16.net(x)).astype(np.float32).ravel()
        l8 = np.asarray(int8.net(x)).astype(np.float32).ravel()
        rp, p16, p8 = int(lr.argmax()), int(l16.argmax()), int(l8.argmax())
        c_ref += (rp - off == labels[i]); c_16 += (p16 - off == labels[i]); c_8 += (p8 - off == labels[i])
        agree16 += (p16 == rp); agree8 += (p8 == rp)
        cos16 += float(l16 @ lr / (np.linalg.norm(l16) * np.linalg.norm(lr) + 1e-9))
        cos8 += float(l8 @ lr / (np.linalg.norm(l8) * np.linalg.norm(lr) + 1e-9))
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1}/{n}  fp16 top-1 so far {100 * c_16 / (i + 1):.2f}%", flush=True)

    rows = {"fp16": {"top1": c_16 / n, "agreement_vs_fp32": agree16 / n, "logit_cosine_vs_fp32": cos16 / n},
            "int8": {"top1": c_8 / n, "agreement_vs_fp32": agree8 / n, "logit_cosine_vs_fp32": cos8 / n},
            "onnxruntime_fp32": {"top1": c_ref / n}, "n": n, "class_offset": off}
    print(f"  ANE fp16              top-1 = {100 * c_16 / n:6.2f}%   (vs fp32: {100 * agree16 / n:5.1f}% agree, cosine {cos16 / n:.4f})")
    print(f"  ANE int8              top-1 = {100 * c_8 / n:6.2f}%   (vs fp32: {100 * agree8 / n:5.1f}% agree, cosine {cos8 / n:.4f})")
    print(f"  onnxruntime-fp32      top-1 = {100 * c_ref / n:6.2f}%")
    print(f"  (MLPerf reference full-val {_REF_TOP1 * 100:.2f}%; Closed needs >= {_REF_TOP1 * 99:.2f}%)")

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", "resnet50_accuracy.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": tag, "images": n, "preprocessing": args.preprocess, "top1": rows}, f, indent=2, allow_nan=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
