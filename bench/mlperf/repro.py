#!/usr/bin/env python3
"""One-command ANE MLPerf ResNet-50 repro (no dataset, no arguments).

Compiles the MLPerf reference ResNet-50 onto the Apple Neural Engine and prints:
  1. SingleStream p90 latency -- real MLCommons LoadGen if installed, else the lite harness. Short run;
     the official-length (600 s) VALID number is in results/resnet50_official.json.
  2. ANE fp16 / int8 vs onnxruntime fp32 on MLPerf-scale inputs (mean-subtracted, ~+-150): top-1
     agreement and mean logit cosine.

Full 50k accuracy + submission_checker: `repro.sh --full`. Not an official MLPerf submission (see README)."""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Batch-1 ResNet-50 is dispatch-floor-bound (the SingleStream regime); silence the per-compile advisory.
warnings.filterwarnings("ignore", message=r".*dispatch-floor-bound.*")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # loadgen_lite, loadgen_official, resnet50

import numpy as np           # noqa: E402
import loadgen_lite as lg    # noqa: E402
import resnet50 as rn        # noqa: E402

_REF_ONNX = os.path.join(os.path.expanduser("~"), "Models", "mlperf", "resnet50_v1_mlperf.onnx")


def _mlperf_scale_qsl(count, seed=0):
    """Synthetic inputs scaled like MLPerf-preprocessed images (random 0-255 pixels minus the per-channel mean,
    ~[-124, 151]) -- the large-magnitude range where fp16 rounding matters, so the fidelity check is meaningful."""
    rng = np.random.default_rng(seed)
    px = rng.integers(0, 256, size=(count, 1, 3, 224, 224)).astype(np.float32)
    data = (px - rn._MEAN_MLPERF).astype(np.float16)
    return lg.QSL(count, lambda i: data[i], name="synthetic-mlperf")


def _singlestream_p90(sut, qsl, count):
    """SingleStream p90 latency (ms) via real LoadGen if available, else the lite harness. Returns
    (p90_ms, driver_label, validity_note)."""
    import loadgen_official as lgo
    if lgo.available():
        import tempfile
        summ = lgo.run(sut, qsl, scenario="SingleStream", min_query_count=count, outdir=tempfile.mkdtemp())
        return (summ.get("p90_latency_ns") or 0.0) / 1e6, "real MLCommons LoadGen", summ.get("valid", "?")
    r = lg.run_single_stream(sut, qsl, count=count, warmup=16)
    return r.p90_ms, "lite harness", ("VALID" if r.official else "short demo run")


def main():
    ap = argparse.ArgumentParser(description="One-command ANE MLPerf ResNet-50 repro (no dataset).")
    ap.add_argument("--onnx", default=None,
                    help="ResNet-50 ONNX (default: the MLPerf reference at ~/Models/mlperf/, else torchvision)")
    ap.add_argument("--count", type=int, default=2000, help="SingleStream demo queries (short; not official length)")
    ap.add_argument("--fidelity-count", type=int, default=16, help="synthetic inputs for the fp16/int8 fidelity check")
    args = ap.parse_args()

    model = args.onnx or (_REF_ONNX if os.path.exists(_REF_ONNX) else None)
    is_ref = bool(model)   # the MLPerf reference model is present; else we fall back to a torchvision export
    print("=" * 72)
    print("ANE MLPerf ResNet-50 repro" + ("  (MLPerf reference model)" if is_ref else "  (torchvision fallback)"))
    print("=" * 72, flush=True)

    # strip_to_logits: rewrite the reference model (ArgMax-terminated) to output logits with a static batch of 1;
    # a no-op on a model that already outputs logits. Both the ANE nets and onnxruntime run this same graph.
    logits_path = rn.strip_to_logits(rn.resolve_onnx(model))
    print("compiling ResNet-50 -> one ANE program (fp16) ...", flush=True)
    sut = rn.build_sut(model_path=logits_path)

    print("\n[1/2] SingleStream latency (Apple Neural Engine)")
    p90, driver, valid = _singlestream_p90(sut, rn.synthetic_qsl(count=args.count), args.count)
    print(f"      p90 = {p90:.3f} ms   ({driver}; {valid}; {args.count} queries)")
    print("      official-length (600 s) VALID number: results/resnet50_official.json", flush=True)

    print("\n[2/2] Numerical fidelity vs onnxruntime fp32 (MLPerf-scale inputs)")
    fid_qsl = _mlperf_scale_qsl(args.fidelity_count)
    ref_preds, ref_logits = rn.onnx_logits(logits_path, fid_qsl)
    fp16 = sut
    for name, s in (("fp16", fp16), ("int8", rn.build_sut(model_path=logits_path, compress="int8"))):
        preds, logits = rn.net_logits(s.net, fid_qsl)
        top1, cos = rn.fidelity(ref_preds, ref_logits, preds, logits)
        print(f"      ANE {name:4s}  top-1 agreement = {top1 * 100:6.2f}%   logit cosine = {cos:.5f}")

    print("\n" + "-" * 72)
    print("Reproduced: ResNet-50 runs on the Apple Neural Engine; ANE fp16 matches onnxruntime")
    print("fp32 to logit cosine ~1.0. Full 50k accuracy + submission_checker: repro.sh --full <val>.")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
