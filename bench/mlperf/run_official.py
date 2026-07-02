#!/usr/bin/env python3
"""Run ResNet-50 on the ANE under the REAL MLCommons LoadGen (writes mlperf_log_summary.txt and the official
early-stopping metrics), and optionally cross-check against loadgen_lite on the same SUT -- the differential
validation that our methodology-shaped numbers track LoadGen's.

Requires `mlperf_loadgen` (pip install mlcommons-loadgen; a compiler if no wheel).

  PYTHONPATH=. python3 bench/mlperf/run_official.py --count 2048 --compare
  PYTHONPATH=. python3 bench/mlperf/run_official.py --scenario Offline --count 4096
  PYTHONPATH=. python3 bench/mlperf/run_official.py --count 1024 --min-duration 600   # official length

Writes a JSON to bench/mlperf/results/. Real LoadGen output goes to --logdir."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))   # loadgen_lite, loadgen_official, resnet50

import loadgen_lite as lg_lite       # noqa: E402
import loadgen_official as lg_off    # noqa: E402
import resnet50 as rn                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="ResNet-50 on the ANE under real MLCommons LoadGen")
    ap.add_argument("--onnx", default=None, help="ResNet-50 ONNX (default: torchvision reference)")
    ap.add_argument("--scenario", default="SingleStream", choices=("SingleStream", "Offline"))
    ap.add_argument("--count", type=int, default=1024, help="min query/sample count")
    ap.add_argument("--min-duration", type=float, default=0.0, help="min run seconds (600 for an official run)")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--compare", action="store_true", help="also run loadgen_lite on the same SUT and compare")
    ap.add_argument("--logdir", default=None, help="LoadGen output dir (mlperf_log_*.txt)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    if not lg_off.available():
        print("mlperf_loadgen not installed. Install it with:  pip install mlcommons-loadgen"); return 1

    compress = "int8" if args.int8 else None
    print(f"building ResNet-50 SUT ({compress or 'fp16'}, ANE) ...", flush=True)
    sut = rn.build_sut(model_path=args.onnx, compress=compress)
    qsl = rn.synthetic_qsl(count=max(args.count, 256))

    logdir = args.logdir or os.path.join(Path(__file__).resolve().parent, "results", "loadgen")
    print(f"running REAL LoadGen ({args.scenario}, PerformanceOnly) ...", flush=True)
    summ = lg_off.run(sut, qsl, scenario=args.scenario, min_query_count=args.count,
                      min_duration_ms=int(args.min_duration * 1000), outdir=logdir)
    p90_ms = (summ.get("p90_latency_ns") or 0) / 1e6
    print(f"  LoadGen: result={summ.get('valid')}  ", end="")
    if args.scenario == "SingleStream":
        print(f"p90 latency = {p90_ms:.4f} ms")
    else:
        print(f"throughput = {summ.get('samples_per_second')} samples/s")

    out = {"scenario": args.scenario, "compress": compress or "fp16", "loadgen": summ}
    if args.compare:
        run = lg_lite.run_single_stream if args.scenario == "SingleStream" else lg_lite.run_offline
        r = run(sut, qsl, count=args.count, min_duration_s=args.min_duration)
        print(r.summary(), flush=True)
        out["lite"] = r.to_dict()
        if args.scenario == "SingleStream" and p90_ms > 0:
            ratio = r.p90_ms / p90_ms
            print(f"\n  differential  lite p90 / loadgen p90 = {r.p90_ms:.4f} / {p90_ms:.4f} = {ratio:.3f}x")
            out["p90_ratio_lite_over_loadgen"] = ratio

    path = args.out or os.path.join(Path(__file__).resolve().parent, "results", f"resnet50_loadgen_{args.scenario.lower()}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"\nwrote {path}  (LoadGen logs in {logdir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
