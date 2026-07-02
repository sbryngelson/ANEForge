#!/usr/bin/env python3
"""Assemble an MLPerf Inference submission tree for ResNet-50 on the ANE (Closed, Edge, SingleStream), using
the REAL MLCommons LoadGen for both the performance and accuracy runs. Produces the directory layout the
`submission-checker` expects:

  <out>/<div>/<submitter>/
    systems/<system>.json                                    # system_description
    code/resnet50/aneforge/README.md
    measurements/<system>/resnet50/SingleStream/{mlperf.conf,user.conf,<system>.json,README.md}
    results/<system>/resnet50/SingleStream/
      performance/run_1/{mlperf_log_summary.txt, mlperf_log_detail.txt}
      accuracy/{mlperf_log_accuracy.json, mlperf_log_summary.txt, mlperf_log_detail.txt, accuracy.txt}

Requires `mlperf_loadgen` and the MLPerf reference ResNet-50 + ImageNet val (see README.md). A full official run
needs `--count 1024 --min-duration 600` (SingleStream floors) and the full 50k val for accuracy.

  PYTHONPATH=. python3 bench/mlperf/run_submission.py --imagenet-val ~/Models/mlperf/val --val-map val_map.txt \
      --count 1024 --min-duration 600
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import resnet50 as rn          # noqa: E402
import loadgen_official as lgo # noqa: E402

_REF = os.path.expanduser("~/Models/mlperf/resnet50_v1_mlperf.onnx")


def _sysctl(k):
    try:
        return subprocess.check_output(["sysctl", "-n", k]).decode().strip()
    except Exception:
        return ""


def system_description(submitter, system_name, division, status):
    """MLPerf system_description.json, populated from this machine + the ANE/aneforge stack. Fills every field
    the submission-checker requires (meaningful, non-empty values where it demands them)."""
    mem = int(_sysctl("hw.memsize") or 0) // (1024 ** 3)
    try:
        st = os.statvfs("/"); disk = st.f_blocks * st.f_frsize // (1024 ** 3)
    except Exception:
        disk = 0
    try:
        osv = subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
        build = subprocess.check_output(["sw_vers", "-buildVersion"]).decode().strip()
    except Exception:
        osv = build = "unknown"
    chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    cores = int(_sysctl("hw.physicalcpu") or 0)
    return {
        "submitter": submitter, "division": division, "system_name": system_name, "system_type": "edge",
        "system_type_detail": "Apple Silicon SoC (on-die Neural Engine)", "status": status, "number_of_nodes": 1,
        "host_processor_model_name": chip, "host_processors_per_node": 1, "host_processor_core_count": cores,
        "host_processor_frequency": "3.9 GHz (P-core max)", "host_processor_caches": "per-Apple-Silicon",
        "host_processor_interconnect": "Apple SoC fabric",
        "host_memory_capacity": f"{mem} GB", "host_memory_configuration": "unified LPDDR5",
        "host_storage_capacity": f"{disk} GB", "host_storage_type": "NVMe SSD (internal)",
        "host_networking": "Wi-Fi / Thunderbolt", "host_networking_topology": "N/A (single node)",
        "host_network_card_count": "0",
        "accelerator_model_name": "Apple Neural Engine", "accelerators_per_node": 1,
        "accelerator_frequency": "on-die (SoC-managed)", "accelerator_host_interconnect": "on-die (SoC)",
        "accelerator_interconnect": "N/A (single accelerator)", "accelerator_interconnect_topology": "N/A",
        "accelerator_memory_capacity": f"shared unified memory ({mem} GB)",
        "accelerator_memory_configuration": "unified LPDDR5", "accelerator_on-chip_memories": "on-die SRAM",
        "cooling": "active", "hw_notes": f"{chip}, {mem} GB unified memory; Apple Neural Engine via e5rt",
        "framework": "aneforge (ANE e5rt dispatch, fp16)", "operating_system": f"macOS {osv} ({build})",
        "other_software_stack": "aneforge; coremltools-free MIL emitter",
        "sw_notes": "Pure Apple Neural Engine execution via aneforge; Conv->BN folded in the importer for fp16 fidelity.",
    }


def measurements_json(division):
    """The per-benchmark measurements/<system>.json (implementation metadata the checker reads)."""
    return {
        "starting_weights_filename": "resnet50_v1.onnx (MLPerf reference, Zenodo record 2592612)",
        "weight_transformations": "Conv->BatchNorm folded into conv (exact); trailing ArgMax stripped to logits",
        "weight_data_types": "fp16", "input_data_types": "fp16",
        "retraining": "none", "framework": "aneforge (ANE e5rt, fp16)", "division": division,
    }


_USER_CONF = ("# user.conf -- SingleStream single-sample latency target (LoadGen refines the schedule)\n"
              "resnet50.SingleStream.target_latency = 1\n")
_MLPERF_CONF = ("# mlperf.conf (subset) -- SingleStream run-length floors for a valid Edge run\n"
                "resnet50.SingleStream.min_query_count = 1024\n"
                "resnet50.SingleStream.min_duration = 600000\n")


def main():
    ap = argparse.ArgumentParser(description="Assemble an MLPerf ResNet-50 submission tree (ANE, Closed, SingleStream)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "submission"))
    ap.add_argument("--submitter", default="ANEForge")
    ap.add_argument("--system-name", default="apple-m5pro-ane")
    ap.add_argument("--division", default="closed", choices=("closed", "open"))
    ap.add_argument("--status", default="preview", choices=("available", "preview", "rdi"))
    ap.add_argument("--model", default=None, help="ResNet-50 ONNX (default: MLPerf reference)")
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--val-map", required=True)
    ap.add_argument("--count", type=int, default=1024, help="min queries (perf) / images (accuracy)")
    ap.add_argument("--min-duration", type=float, default=0.0, help="perf min seconds (600 for an official run)")
    ap.add_argument("--version", default="v5.1", help="MLPerf round (selects the official LoadGen RNG seeds)")
    args = ap.parse_args()
    seeds = lgo.SEEDS.get(args.version)

    if not lgo.available():
        print("mlperf_loadgen not installed. pip install mlcommons-loadgen"); return 1
    model = args.model or (_REF if os.path.exists(_REF) else None)
    if args.division == "closed" and model is None:
        print("closed division needs the MLPerf reference model (--model or ~/Models/mlperf/resnet50_v1_mlperf.onnx)"); return 1

    sysn = args.system_name
    base = Path(args.out) / args.division / args.submitter
    res = base / "results" / sysn / "resnet50" / "SingleStream"
    meas = base / "measurements" / sysn / "resnet50" / "SingleStream"
    perf_dir = res / "performance" / "run_1"; acc_dir = res / "accuracy"
    for d in (perf_dir, acc_dir, meas, base / "systems", base / "code" / "resnet50" / "aneforge"):
        d.mkdir(parents=True, exist_ok=True)

    print(f"building SUT (reference ResNet-50, fp16, ANE) ...", flush=True)
    sut = rn.build_sut(model_path=model, compress=None, strip_logits=True)

    # performance run (SingleStream, PerformanceOnly)
    print("performance run (real LoadGen, SingleStream) ...", flush=True)
    perf_qsl = rn.synthetic_qsl(count=max(args.count, 1024))
    psumm = lgo.run(sut, perf_qsl, scenario="SingleStream", mode="PerformanceOnly",
                    min_query_count=args.count, min_duration_ms=int(args.min_duration * 1000), outdir=str(perf_dir), seeds=seeds)

    # accuracy run (AccuracyOnly over the val set)
    print("accuracy run (real LoadGen AccuracyOnly, ImageNet val) ...", flush=True)
    acc_qsl, labels = rn.imagenet_val_qsl(args.imagenet_val, args.val_map, count=(args.count if args.count > 1024 else None), pre=rn.preprocess_mlperf)
    lgo.run(sut, acc_qsl, scenario="SingleStream", mode="AccuracyOnly", min_query_count=acc_qsl.count, outdir=str(acc_dir), seeds=seeds)
    top1, n = lgo.score_accuracy(str(acc_dir / "mlperf_log_accuracy.json"), labels)
    (acc_dir / "accuracy.txt").write_text(f"accuracy={top1 * 100:.3f}% ({int(round(top1 * n))}/{n}) top-1\n")

    # metadata files
    (base / "systems" / f"{sysn}.json").write_text(json.dumps(system_description(args.submitter, sysn, args.division, args.status), indent=2))
    (meas / f"{sysn}.json").write_text(json.dumps(measurements_json(args.division), indent=2))
    (meas / "user.conf").write_text(_USER_CONF)
    (meas / "mlperf.conf").write_text(_MLPERF_CONF)
    (meas / "README.md").write_text(f"# ResNet-50 SingleStream on the Apple Neural Engine (aneforge)\n\n"
                                    f"Reference ResNet-50 + MLPerf preprocessing, fp16 on the ANE. Conv->BN folded in the importer.\n"
                                    f"top-1 {top1 * 100:.2f}% over {n} val images. See bench/mlperf/.\n")
    (base / "code" / "resnet50" / "aneforge" / "README.md").write_text(
        "Implementation: `bench/mlperf/` in the aneforge repo (loadgen_official.py drives real LoadGen;\n"
        "resnet50.py is the SUT/QSL; the ANE runs fp16 via e5rt).\n")

    p90 = (psumm.get("p90_latency_ns") or 0) / 1e6
    print(f"\n=== submission tree at {base} ===")
    print(f"  performance: SingleStream p90 = {p90:.3f} ms   LoadGen result = {psumm.get('valid')}")
    print(f"  accuracy   : top-1 = {top1 * 100:.2f}% over {n}  (Closed needs >= 75.70%)")
    print(f"  official-length: {'YES' if psumm.get('valid') == 'VALID' and n >= 50000 else 'NO (short run; use --count 1024 --min-duration 600 and full 50k val)'}")
    print("Validate with the upstream checker:  python3 inference/tools/submission/submission_checker.py --input " + str(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
