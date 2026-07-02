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


_SCENARIOS = ["SingleStream", "MultiStream", "Offline"]


def _scn_kwargs(scenario, offline_qps):
    kw = {}
    if scenario == "MultiStream":
        kw["samples_per_query"] = 8
    if scenario == "Offline":
        kw["offline_qps"] = offline_qps
    return kw


def _compliance(inf_repo, sut, scenario, perf_qsl, count, dur_ms, seeds, offline_qps, res_scn, comp_scn, tmp):
    """Generate TEST01 (accuracy+performance) and TEST04 (determinism) compliance dirs for one scenario, using
    each test's audit.config and the upstream run_verification.py."""
    import subprocess
    comp = os.path.join(inf_repo, "compliance")
    kw = _scn_kwargs(scenario, offline_qps)
    # TEST01: one audit run (mode=2 + sampled accuracy log) -> compliance_dir has mlperf_log_{accuracy.json,detail.txt}
    t1 = os.path.join(tmp, "TEST01"); os.makedirs(t1, exist_ok=True)
    lgo.run(sut, perf_qsl, scenario=scenario, mode="PerformanceOnly", min_query_count=count,
            min_duration_ms=dur_ms, outdir=t1, seeds=seeds, audit_config=os.path.join(comp, "TEST01", "resnet50", "audit.config"), **kw)
    subprocess.run([sys.executable, os.path.join(comp, "TEST01", "run_verification.py"),
                    "-r", res_scn, "-c", t1, "-o", comp_scn], check=False)
    # TEST04: one determinism re-run -> compliance_dir has mlperf_log_{summary,detail}; the verifier compares
    # it against the submission's own performance/run_1.
    t4 = os.path.join(tmp, "TEST04"); os.makedirs(t4, exist_ok=True)
    lgo.run(sut, perf_qsl, scenario=scenario, mode="PerformanceOnly", min_query_count=count,
            min_duration_ms=dur_ms, outdir=t4, seeds=seeds, audit_config=os.path.join(comp, "TEST04", "audit.config"), **kw)
    subprocess.run([sys.executable, os.path.join(comp, "TEST04", "run_verification.py"),
                    "-r", res_scn, "-c", t4, "-o", comp_scn], check=False)


def main():
    ap = argparse.ArgumentParser(description="Assemble an MLPerf ResNet-50 submission tree (ANE, Closed, Edge)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "submission"))
    ap.add_argument("--submitter", default="ANEForge")
    ap.add_argument("--system-name", default="apple-m5pro-ane")
    ap.add_argument("--division", default="closed", choices=("closed", "open"))
    ap.add_argument("--status", default="preview", choices=("available", "preview", "rdi"))
    ap.add_argument("--model", default=None, help="ResNet-50 ONNX (default: MLPerf reference)")
    ap.add_argument("--imagenet-val", required=True)
    ap.add_argument("--val-map", required=True)
    ap.add_argument("--count", type=int, default=1024, help="min queries per perf run")
    ap.add_argument("--acc-count", type=int, default=0, help="accuracy images (0 = full val set)")
    ap.add_argument("--min-duration", type=float, default=0.0, help="perf min seconds (600 for an official run)")
    ap.add_argument("--compliance-duration", type=float, default=0.0, help="compliance perf min seconds")
    ap.add_argument("--offline-qps", type=float, default=1200.0)
    ap.add_argument("--no-compliance", action="store_true")
    ap.add_argument("--inference-repo", default=os.path.expanduser("~/Downloads/inference"))
    ap.add_argument("--version", default="v5.1", help="MLPerf round (selects the official LoadGen RNG seeds)")
    args = ap.parse_args()
    seeds = lgo.SEEDS.get(args.version)

    if not lgo.available():
        print("mlperf_loadgen not installed. pip install mlcommons-loadgen"); return 1
    model = args.model or (_REF if os.path.exists(_REF) else None)
    if args.division == "closed" and model is None:
        print("closed division needs the MLPerf reference model"); return 1

    sysn = args.system_name
    base = Path(args.out) / args.division / args.submitter
    (base / "systems").mkdir(parents=True, exist_ok=True)
    (base / "code" / "resnet50" / "aneforge").mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out) / "_compliance_runs"

    print("building SUT (reference ResNet-50, fp16, ANE) ...", flush=True)
    sut = rn.build_sut(model_path=model, compress=None, strip_logits=True)
    # one val QSL for perf + accuracy + compliance: preprocessing happens at LoadGen load time (untimed, cached,
    # bounded to the perf-sample window), so perf latency is inference-only and consistent with the TEST01 run.
    val_qsl, labels = rn.imagenet_val_qsl(args.imagenet_val, args.val_map,
                                          count=(args.acc_count or None), pre=rn.preprocess_mlperf, cache=True)
    perf_qsl = acc_qsl = val_qsl
    dur_ms = int(args.min_duration * 1000); cdur_ms = int(args.compliance_duration * 1000)
    summary = {}
    for scn in _SCENARIOS:
        res = base / "results" / sysn / "resnet50" / scn
        meas = base / "measurements" / sysn / "resnet50" / scn
        (res / "performance" / "run_1").mkdir(parents=True, exist_ok=True)
        (res / "accuracy").mkdir(parents=True, exist_ok=True); meas.mkdir(parents=True, exist_ok=True)
        kw = _scn_kwargs(scn, args.offline_qps)
        print(f"[{scn}] performance run ...", flush=True)
        ps = lgo.run(sut, perf_qsl, scenario=scn, mode="PerformanceOnly", min_query_count=args.count,
                     min_duration_ms=dur_ms, outdir=str(res / "performance" / "run_1"), seeds=seeds, **kw)
        print(f"[{scn}] accuracy run ...", flush=True)
        lgo.run(sut, acc_qsl, scenario=scn, mode="AccuracyOnly", min_query_count=acc_qsl.count,
                outdir=str(res / "accuracy"), seeds=seeds, **kw)
        top1, n = lgo.score_accuracy(str(res / "accuracy" / "mlperf_log_accuracy.json"), labels)
        (res / "accuracy" / "accuracy.txt").write_text(f"accuracy={top1 * 100:.3f}% ({int(round(top1 * n))}/{n}) top-1\n")
        (meas / f"{sysn}.json").write_text(json.dumps(measurements_json(args.division), indent=2))
        (meas / "user.conf").write_text(_USER_CONF)
        (meas / "mlperf.conf").write_text(_MLPERF_CONF)
        (meas / "README.md").write_text(f"# ResNet-50 {scn} on the Apple Neural Engine (aneforge)\n\n"
                                        f"Reference ResNet-50 + MLPerf preprocessing, fp16 on the ANE (Conv->BN folded). "
                                        f"top-1 {top1 * 100:.2f}% over {n} val images.\n")
        summary[scn] = {"valid": ps.get("valid"), "p90_ms": (ps.get("p90_latency_ns") or 0) / 1e6,
                        "sps": ps.get("samples_per_second"), "top1": top1, "n": n}
        if not args.no_compliance:
            print(f"[{scn}] compliance TEST01 + TEST04 ...", flush=True)
            comp_scn = str(base / "compliance" / sysn / "resnet50" / scn)
            _compliance(args.inference_repo, sut, scn, perf_qsl, args.count, cdur_ms, seeds, args.offline_qps,
                        str(res), comp_scn, str(tmp / scn))

    (base / "systems" / f"{sysn}.json").write_text(json.dumps(system_description(args.submitter, sysn, args.division, args.status), indent=2))
    (base / "code" / "resnet50" / "aneforge" / "README.md").write_text(
        "Implementation: `bench/mlperf/` in the aneforge repo (loadgen_official.py drives real LoadGen;\n"
        "resnet50.py is the SUT/QSL; the ANE runs fp16 via e5rt).\n")

    # truncate accuracy logs + write hashes (MLPerf requirement)
    trunc = os.path.join(args.inference_repo, "tools", "submission", "truncate_accuracy_log.py")
    if os.path.exists(trunc):
        import subprocess
        print("truncating accuracy logs ...", flush=True)
        subprocess.run([sys.executable, trunc, "--input", str(base.parent.parent), "--submitter", args.submitter,
                        "--backup", str(tmp / "accuracy_backup")], check=False)

    print(f"\n=== submission tree at {base} ===")
    for scn, s in summary.items():
        m = f"p90 {s['p90_ms']:.3f} ms" if scn != "Offline" else f"{s['sps'] or 0:.0f} samples/s"
        print(f"  {scn:13s} perf={s['valid']} ({m})   top-1={s['top1'] * 100:.2f}% / {s['n']}")
    print("Validate:  python3 -m submission_checker.main --input " + str(Path(args.out)) +
          " --version " + args.version + " --submitter " + args.submitter + "  (from inference/tools/submission)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
