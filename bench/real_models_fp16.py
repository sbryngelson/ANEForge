#!/usr/bin/env python3
"""fp16 GPU baselines for the real models (addresses the fp16-ANE-vs-fp32-GPU
asymmetry in the real-model table).

The single-stream real-model rows compared an fp16 ANE against an fp32 torch-MPS
GPU, so the headline energy ratios (e.g. ViT-B/16 ~18x mJ/inference) were not
like-for-like. This script re-runs ResNet-18, ViT-B/16, and MiniLM on the GPU in
BOTH fp32 and fp16 (torch-MPS .half()), under the same watt-complete harness, so
the fp16-vs-fp16 GPU/ANE energy ratio can be reported. It reuses the exact model
builders from device_compare_wattcomplete.py.

Run from repo root (energy needs passwordless sudo):
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 bench/real_models_fp16.py --window 6
Writes bench/results/real_models_fp16_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import aneforge as af  # noqa: E402
import device_compare as dc  # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402

HAVE_SUDO = dc.HAVE_SUDO
# GPU-only: the ANE real-model numbers are unchanged and read from the committed
# device-map JSON, so we never re-dispatch the ANE here (avoids session-state hangs
# and keeps the ANE pairing identical to the headline table).
HAVE_ANE = False
min_latency_with_out = dc.min_latency_with_out
relerr = dc.relerr

# ANE real-model energy from the committed single-stream run (the paper's M5 table).
_ANE_DEVMAP = json.loads((Path(__file__).resolve().parent / "results" /
    "device_compare_wattcomplete_results_M5.json").read_text())
def _ane_mJ(wl_substr):
    for k, v in _ANE_DEVMAP["results"].items():
        if wl_substr in k:
            e = v.get("energy", {}).get("ANE", {})
            return e.get("mJ_per_inf"), e.get("active_pkg_W")
    return None, None

RESULTS: dict[str, dict] = {}


def _energy(run_once, *, window, tag):
    if not (HAVE_SUDO and window > 0):
        return None
    return wc.measure_energy(run_once, tag=tag, window=window)


def _row(wl, device, dtype, *, lat_s, relerr_v, energy=None):
    r = {"device": device, "dtype": dtype, "lat_ms": lat_s * 1e3, "relerr": relerr_v}
    if energy and energy.get("active_pkg_W"):
        r["active_pkg_W"] = energy["active_pkg_W"]
        r["pkg_cv_pct"] = energy.get("active_pkg_cv_pct")
        r["mJ_per_inf"] = energy["active_pkg_W"] * energy["iter_ms"]
    RESULTS.setdefault(wl, {"rows": []})["rows"].append(r)
    extra = ""
    if "mJ_per_inf" in r:
        extra = f"  {r['active_pkg_W']:5.2f} W (CV {r.get('pkg_cv_pct',0):.0f}%)  {r['mJ_per_inf']:7.1f} mJ/inf"
    print(f"  {device+' '+dtype:<16} {lat_s*1e3:8.3f} ms{extra}"
          f"  relerr {relerr_v:.2e}" if relerr_v is not None else
          f"  {device+' '+dtype:<16} {lat_s*1e3:8.3f} ms{extra}")


def resnet18(window):
    import torch, torchvision as tv
    wl = "ResNet-18 forward (1x3x224x224)"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(6)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    m = tv.models.resnet18(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0].astype(np.float64)
    if torch.backends.mps.is_available():
        for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            mm = m.to("mps").to(dt)
            ti = torch.from_numpy(img).to("mps").to(dt)
            def fwd():
                with torch.no_grad():
                    o = mm(ti); torch.mps.synchronize()
                    return o.float().to("cpu").numpy()[0]
            lat, out = min_latency_with_out(fwd, reps=15, warmup=5)
            e = _energy(fwd, window=window, tag=f"resnet_mps_{name}")
            _row(wl, "GPU(MPS)", name, lat_s=lat, relerr_v=relerr(out, ref), energy=e)
    if HAVE_ANE:
        clf = af.load_resnet18()
        lat, out = min_latency_with_out(lambda: clf(img), reps=15, warmup=5)
        e = _energy(lambda: clf(img), window=window, tag="resnet_ane")
        _row(wl, "ANE", "fp16", lat_s=lat, relerr_v=relerr(out, ref), energy=e)


def vit_b16(window):
    """Full ViT-B/16 (torchvision, 12-layer) on GPU fp32 + fp16. GPU-only: the ANE
    ViT-B/16 energy is read from the committed device-map run (the paper's table)."""
    import torch, torchvision as tv
    wl = "ViT-B/16 forward (1x3x224x224, 197 tokens)"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    m = tv.models.vit_b_16(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0].astype(np.float64)
    if torch.backends.mps.is_available():
        for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            mm = m.to("mps").to(dt)
            ti = torch.from_numpy(img).to("mps").to(dt)
            def fwd():
                with torch.no_grad():
                    o = mm(ti); torch.mps.synchronize()
                    return o.float().to("cpu").numpy()[0]
            lat, out = min_latency_with_out(fwd, reps=12, warmup=4)
            e = _energy(fwd, window=window, tag=f"vit_mps_{name}")
            _row(wl, "GPU(MPS)", name, lat_s=lat, relerr_v=relerr(out, ref), energy=e)


def minilm(window):
    import torch
    wl = "MiniLM encoder (1 sentence)"
    print(f"\n=== {wl} ===", flush=True)
    NAME = "sentence-transformers/all-MiniLM-L6-v2"
    text = "The Apple Neural Engine is a specialized accelerator for matrix math."
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    hf = AutoModel.from_pretrained(NAME).eval()
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        hs = hf(**ids).last_hidden_state[0].numpy()
    ref = hs.mean(0); ref = (ref / np.linalg.norm(ref)).astype(np.float64)
    if torch.backends.mps.is_available():
        for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            mm = hf.to("mps").to(dt)
            mids = {k: v.to("mps") for k, v in ids.items()}
            def fwd():
                with torch.no_grad():
                    v = mm(**mids).last_hidden_state[0].mean(0).float().to("cpu").numpy()
                    torch.mps.synchronize()
                    return v / np.linalg.norm(v)
            lat, out = min_latency_with_out(fwd, reps=15, warmup=5)
            e = _energy(fwd, window=window, tag=f"minilm_mps_{name}")
            _row(wl, "GPU(MPS)", name, lat_s=lat, relerr_v=relerr(out.astype(np.float64), ref), energy=e)
        hf = hf.to("cpu").float()
    if HAVE_ANE:
        enc = af.load(NAME); enc(text)
        lat, _ = min_latency_with_out(lambda: enc(text), reps=15, warmup=3)
        out = enc(text)[0].astype(np.float64)
        e = _energy(lambda: enc(text), window=window, tag="minilm_ane")
        _row(wl, "ANE", "fp16", lat_s=lat, relerr_v=relerr(out, ref), energy=e)


def summarize():
    print("\n" + "=" * 80)
    print(" fp16-vs-fp16 GPU/ANE energy ratios (the like-for-like comparison)")
    print("=" * 80)
    summ = {}
    sub = {"ResNet-18 forward (1x3x224x224)": "ResNet",
           "ViT-B/16 forward (1x3x224x224, 197 tokens)": "ViT",
           "MiniLM encoder (1 sentence)": "MiniLM"}
    for wl, data in RESULTS.items():
        rows = data["rows"]
        ane_mJ, ane_W = _ane_mJ(sub.get(wl, "###"))
        gpu16 = next((r for r in rows if r["device"] == "GPU(MPS)" and r["dtype"] == "fp16"), None)
        gpu32 = next((r for r in rows if r["device"] == "GPU(MPS)" and r["dtype"] == "fp32"), None)
        s = {"ane_mJ_per_inf": ane_mJ}
        if ane_mJ:
            if gpu32 and gpu32.get("mJ_per_inf"):
                s["energy_ratio_vs_fp32"] = gpu32["mJ_per_inf"] / ane_mJ
            if gpu16 and gpu16.get("mJ_per_inf"):
                s["energy_ratio_vs_fp16"] = gpu16["mJ_per_inf"] / ane_mJ
        summ[wl] = s
        if s.get("energy_ratio_vs_fp16") or s.get("energy_ratio_vs_fp32"):
            print(f"\n {wl}  (ANE {ane_mJ:.1f} mJ/inf, committed)")
            if "energy_ratio_vs_fp32" in s:
                print(f"   ANE mJ/inf vs GPU-fp32: {s['energy_ratio_vs_fp32']:.1f}x")
            if "energy_ratio_vs_fp16" in s:
                print(f"   ANE mJ/inf vs GPU-fp16: {s['energy_ratio_vs_fp16']:.1f}x")
    RESULTS["_summary"] = summ
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=6.0)
    args = ap.parse_args()
    print("=" * 80)
    print(" real_models_fp16 — fp32 vs fp16 GPU baselines vs ANE")
    print("=" * 80)
    if HAVE_SUDO and args.window > 0:
        wc.sample_idle(3.0)
        print(f" idle pkg {wc.IDLE_PKG:.0f} mW")
    for fn in (resnet18, minilm, vit_b16):
        try:
            fn(args.window)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {fn.__name__} FAILED: {type(e).__name__}: {e}")
    summarize()
    out = Path(__file__).resolve().parent / "results" / "real_models_fp16_results.json"
    out.write_text(json.dumps({"window_s": args.window, "idle_pkg_mW": wc.IDLE_PKG,
                               "results": RESULTS}, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
