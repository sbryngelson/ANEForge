#!/usr/bin/env python3
"""Fused-GPU (MLX) baseline (addresses the "own-tool vs MLX" confound).

Peer-review blocker: the ANE is driven by our fused compiler (aneforge, one program)
while the GPU baseline is *default* MLX, which runs some ops UNFUSED (e.g. layer_norm
as several separate kernel passes). So the device map might reflect TOOLCHAIN
(fused-vs-unfused) rather than SILICON. We fix it with measurement: re-run the
fusion-sensitive workloads with the GPU ALSO fused via mx.compile (MLX graph
capture/kernel fusion), and put default-MLX, fused-MLX, and the ANE side by side.

Workloads (same shapes the paper uses, pulled from device_compare_wattcomplete.py):
  * layer_norm  (197, 768)         - the canonical multi-pass op MLX runs unfused
  * gelu        (197, 768)         - transcendental elementwise
  * attention   (197, 768, 12)     - ViT self-attention block (qkv->softmax->out)
  * conv stack  (64->256, 32x32, k3, depth 16)  - resnet-ish 3x3 stack

For each: default-MLX vs mx.compile-fused-MLX vs ANE (aneforge), reporting
latency (min over reps), idle-subtracted total-package active power + CV%, and
GFLOP/s-or-items/s perf/W. Power harness imported from device_compare_wattcomplete.

THE QUESTION: does fusing the GPU materially change any device-map verdict - does
the GPU close the perf/watt gap, or flip any ANE win? We report it plainly either way.

Run from repo root (energy needs passwordless sudo):

    PYTHONPATH=. python3 bench/fused_gpu_baseline.py

Writes bench/results/fused_gpu_baseline_results.json. --quick reduces the window.
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

import numpy as np

import device_compare as dc            # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
min_latency = dc.min_latency
min_latency_with_out = dc.min_latency_with_out
relerr = dc.relerr

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx

RESULTS: dict[str, dict] = {}


def _mlx_min_latency(build, reps=30, warmup=10):
    """min wall time forcing eval each rep; returns (lat_s, np_out)."""
    def fn():
        mx.eval(build())
    lat = min_latency(fn, reps=reps, warmup=warmup)
    o = build(); mx.eval(o)
    return lat, np.array(o, copy=False)


def _record(wl, variant, *, lat_s, out, ref, flops=None, items=None,
            energy=None, per_inf=False):
    row = {"variant": variant, "lat_ms": lat_s * 1e3,
           "relerr": relerr(out, ref) if out is not None and ref is not None else None}
    if flops is not None:
        row["GFLOPs"] = flops / lat_s / 1e9
    if items is not None:
        row["Gelem_s"] = items / lat_s / 1e9
    if energy:
        apw = energy.get("active_pkg_W")
        row["active_pkg_W"] = apw
        row["pkg_cv_pct"] = energy.get("active_pkg_cv_pct")
        row["energy_iter_ms"] = energy.get("iter_ms")
        row["flags"] = energy.get("flags")
        if apw and apw > 0:
            if flops is not None:
                row["perf_per_W"] = (flops / (energy["iter_ms"] / 1e3)) / 1e9 / apw
                row["perf_unit"] = "GFLOP/s/W"
            elif items is not None:
                row["perf_per_W"] = (items / (energy["iter_ms"] / 1e3)) / 1e9 / apw
                row["perf_unit"] = "Gelem/s/W"
            if per_inf:
                row["mJ_per_inf"] = apw * energy["iter_ms"]
    RESULTS.setdefault(wl, {"rows": []})["rows"].append(row)
    extra = ""
    if "GFLOPs" in row: extra += f"  {row['GFLOPs']:8.1f} GFLOP/s"
    if "Gelem_s" in row: extra += f"  {row['Gelem_s']:7.2f} Gelem/s"
    if energy and energy.get("active_pkg_W"):
        extra += f"  {energy['active_pkg_W']:5.2f} W (CV {energy.get('active_pkg_cv_pct',0):.0f}%)"
        if "perf_per_W" in row: extra += f"  {row['perf_per_W']:7.2f} {row['perf_unit']}"
    re_s = f"  relerr {row['relerr']:.2e}" if row["relerr"] is not None else ""
    print(f"  {variant:<18} {lat_s*1e3:8.3f} ms{extra}{re_s}")


# layer_norm (197,768) - the multi-pass op
def wl_layer_norm(shape=(197, 768), window=6.0):
    wl = f"layer_norm {shape}"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(7)
    x32 = rng.standard_normal(shape).astype(np.float32)
    D = shape[-1]
    g = rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0
    b = rng.standard_normal(D).astype(np.float32) * 0.1
    mu = x32.mean(-1, keepdims=True); var = ((x32 - mu) ** 2).mean(-1, keepdims=True)
    ref = ((x32 - mu) / np.sqrt(var + 1e-5)) * g + b
    items = float(np.prod(shape))

    if HAVE_MLX:
        xg = mx.array(x32.astype(np.float16)); gg = mx.array(g.astype(np.float16)); bb = mx.array(b.astype(np.float16))
        def ln(xg, gg, bb):
            mu = mx.mean(xg, axis=-1, keepdims=True)
            var = mx.mean((xg - mu) ** 2, axis=-1, keepdims=True)
            return (xg - mu) * mx.rsqrt(var + 1e-5) * gg + bb
        # default (unfused: each mx op a separate kernel launch)
        lat, out = _mlx_min_latency(lambda: ln(xg, gg, bb))
        e = wc.measure_energy(lambda: mx.eval(ln(xg, gg, bb)), tag="ln_gpu_default", window=window) if HAVE_SUDO else None
        _record(wl, "GPU default", lat_s=lat, out=out, ref=ref, items=items, energy=e)
        # fused via mx.compile
        cln = mx.compile(ln)
        lat, out = _mlx_min_latency(lambda: cln(xg, gg, bb))
        e = wc.measure_energy(lambda: mx.eval(cln(xg, gg, bb)), tag="ln_gpu_fused", window=window) if HAVE_SUDO else None
        _record(wl, "GPU mx.compile", lat_s=lat, out=out, ref=ref, items=items, energy=e)
    if HAVE_ANE:
        net = af.compile(af.input(shape).layer_norm(g.astype(np.float16), b.astype(np.float16)))
        xf = x32.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        e = wc.measure_energy(lambda: net(xf), tag="ln_ane", window=window) if HAVE_SUDO else None
        _record(wl, "ANE (aneforge)", lat_s=lat, out=np.asarray(out), ref=ref, items=items, energy=e)


# gelu (197,768)
def wl_gelu(shape=(197, 768), window=6.0):
    wl = f"gelu {shape}"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(11)
    x32 = rng.standard_normal(shape).astype(np.float32)
    from math import sqrt, pi
    c = sqrt(2.0 / pi)
    xd = x32.astype(np.float64)
    ref = 0.5 * xd * (1.0 + np.tanh(c * (xd + 0.044715 * xd ** 3)))
    items = float(np.prod(shape))

    if HAVE_MLX:
        xg = mx.array(x32.astype(np.float16))
        def gelu(xg):
            return 0.5 * xg * (1.0 + mx.tanh(c * (xg + 0.044715 * xg ** 3)))
        lat, out = _mlx_min_latency(lambda: gelu(xg))
        e = wc.measure_energy(lambda: mx.eval(gelu(xg)), tag="gelu_gpu_default", window=window) if HAVE_SUDO else None
        _record(wl, "GPU default", lat_s=lat, out=out, ref=ref, items=items, energy=e)
        cg = mx.compile(gelu)
        lat, out = _mlx_min_latency(lambda: cg(xg))
        e = wc.measure_energy(lambda: mx.eval(cg(xg)), tag="gelu_gpu_fused", window=window) if HAVE_SUDO else None
        _record(wl, "GPU mx.compile", lat_s=lat, out=out, ref=ref, items=items, energy=e)
    if HAVE_ANE:
        try:
            net = af.compile(af.input(shape).gelu())
            xf = x32.astype(np.float16)
            lat, out = min_latency_with_out(lambda: net(xf))
            e = wc.measure_energy(lambda: net(xf), tag="gelu_ane", window=window) if HAVE_SUDO else None
            _record(wl, "ANE (aneforge)", lat_s=lat, out=np.asarray(out), ref=ref, items=items, energy=e)
        except Exception as ex:
            RESULTS.setdefault(wl, {"rows": []})["rows"].append({"variant": "ANE", "error": f"{type(ex).__name__}: {ex}"})
            print(f"  ANE FAILED: {type(ex).__name__}: {ex}")


# attention block (197,768,12)
def wl_attention(SEQ=197, DIM=768, HEADS=12, window=6.0):
    wl = f"attention (S={SEQ},D={DIM},H={HEADS})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(5)
    dh = DIM // HEADS; sc = 1.0 / np.sqrt(DIM)
    def mkw(): return rng.standard_normal((DIM, DIM)).astype(np.float32) * sc
    def mkb(): return rng.standard_normal(DIM).astype(np.float32) * 0.01
    Wq, Wk, Wv, Wo = (mkw() for _ in range(4))
    bq, bk, bv, bo = (mkb() for _ in range(4))
    x = rng.standard_normal((SEQ, DIM)).astype(np.float32)

    def ref_attn(xx, dt):
        xx = xx.astype(dt)
        def lin(a, W, b): return a @ W.astype(dt).T + b.astype(dt)
        q = lin(xx, Wq, bq).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        kk = lin(xx, Wk, bk).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        v = lin(xx, Wv, bv).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
        s = (q @ kk.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
        s = s - s.max(-1, keepdims=True); a = np.exp(s); a = a / a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
        return lin(o, Wo, bo)
    ref = ref_attn(x, np.float64)
    flops = 4.0 * 2 * SEQ * DIM * DIM + 2.0 * 2 * HEADS * SEQ * SEQ * dh

    if HAVE_MLX:
        xg = mx.array(x.astype(np.float16))
        Wqg, Wkg, Wvg, Wog = (mx.array(w.T.astype(np.float16)) for w in (Wq, Wk, Wv, Wo))
        bqg, bkg, bvg, bog = (mx.array(b.astype(np.float16)) for b in (bq, bk, bv, bo))
        def attn(xg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog):
            q = (xg @ Wqg + bqg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
            kk = (xg @ Wkg + bkg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
            v = (xg @ Wvg + bvg).reshape(SEQ, HEADS, dh).transpose(1, 0, 2)
            s = (q @ kk.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
            a = mx.softmax(s, axis=-1)
            o = (a @ v).transpose(1, 0, 2).reshape(SEQ, DIM)
            return o @ Wog + bog
        args = (xg, Wqg, Wkg, Wvg, Wog, bqg, bkg, bvg, bog)
        lat, out = _mlx_min_latency(lambda: attn(*args))
        e = wc.measure_energy(lambda: mx.eval(attn(*args)), tag="attn_gpu_default", window=window) if HAVE_SUDO else None
        _record(wl, "GPU default", lat_s=lat, out=np.array(out, copy=False), ref=ref, flops=flops, energy=e)
        ca = mx.compile(attn)
        lat, out = _mlx_min_latency(lambda: ca(*args))
        e = wc.measure_energy(lambda: mx.eval(ca(*args)), tag="attn_gpu_fused", window=window) if HAVE_SUDO else None
        _record(wl, "GPU mx.compile", lat_s=lat, out=np.array(out, copy=False), ref=ref, flops=flops, energy=e)
    if HAVE_ANE:
        y = af.mha(af.input((SEQ, DIM)), Wq.astype(np.float16), bq, Wk.astype(np.float16), bk,
                   Wv.astype(np.float16), bv, Wo.astype(np.float16), bo, HEADS)
        net = af.compile(y)
        xf = x.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        e = wc.measure_energy(lambda: net(xf), tag="attn_ane", window=window) if HAVE_SUDO else None
        _record(wl, "ANE (aneforge)", lat_s=lat, out=np.asarray(out), ref=ref, flops=flops, energy=e)


# resnet-ish conv stack (64->256, 32x32, k3, depth 16)
def wl_conv_stack(Cin=64, Cout=256, H=32, W=32, k=3, depth=16, window=6.0):
    wl = f"conv stack (C={Cout},{H}x{W},k={k},depth={depth})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(1)
    pad = k // 2
    x32 = rng.standard_normal((1, Cin, H, W)).astype(np.float32)
    Ws = [rng.standard_normal((Cout, (Cin if d == 0 else Cout), k, k)).astype(np.float32)
          * np.sqrt(2.0 / ((Cin if d == 0 else Cout) * k * k)) for d in range(depth)]
    flops = sum(2.0 * (Cin if d == 0 else Cout) * Cout * k * k * H * W for d in range(depth))
    ref = dc._np_conv_stack(x32, Ws, pad, relu=True)

    if HAVE_MLX:
        xg = mx.array(np.transpose(x32, (0, 2, 3, 1)).astype(np.float16))
        Wg = [mx.array(np.transpose(w, (0, 2, 3, 1)).astype(np.float16)) for w in Ws]
        def stack(xg, *Wg):
            h = xg
            for w in Wg:
                h = mx.maximum(mx.conv2d(h, w, stride=1, padding=pad), 0)
            return h
        lat, out = _mlx_min_latency(lambda: stack(xg, *Wg))
        outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
        e = wc.measure_energy(lambda: mx.eval(stack(xg, *Wg)), tag="conv_gpu_default", window=window) if HAVE_SUDO else None
        _record(wl, "GPU default", lat_s=lat, out=outn, ref=ref, flops=flops, energy=e)
        cs = mx.compile(stack)
        lat, out = _mlx_min_latency(lambda: cs(xg, *Wg))
        outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
        e = wc.measure_energy(lambda: mx.eval(cs(xg, *Wg)), tag="conv_gpu_fused", window=window) if HAVE_SUDO else None
        _record(wl, "GPU mx.compile", lat_s=lat, out=outn, ref=ref, flops=flops, energy=e)
    if HAVE_ANE:
        h = af.input((1, Cin, H, W))
        for w in Ws:
            h = af.conv(h, w.astype(np.float16), stride=1, pad=pad).relu()
        net = af.compile(h)
        xf = x32.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(xf))
        e = wc.measure_energy(lambda: net(xf), tag="conv_ane", window=window) if HAVE_SUDO else None
        _record(wl, "ANE (aneforge)", lat_s=lat, out=np.asarray(out), ref=ref, flops=flops, energy=e)


# 5-point stencil, 32 chained steps (256x256) - the largest perf/W gap (49x)   #
# in the device map, and exactly the asymmetric-fusion case: default MLX runs   #
# 32 separate conv kernels, the ANE fuses all 32 into one program. Re-test the  #
# GPU with mx.compile fusing the whole 32-step chain so the comparison is fair. #
def wl_stencil(H=256, W=256, steps=32, window=6.0):
    wl = f"stencil 5pt ({H}x{W}, steps={steps})"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(3)
    u0 = rng.standard_normal((1, 1, H, W)).astype(np.float32)
    dt = 0.1
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    K = np.zeros((1, 1, 3, 3), dtype=np.float32)
    K[0, 0] = dt * lap
    K[0, 0, 1, 1] += 1.0                      # identity + dt*Laplacian, one 3x3
    flops = float(steps * 2 * 1 * 1 * 9 * H * W)

    def np_step(u):
        u = u.astype(np.float64)
        for _ in range(steps):
            u = dc._np_conv2d(u.astype(np.float32), K, 1).astype(np.float64)
        return u
    ref = np_step(u0)

    if HAVE_MLX:
        xg = mx.array(np.transpose(u0, (0, 2, 3, 1)).astype(np.float16))
        Kg = mx.array(np.transpose(K, (0, 2, 3, 1)).astype(np.float16))
        def stencil(xg, Kg):
            h = xg
            for _ in range(steps):
                h = mx.conv2d(h, Kg, stride=1, padding=1)
            return h
        lat, out = _mlx_min_latency(lambda: stencil(xg, Kg))
        outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
        e = wc.measure_energy(lambda: mx.eval(stencil(xg, Kg)), tag="stencil_gpu_default", window=window) if HAVE_SUDO else None
        _record(wl, "GPU default", lat_s=lat, out=outn, ref=ref, flops=flops, energy=e)
        cs = mx.compile(stencil)
        lat, out = _mlx_min_latency(lambda: cs(xg, Kg))
        outn = np.transpose(np.array(out, copy=False), (0, 3, 1, 2))
        e = wc.measure_energy(lambda: mx.eval(cs(xg, Kg)), tag="stencil_gpu_fused", window=window) if HAVE_SUDO else None
        _record(wl, "GPU mx.compile", lat_s=lat, out=outn, ref=ref, flops=flops, energy=e)
    if HAVE_ANE:
        h = af.input((1, 1, H, W))
        for _ in range(steps):
            h = af.conv(h, K.astype(np.float16), stride=1, pad=1)
        net = af.compile(h)
        uf = u0.astype(np.float16)
        lat, out = min_latency_with_out(lambda: net(uf))
        e = wc.measure_energy(lambda: net(uf), tag="stencil_ane", window=window) if HAVE_SUDO else None
        _record(wl, "ANE (aneforge)", lat_s=lat, out=np.asarray(out), ref=ref, flops=flops, energy=e)


def _verdict():
    """Compare GPU default vs fused vs ANE; flag any flipped device-map verdict."""
    print("\n" + "=" * 90)
    print(" VERDICT: did fusing the GPU change the device map?")
    print("=" * 90)
    summary = {}
    for wl, data in RESULTS.items():
        rows = {r.get("variant"): r for r in data["rows"] if "error" not in r}
        d = rows.get("GPU default"); f = rows.get("GPU mx.compile"); a = rows.get("ANE (aneforge)")
        if not (d and f):
            continue
        s = {"gpu_default_ms": d["lat_ms"], "gpu_fused_ms": f["lat_ms"],
             "fusion_speedup": d["lat_ms"] / f["lat_ms"] if f["lat_ms"] else None,
             "gpu_default_W": d.get("active_pkg_W"), "gpu_fused_W": f.get("active_pkg_W"),
             "gpu_default_perfW": d.get("perf_per_W"), "gpu_fused_perfW": f.get("perf_per_W")}
        if a:
            s["ane_ms"] = a["lat_ms"]; s["ane_W"] = a.get("active_pkg_W")
            s["ane_perfW"] = a.get("perf_per_W")
            s["ane_faster_than_gpu_default"] = a["lat_ms"] < d["lat_ms"]
            s["ane_faster_than_gpu_fused"] = a["lat_ms"] < f["lat_ms"]
            s["latency_verdict_flipped"] = (a["lat_ms"] < d["lat_ms"]) != (a["lat_ms"] < f["lat_ms"])
            if a.get("perf_per_W") and d.get("perf_per_W") and f.get("perf_per_W"):
                s["ane_efficiency_wins_vs_default"] = a["perf_per_W"] > d["perf_per_W"]
                s["ane_efficiency_wins_vs_fused"] = a["perf_per_W"] > f["perf_per_W"]
                s["efficiency_verdict_flipped"] = \
                    (a["perf_per_W"] > d["perf_per_W"]) != (a["perf_per_W"] > f["perf_per_W"])
        summary[wl] = s
        spd = s["fusion_speedup"]
        print(f"\n {wl}")
        print(f"   GPU default {d['lat_ms']:.3f} ms -> GPU fused {f['lat_ms']:.3f} ms "
              f"(fusion {spd:.2f}x)" + (f" | ANE {a['lat_ms']:.3f} ms" if a else ""))
        flips = [k for k in ("latency_verdict_flipped", "efficiency_verdict_flipped") if s.get(k)]
        if flips:
            print(f"   ** VERDICT FLIPPED: {', '.join(flips)} **")
        elif a:
            lw = "ANE" if s.get("ane_faster_than_gpu_fused") else "GPU"
            ew = ("ANE" if s.get("ane_efficiency_wins_vs_fused") else "GPU") if "ane_efficiency_wins_vs_fused" in s else "?"
            print(f"   no flip: latency winner={lw}, perf/W winner={ew} (same vs default and fused)")
    RESULTS["_verdict_summary"] = summary
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--window", type=float, default=6.0)
    args = ap.parse_args()
    window = 2.0 if args.quick else args.window

    print("=" * 90)
    print(" fused_gpu_baseline - default-MLX vs mx.compile-fused-MLX vs ANE")
    print("=" * 90)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}  "
          f"sudo={'yes' if HAVE_SUDO else 'NO (no power)'}  window={window}s")

    if HAVE_SUDO:
        print("\n sampling idle baseline...", flush=True)
        wc.sample_idle(3.0)
        print(f" idle pkg {wc.IDLE_PKG:.0f} mW")

    wl_layer_norm(window=window)
    wl_gelu(window=window)
    wl_attention(window=window)
    wl_conv_stack(window=window)
    wl_stencil(window=window)

    _verdict()

    out = Path(__file__).resolve().parent / "results" / "fused_gpu_baseline_results.json"
    out.write_text(json.dumps({
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
        "window_s": window, "idle_pkg_mW": wc.IDLE_PKG, "idle_mW": wc.IDLE,
        "results": RESULTS,
    }, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
