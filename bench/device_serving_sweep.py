#!/usr/bin/env python3
"""Batched SERVING sweep - ANE (aneforge fp16) vs GPU (MLX fp16) across BATCH SIZE.

The serving-regime complement to the single-stream device map
(``device_compare_wattcomplete.py``). That harness answers "which device per op at
B=1". This one answers the serving question: *as you batch, where does the GPU's
throughput (and its throughput/watt) overtake the ANE's?* - the crossover that
decides which accelerator a serving deployment should target at a given batch size.

METHODOLOGY - reused verbatim from device_compare_wattcomplete (see that file's
docstring for the full rationale); this script imports its energy harness so the
numbers are produced by the SAME code:

  * idle-subtracted ACTIVE power; headline = total-package power from powermetrics'
    own ``Combined Power (CPU + GPU + ANE)`` line (NOT a hand-summed rail total -
    the GPU rail prints twice per sample block);
  * median + CV% over the per-sample package totals (CV>35% is FLAGGED low-confidence);
  * sustained loop driven until the sampler process EXITS (the fix that got CV to
    1-25%); warmup before the window;
  * accuracy (relerr vs an fp32/fp64 numpy reference) reported alongside; fp16 both
    sides; forced device sync (compiled-net call on ANE, ``mx.eval`` on GPU).

TRUE BATCHING ON BOTH SIDES: each workload is ONE compiled program with a real
batch dimension B (NOT multi-stream). The ANE true-batches - dispatch amortizes
over B - so a real batch dim is the fair apples-to-apples vs GPU batching. We sweep
B in {1, 4, 16, 64, 256}; if an ANE workload fails to compile / OOMs at some B we
report the cap as a finding (we do NOT silently drop it or fall back to multi-stream).

WORKLOADS (4, serving-relevant):
  1. vision    - a 3x3 conv stack -> GAP -> FC, image-classifier serving (N=B images,
                 conv's native batch dim).
  2. encoder   - a transformer-encoder block (attn + MLP + 2 layernorms) at S=128,
                 batched over B sequences; embedding serving. Batched as a rank-3/4
                 graph (layernorm folds B into rows [B*S, D]).
  3. attention - the self-attention block alone at S=128, batched over B (rank-4
                 q@k / softmax / @v). The decomposed-SDPA route, true-batched.
  4. gemm      - the batched GEMM [B,M,K] @ [K,N] underlying serving, the throughput
                 primitive.

METRICS per (workload, B, device):
  * throughput = items/s (images or sequences per second) and latency/call (ms);
  * total-package ACTIVE power (idle-subtracted, median + CV%), sustained loop;
  * throughput/watt = items/s / active_W;  energy/item = active_W * call_ms / B (mJ);
  * accuracy = relerr vs fp32 reference.

DELIVERABLE = the two crossovers per workload: the B where GPU THROUGHPUT overtakes
ANE, and the B where GPU THROUGHPUT/WATT overtakes ANE (typically a larger B - the
ANE's watt advantage persists past its throughput advantage). Printed explicitly.

Run from repo root (energy needs passwordless sudo for powermetrics)::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 bench/device_serving_sweep.py

Writes bench/results/device_serving_sweep_results.json. --quick = short window;
--batches "1,4,16" overrides the sweep; --window N overrides the per-loop seconds.
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

import numpy as np

# Reuse the wattcomplete energy harness + device_compare timing/precision helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_compare as dc          # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
relerr = dc.relerr
min_latency_with_out = dc.min_latency_with_out

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx

BATCHES = [1, 4, 16, 64, 256]

# Results: per workload, per (device, B) point with latency/throughput/relerr/energy.
RESULTS: dict[str, dict] = {}


# one measured point: latency (min over reps) + sustained-loop energy
def measure_point(wl, device, B, run_once, *, items_per_call, relerr_val, tag):
    """Time one (device,B) point and run the reused energy harness on it.

    items_per_call = the number of serving items processed by one run_once() call
    (= B here, since the whole batch is one call). throughput = items/s, perf/watt
    and energy/item are derived from the sustained-loop active package power."""
    lat, _out = min_latency_with_out(run_once, reps=20, warmup=6)
    thr = items_per_call / lat                                    # items/s
    point = {"device": device, "B": B, "latency_ms": lat * 1e3,
             "throughput_items_s": thr, "relerr": relerr_val}
    # reuse the wattcomplete sustained-loop power harness verbatim
    e = wc.measure_energy(run_once, tag=tag, window=wc.WINDOW)
    if e is not None:
        apw = e.get("active_pkg_W", float("nan"))
        sane = apw == apw and apw > 0
        # plausibility guard, same as wattcomplete's _attach_energy
        if device == "ANE" and e.get("ane_active_mW", 0.0) < 5.0 and not e["flags"]:
            e["flags"].append("ANE rail ~0 mW during ANE workload - likely a 100ms sampling miss")
        # the loop's own per-iter time is the steady-state call time; throughput/watt
        # and energy/item use it (the min-latency above is the headline call time).
        loop_items_s = items_per_call / (e["iter_ms"] / 1e3)
        if sane:
            e["throughput_items_s"] = loop_items_s
            e["perf_per_W"] = loop_items_s / apw                  # items/s/W
            e["mJ_per_item"] = apw * e["iter_ms"] / items_per_call  # W*ms/items = mJ/item
        point["energy"] = e
        point["active_pkg_W"] = apw
        point["perf_per_W"] = e.get("perf_per_W")
        point["mJ_per_item"] = e.get("mJ_per_item")
        point["pkg_cv_pct"] = e.get("active_pkg_cv_pct")
        point["flags"] = e.get("flags", [])
    RESULTS.setdefault(wl, {"points": [], "note": ""})["points"].append(point)
    fl = (" FLAG:" + ";".join(point.get("flags", []))) if point.get("flags") else ""
    print(f"  {device:<4} B={B:<4} lat={lat*1e3:8.3f}ms  thr={thr:10.1f} it/s  "
          f"pkgW={point.get('active_pkg_W', float('nan')):5.2f}  "
          f"perf/W={point.get('perf_per_W') or float('nan'):8.1f} it/s/W  "
          f"mJ/it={point.get('mJ_per_item') or float('nan'):7.2f}  relerr={relerr_val:.2e}{fl}",
          flush=True)


def note(wl, txt):
    RESULTS.setdefault(wl, {"points": [], "note": ""})["note"] = txt


def cap(wl, device, B, exc):
    print(f"  {device:<4} B={B:<4} CAP/FAIL: {type(exc).__name__}: {exc}", flush=True)
    RESULTS.setdefault(wl, {"points": [], "note": ""}).setdefault("caps", []).append(
        {"device": device, "B": B, "error": f"{type(exc).__name__}: {exc}"})


# WORKLOADS - TRUE-batched on both devices (one program, real batch dim B)
def wl_vision(batches):
    """Conv classifier: 3x3 conv stack -> GAP -> FC. N=B images (conv native batch)."""
    wl = "vision (conv-stack->GAP->FC classifier)"
    print(f"\n=== {wl} ===", flush=True)
    rng = np.random.default_rng(1)
    Cin, Cout, Hh, Ww, k, depth, classes = 16, 64, 32, 32, 3, 6, 100
    pad = k // 2
    Ws = [rng.standard_normal((Cout, (Cin if d == 0 else Cout), k, k)).astype(np.float32)
          * np.sqrt(2.0 / ((Cin if d == 0 else Cout) * k * k)) for d in range(depth)]
    Wfc = (rng.standard_normal((classes, Cout)).astype(np.float32) / np.sqrt(Cout))
    note(wl, f"{depth}x 3x3 conv (Cin={Cin}->Cout={Cout}, {Hh}x{Ww}) + GAP + FC[{classes}]. "
             f"items = images. ref = fp32 numpy. seq=N/A.")

    def np_ref(x):
        h = dc._np_conv_stack(x.astype(np.float32), Ws, pad, relu=True)
        g = h.mean((2, 3))                        # GAP -> [B, Cout]
        return g @ Wfc.T

    for B in batches:
        xb = rng.standard_normal((B, Cin, Hh, Ww)).astype(np.float32)
        ref = np_ref(xb).astype(np.float64)
        if HAVE_ANE:
            try:
                h = af.input((B, Cin, Hh, Ww))
                for w in Ws:
                    h = af.conv(h, w.astype(np.float16), stride=1, pad=pad).relu()
                g = h.mean((2, 3)).reshape(B, Cout)          # GAP
                net = af.compile(g.linear(Wfc.astype(np.float16)))
                xf = xb.astype(np.float16)
                out = net(xf)
                measure_point(wl, "ANE", B, lambda xf=xf, net=net: net(xf),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"vision_ane_B{B}")
            except Exception as e:
                cap(wl, "ANE", B, e)
        if HAVE_MLX:
            try:
                xg = mx.array(np.transpose(xb, (0, 2, 3, 1)).astype(np.float16))
                Wg = [mx.array(np.transpose(w, (0, 2, 3, 1)).astype(np.float16)) for w in Ws]
                Wfcg = mx.array(Wfc.T.astype(np.float16))

                def run(xg=xg, Wg=Wg, Wfcg=Wfcg):
                    hh = xg
                    for w in Wg:
                        hh = mx.maximum(mx.conv2d(hh, w, stride=1, padding=pad), 0)
                    g = mx.mean(hh, axis=(1, 2))             # GAP over H,W (NHWC)
                    return g @ Wfcg
                out = np.array(run(), copy=False)
                measure_point(wl, "GPU", B, lambda run=run: mx.eval(run()),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"vision_gpu_B{B}")
            except Exception as e:
                cap(wl, "GPU", B, e)


def _build_encoder_ane(B, S, D, H, Wq, bq, Wk, bk, Wv, bv, Wo, bo, W1, b1, W2, b2,
                       g1, bn1, g2, bn2):
    """One batched transformer-encoder block as an aneforge graph (rank-4 attn,
    layernorms folded to [B*S, D]). Returns the compiled Model. fp16 weights."""
    dh = D // H
    x = af.input((B, S, D))
    # pre-LN attention
    xf = x.reshape(B * S, D).layer_norm(g1, bn1).reshape(B, S, D)
    q, k, v = xf.linear(Wq, bq), xf.linear(Wk, bk), xf.linear(Wv, bv)

    def heads(t):
        return t.reshape(B, S, H, dh).transpose([0, 2, 1, 3])   # [B,H,S,dh]
    qh, kh, vh = heads(q), heads(k), heads(v)
    a = ((qh @ kh.transpose([0, 1, 3, 2])) * (1.0 / dh ** 0.5)).softmax(-1)  # [B,H,S,S]
    o = (a @ vh).transpose([0, 2, 1, 3]).reshape(B, S, D).linear(Wo, bo)
    x = x + o
    # pre-LN MLP (GELU)
    xf2 = x.reshape(B * S, D).layer_norm(g2, bn2).reshape(B, S, D)
    m = xf2.linear(W1, b1).gelu().linear(W2, b2)
    return af.compile(x + m)


def wl_encoder(batches):
    """Transformer-encoder block (ViT-ish), S=128, batched over B sequences."""
    wl = "encoder (transformer block, S=128)"
    print(f"\n=== {wl} ===", flush=True)
    S, D, H, Dff = 128, 768, 12, 3072
    dh = D // H
    rng = np.random.default_rng(5)
    sc = 1.0 / np.sqrt(D)

    def mkw(o, i): return (rng.standard_normal((o, i)).astype(np.float32) * sc)
    def mkb(o): return (rng.standard_normal(o).astype(np.float32) * 0.01)
    Wq, Wk, Wv, Wo = (mkw(D, D) for _ in range(4))
    bq, bk, bv, bo = (mkb(D) for _ in range(4))
    W1, b1 = mkw(Dff, D), mkb(Dff)
    W2, b2 = mkw(D, Dff), mkb(D)
    g1 = rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0
    bn1 = rng.standard_normal(D).astype(np.float32) * 0.1
    g2 = rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0
    bn2 = rng.standard_normal(D).astype(np.float32) * 0.1
    note(wl, f"pre-LN encoder block S={S}, D={D}, H={H}, Dff={Dff}; items = sequences. "
             f"layernorm folds B into rows [B*S,D]. ref = fp64 numpy.")

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * g + b

    def np_ref(x):
        x = x.astype(np.float64)
        def lin(a, W, bb): return a @ W.astype(np.float64).T + bb.astype(np.float64)
        B = x.shape[0]
        xf = ln(x, g1, bn1)
        q = lin(xf, Wq, bq).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        k = lin(xf, Wk, bk).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        v = lin(xf, Wv, bv).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
        s = s - s.max(-1, keepdims=True); ea = np.exp(s); a = ea / ea.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        o = lin(o, Wo, bo); x = x + o
        xf2 = ln(x, g2, bn2)
        gelu = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
        m = lin(gelu(lin(xf2, W1, b1)), W2, b2)
        return x + m

    f16 = lambda a: a.astype(np.float16)
    for B in batches:
        xb = rng.standard_normal((B, S, D)).astype(np.float32)
        ref = np_ref(xb)
        if HAVE_ANE:
            try:
                net = _build_encoder_ane(B, S, D, H, f16(Wq), bq, f16(Wk), bk, f16(Wv), bv,
                                         f16(Wo), bo, f16(W1), b1, f16(W2), b2,
                                         f16(g1), f16(bn1), f16(g2), f16(bn2))
                xf = xb.astype(np.float16)
                out = net(xf)
                measure_point(wl, "ANE", B, lambda xf=xf, net=net: net(xf),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"encoder_ane_B{B}")
            except Exception as e:
                cap(wl, "ANE", B, e)
        if HAVE_MLX:
            try:
                xg = mx.array(xb.astype(np.float16))
                Wqg, Wkg, Wvg, Wog = (mx.array(w.T.astype(np.float16)) for w in (Wq, Wk, Wv, Wo))
                bqg, bkg, bvg, bog = (mx.array(b.astype(np.float16)) for b in (bq, bk, bv, bo))
                W1g, W2g = mx.array(W1.T.astype(np.float16)), mx.array(W2.T.astype(np.float16))
                b1g, b2g = mx.array(b1.astype(np.float16)), mx.array(b2.astype(np.float16))
                g1g, bn1g = mx.array(g1.astype(np.float16)), mx.array(bn1.astype(np.float16))
                g2g, bn2g = mx.array(g2.astype(np.float16)), mx.array(bn2.astype(np.float16))

                def mln(z, g, b):
                    mu = mx.mean(z, axis=-1, keepdims=True)
                    var = mx.mean((z - mu) ** 2, axis=-1, keepdims=True)
                    return (z - mu) * mx.rsqrt(var + 1e-5) * g + b

                def mgelu(z):  # tanh approximation - matches the fp64 reference's gelu
                    return 0.5 * z * (1 + mx.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

                def run():
                    xf = mln(xg, g1g, bn1g)
                    q = (xf @ Wqg + bqg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    k = (xf @ Wkg + bkg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    v = (xf @ Wvg + bvg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
                    a = mx.softmax(s, axis=-1)
                    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
                    o = o @ Wog + bog
                    x2 = xg + o
                    xf2 = mln(x2, g2g, bn2g)
                    m = mgelu(xf2 @ W1g + b1g) @ W2g + b2g
                    return x2 + m
                out = np.array(run(), copy=False)
                measure_point(wl, "GPU", B, lambda run=run: mx.eval(run()),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"encoder_gpu_B{B}")
            except Exception as e:
                cap(wl, "GPU", B, e)


def wl_attention(batches):
    """Self-attention block alone (Q/K/V proj -> SDPA -> O proj), S=128, batched."""
    wl = "attention (self-attn block, S=128)"
    print(f"\n=== {wl} ===", flush=True)
    S, D, H = 128, 768, 12
    dh = D // H
    rng = np.random.default_rng(7)
    sc = 1.0 / np.sqrt(D)
    def mkw(): return (rng.standard_normal((D, D)).astype(np.float32) * sc)
    def mkb(): return (rng.standard_normal(D).astype(np.float32) * 0.01)
    Wq, Wk, Wv, Wo = (mkw() for _ in range(4))
    bq, bk, bv, bo = (mkb() for _ in range(4))
    note(wl, f"self-attention block S={S}, D={D}, H={H}; items = sequences. "
             f"decomposed-SDPA, rank-4 true batch. ref = fp64 numpy.")

    def np_ref(x):
        x = x.astype(np.float64)
        def lin(a, W, bb): return a @ W.astype(np.float64).T + bb.astype(np.float64)
        B = x.shape[0]
        q = lin(x, Wq, bq).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        k = lin(x, Wk, bk).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        v = lin(x, Wv, bv).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
        s = s - s.max(-1, keepdims=True); ea = np.exp(s); a = ea / ea.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
        return lin(o, Wo, bo)

    f16 = lambda a: a.astype(np.float16)
    for B in batches:
        xb = rng.standard_normal((B, S, D)).astype(np.float32)
        ref = np_ref(xb)
        if HAVE_ANE:
            try:
                x = af.input((B, S, D))
                q, k, v = x.linear(f16(Wq), bq), x.linear(f16(Wk), bk), x.linear(f16(Wv), bv)
                def heads(t): return t.reshape(B, S, H, dh).transpose([0, 2, 1, 3])
                qh, kh, vh = heads(q), heads(k), heads(v)
                a = ((qh @ kh.transpose([0, 1, 3, 2])) * (1.0 / dh ** 0.5)).softmax(-1)
                o = (a @ vh).transpose([0, 2, 1, 3]).reshape(B, S, D)
                net = af.compile(o.linear(f16(Wo), bo))
                xf = xb.astype(np.float16)
                out = net(xf)
                measure_point(wl, "ANE", B, lambda xf=xf, net=net: net(xf),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"attn_ane_B{B}")
            except Exception as e:
                cap(wl, "ANE", B, e)
        if HAVE_MLX:
            try:
                xg = mx.array(xb.astype(np.float16))
                Wqg, Wkg, Wvg, Wog = (mx.array(w.T.astype(np.float16)) for w in (Wq, Wk, Wv, Wo))
                bqg, bkg, bvg, bog = (mx.array(b.astype(np.float16)) for b in (bq, bk, bv, bo))

                def run():
                    q = (xg @ Wqg + bqg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    k = (xg @ Wkg + bkg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    v = (xg @ Wvg + bvg).reshape(B, S, H, dh).transpose(0, 2, 1, 3)
                    s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
                    a = mx.softmax(s, axis=-1)
                    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
                    return o @ Wog + bog
                out = np.array(run(), copy=False)
                measure_point(wl, "GPU", B, lambda run=run: mx.eval(run()),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"attn_gpu_B{B}")
            except Exception as e:
                cap(wl, "GPU", B, e)


def wl_gemm(batches):
    """Batched GEMM [B,M,K] @ [K,N], the serving throughput primitive. items = B*M
    rows (we report items = B 'sequences' of M rows for a clean serving metric)."""
    wl = "gemm (batched [B,M,K]@[K,N], M=128,K=1024,N=1024)"
    print(f"\n=== {wl} ===", flush=True)
    M, K, N = 128, 1024, 1024
    rng = np.random.default_rng(0)
    W32 = (rng.standard_normal((N, K)).astype(np.float32) / np.sqrt(K))
    note(wl, f"M={M}, K={K}, N={N}. items = batch elements (each = M x K @ K x N). "
             f"ref = fp64 numpy. seq=N/A.")
    for B in batches:
        xb = (rng.standard_normal((B, M, K)).astype(np.float32) / np.sqrt(K))
        ref = (xb.astype(np.float64) @ W32.astype(np.float64).T)
        if HAVE_ANE:
            try:
                net = af.compile(af.input((B, M, K)).linear(W32.astype(np.float16)))
                xf = xb.astype(np.float16)
                out = net(xf)
                measure_point(wl, "ANE", B, lambda xf=xf, net=net: net(xf),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"gemm_ane_B{B}")
            except Exception as e:
                cap(wl, "ANE", B, e)
        if HAVE_MLX:
            try:
                xg = mx.array(xb.astype(np.float16))
                Wg = mx.array(W32.T.astype(np.float16))
                out = np.array(xg @ Wg, copy=False)
                measure_point(wl, "GPU", B, lambda xg=xg, Wg=Wg: mx.eval(xg @ Wg),
                              items_per_call=B, relerr_val=relerr(out, ref),
                              tag=f"gemm_gpu_B{B}")
            except Exception as e:
                cap(wl, "GPU", B, e)


# crossover analysis + reporting
def _series(points, device, key):
    """sorted [(B, value)] for one device/metric, dropping low-confidence flagged."""
    out = []
    for p in points:
        if p["device"] != device:
            continue
        v = p.get(key)
        if v is None or (isinstance(v, float) and v != v):
            continue
        out.append((p["B"], v))
    return sorted(out)


def _crossover(ane, gpu):
    """First B where GPU >= ANE (linear-interpolated in log2 B). Returns
    (crossover_B or None, verdict_string)."""
    ad = dict(ane); gd = dict(gpu)
    common = sorted(set(ad) & set(gd))
    if not common:
        return None, "no common B"
    diffs = [(B, gd[B] - ad[B]) for B in common]   # GPU - ANE
    if all(d <= 0 for _, d in diffs):
        return None, "ANE wins at all measured B (never overtaken in range)"
    if all(d >= 0 for _, d in diffs):
        return common[0], f"GPU wins at all measured B (>= B={common[0]})"
    # find first sign change from ANE-wins (d<0) to GPU-wins (d>=0)
    for i in range(1, len(diffs)):
        b0, d0 = diffs[i - 1]; b1, d1 = diffs[i]
        if d0 < 0 <= d1:
            # log2-B interpolation of the crossover point
            l0, l1 = np.log2(b0), np.log2(b1)
            frac = -d0 / (d1 - d0) if (d1 - d0) != 0 else 0.0
            bx = 2 ** (l0 + frac * (l1 - l0))
            return float(bx), f"GPU overtakes ANE between B={b0} and B={b1} (~B={bx:.0f})"
    return None, "ANE wins at all measured B (never overtaken in range)"


def analyze_and_print():
    print("\n" + "=" * 100)
    print(" BATCHED SERVING SWEEP - crossover analysis")
    print(" throughput = items/s (steady-state loop); perf/W = items/s per active-package-W")
    print("=" * 100)
    if HAVE_SUDO:
        print(f" idle baseline (median mW): ANE {wc.IDLE.get('ane',0):.0f} | GPU {wc.IDLE.get('gpu',0):.0f}"
              f" | CPU {wc.IDLE.get('cpu',0):.0f} | package {wc.IDLE_PKG:.0f}")
    summary = {}
    for wl, data in RESULTS.items():
        pts = data.get("points", [])
        print(f"\n{wl}")
        if data.get("note"):
            print(f"  {data['note']}")
        # per-B table
        print(f"  {'B':>5} | {'ANE thr':>11} {'GPU thr':>11} | {'ANE p/W':>10} {'GPU p/W':>10}"
              f" | {'ANE mJ/it':>9} {'GPU mJ/it':>9} | {'ANE rel':>8} {'GPU rel':>8}")
        print("  " + "-" * 96)
        Bs = sorted({p["B"] for p in pts})
        for B in Bs:
            a = next((p for p in pts if p["device"] == "ANE" and p["B"] == B), None)
            g = next((p for p in pts if p["device"] == "GPU" and p["B"] == B), None)
            def gv(p, k, f="{:.1f}"):
                if p is None or p.get(k) is None or (isinstance(p.get(k), float) and p[k] != p[k]):
                    return "   -   "
                return f.format(p[k])
            print(f"  {B:>5} | {gv(a,'throughput_items_s'):>11} {gv(g,'throughput_items_s'):>11} |"
                  f" {gv(a,'perf_per_W'):>10} {gv(g,'perf_per_W'):>10} |"
                  f" {gv(a,'mJ_per_item','{:.2f}'):>9} {gv(g,'mJ_per_item','{:.2f}'):>9} |"
                  f" {gv(a,'relerr','{:.1e}'):>8} {gv(g,'relerr','{:.1e}'):>8}")
        # caps
        for c in data.get("caps", []):
            print(f"  CAP: {c['device']} B={c['B']}: {c['error']}")
        # crossovers (use steady-state loop throughput for thr; perf_per_W for watts)
        thr_x, thr_v = _crossover(_series(pts, "ANE", "throughput_items_s"),
                                  _series(pts, "GPU", "throughput_items_s"))
        pw_x, pw_v = _crossover(_series(pts, "ANE", "perf_per_W"),
                                _series(pts, "GPU", "perf_per_W"))
        print(f"  >> THROUGHPUT crossover: {thr_v}")
        print(f"  >> THROUGHPUT/WATT crossover: {pw_v}")
        # flag any low-confidence points
        flagged = [(p["device"], p["B"], p.get("flags")) for p in pts if p.get("flags")]
        for dev, B, fl in flagged:
            print(f"  !! flagged {dev} B={B}: {';'.join(fl)}")
        summary[wl] = {"throughput_crossover_B": thr_x, "throughput_verdict": thr_v,
                       "perf_per_W_crossover_B": pw_x, "perf_per_W_verdict": pw_v,
                       "flagged": [{"device": d, "B": b, "flags": f} for d, b, f in flagged]}
    RESULTS["_summary"] = summary
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--batches", type=str, default=None, help='e.g. "1,4,16"')
    args = ap.parse_args()
    if args.quick:
        wc.WINDOW = 2.5
    if args.window:
        wc.WINDOW = args.window
    batches = [int(x) for x in args.batches.split(",")] if args.batches else BATCHES

    print("=" * 100)
    print(" device_serving_sweep - ANE vs MLX-GPU, batched-serving crossover")
    print("=" * 100)
    print(f" backends: ANE={'yes' if HAVE_ANE else 'NO'}  MLX={'yes' if HAVE_MLX else 'NO'}")
    print(f" powermetrics(sudo)={'yes' if HAVE_SUDO else 'NO - energy skipped'}  "
          f"window={wc.WINDOW}s  batches={batches}")

    if HAVE_SUDO:
        print("\n sampling idle baseline (no workload)...", flush=True)
        wc.sample_idle(3.0)
        print(f" idle: ANE {wc.IDLE.get('ane',0):.0f} / GPU {wc.IDLE.get('gpu',0):.0f} / "
              f"CPU {wc.IDLE.get('cpu',0):.0f} mW (pkg {wc.IDLE_PKG:.0f} mW)")

    wl_vision(batches)
    wl_encoder(batches)
    wl_attention(batches)
    wl_gemm(batches)

    summary = analyze_and_print()

    out = Path(__file__).resolve().parent / "results" / "device_serving_sweep_results.json"
    out.write_text(json.dumps({
        "backends": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
        "window_s": wc.WINDOW, "pm_interval_ms": wc.PM_INTERVAL_MS, "batches": batches,
        "idle_mW": wc.IDLE, "idle_pkg_mW": wc.IDLE_PKG,
        "results": RESULTS,
    }, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    print("\n=== CROSSOVER SUMMARY ===")
    for wl, s in summary.items():
        print(f" {wl}")
        print(f"   throughput:  {s['throughput_verdict']}")
        print(f"   throughput/W: {s['perf_per_W_verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
