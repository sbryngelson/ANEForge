#!/usr/bin/env python3
"""Capstone roofline analysis - unify the compute-ceiling and bandwidth-ceiling sweeps
into a per-device roofline that mechanistically EXPLAINS the device map.

Two measured ceilings already exist (we do NOT re-run them):
  * COMPUTE roof  - peak GFLOP/s per device, from device_saturation_sweep_results.json
                    ("peaks": GEMM/conv). GPU ~30.9/31.8 TFLOP/s; ANE ~10.2/18.8;
                    CPU ~1.9 (fp32 AMX).
  * BANDWIDTH roof - peak GB/s per device, from device_bandwidth_roofline_results.json
                    (streaming archetype). GPU ~230, CPU ~130, ANE ~24 (the STANDALONE
                    single-program path ceiling, flat from small - NOT silicon DMA).

This script:
  1. Computes arithmetic intensity (AI = FLOPs / bytes-moved) for each op archetype from
     real FLOP and byte counts, showing the formula. fp16 = 2 B/elem; CPU fp32 = 4 B.
  2. Pulls the two ceilings from the JSONs and computes each device's ridge point
     (ridge = peak_compute / peak_bw, in FLOP/byte).
  3. Places every archetype on EACH device's roofline:
        applicable_roof(AI) = min(compute_roof, bw_roof * AI)
        %roof = achieved / applicable_roof
     where achieved is: the saturation sweep for compute-bound ops; AI x achieved_GB/s
     (DERIVED) for bandwidth-bound ops; and a fresh MEASUREMENT for attention, a real
     fused block, and a GEMV/decode point (the gaps not covered by the two sweeps).
  4. Emits roofline_analysis_results.json + prints per-device tables and ASCII rooflines.

Only the GAPS are measured here (attention achieved GFLOP/s per device, a real fused
conv block achieved GFLOP/s on the ANE, a GEMV decode point). GEMM/conv/bandwidth peaks
are READ from the existing JSONs.

Run from repo root:
    PYTHONPATH=. python3 bench/roofline_analysis.py
    # --no-measure  : skip the gap measurements (uses analytic AIxBW placeholders, labeled)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

TOOLS = Path(__file__).resolve().parent / "results"
SAT_JSON = TOOLS / "device_saturation_sweep_results.json"
BW_JSON = TOOLS / "device_bandwidth_roofline_results.json"

FP16 = 2  # bytes/elem
FP32 = 4

# Part 0 - pull the measured ceilings from the two JSONs
def load_ceilings():
    sat = json.loads(SAT_JSON.read_text())
    bw = json.loads(BW_JSON.read_text())

    peaks = sat["peaks"]
    # Compute roof (GFLOP/s). GEMM (square) and conv are reported separately; the ANE in
    # particular has DIFFERENT compute roofs per op (conv >> square-matmul). Keep both.
    compute = {}
    for dev in ("CPU", "GPU", "ANE"):
        compute[dev] = {
            "gemm_peak_gflops": peaks["gemm"][dev]["peak_gflops"],
            "gemm_peak_at_N": peaks["gemm"][dev]["peak_gflops_size"],
            "conv_peak_gflops": peaks["conv"][dev]["peak_gflops"],
            "conv_peak_at_C": peaks["conv"][dev]["peak_gflops_size"],
        }

    # Bandwidth roof (GB/s) - use the STREAMING archetype peak (relu / x*2, byte_factor=2),
    # the cleanest pure read+write traffic op. This is the standalone-op bandwidth ceiling.
    stream = bw["results"]["roofline"]["streaming (relu / x*2)"]["peak"]
    bandwidth = {dev.upper(): stream[dev.lower()]["gbps"] for dev in ("CPU", "GPU", "ANE")}

    # achieved GB/s per device per archetype (the saturated tail of each curve) - used to
    # DERIVE achieved GFLOP/s for the bandwidth-bound ops.
    arche_bw = {}
    for name, blk in bw["results"]["roofline"].items():
        arche_bw[name] = {dev.upper(): blk["peak"][dev.lower()]["gbps"]
                          for dev in ("CPU", "GPU", "ANE")}
    return sat, bw, compute, bandwidth, arche_bw


# Part 1 - arithmetic intensity of the op archetypes (formula-driven)
def ai_matmul(M, K, N, bpe):
    """C[MxN] = A[MxK] @ B[KxN]. flops = 2*M*K*N. bytes = (M*K + K*N + M*N)*bpe."""
    flops = 2.0 * M * K * N
    nbytes = (M * K + K * N + M * N) * bpe
    return flops, nbytes, flops / nbytes


def ai_conv3x3(B, Cin, Cout, H, W, bpe):
    """3x3 same-pad conv. flops = 2*B*Cout*Cin*9*H*W.
    bytes moved = input(B*Cin*H*W) + weights(Cout*Cin*9) + output(B*Cout*H*W), *bpe.
    The reuse factor (why conv has decent AI): each input element participates in
    ~9*Cout MACs, each weight in B*H*W MACs - so flops/bytes climbs with channels."""
    k = 3
    flops = 2.0 * B * Cout * Cin * k * k * H * W
    in_el = B * Cin * H * W
    w_el = Cout * Cin * k * k
    out_el = B * Cout * H * W
    nbytes = (in_el + w_el + out_el) * bpe
    return flops, nbytes, flops / nbytes


def ai_elementwise(reads, writes, flops_per_elem, n, bpe):
    """Generic bandwidth-bound op over n elements. AI = flops_per_elem / ((reads+writes)*bpe)."""
    flops = flops_per_elem * n
    nbytes = (reads + writes) * n * bpe
    return flops, nbytes, flops / nbytes


def build_archetype_ai():
    """Return a list of archetype dicts with AI computed from real counts + the formula
    string, for fp16 (GPU/ANE) traffic. AI is dtype-dependent only through bytes-per-
    element: CPU fp32 doubles the bytes/elem, halving AI; handled per device in placement."""
    rows = []

    # ---- matmul: square N (compute archetype) + GEMV/decode (the key low-AI case) ----
    for N in (512, 2048, 8192):
        f, b, ai = ai_matmul(N, N, N, FP16)
        rows.append({
            "archetype": f"matmul square N={N}", "class": "compute",
            "formula": "AI = 2N^3 / (3N^2 * 2B) = N/3",
            "flops": f, "bytes_fp16": b, "ai_fp16": ai,
            "note": "AI = 2N^3 / (3N^2 * 2) = N/3. Grows with N -> large GEMM is compute-bound.",
            "represents": ["matmul(large)", "bmm", "linear"],
        })
    # GEMV / M=1 decode matmul, K=4096, N=4096 (a transformer projection at decode)
    M, K, N = 1, 4096, 4096
    f, b, ai = ai_matmul(M, K, N, FP16)
    rows.append({
        "archetype": "matmul GEMV M=1 (decode), K=N=4096", "class": "memory",
        "formula": "2*1*K*N / (1*K + K*N + 1*N)*2B  ~ 2KN/(KN*2B) = 1/bpe = 1.0 FLOP/byte (fp16)",
        "flops": f, "bytes_fp16": b, "ai_fp16": ai,
        "note": "weight matrix (KxN) read once, used once -> the AI floor. With the standard "
                "2-flop/MAC count it is 1.0 FLOP/byte fp16 (the often-quoted '~0.5' uses a "
                "1-flop/MAC convention; same physics, factor-2 bookkeeping). Memory-bound on "
                "EVERY device: well below the lowest ridge (CPU ~15).",
        "represents": ["GEMV", "LLM decode matmul", "M=1 projection"],
    })

    # ---- conv 3x3, the saturating config from the sweep (C=512,B=4,64x64) ----
    f, b, ai = ai_conv3x3(B=4, Cin=512, Cout=512, H=64, W=64, bpe=FP16)
    rows.append({
        "archetype": "conv 3x3 (C=512,B=4,64x64)", "class": "compute",
        "formula": "2*B*Cout*Cin*9*H*W / (in+wt+out)*2B",
        "flops": f, "bytes_fp16": b, "ai_fp16": ai,
        "note": "reuse: each input elem in ~9*Cout MACs -> high AI, clears every ridge.",
        "represents": ["conv", "conv_transpose"],
    })
    # a small conv too (C=64,B=16) to show AI is config-dependent but still high
    f, b, ai = ai_conv3x3(B=16, Cin=64, Cout=64, H=64, W=64, bpe=FP16)
    rows.append({
        "archetype": "conv 3x3 (C=64,B=16,64x64)", "class": "compute",
        "formula": "2*B*Cout*Cin*9*H*W / (in+wt+out)*2B",
        "flops": f, "bytes_fp16": b, "ai_fp16": ai,
        "note": "fewer channels -> lower reuse -> lower AI than the C=512 config, still >> ridge.",
        "represents": ["conv (small-channel)"],
    })

    # ---- bandwidth-bound elementwise archetypes (AI from real op counts) ----
    # relu/copy: 1 read + 1 write, ~1 flop (compare/max). ~0.2 FLOP/byte
    f, b, ai = ai_elementwise(reads=1, writes=1, flops_per_elem=1, n=16_777_216, bpe=FP16)
    rows.append({"archetype": "relu / copy (1R+1W, ~1 flop)", "class": "memory",
                 "formula": "1 flop / (2 elem * 2B) = 0.25; counting copy as ~0 flop -> ~0.2",
                 "flops": f, "bytes_fp16": b, "ai_fp16": ai, "bw_archetype": "streaming (relu / x*2)",
                 "represents": ["relu", "abs", "copy", "reshape", "transpose", "add", "sub", "mul"]})
    # gelu/silu: ~10 flops/elem (erf/tanh approx), 1R+1W -> ~2.5 FLOP/byte
    f, b, ai = ai_elementwise(reads=1, writes=1, flops_per_elem=10, n=16_777_216, bpe=FP16)
    rows.append({"archetype": "gelu / silu (~10 flop, 1R+1W)", "class": "memory",
                 "formula": "~10 flop / (2 elem * 2B) = 2.5 FLOP/byte",
                 "flops": f, "bytes_fp16": b, "ai_fp16": ai, "bw_archetype": "gelu (light compute)",
                 "represents": ["gelu", "silu", "erf", "tanh", "sigmoid", "exp"]})
    # softmax: ~5 flops/elem (max,sub,exp,sum,div), 1R+1W effective -> ~1.2 FLOP/byte
    f, b, ai = ai_elementwise(reads=1, writes=1, flops_per_elem=5, n=16_777_216, bpe=FP16)
    rows.append({"archetype": "softmax (~5 flop, 1R+1W)", "class": "memory",
                 "formula": "~5 flop / (2 elem * 2B) = 1.25 FLOP/byte",
                 "flops": f, "bytes_fp16": b, "ai_fp16": ai, "bw_archetype": "softmax",
                 "represents": ["softmax"]})
    # layer_norm: ~10 flops/elem (mean,var,normalize,scale,shift), 1R+1W -> ~2.5 FLOP/byte
    f, b, ai = ai_elementwise(reads=1, writes=1, flops_per_elem=10, n=16_777_216, bpe=FP16)
    rows.append({"archetype": "layer_norm (~10 flop, 1R+1W)", "class": "memory",
                 "formula": "~10 flop / (2 elem * 2B) = 2.5 FLOP/byte",
                 "flops": f, "bytes_fp16": b, "ai_fp16": ai, "bw_archetype": "layer_norm",
                 "represents": ["layer_norm", "rms_norm", "group_norm", "batch_norm", "l2_norm"]})
    return rows


# Part 1b - attention block: blended AI (matmul-dominated)
def attention_flops_bytes(S, D, H, bpe):
    """One attention block QK^T + softmax + AV for [1,H,S,D] (per the af.sdpa shape).
    QK^T: 2*H*S*S*D flops.  AV: 2*H*S*S*D flops.  softmax over H*S*S scores: ~5 flop each.
    bytes moved (standalone, intermediates spill): Q,K,V in = 3*H*S*D; scores S^2*H
    written+read for softmax = 2*H*S*S; out = H*S*D. *bpe."""
    qkt = 2.0 * H * S * S * D
    av = 2.0 * H * S * S * D
    sm = 5.0 * H * S * S
    flops = qkt + av + sm
    in_el = 3 * H * S * D
    score_traffic = 2 * H * S * S  # write then read scores (standalone, not fused)
    out_el = H * S * D
    nbytes = (in_el + score_traffic + out_el) * bpe
    return flops, nbytes, flops / nbytes


# Part 2 - gap measurements (attention per device, a real fused block, GEMV)
def _min_lat(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_gaps(no_measure):
    out = {"attention": {}, "fused_block": {}, "gemv": {}, "available": {}}
    try:
        import aneforge as af  # noqa
        HAVE_ANE = True
    except Exception as e:
        HAVE_ANE = False
        out["available"]["ane_err"] = str(e)
    try:
        import mlx.core as mx  # noqa
        HAVE_MLX = True
    except Exception as e:
        HAVE_MLX = False
        out["available"]["mlx_err"] = str(e)
    out["available"]["ane"] = HAVE_ANE
    out["available"]["mlx"] = HAVE_MLX
    if no_measure:
        out["skipped"] = True
        return out

    # ---------- ATTENTION block [1,H,S,D] ----------
    S, D, H = 512, 64, 12   # S=512 seq, d_head=64, 12 heads (ViT-B-ish)
    flops, _, _ = attention_flops_bytes(S, D, H, FP16)
    out["attention"]["config"] = {"S": S, "D": D, "H": H, "flops": flops}
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((1, H, S, D)) / np.sqrt(D)).astype(np.float16)
    k = (rng.standard_normal((1, H, S, D)) / np.sqrt(D)).astype(np.float16)
    v = (rng.standard_normal((1, H, S, D)) / np.sqrt(D)).astype(np.float16)

    # ANE native sdpa
    if HAVE_ANE:
        try:
            import aneforge as af
            net = af.compile(af.sdpa(af.input((1, H, S, D)), af.input((1, H, S, D)),
                                     af.input((1, H, S, D))))
            net(q, k, v)
            lat = _min_lat(lambda: net(q, k, v))
            out["attention"]["ANE"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                       "gflops": flops / lat / 1e9}
        except Exception as e:
            out["attention"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}
    # GPU MLX fast SDPA
    if HAVE_MLX:
        try:
            import mlx.core as mx
            qg, kg, vg = (mx.array(t) for t in (q, k, v))
            scale = 1.0 / math.sqrt(D)
            def grun(qg=qg, kg=kg, vg=vg, scale=scale):
                o = mx.fast.scaled_dot_product_attention(qg, kg, vg, scale=scale)
                mx.eval(o)
                return o
            grun()
            lat = _min_lat(grun)
            out["attention"]["GPU"] = {"dtype": "fp16", "lat_ms": lat * 1e3,
                                       "gflops": flops / lat / 1e9}
        except Exception as e:
            out["attention"]["GPU"] = {"error": f"{type(e).__name__}: {e}"}
    # CPU numpy fp32
    try:
        qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
        scale = 1.0 / math.sqrt(D)
        def crun(qf=qf, kf=kf, vf=vf, scale=scale):
            scores = np.einsum("bhsd,bhtd->bhst", qf, kf) * scale
            scores -= scores.max(-1, keepdims=True)
            e = np.exp(scores); p = e / e.sum(-1, keepdims=True)
            return np.einsum("bhst,bhtd->bhsd", p, vf)
        crun()
        lat = _min_lat(crun, reps=10)
        out["attention"]["CPU"] = {"dtype": "fp32", "lat_ms": lat * 1e3,
                                   "gflops": flops / lat / 1e9}
    except Exception as e:
        out["attention"]["CPU"] = {"error": f"{type(e).__name__}: {e}"}

    # ---------- REAL FUSED BLOCK on the ANE: a 3-conv stack fused into ONE program ----
    # This is the fusion-lever evidence: 3 chained 3x3 convs, intermediates never leave
    # the chip. Compare its achieved GFLOP/s to ONE standalone conv's (from the sweep).
    if HAVE_ANE:
        try:
            import aneforge as af
            Bc, C, Hc, Wc = 4, 256, 64, 64
            k3 = 3
            flops_one = 2.0 * Bc * C * C * k3 * k3 * Hc * Wc
            n_conv = 3
            fused_flops = flops_one * n_conv
            w1 = (rng.standard_normal((C, C, k3, k3)) * np.sqrt(2.0 / (C * k3 * k3))).astype(np.float16)
            w2 = (rng.standard_normal((C, C, k3, k3)) * np.sqrt(2.0 / (C * k3 * k3))).astype(np.float16)
            w3 = (rng.standard_normal((C, C, k3, k3)) * np.sqrt(2.0 / (C * k3 * k3))).astype(np.float16)
            xin = af.input((Bc, C, Hc, Wc))
            g = af.conv(xin, w1, stride=1, pad=1).relu()
            g = af.conv(g, w2, stride=1, pad=1).relu()
            g = af.conv(g, w3, stride=1, pad=1)
            net = af.compile(g)
            xf = rng.standard_normal((Bc, C, Hc, Wc)).astype(np.float16)
            net(xf)
            lat = _min_lat(lambda: net(xf))
            # standalone bytes for ONE conv (input+wt+out) vs fused (read input once, write
            # output once, 3x weights, intermediates stay on-chip): estimate effective AI.
            f1, b_standalone_one, ai_one = ai_conv3x3(Bc, C, C, Hc, Wc, FP16)
            in_el = Bc * C * Hc * Wc; out_el = Bc * C * Hc * Wc
            w_el = 3 * C * C * k3 * k3
            fused_bytes = (in_el + w_el + out_el) * FP16  # intermediates NOT counted (on-chip)
            standalone_bytes_3 = 3 * b_standalone_one      # 3 separate dispatches spill each
            out["fused_block"]["ANE"] = {
                "config": {"B": Bc, "C": C, "HxW": f"{Hc}x{Wc}", "n_conv": n_conv},
                "dtype": "fp16", "lat_ms": lat * 1e3, "gflops": fused_flops / lat / 1e9,
                "fused_flops": fused_flops,
                "ai_one_conv_standalone": ai_one,
                "ai_fused_est": fused_flops / fused_bytes,
                "ai_3conv_standalone_est": fused_flops / standalone_bytes_3,
                "note": "fused AI is an ESTIMATE: intermediates stay on-chip so not counted "
                        "in fused_bytes; standalone-3x would spill each intermediate.",
            }
        except Exception as e:
            out["fused_block"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}

    # ---------- GEMV / decode point (M=1) per device ----------
    M, K, N = 1, 4096, 4096
    flops_gemv = 2.0 * M * K * N
    xg = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float16)
    Wg = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16)  # [out,in]
    if HAVE_ANE:
        try:
            import aneforge as af
            net = af.compile(af.input((M, K)).linear(Wg))
            net(xg)
            lat = _min_lat(lambda: net(xg))
            out["gemv"]["ANE"] = {"dtype": "fp16", "lat_ms": lat * 1e3, "gflops": flops_gemv / lat / 1e9}
        except Exception as e:
            out["gemv"]["ANE"] = {"error": f"{type(e).__name__}: {e}"}
    if HAVE_MLX:
        try:
            import mlx.core as mx
            xm = mx.array(xg); Wm = mx.array(Wg.T)
            def grun(xm=xm, Wm=Wm):
                o = xm @ Wm; mx.eval(o); return o
            grun()
            lat = _min_lat(grun)
            out["gemv"]["GPU"] = {"dtype": "fp16", "lat_ms": lat * 1e3, "gflops": flops_gemv / lat / 1e9}
        except Exception as e:
            out["gemv"]["GPU"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        xf = xg.astype(np.float32); Wf = np.ascontiguousarray(Wg.T.astype(np.float32))
        lat = _min_lat(lambda: xf @ Wf)
        out["gemv"]["CPU"] = {"dtype": "fp32", "lat_ms": lat * 1e3, "gflops": flops_gemv / lat / 1e9}
    except Exception as e:
        out["gemv"]["CPU"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# Part 2b - placement: roof at AI, achieved, %roof
def applicable_roof(ai, compute_roof, bw_roof):
    return min(compute_roof, bw_roof * ai)


def _sat_achieved(sat, dev, archetype):
    """Pull the size-specific achieved GFLOP/s straight from the saturation sweep JSON
    for the exact size a compute archetype names (not the peak)."""
    if "square N=" in archetype:
        N = int(archetype.split("N=")[1].split()[0])
        for r in sat["gemm"]:
            if r["N"] == N:
                return r["devices"].get(dev, {}).get("gflops")
    if "conv" in archetype and "C=" in archetype:
        C = int(archetype.split("C=")[1].split(",")[0])
        for r in sat["conv"]:
            if r["C"] == C:
                return r["devices"].get(dev, {}).get("gflops")
    return None


def build_placement(arche, compute, bandwidth, arche_bw, gaps, sat):
    """For each (op, device): AI (dtype-adjusted), applicable roof, achieved, %roof."""
    placement = {dev: [] for dev in ("CPU", "GPU", "ANE")}
    for dev in ("CPU", "GPU", "ANE"):
        bpe = FP32 if dev == "CPU" else FP16
        bw_roof = bandwidth[dev]
        for row in arche:
            # AI for this device's dtype: recompute byte count with this bpe -> AI scales by 2/bpe
            ai = row["ai_fp16"] * (FP16 / bpe)
            cls = row["class"]
            # pick compute roof: conv archetypes use conv peak, matmul archetypes use gemm peak
            if "conv" in row["archetype"]:
                comp_roof = compute[dev]["conv_peak_gflops"]
            else:
                comp_roof = compute[dev]["gemm_peak_gflops"]
            roof = applicable_roof(ai, comp_roof, bw_roof)
            achieved = None
            src = None
            if cls == "compute":
                # size-SPECIFIC achieved from the saturation sweep (NOT the device peak)
                achieved = _sat_achieved(sat, dev, row["archetype"])
                src = "saturation sweep (size-specific)"
            else:
                # bandwidth-bound: achieved = AI * achieved_GB/s (DERIVED)
                bwname = row.get("bw_archetype")
                if bwname and bwname in arche_bw:
                    gbps = arche_bw[bwname][dev]
                    achieved = ai * gbps  # GFLOP/s = (FLOP/byte) * (GB/s)
                    src = f"DERIVED: AI x achieved {gbps:.1f} GB/s ({bwname})"
            pct = (achieved / roof * 100.0) if (achieved and roof) else None
            placement[dev].append({
                "archetype": row["archetype"], "class": cls, "ai": ai,
                "applicable_roof_gflops": roof, "achieved_gflops": achieved,
                "pct_roof": pct, "source": src,
            })

        # attention (measured)
        S = gaps["attention"].get("config", {}).get("S", 512)
        D = gaps["attention"].get("config", {}).get("D", 64)
        Hh = gaps["attention"].get("config", {}).get("H", 12)
        _, _, ai_att = attention_flops_bytes(S, D, Hh, bpe)
        comp_roof = compute[dev]["gemm_peak_gflops"]  # attention is matmul-dominated
        roof = applicable_roof(ai_att, comp_roof, bw_roof)
        adev = gaps["attention"].get(dev, {})
        ach = adev.get("gflops")
        pct = (ach / roof * 100.0) if (ach and roof) else None
        placement[dev].append({
            "archetype": f"attention block [1,{Hh},{S},{D}]", "class": "composite (matmul-dom)",
            "ai": ai_att, "applicable_roof_gflops": roof, "achieved_gflops": ach,
            "pct_roof": pct, "source": "MEASURED attention" + (f" ({adev.get('error')})" if 'error' in adev else "")})

        # GEMV (measured)
        gdev = gaps["gemv"].get(dev, {})
        ai_gemv = ai_matmul(1, 4096, 4096, bpe)[2]
        roof = applicable_roof(ai_gemv, compute[dev]["gemm_peak_gflops"], bw_roof)
        ach = gdev.get("gflops")
        pct = (ach / roof * 100.0) if (ach and roof) else None
        placement[dev].append({
            "archetype": "GEMV M=1 K=N=4096 (decode)", "class": "memory",
            "ai": ai_gemv, "applicable_roof_gflops": roof, "achieved_gflops": ach,
            "pct_roof": pct, "source": "MEASURED GEMV" + (f" ({gdev.get('error')})" if 'error' in gdev else "")})
    return placement


# ASCII roofline sketch
def ascii_roofline(dev, compute, bandwidth, placement_rows, ridge):
    """log-log ASCII: x=AI (FLOP/byte), y=GFLOP/s. Draw the roof (bw slope -> flat) and
    plot each op as a point at its (AI, achieved)."""
    comp = compute[dev]["gemm_peak_gflops"]
    comp_conv = compute[dev]["conv_peak_gflops"]
    bw = bandwidth[dev]
    W, Hh = 64, 18
    # axis ranges (log10)
    ais = [r["ai"] for r in placement_rows if r["ai"] and r["ai"] > 0]
    achs = [r["achieved_gflops"] for r in placement_rows if r.get("achieved_gflops")]
    x_lo, x_hi = math.log10(0.1), math.log10(max(2000, max(ais) * 2))
    y_hi = math.log10(max(comp, comp_conv) * 1.5)
    y_lo = math.log10(max(0.5, (min(achs) if achs else 1) * 0.3))

    def px(ai):
        return int((math.log10(ai) - x_lo) / (x_hi - x_lo) * (W - 1))
    def py(g):
        return int((y_hi - math.log10(g)) / (y_hi - y_lo) * (Hh - 1))

    grid = [[" "] * W for _ in range(Hh)]
    # roof line: for AI from x_lo..x_hi, y = min(comp, bw*AI)
    for c in range(W):
        ai = 10 ** (x_lo + c / (W - 1) * (x_hi - x_lo))
        y = min(comp, bw * ai)
        if y <= 0:
            continue
        r = py(y)
        if 0 <= r < Hh:
            grid[r][c] = "_" if (bw * ai >= comp) else "/"
    # ridge marker
    rc = px(ridge)
    rr = py(comp)
    if 0 <= rc < W and 0 <= rr < Hh:
        grid[rr][rc] = "R"
    # plot ops
    labels = []
    mk = "0123456789abcdef"
    for i, r in enumerate(placement_rows):
        if not (r["ai"] and r.get("achieved_gflops")):
            continue
        c = px(r["ai"]); rr = py(r["achieved_gflops"])
        if 0 <= rr < Hh and 0 <= c < W:
            grid[rr][c] = mk[i % len(mk)]
        labels.append(f"   [{mk[i % len(mk)]}] {r['archetype']:<34} AI={r['ai']:8.3f}  "
                      f"{(str(round(r['achieved_gflops'],1))+' GFLOP/s'):>16}  "
                      f"{(str(round(r['pct_roof'],1))+'%roof' if r.get('pct_roof') else '-')}")
    def _tf(x):
        return f"{x/1e3:.1f} TF/s" if x >= 1000 else f"{x:.0f} GF/s"
    lines = [f" {dev} roofline  (compute roof gemm={_tf(comp)} conv={_tf(comp_conv)}, "
             f"bw={bw:.0f} GB/s, ridge R={ridge:.0f} FLOP/byte)",
             f" y=GFLOP/s [{10**y_lo:.0f}..{10**y_hi:.0f}]  x=AI [0.1..{10**x_hi:.0f}]  (log-log)"]
    for r in range(Hh):
        lines.append(" |" + "".join(grid[r]))
    lines.append(" +" + "-" * W + "  AI->")
    lines.extend(labels)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the gap measurements (attention/fused/gemv)")
    args = ap.parse_args()

    sat, bw, compute, bandwidth, arche_bw = load_ceilings()

    # ridge points
    ridges = {}
    for dev in ("CPU", "GPU", "ANE"):
        # use the GEMM compute peak as the headline compute roof for the ridge
        ridges[dev] = compute[dev]["gemm_peak_gflops"] / bandwidth[dev]

    print("=" * 88)
    print(" ROOFLINE ANALYSIS - per-device ridge points (recomputed from the two JSONs)")
    print("=" * 88)
    print(f" {'device':<5} {'dtype':<5} {'compute roof (GFLOP/s)':>26} {'bw roof (GB/s)':>15} {'ridge FLOP/byte':>17}")
    for dev in ("CPU", "GPU", "ANE"):
        dtype = "fp32" if dev == "CPU" else "fp16"
        cg = compute[dev]["gemm_peak_gflops"]; cc = compute[dev]["conv_peak_gflops"]
        print(f" {dev:<5} {dtype:<5} {f'gemm {cg:.0f} / conv {cc:.0f}':>26} "
              f"{bandwidth[dev]:>15.1f} {ridges[dev]:>17.1f}")

    arche = build_archetype_ai()
    print("\n" + "=" * 88)
    print(" ARITHMETIC INTENSITY of op archetypes (fp16; CPU fp32 halves AI)")
    print("=" * 88)
    for r in arche:
        print(f"  {r['archetype']:<34} AI={r['ai_fp16']:9.4f} FLOP/byte   [{r['class']}]")
        print(f"        {r['formula']}")

    gaps = measure_gaps(args.no_measure)
    print("\n" + "=" * 88)
    print(" GAP MEASUREMENTS (attention / fused block / GEMV)")
    print("=" * 88)
    for k in ("attention", "gemv"):
        print(f"\n {k}:")
        for dev in ("CPU", "GPU", "ANE"):
            d = gaps[k].get(dev, {})
            if "gflops" in d:
                print(f"   {dev:<4} {d.get('dtype'):<5} {d['lat_ms']:8.3f} ms  {d['gflops']:9.1f} GFLOP/s")
            elif d:
                print(f"   {dev:<4} {d.get('error', d)}")
    fb = gaps["fused_block"].get("ANE", {})
    if "gflops" in fb:
        print(f"\n fused 3-conv block (ANE): {fb['lat_ms']:.3f} ms  {fb['gflops']:.1f} GFLOP/s  "
              f"| AI: 1-conv standalone={fb['ai_one_conv_standalone']:.1f}, "
              f"3x-standalone-est={fb['ai_3conv_standalone_est']:.1f}, fused-est={fb['ai_fused_est']:.1f}")
    elif fb:
        print(f"\n fused block (ANE): {fb.get('error', fb)}")

    placement = build_placement(arche, compute, bandwidth, arche_bw, gaps, sat)

    print("\n" + "=" * 88)
    print(" OP PLACEMENT per device - AI / applicable roof / achieved / %roof")
    print("=" * 88)
    for dev in ("CPU", "GPU", "ANE"):
        print(f"\n--- {dev} ---")
        print(f" {'archetype':<36} {'AI':>9} {'roof@AI':>10} {'achieved':>10} {'%roof':>7}")
        for r in placement[dev]:
            ach = f"{r['achieved_gflops']:.1f}" if r.get("achieved_gflops") else "-"
            pct = f"{r['pct_roof']:.1f}" if r.get("pct_roof") else "-"
            print(f" {r['archetype']:<36} {r['ai']:9.3f} {r['applicable_roof_gflops']:10.0f} {ach:>10} {pct:>7}")

    print("\n" + "=" * 88)
    print(" ASCII ROOFLINES")
    print("=" * 88)
    sketches = {}
    for dev in ("CPU", "GPU", "ANE"):
        s = ascii_roofline(dev, compute, bandwidth, placement[dev], ridges[dev])
        sketches[dev] = s
        print("\n" + s)

    results = {
        "ceilings": {"compute": compute, "bandwidth_gbps": bandwidth, "ridge_flop_per_byte": ridges,
                     "archetype_achieved_gbps": arche_bw},
        "archetype_ai": arche,
        "gap_measurements": gaps,
        "placement": placement,
        "ascii_rooflines": sketches,
        "meta": {"source_compute_json": str(SAT_JSON), "source_bandwidth_json": str(BW_JSON),
                 "fp16_bytes": FP16, "cpu_fp32_bytes": FP32},
    }
    outp = TOOLS / "roofline_analysis_results.json"
    outp.write_text(json.dumps(results, indent=2, default=lambda o: None))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
