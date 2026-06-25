#!/usr/bin/env python3
"""End-to-end encoder-block serving, cross-path (ANE vs MLX-GPU): does the ANE int4 win survive a full block? Run: PYTHONPATH=<repo> python3 bench/encoder_serving_crosspath.py"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import aneforge as af
import mlx.core as mx

from _mil import mil_encoding_tally as _lut_tally

# config
D, H, DFF, S = 768, 12, 3072, 128
DH = D // H
SCALE = 1.0 / (DH ** 0.5)
BATCHES = [1, 8, 16, 32, 64]
GROUP = 64                     # MLX 4-bit group size (D and DFF divisible)
SEED = 0
WARMUP, REPS = 10, 30
EPS = 1e-12
OUT_JSON = Path(__file__).resolve().parent / "results" / (Path(__file__).stem + ".json")


def _rng():
    return np.random.default_rng(SEED)


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def median_ms(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


# shared weights (PyTorch [out,in] convention; identical for ANE & GPU)
def make_weights(rng):
    def lin(out, inp):
        return (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)

    def b(n):
        return (rng.standard_normal(n) * 0.01).astype(np.float32)

    return {
        "Wq": lin(D, D), "bq": b(D), "Wk": lin(D, D), "bk": b(D),
        "Wv": lin(D, D), "bv": b(D), "Wo": lin(D, D), "bo": b(D),
        "ln1w": (rng.standard_normal(D) * 0.05 + 1).astype(np.float32), "ln1b": b(D),
        "Wi": lin(DFF, D), "bi": b(DFF), "Wd": lin(D, DFF), "bd": b(D),
        "ln2w": (rng.standard_normal(D) * 0.05 + 1).astype(np.float32), "ln2b": b(D),
    }


# ANE: pre-LN encoder block as one aneforge graph
def build_ane(w, B):
    """Batched [B,S,D] pre-LN encoder block; attention bmm+softmax, one fused e5rt program."""
    x = af.input((B, S, D))
    xf = x.reshape(B * S, D)
    xn = xf.layer_norm(w["ln1w"], w["ln1b"], EPS)
    q = xn.linear(w["Wq"], w["bq"]).reshape(B, S, H, DH).transpose([0, 2, 1, 3])  # [B,H,S,dh]
    k = xn.linear(w["Wk"], w["bk"]).reshape(B, S, H, DH).transpose([0, 2, 1, 3])
    v = xn.linear(w["Wv"], w["bv"]).reshape(B, S, H, DH).transpose([0, 2, 1, 3])
    a = ((q @ k.transpose([0, 1, 3, 2])) * SCALE).softmax(-1)                     # [B,H,S,S]
    o = (a @ v).transpose([0, 2, 1, 3]).reshape(B * S, D)                         # [B*S,D]
    attn = o.linear(w["Wo"], w["bo"])
    h1 = xf + attn
    yn = h1.layer_norm(w["ln2w"], w["ln2b"], EPS)
    ff = yn.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
    out = (h1 + ff).reshape(B, S, D)
    return out


def run_ane(w, B, x, label, compress, atol):
    out = build_ane(w, B)
    d = tempfile.mkdtemp(prefix=f"encx_{label}_b{B}_")
    net = af.compile(out, compress=compress, compress_atol=atol, build_dir=d)
    y = net(x)
    ms = median_ms(lambda: net(x))
    enc = _lut_tally(d)
    net.release()
    return ms, np.asarray(y, np.float32), enc


# GPU: same block in MLX (fp16 and 4-bit group-affine)
def _mx_w(w):
    """upload weights as mx float16; linears transposed to [in,out] for x @ Wt."""
    g = {}
    for kk, vv in w.items():
        g[kk] = mx.array(vv).astype(mx.float16)
    for nm in ("Wq", "Wk", "Wv", "Wo", "Wi", "Wd"):
        g[nm + "_t"] = g[nm].T            # [in,out] for x @ Wt
    return g


def _gpu_block(xm, g, lin):
    """pre-LN encoder block in MLX; `lin(name, x)` does x @ W.T (+ bias)."""
    B = xm.shape[0]

    def layernorm(t, gw, gb):
        mu = mx.mean(t, axis=-1, keepdims=True)
        var = mx.mean((t - mu) ** 2, axis=-1, keepdims=True)
        return (t - mu) * mx.rsqrt(var + EPS) * gw + gb

    xn = layernorm(xm, g["ln1w"], g["ln1b"])
    q = lin("Wq", xn).reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    k = lin("Wk", xn).reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    v = lin("Wv", xn).reshape(B, S, H, DH).transpose(0, 2, 1, 3)
    a = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * SCALE, axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, S, D)
    attn = lin("Wo", o)
    h1 = xm + attn
    yn = layernorm(h1, g["ln2w"], g["ln2b"])
    up = lin("Wi", yn)
    ff = lin("Wd", _gelu_mx(up))
    return h1 + ff


def _gelu_mx(t):
    return t * 0.5 * (1.0 + mx.erf(t / mx.sqrt(mx.array(2.0, dtype=t.dtype))))


def run_gpu_fp16(g, B, xnp):
    xm = mx.array(xnp).astype(mx.float16)

    def lin(name, t):
        return t @ g[name + "_t"] + g["b" + name[1].lower()]

    def run():
        y = _gpu_block(xm, g, lin)
        mx.eval(y)
        return y
    out = np.array(run().astype(mx.float32))
    return median_ms(run), out


def run_gpu_int4(g, B, xnp):
    xm = mx.array(xnp).astype(mx.float16)
    # quantize the six linears 4-bit, group_size 64
    q4 = {}
    for nm in ("Wq", "Wk", "Wv", "Wo", "Wi", "Wd"):
        wq, sc, bi = mx.quantize(g[nm + "_t"], bits=4, group_size=GROUP)
        q4[nm] = (wq, sc, bi)

    def lin(name, t):
        wq, sc, bi = q4[name]
        y = mx.quantized_matmul(t, wq, scales=sc, biases=bi, transpose=False,
                                bits=4, group_size=GROUP)
        return y + g["b" + name[1].lower()]

    def run():
        y = _gpu_block(xm, g, lin)
        mx.eval(y)
        return y
    out = np.array(run().astype(mx.float32))
    return median_ms(run), out


# driver
def main():
    print("END-TO-END encoder-block serving, cross-path (ANE vs MLX-GPU)")
    print("=" * 78)
    print(f"D={D} H={H} Dff={DFF} S={S}  batches={BATCHES}  attention=bmm+softmax")
    print(f"warmup={WARMUP} reps={REPS} (median ms)  GROUP={GROUP}  seed={SEED}\n")

    rng = _rng()
    w = make_weights(rng)
    g = _mx_w(w)

    results = {"meta": {"D": D, "H": H, "Dff": DFF, "S": S, "batches": BATCHES,
                        "group_size": GROUP, "warmup": WARMUP, "reps": REPS,
                        "seed": SEED, "attention": "bmm+softmax (no af.sdpa cut)",
                        "weights": "deterministic random; latency faithful, int4 "
                                   "default-gate pessimistic on gaussian weights",
                        "host_python": os.sys.version.split()[0],
                        "mlx": getattr(mx, "__version__", "?")},
               "rows": []}

    hdr = (f"{'B':>3} | {'ANEfp16':>8} {'ANEi4d':>8} {'ANEi4L':>8} {'ANEauto':>8} | "
           f"{'GPUfp16':>8} {'GPUi4':>8} | {'i4L/GPUi4':>9} {'i4L/fp16':>8} | "
           f"{'lutD':>4} {'lutL':>4}")
    print(hdr)
    print("-" * len(hdr))

    for B in BATCHES:
        in_rng = np.random.default_rng(SEED + 1)
        xnp = (in_rng.standard_normal((B, S, D)) * 0.5).astype(np.float32)

        a16_ms, a16, _ = run_ane(w, B, xnp, "fp16", None, 0.05)
        a4d_ms, a4d, encd = run_ane(w, B, xnp, "int4d", "int4", 0.05)
        a4l_ms, a4l, encl = run_ane(w, B, xnp, "int4L", "int4", 0.5)
        au_ms, au, enca = run_ane(w, B, xnp, "auto", "auto", 0.05)

        g16_ms, g16 = run_gpu_fp16(g, B, xnp)
        g4_ms, g4 = run_gpu_int4(g, B, xnp)

        row = {
            "B": B,
            "ane_fp16_ms": a16_ms, "ane_int4_default_ms": a4d_ms,
            "ane_int4_loose_ms": a4l_ms, "ane_auto_ms": au_ms,
            "gpu_fp16_ms": g16_ms, "gpu_int4_ms": g4_ms,
            "throughput_blocks_per_s": {
                "ane_fp16": B / (a16_ms / 1e3), "ane_int4_loose": B / (a4l_ms / 1e3),
                "gpu_fp16": B / (g16_ms / 1e3), "gpu_int4": B / (g4_ms / 1e3)},
            "ratio_aneI4loose_over_gpuI4": g4_ms / a4l_ms,   # >1 => ANE int4 faster
            "ratio_aneI4loose_over_aneFp16": a16_ms / a4l_ms,  # >1 => int4 helps ANE
            "lut_nodes": {"int4_default": encd, "int4_loose": encl, "auto": enca},
            "cosine": {
                "ane_int4_default_vs_ane_fp16": _cos(a4d, a16),
                "ane_int4_loose_vs_ane_fp16": _cos(a4l, a16),
                "ane_auto_vs_ane_fp16": _cos(au, a16),
                "gpu_int4_vs_gpu_fp16": _cos(g4, g16),
                "ane_fp16_vs_gpu_fp16": _cos(a16, g16)},
        }
        results["rows"].append(row)
        print(f"{B:>3} | {a16_ms:>8.3f} {a4d_ms:>8.3f} {a4l_ms:>8.3f} {au_ms:>8.3f} | "
              f"{g16_ms:>8.3f} {g4_ms:>8.3f} | "
              f"{row['ratio_aneI4loose_over_gpuI4']:>9.2f} "
              f"{row['ratio_aneI4loose_over_aneFp16']:>8.2f} | "
              f"{encd['int4_lut']:>4} {encl['int4_lut']:>4}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    print("\ncosine sanity (loose int4 vs own fp16; cross-path fp16 agreement):")
    for r in results["rows"]:
        c = r["cosine"]
        print(f"  B={r['B']:>3}  ANE i4L/fp16 {c['ane_int4_loose_vs_ane_fp16']:.4f}  "
              f"GPU i4/fp16 {c['gpu_int4_vs_gpu_fp16']:.4f}  "
              f"ANE-fp16/GPU-fp16 {c['ane_fp16_vs_gpu_fp16']:.4f}")


if __name__ == "__main__":
    main()
