#!/usr/bin/env python3
"""End-to-end int4 / auto weight-compression benchmark: does the per-matmul 4-bit win carry to whole-model latency, cosine, and footprint? Run: PYTHONPATH=<repo> python3 bench/model_int4_bench.py"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import aneforge as af

from _mil import mil_encoding_tally as _encoding_tally

SEED = 0
WARMUP = 8
REPS = 30
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "model_int4_bench.json"


def _rng():
    return np.random.default_rng(SEED)


def W(rng, *shape, fan_in=None):
    """Xavier-ish weight: scale by 1/sqrt(fan_in) so activations stay O(1)."""
    fan = fan_in if fan_in is not None else shape[-1]
    return (rng.standard_normal(shape) / np.sqrt(fan)).astype(np.float32)


def B(rng, n):
    return (rng.standard_normal(n) * 0.01).astype(np.float32)


# model graph builders (full architecture, deterministic random weights)
def build_resnet18(rng):
    """torchvision ResNet-18 forward (ImageNet): conv-dominated, BN folded into conv. [1,3,224,224] -> [1,1000]."""
    def conv_w(cout, cin, k):
        return W(rng, cout, cin, k, k, fan_in=cin * k * k)

    def block(x, cin, cout, stride, downsample):
        out = af.conv(x, conv_w(cout, cin, 3), stride=stride, pad=1, bias=B(rng, cout)).relu()
        out = af.conv(out, conv_w(cout, cout, 3), stride=1, pad=1, bias=B(rng, cout))
        idn = x
        if downsample:
            idn = af.conv(x, conv_w(cout, cin, 1), stride=stride, pad=0, bias=B(rng, cout))
        return (out + idn).relu()

    x = af.input((1, 3, 224, 224))
    h = af.conv(x, conv_w(64, 3, 7), stride=2, pad=3, bias=B(rng, 64)).relu().max_pool(3, stride=2, pad=1)
    chans = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
    for li, (cin, cout, stride) in enumerate(chans):
        for i in range(2):
            s = stride if i == 0 else 1
            ci = cin if i == 0 else cout
            h = block(h, ci, cout, s, downsample=(i == 0 and li != 0))
    h = h.mean((2, 3)).reshape(1, 512)
    return h.linear(W(rng, 1000, 512), B(rng, 1000)), [(1, 3, 224, 224)]


def build_vit_b16(rng, n_layers=12):
    """ViT-B/16 encoder forward (12 layers x 768 dim x 12 heads, 197 tokens) -> [1,1000]; matmul-dominated."""
    DIM, HEADS, PATCH, IMG = 768, 12, 16, 224
    NP = (IMG // PATCH) ** 2          # 196
    SEQ = NP + 1                       # 197

    x = af.input((1, 3, IMG, IMG))
    cls = af.input((1, DIM))
    pos = af.input((SEQ, DIM))

    h = af.conv(x, W(rng, DIM, 3, PATCH, PATCH, fan_in=3 * PATCH * PATCH), stride=PATCH, bias=B(rng, DIM))
    patches = h.reshape(1, DIM, NP).transpose([0, 2, 1]).reshape(NP, DIM)
    seq = af.concat([cls, patches], axis=0) + pos

    for _ in range(n_layers):
        x_ln = seq.layer_norm(W(rng, DIM), B(rng, DIM), eps=1e-6)
        attn = af.mha(x_ln, W(rng, DIM, DIM), B(rng, DIM), W(rng, DIM, DIM), B(rng, DIM),
                      W(rng, DIM, DIM), B(rng, DIM), W(rng, DIM, DIM), B(rng, DIM), HEADS)
        seq = seq + attn
        y_ln = seq.layer_norm(W(rng, DIM), B(rng, DIM), eps=1e-6)
        y = y_ln.linear(W(rng, DIM * 4, DIM), B(rng, DIM * 4)).gelu().linear(W(rng, DIM, DIM * 4), B(rng, DIM))
        seq = seq + y

    seq = seq.layer_norm(W(rng, DIM), B(rng, DIM), eps=1e-6)
    # classifier on CLS row via one-hot picker matmul
    sel = np.eye(1, SEQ, dtype=np.float32)
    cls_row = seq.transpose([1, 0]).linear(sel).transpose([1, 0])    # [1,768]
    out = cls_row.linear(W(rng, 1000, DIM), B(rng, 1000))
    cls_const = (rng.standard_normal((1, DIM)) * 0.02).astype(np.float32)
    pos_const = (rng.standard_normal((SEQ, DIM)) * 0.02).astype(np.float32)
    return out, [(1, 3, IMG, IMG), (1, DIM), (SEQ, DIM)], (cls_const, pos_const)


def build_minilm(rng, S=64, L=6, DIM=384, HEADS=12, ff=1536):
    """all-MiniLM-L6-v2 encoder: 6 post-norm MHA+LayerNorm+(Linear-GELU-Linear) layers; matmul-dominated."""
    eps = 1e-12
    h = af.input((S, DIM))
    for _ in range(L):
        attn = af.mha(h, W(rng, DIM, DIM), B(rng, DIM), W(rng, DIM, DIM), B(rng, DIM),
                      W(rng, DIM, DIM), B(rng, DIM), W(rng, DIM, DIM), B(rng, DIM), HEADS)
        h = (h + attn).layer_norm(W(rng, DIM), B(rng, DIM), eps)
        f = h.linear(W(rng, ff, DIM), B(rng, ff)).gelu().linear(W(rng, DIM, ff), B(rng, DIM))
        h = (h + f).layer_norm(W(rng, DIM), B(rng, DIM), eps)
    return h, [(S, DIM)]


MODELS = {
    "resnet18": ("conv", build_resnet18),
    "vit_b16": ("matmul+conv", build_vit_b16),
    "minilm_l6": ("matmul", build_minilm),
}


# bench harness
def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def median_ms(net, inputs):
    for _ in range(WARMUP):
        net(*inputs)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        net(*inputs)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def run_model(name, kind, builder):
    print(f"\n=== {name} ({kind}) ===")
    # build inputs once so all variants see identical feeds
    rng = _rng()
    built = builder(rng)
    in_shapes = built[1]
    extra_consts = built[2] if len(built) > 2 else ()
    in_rng = np.random.default_rng(SEED + 1)
    arrays = [(in_rng.standard_normal(s) * 0.5).astype(np.float32) for s in in_shapes[:len(in_shapes) - len(extra_consts)]]
    arrays += list(extra_consts)

    res = {"kind": kind, "n_ops": None, "fp16_ms": None, "variants": {}}
    fp16_out = None
    # (label, compress, atol); int4_forced loosens the gate to 0.30 so the LUT fires on random weights
    plan = [("fp16", None, 0.05), ("int4", "int4", 0.05),
            ("int4_forced", "int4", 0.30), ("auto", "auto", 0.05)]
    for label, variant, atol in plan:
        # rebuild graph fresh per variant from the same seed; only the encoding differs
        rng_v = _rng()
        b = builder(rng_v)
        out_v = b[0]
        d = tempfile.mkdtemp(prefix=f"int4bench_{name}_")
        t0 = time.perf_counter()
        try:
            net = af.compile(out_v, compress=variant, compress_atol=atol, build_dir=d)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:12} COMPILE FAILED: {type(e).__name__}: {str(e)[:80]}")
            res["variants"][label] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            continue
        compile_s = time.perf_counter() - t0
        y = net(*arrays)
        ms = median_ms(net, arrays)
        wb = os.path.getsize(os.path.join(d, "weights.bin"))
        n_ops = getattr(net, "n_ops", None)
        enc = _encoding_tally(d)
        net.release()

        if variant is None:
            fp16_out = y
            res["fp16_ms"] = ms
            res["fp16_weights_bytes"] = wb
            res["n_ops"] = n_ops
            print(f"  {label:12} {ms:8.3f} ms   weights.bin {wb/1e6:7.2f} MB   "
                  f"(compile {compile_s:.1f}s, {res['n_ops']} ops)")
        else:
            cos = cosine(y, fp16_out)
            spd = res["fp16_ms"] / ms if ms else float("nan")
            frac = wb / res["fp16_weights_bytes"]
            res["variants"][label] = {
                "compress": variant, "compress_atol": atol,
                "ms": ms, "speedup": spd, "cosine": cos,
                "weights_bytes": wb, "size_frac": frac, "encoding": enc,
            }
            print(f"  {label:12} {ms:8.3f} ms   {spd:4.2f}x   cos {cos:.4f}   "
                  f"weights.bin {wb/1e6:7.2f} MB ({frac:.2f}x)   enc {enc}")
    return res


def main():
    print("END-TO-END int4 / auto weight-compression benchmark (ANE, aneforge)")
    print("=" * 72)
    print(f"warmup={WARMUP} reps={REPS} (median) seed={SEED} compress_atol=lib-default(0.05)")
    results = {"meta": {"warmup": WARMUP, "reps": REPS, "seed": SEED,
                        "weights": "deterministic random (no torch on host); "
                                   "latency+footprint faithful, int4 accuracy-gate pessimistic",
                        "host_python": os.sys.version.split()[0]},
               "models": {}}
    for name, (kind, builder) in MODELS.items():
        try:
            results["models"][name] = run_model(name, kind, builder)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # compact verdict table
    print("\n" + "=" * 72)
    hdr = f"{'ms':>8} {'spd':>5} {'cos':>6} {'sz':>5}"
    print(f"{'model':11}{'kind':13}{'fp16ms':>8} | int4(gate) {hdr} | int4_forced {hdr} | auto {hdr}")

    def fmt(v):
        if not v or "error" in v:
            return f"{'--':>8} {'--':>5} {'--':>6} {'--':>5}"
        return f"{v['ms']:8.3f} {v['speedup']:4.2f}x {v['cosine']:6.3f} {v['size_frac']:4.2f}x"

    for name, r in results["models"].items():
        if "error" in r:
            print(f"{name:11} ERROR: {r['error']}")
            continue
        i4 = r["variants"].get("int4", {})
        i4f = r["variants"].get("int4_forced", {})
        au = r["variants"].get("auto", {})
        print(f"{name:11}{r['kind']:13}{r['fp16_ms']:8.3f} |            {fmt(i4)} |             {fmt(i4f)} |      {fmt(au)}")


if __name__ == "__main__":
    main()
