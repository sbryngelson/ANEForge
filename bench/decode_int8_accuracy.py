#!/usr/bin/env python3
"""int8 decode accuracy: token-agreement / top-5 / logit relerr / softmax KL of int8 weight-stream vs fp16/fp32 on shared weights. Run: PYTHONPATH=. python3 bench/decode_int8_accuracy.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import decode_measurement as dm  # noqa: E402

HAVE_ANE = dm.HAVE_ANE


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def kl(p, q, eps=1e-9):
    p = p + eps; q = q + eps
    return float((p * np.log(p / q)).sum(-1).mean())


def topk_overlap(a, b, k=5):
    ta = np.argpartition(-a, k, axis=-1)[:, :k]
    tb = np.argpartition(-b, k, axis=-1)[:, :k]
    return float(np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(a.shape[0])]))


def main() -> int:
    if not HAVE_ANE:
        print("ANE unavailable; nothing to do.")
        return 0
    cfg = dm.Cfg(d_model=2048, n_layers=8, n_heads=16, d_ff=8192, vocab=32000, s_kv=256)
    print(f"=== int8 decode accuracy: {cfg.asdict()} ===")
    W = dm.make_weights(cfg)
    B = 64                                   # decode positions to average over
    rng = np.random.default_rng(123)
    x = (rng.standard_normal((B, cfg.d)).astype(np.float32) * 0.1)

    ref = np.asarray(dm.ref_decode(cfg, W, x, dt=np.float64), dtype=np.float32)  # [B, vocab]

    import aneforge as af  # noqa: F401
    net16, c16 = dm.build_ane(cfg, W, B, int8=False)
    o16 = np.asarray(net16(x.astype(np.float16), *c16), dtype=np.float32)
    net8, c8 = dm.build_ane(cfg, W, B, int8=True)
    o8 = np.asarray(net8(x.astype(np.float16), *c8), dtype=np.float32)

    am_ref = ref.argmax(-1)
    am16 = o16.argmax(-1)
    am8 = o8.argmax(-1)
    p_ref, p16, p8 = softmax(ref), softmax(o16), softmax(o8)

    res = {
        "config": cfg.asdict(), "B_positions": B,
        "logit_relerr": {"fp16_vs_fp32": dm.relerr(o16, ref),
                         "int8_vs_fp32": dm.relerr(o8, ref),
                         "int8_vs_fp16": dm.relerr(o8, o16)},
        "argmax_agreement": {
            "fp16_vs_fp32": float((am16 == am_ref).mean()),
            "int8_vs_fp32": float((am8 == am_ref).mean()),
            "int8_vs_fp16": float((am8 == am16).mean())},
        "top5_overlap": {
            "fp16_vs_fp32": topk_overlap(o16, ref, 5),
            "int8_vs_fp32": topk_overlap(o8, ref, 5),
            "int8_vs_fp16": topk_overlap(o8, o16, 5)},
        "softmax_kl": {
            "fp16_vs_fp32": kl(p16, p_ref),
            "int8_vs_fp32": kl(p8, p_ref),
            "int8_vs_fp16": kl(p8, p16)},
    }
    print(json.dumps(res, indent=2))
    outp = Path(__file__).resolve().parent / "results" / "decode_int8_accuracy.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {outp}")
    print("\nSUMMARY:")
    a = res["argmax_agreement"]
    print(f"  next-token argmax agreement: fp16-vs-fp32 {a['fp16_vs_fp32']*100:.1f}%, "
          f"int8-vs-fp32 {a['int8_vs_fp32']*100:.1f}%, int8-vs-fp16 {a['int8_vs_fp16']*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
