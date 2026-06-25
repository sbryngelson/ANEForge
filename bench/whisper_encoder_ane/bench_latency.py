#!/usr/bin/env python3
"""Whisper-tiny encoder latency (ANE vs PyTorch CPU/MPS). Run: PYTHONPATH=. python3 bench/whisper_encoder_ane/bench_latency.py"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import encoder as E  # noqa: E402


def time_call(call, reps):
    out = call()                      # warmup (also the returned sample)
    t = time.perf_counter()
    for _ in range(reps):
        out = call()
    return (time.perf_counter() - t) / reps * 1e3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--real", action="store_true",
                    help="trained whisper-tiny weights (downloads; the representative number)")
    args = ap.parse_args()

    enc, sd = E.real_encoder() if args.real else E.make_encoder()
    print(f"weights: {'trained whisper-tiny' if args.real else 'random init'}")
    mel = E.mel_input()
    ref = E.torch_reference(enc, mel)

    print(f"{'engine':22s} {'ms/call':>8} {'cosine':>9}")
    net_cf = E.build_cf(sd)
    ms, out = time_call(lambda: E.run_cf(net_cf, sd, mel), args.reps)
    print(f"{'ANE channels-first':22s} {ms:8.2f} {E.cosine(out, ref):9.6f}")
    net_sd = E.build(sd, attn="mha")
    ms, out = time_call(lambda: E.run(net_sd, sd, mel), args.reps)
    print(f"{'ANE [seq,d] (old)':22s} {ms:8.2f} {E.cosine(out, ref):9.6f}")

    mt = torch.from_numpy(mel)

    def torch_call(dev, half):
        m = enc.to(dev)
        x = mt.to(dev)
        if half:
            m = m.half(); x = x.half()
        def call():
            with torch.no_grad():
                m(x)
            if dev == "mps":
                torch.mps.synchronize()
        return call

    ms, _ = time_call(torch_call("cpu", False), args.reps)
    print(f"{'CPU [fp32]':18s} {ms:8.2f}")
    enc.to("cpu").float()
    if torch.backends.mps.is_available():
        ms, _ = time_call(torch_call("mps", True), args.reps)
        print(f"{'MPS [fp16]':18s} {ms:8.2f}   (Metal GPU)")
        enc.to("cpu").float()
    else:
        print("MPS not available")


if __name__ == "__main__":
    main()
