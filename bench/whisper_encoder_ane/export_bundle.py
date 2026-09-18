#!/usr/bin/env python3
"""Compile and persist a Whisper encoder bundle plus the I/O the standalone C++ runner needs.

The bundle is what whisper.cpp's ANEForge backend loads via ANEFORGE_ENCODER (ggml-org/whisper.cpp#3905).
Defaults to the trained whisper-tiny checkpoint in the benchmarked channels-first layout.
Run: PYTHONPATH=. python3 bench/whisper_encoder_ane/export_bundle.py [--model openai/whisper-base]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import encoder as E  # noqa: E402


def find_composite_bundle(build_dir: Path):
    """The composite '*.bundle' dir under the cache (a '*.bundle' that contains '*.bundle' children); what e5rt_program_library_create loads."""
    for d in build_dir.glob("cache/**/*.bundle"):
        if any(d.glob("*.bundle")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/whisper_enc_bundle", help="bundle + I/O output dir")
    ap.add_argument("--model", default="openai/whisper-tiny",
                    help="HF Whisper repo id whose trained encoder weights to compile (tiny..medium)")
    ap.add_argument("--layout", choices=("cf", "seq"), default="cf",
                    help="cf = channels-first + query-tiled (the benchmarked fast encoder); "
                         "seq = the generic [seq,d] build, ~3x slower, kept for comparison")
    ap.add_argument("--compress", choices=("int4", "int8"), default=None,
                    help="stream quantized weights (medium is benchmarked at int4)")
    ap.add_argument("--random", action="store_true",
                    help="randomly-initialised weights instead of the checkpoint: a no-download smoke "
                         "test only. The bundle transcribes noise, and ANE latency is weight-dependent, "
                         "so it is not representative.")
    args = ap.parse_args()
    build_dir = Path(args.out)
    io_dir = build_dir / "io"
    io_dir.mkdir(parents=True, exist_ok=True)

    if args.random:
        enc, sd = E.make_encoder()
        print("WARNING: --random weights; this bundle does not transcribe and its latency is not representative")
    else:
        enc, sd = E.real_encoder(args.model)
        E.set_dims(enc.config)                       # the builders read module-level dims
        print(f"model {args.model}: d={E.D} layers={E.LAYERS} heads={E.HEADS} ffn={E.FFN}")
    mel = E.mel_input()
    ref = E.torch_reference(enc, mel)

    if args.layout == "cf":
        net = E.build_cf(sd, build_dir=str(build_dir), compress=args.compress)
        out = E.run_cf(net, sd, mel)
    else:
        net = E.build(sd, attn="mha", build_dir=str(build_dir))
        out = E.run(net, sd, mel)

    pos = (sd["embed_positions.weight"].T.reshape(1, E.D, 1, E.CTX) if args.layout == "cf"
           else sd["embed_positions.weight"])
    feed = {(1, E.MELS, 1, E.FRAMES): mel[:, :, None, :].astype(np.float16),
            tuple(pos.shape): pos.astype(np.float16)}
    for name, shape in net._inputs:
        feed[tuple(shape)].tofile(io_dir / f"{name}.f16")
    ref.astype(np.float32).tofile(io_dir / "ref.f32")
    out.astype(np.float32).tofile(io_dir / "out_python.f32")

    composite = find_composite_bundle(build_dir)
    manifest = {
        "build_dir": str(build_dir),
        "bundle": str(composite) if composite else None,
        "inputs": [[name, int(np.prod(shape))] for name, shape in net._inputs],
        "output": [net._out_name, int(np.prod(out.shape))],
        "out_shape": list(out.shape),
        "model": "random-init" if args.random else args.model,
        "layout": args.layout,
        "compress": args.compress,
        "dims": {"D": E.D, "LAYERS": E.LAYERS, "HEADS": E.HEADS, "FFN": E.FFN,
                 "MELS": E.MELS, "CTX": E.CTX, "FRAMES": E.FRAMES},
    }
    (io_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"cosine (python ANE vs torch): {E.cosine(out, ref):.6f}")
    print(f"wrote {build_dir}/model.mil, weights.bin, cache/, and io/ (vectors + manifest)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
