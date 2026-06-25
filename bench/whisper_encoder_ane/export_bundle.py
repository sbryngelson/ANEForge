#!/usr/bin/env python3
"""Compile and persist the whisper-tiny encoder bundle plus the I/O the standalone C++ runner needs. Run: PYTHONPATH=. python3 bench/whisper_encoder_ane/export_bundle.py"""
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
    """The composite '*.bundle' dir under the cache (it holds the per-hardware
    H17*.bundle / universal.bundle variants); that is what e5rt_program_library_create
    loads. Identify it as a '*.bundle' that itself contains '*.bundle' children."""
    for d in build_dir.glob("cache/**/*.bundle"):
        if any(d.glob("*.bundle")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/whisper_enc_bundle", help="bundle + I/O output dir")
    args = ap.parse_args()
    build_dir = Path(args.out)
    io_dir = build_dir / "io"
    io_dir.mkdir(parents=True, exist_ok=True)

    enc, sd = E.make_encoder()
    mel = E.mel_input()
    ref = E.torch_reference(enc, mel)

    net = E.build(sd, attn="mha", build_dir=str(build_dir))
    out = E.run(net, sd, mel)

    feed = {(1, E.MELS, 1, E.FRAMES): mel[:, :, None, :].astype(np.float16),
            (E.CTX, E.D): sd["embed_positions.weight"].astype(np.float16)}
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
    }
    (io_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"cosine (python ANE vs torch): {E.cosine(out, ref):.6f}")
    print(f"wrote {build_dir}/model.mil, weights.bin, cache/, and io/ (vectors + manifest)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
