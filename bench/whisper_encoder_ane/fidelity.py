#!/usr/bin/env python3
"""Fidelity gate: ANEForge whisper-tiny encoder vs PyTorch reference (cosine + relative error). Run: PYTHONPATH=. python3 bench/whisper_encoder_ane/fidelity.py"""
from __future__ import annotations

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

enc, sd = E.make_encoder()
mel = E.mel_input()
ref = E.torch_reference(enc, mel)
out = E.run(E.build(sd, attn="mha"), sd, mel)

rel = float(np.linalg.norm(out.ravel().astype(np.float64) - ref.ravel().astype(np.float64))
            / np.linalg.norm(ref.ravel().astype(np.float64)))
print(f"whisper-tiny encoder  shape {tuple(out.shape)}")
print(f"  cosine vs torch : {E.cosine(out, ref):.6f}")
print(f"  relative error  : {rel:.4f}")
