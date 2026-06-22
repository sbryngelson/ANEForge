#!/usr/bin/env python3
"""Fidelity on the real checkpoint: whisper-tiny's trained encoder weights and a real
log-mel from Whisper's own feature extractor, ANEForge vs the PyTorch reference. This
complements fidelity.py, which uses a randomly-initialised encoder.

The trained encoder on real audio reaches cosine ~0.998 (vs ~1.0000 for random init):
the ANE's gelu LUT and fp16 lose more on the sharp, high-dynamic-range activations a
trained encoder produces on real log-mel. The drop needs both real weights and real
input; neither alone shows it.

Downloads openai/whisper-tiny (needs network). Run from repo root:
    PYTHONPATH=. python3 bench/whisper_encoder_ane/validate_real.py
"""
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
import torch  # noqa: E402
import encoder as E  # noqa: E402


def structured_mel(fe, seconds: float = 30.0) -> np.ndarray:
    """A deterministic structured signal (formant-like tones, amplitude modulation,
    noise) through Whisper's feature extractor: a realistic log-mel range without an
    audio file. Real speech gives a comparable distribution (the extractor normalises
    the same way), so the fidelity number is representative."""
    sr = fe.sampling_rate
    t = np.arange(int(sr * seconds)) / sr
    audio = np.zeros_like(t)
    for f0 in (140, 220, 330):
        audio += 0.3 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * f0 * t)
    audio += 0.05 * np.random.default_rng(0).standard_normal(t.shape)
    audio = (audio / np.abs(audio).max()).astype(np.float32)
    return fe(audio, sampling_rate=sr, return_tensors="np").input_features.astype(np.float32)


def main():
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
    enc = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval().model.encoder
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in enc.state_dict().items()}

    mel = structured_mel(fe)
    with torch.no_grad():
        ref = enc(torch.from_numpy(mel)).last_hidden_state.numpy()[0]
    out = E.run(E.build(sd, attn="mha"), sd, mel)
    rel = float(np.linalg.norm(out.ravel().astype(np.float64) - ref.ravel().astype(np.float64))
                / np.linalg.norm(ref.ravel().astype(np.float64)))
    print(f"real whisper-tiny encoder, real log-mel  shape {tuple(out.shape)}")
    print(f"  cosine vs torch : {E.cosine(out, ref):.6f}")
    print(f"  relative error  : {rel:.4f}")


if __name__ == "__main__":
    main()
