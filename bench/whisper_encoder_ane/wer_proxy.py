#!/usr/bin/env python3
"""Does the encoder's fidelity gap change the transcript? Feed the ANE encoder output
into the HF Whisper decoder on real speech and compare against the reference encoder's
transcript. A word-error-rate proxy that skips the whisper.cpp C++ wiring: same
decoder weights either way, only the encoder differs.

On the canonical jfk.wav clip the encoder reaches cosine ~0.9998 (real speech is
easier than the synthetic signal in validate_real.py) and the two transcripts are
identical, so the feature error costs no accuracy here. This is one clip, not a full
dataset WER; it answers "does the gap matter" for a real sample, not "what is the WER
over LibriSpeech".

Downloads openai/whisper-tiny and jfk.wav (needs network). Run from repo root:
    PYTHONPATH=. python3 bench/whisper_encoder_ane/wer_proxy.py [--audio path.wav]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import wave
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import encoder as E  # noqa: E402

JFK_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"


def load_wav_16k_mono(path):
    w = wave.open(path, "rb")
    if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
        raise ValueError(f"expected 16 kHz mono 16-bit wav; got {w.getframerate()} Hz, "
                         f"{w.getnchannels()} ch, {w.getsampwidth() * 8}-bit")
    audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return audio, w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="16 kHz mono wav (default: download jfk.wav)")
    args = ap.parse_args()

    path = args.audio
    if path is None:
        path = "/tmp/jfk.wav"
        if not os.path.exists(path):
            urllib.request.urlretrieve(JFK_URL, path)
    audio, seconds = load_wav_16k_mono(path)
    print(f"audio: {seconds:.1f} s")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    enc = model.model.encoder
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in enc.state_dict().items()}

    mel = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
    with torch.no_grad():
        ref = enc(torch.from_numpy(mel)).last_hidden_state.numpy()[0]
    ours = E.run(E.build(sd, attn="mha"), sd, mel)
    print(f"encoder cosine (real speech): {E.cosine(ours, ref):.6f}")

    def transcribe(enc_out):
        eo = BaseModelOutput(last_hidden_state=torch.from_numpy(enc_out[None].astype(np.float32)))
        with torch.no_grad():
            ids = model.generate(encoder_outputs=eo, language="en", task="transcribe")
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    t_ref, t_ours = transcribe(ref), transcribe(ours)
    print("\nreference encoder:\n  " + t_ref)
    print("ANE encoder:\n  " + t_ours)
    print("\n" + ("IDENTICAL transcript (no WER cost on this clip)" if t_ref == t_ours
                  else "transcripts DIFFER"))


if __name__ == "__main__":
    main()
