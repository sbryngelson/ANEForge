"""Whisper speech-to-text on the Apple Neural Engine (aneforge): both the audio encoder and the
autoregressive text decoder run on the ANE, validated against Hugging Face.

  python3 examples/whisper.py                 # transcribe a fetched librispeech sample clip
  python3 examples/whisper.py --audio clip.wav   # transcribe your own 16 kHz-ish mono file
"""
import argparse
import io
import sys
import time
import warnings

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

warnings.filterwarnings("ignore", "aneforge.compile: dispatch-floor")   # per-call dispatch note is noise here

NAME = "openai/whisper-base.en"


def _load_audio(path: str | None) -> np.ndarray:
  """Return a 16 kHz mono float32 waveform: the user's --audio file, else a fetched sample clip."""
  import soundfile as sf
  if path:
    data, sr = sf.read(path)
  else:
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(decode=False))
    a = ds[0]["audio"]
    data, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
  data = np.asarray(data, np.float32)
  if data.ndim == 2:
    data = data.mean(1)                                     # mix down to mono
  if sr != 16000:                                           # linear resample to Whisper's 16 kHz
    n = int(round(len(data) * 16000 / sr))
    data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
  return data


def main(argv) -> int:
  ap = argparse.ArgumentParser(description="Whisper transcription on the Apple Neural Engine.")
  ap.add_argument("--audio", help="path to an audio file (default: a fetched librispeech sample)")
  a = ap.parse_args(argv)

  import transformers
  transformers.logging.set_verbosity_error()               # HF's tokenizer/generate notes are noise for a demo

  _common.head("Whisper speech-to-text on the Apple Neural Engine (aneforge)")
  print(f"config: {NAME} | 512-dim, 6+6 layers | audio encoder + text decoder both on the ANE")

  audio = _load_audio(a.audio)
  print(f"\naudio: {len(audio) / 16000:.1f}s @ 16 kHz mono")

  print("\nloading Whisper via aneforge (compiling encoder + decoder programs for ANE) ...", end="", flush=True)
  t0 = time.perf_counter()
  w = af.load_whisper(NAME)
  print(f" done ({time.perf_counter() - t0:.2f}s)")

  text = w.transcribe(audio)                                 # first call also compiles the decode program
  t0 = time.perf_counter()
  text = w.transcribe(audio)                                 # steady-state: resident KV cache, program already built
  dt = time.perf_counter() - t0

  print("\n" + "=" * 60)
  print("TRANSCRIPT (ANE):")
  print(f"  {text.strip()}")
  print("=" * 60)

  # Reference: HF encoder features (cosine) and HF greedy transcript (exact match).
  import torch
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  proc = WhisperProcessor.from_pretrained(NAME)
  hf = WhisperForConditionalGeneration.from_pretrained(NAME).eval()
  feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
  with torch.no_grad():
    ref_feat = hf.model.encoder(feats).last_hidden_state[0].numpy()
    ref_text = proc.batch_decode(hf.generate(feats), skip_special_tokens=True)[0]
  ane_feat = w.encode(audio)
  cos = float(ane_feat.ravel() @ ref_feat.ravel() / (np.linalg.norm(ane_feat) * np.linalg.norm(ref_feat)))
  match = text.strip().lower() == ref_text.strip().lower()

  print("\nVALIDATION & BENCHMARKS:")
  print(f"  Encoder features cosine vs HF fp32: {cos:.5f}")
  print(f"  Transcript matches HF greedy:       {match}")
  print(f"  Transcribe latency on ANE (warm):   {dt * 1e3:.0f} ms")
  if not match:
    print(f"  HF transcript: {ref_text.strip()!r}")

  ok = cos > 0.99 and match
  print("\nRESULT:", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
