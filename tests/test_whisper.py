"""Whisper (whisper-base.en) on the ANE: weight mapping (off-device), and encoder/transcript
parity vs Hugging Face (requires_ane). See docs/superpowers/specs/2026-08-18-whisper-on-ane-design.md."""
import numpy as np

from _helpers import requires_ane
from aneforge.models import _whisper_layers

D, DFF = 512, 2048


def _synthetic_sd(prefix: str, n: int) -> dict:
  """A Whisper state dict with the real HF key names and shapes but random weights -- enough to exercise
  the mapping off-device (no transformers, no download). k_proj has no bias, matching Whisper."""
  rng = np.random.default_rng(0)
  w = lambda *s: rng.standard_normal(s).astype(np.float32)
  sd = {}
  for i in range(n):
    p = f"model.{prefix}.layers.{i}."
    for proj, bias in [("q_proj", True), ("k_proj", False), ("v_proj", True), ("out_proj", True)]:
      sd[p + f"self_attn.{proj}.weight"] = w(D, D)
      if bias:
        sd[p + f"self_attn.{proj}.bias"] = w(D)
    for ln in ("self_attn_layer_norm", "final_layer_norm"):
      sd[p + ln + ".weight"], sd[p + ln + ".bias"] = w(D), w(D)
    sd[p + "fc1.weight"], sd[p + "fc1.bias"] = w(DFF, D), w(DFF)
    sd[p + "fc2.weight"], sd[p + "fc2.bias"] = w(D, DFF), w(D)
    if prefix == "decoder":
      for proj, bias in [("q_proj", True), ("k_proj", False), ("v_proj", True), ("out_proj", True)]:
        sd[p + f"encoder_attn.{proj}.weight"] = w(D, D)
        if bias:
          sd[p + f"encoder_attn.{proj}.bias"] = w(D)
      sd[p + "encoder_attn_layer_norm.weight"] = w(D)
      sd[p + "encoder_attn_layer_norm.bias"] = w(D)
  return sd


def test_whisper_layer_mapping():
  enc_sd, dec_sd = _synthetic_sd("encoder", 6), _synthetic_sd("decoder", 6)
  enc = _whisper_layers(enc_sd, "encoder", 6)
  dec = _whisper_layers(dec_sd, "decoder", 6)
  assert len(enc) == 6 and len(dec) == 6
  # self-attn q/o carry bias, k has none (Whisper convention)
  assert enc[0]["Wq"].shape == (D, D) and enc[0]["bq"].shape == (D,)
  assert "bk" not in enc[0] and enc[0]["Wk"].shape == (D, D)
  assert enc[0]["Wi"].shape == (DFF, D) and enc[0]["Wd"].shape == (D, DFF)
  # the decoder layer also carries the cross-attn set (k again unbiased)
  assert dec[0]["CWq"].shape == (D, D) and dec[0]["CWk"].shape == (D, D)
  assert "Cbk" not in dec[0] and dec[0]["cln_w"].shape == (D,)
  # values come straight from the state dict, unmodified
  assert np.array_equal(enc[0]["Wq"], enc_sd["model.encoder.layers.0.self_attn.q_proj.weight"])
  assert np.array_equal(dec[0]["CWo"], dec_sd["model.decoder.layers.0.encoder_attn.out_proj.weight"])


@requires_ane
def test_whisper_encoder_matches_hf():
  import torch
  from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

  from aneforge.models import load_whisper
  w = load_whisper("openai/whisper-base.en")
  audio = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.05).astype(np.float32)
  feat = w.encode(audio)                                      # [1500, 512] on the ANE
  fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base.en")
  mel = fe(audio, sampling_rate=16000, return_tensors="pt")["input_features"]
  hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
  with torch.no_grad():
    ref = hf.model.encoder(mel).last_hidden_state[0].numpy()
  cos = float((feat.ravel() @ ref.ravel()) / (np.linalg.norm(feat) * np.linalg.norm(ref) + 1e-9))
  assert cos > 0.999, f"encoder cosine {cos}"


def _librispeech_clips():
  """The first few librispeech-dummy clips as 16 kHz mono float32 waveforms. decode=False keeps the raw FLAC
  bytes so soundfile reads them (the dataset's own decoder needs torchcodec)."""
  import io

  import soundfile as sf
  from datasets import Audio, load_dataset
  ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                    split="validation").cast_column("audio", Audio(decode=False))

  def clip(i):
    a = ds[i]["audio"]
    data, _ = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
    return np.asarray(data, dtype=np.float32)
  return [clip(i) for i in range(4)]


@requires_ane
def test_whisper_transcribes_like_hf():
  """Greedy transcript matches HF `generate` on several clips -- a single clip can dodge the logit
  suppressions HF applies, so this checks a handful to exercise the begin/step suppression paths."""
  import torch
  from transformers import WhisperForConditionalGeneration, WhisperProcessor

  from aneforge.models import load_whisper
  clips = _librispeech_clips()
  w = load_whisper("openai/whisper-base.en")
  proc = WhisperProcessor.from_pretrained("openai/whisper-base.en")
  hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
  for audio in clips:
    text = w.transcribe(audio)
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
      ref = proc.batch_decode(hf.generate(feats), skip_special_tokens=True)[0]
    assert text.strip().lower() == ref.strip().lower(), f"\nane: {text!r}\nhf:  {ref!r}"


@requires_ane
def test_whisper_kv_cache_resets_between_clips():
  """The per-clip cross-attention K/V (fed into every decode step) must be refreshed each call: transcribing
  clip A after clip B must equal transcribing A alone -- otherwise A would decode against B's audio."""
  from aneforge.models import load_whisper
  clips = _librispeech_clips()
  w = load_whisper("openai/whisper-base.en")
  first = w.transcribe(clips[0])
  w.transcribe(clips[1])                                       # a different-length clip in between
  again = w.transcribe(clips[0])
  assert first.strip() and first == again, f"\nfirst: {first!r}\nagain: {again!r}"
