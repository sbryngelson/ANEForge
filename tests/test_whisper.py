"""Whisper (whisper-base.en) on the ANE: weight mapping (off-device), and encoder/transcript
parity vs Hugging Face (requires_ane). See docs/superpowers/specs/2026-08-18-whisper-on-ane-design.md."""
import numpy as np

from _helpers import requires_ane
from aneforge.models import _whisper_layers


def _sd():
  from transformers import WhisperForConditionalGeneration
  m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
  return {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}


def test_whisper_layer_mapping():
  sd = _sd()
  enc = _whisper_layers(sd, "encoder", 6)
  dec = _whisper_layers(sd, "decoder", 6)
  assert len(enc) == 6 and len(dec) == 6
  # self-attn q/o carry bias, k has none (Whisper convention)
  assert enc[0]["Wq"].shape == (512, 512) and enc[0]["bq"].shape == (512,)
  assert "bk" not in enc[0] and enc[0]["Wk"].shape == (512, 512)
  assert enc[0]["Wi"].shape == (2048, 512) and enc[0]["Wd"].shape == (512, 2048)
  # the decoder layer also carries the cross-attn set (k again unbiased)
  assert dec[0]["CWq"].shape == (512, 512) and dec[0]["CWk"].shape == (512, 512)
  assert "Cbk" not in dec[0] and dec[0]["cln_w"].shape == (512,)
  # values come straight from the real weights
  assert np.array_equal(enc[0]["Wq"], sd["model.encoder.layers.0.self_attn.q_proj.weight"])
  assert np.array_equal(dec[0]["CWo"], sd["model.decoder.layers.0.encoder_attn.out_proj.weight"])


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
  assert cos > 0.99, f"encoder cosine {cos}"


@requires_ane
def test_whisper_transcribes_like_hf():
  import io

  import soundfile as sf
  import torch
  from datasets import Audio, load_dataset
  from transformers import WhisperForConditionalGeneration, WhisperProcessor

  from aneforge.models import load_whisper
  # decode=False keeps the raw FLAC bytes so we read them with soundfile (the dataset's own decoder
  # needs torchcodec); librispeech-dummy is already 16 kHz mono, matching Whisper's expected rate.
  ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                    split="validation").cast_column("audio", Audio(decode=False))
  a = ds[0]["audio"]
  data, _ = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
  audio = np.asarray(data, dtype=np.float32)
  text = load_whisper("openai/whisper-base.en").transcribe(audio)
  proc = WhisperProcessor.from_pretrained("openai/whisper-base.en")
  hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
  feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
  with torch.no_grad():
    ref = proc.batch_decode(hf.generate(feats), skip_special_tokens=True)[0]
  assert text.strip().lower() == ref.strip().lower(), f"\nane: {text!r}\nhf:  {ref!r}"
