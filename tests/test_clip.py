"""CLIP dual-encoder tests: vision and text encoder graph construction, off-device MIL lowering,
and on-device validation vs Hugging Face reference (examples/clip_zero_shot.py)."""
import numpy as np
import pytest

import aneforge as af
from aneforge._compile import _lower_fused_to_dir
from aneforge.models import _quick_gelu
from _helpers import requires_ane


def test_quick_gelu_lowers():
  """QuickGELU activation (x * sigmoid(1.702 * x)) lowers off-device with native ops."""
  x = af.input((4, 64))
  _lower_fused_to_dir(_quick_gelu(x), None)


def test_clip_vision_graph_lowers():
  """A minimal CLIP vision encoder block lowers to MIL off-device."""
  D, H = 64, 4
  x = af.input((1, 3, 64, 64))
  cls_in = af.input((1, D))
  pos_in = af.input((5, D))
  w_pe = np.zeros((D, 3 * 32 * 32, 1, 1), np.float32)
  h = af.conv(af.space_to_depth(x, 32), w_pe, bias=None)
  patches = h.reshape(1, D, 4).transpose([0, 2, 1]).reshape(4, D)
  seq = af.concat([cls_in, patches], axis=0) + pos_in
  seq = seq.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  # One transformer layer
  xn = seq.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  attn = af.mha(xn,
                np.zeros((D, D), np.float32), np.zeros(D, np.float32),
                np.zeros((D, D), np.float32), np.zeros(D, np.float32),
                np.zeros((D, D), np.float32), np.zeros(D, np.float32),
                np.zeros((D, D), np.float32), np.zeros(D, np.float32), H)
  seq = seq + attn
  yn = seq.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  mlp = _quick_gelu(yn.linear(np.zeros((128, D), np.float32), np.zeros(128, np.float32))).linear(
    np.zeros((D, 128), np.float32), np.zeros(D, np.float32))
  seq = seq + mlp
  cls_out = af.models._row0(seq).layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  v_proj = cls_out.linear(np.zeros((32, D), np.float32))
  _lower_fused_to_dir(v_proj.l2_norm(axis=-1), None)


def test_clip_text_graph_lowers():
  """A minimal causal CLIP text encoder block lowers to MIL off-device."""
  S, D, H = 8, 64, 4
  dh = D // H
  x = af.input((S, D))
  xn = x.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  q = xn.linear(np.zeros((D, D), np.float32), np.zeros(D, np.float32)).reshape(1, S, H, dh).transpose([0, 2, 1, 3])
  k = xn.linear(np.zeros((D, D), np.float32), np.zeros(D, np.float32)).reshape(1, S, H, dh).transpose([0, 2, 1, 3])
  v = xn.linear(np.zeros((D, D), np.float32), np.zeros(D, np.float32)).reshape(1, S, H, dh).transpose([0, 2, 1, 3])
  attn = af.models._causal_attn(q, k, v).transpose([0, 2, 1, 3]).reshape(S, D)
  seq = x + attn.linear(np.zeros((D, D), np.float32), np.zeros(D, np.float32))
  yn = seq.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  mlp = _quick_gelu(yn.linear(np.zeros((128, D), np.float32), np.zeros(128, np.float32))).linear(
    np.zeros((D, 128), np.float32), np.zeros(D, np.float32))
  seq = seq + mlp
  seq = seq.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), 1e-5)
  _lower_fused_to_dir(seq, None)


def test_clip_fallback_pixel_resizing():
  """Fallback pixel normalization handles non-224 raw image dimensions."""
  clip = af.models.CLIP.__new__(af.models.CLIP)
  clip.img = 224
  clip.proc = None
  # Pass raw [100, 150, 3] image array
  img = np.random.default_rng(0).integers(0, 256, (100, 150, 3), dtype=np.uint8)
  px = clip._pixels(img)
  assert px.shape == (1, 3, 224, 224)
  assert px.dtype == np.float32


@requires_ane
def test_clip_matches_huggingface():
  """On-device test: CLIP vision and text encoder embeddings have cosine > 0.99 vs HF reference."""
  import torch
  from transformers import CLIPModel, AutoTokenizer
  name = "openai/clip-vit-base-patch32"
  hf = CLIPModel.from_pretrained(name).eval()
  tok = AutoTokenizer.from_pretrained(name)
  clip = af.load_clip(name)

  # Test vision encoder
  dummy_px = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
  ane_img = clip.encode_image(dummy_px)

  # Test text encoder
  labels = ["a photo of a cat", "a photo of a dog"]
  ane_txt = clip.encode_text(labels)

  with torch.no_grad():
    txt_ids = tok(labels, padding="max_length", max_length=77, return_tensors="pt")["input_ids"]
    hf_out = hf(input_ids=txt_ids, pixel_values=torch.tensor(dummy_px))
    hf_img = hf_out.image_embeds.numpy()
    hf_txt = hf_out.text_embeds.numpy()

  cos_img = float(ane_img.ravel() @ hf_img.ravel() / (np.linalg.norm(ane_img) * np.linalg.norm(hf_img)))
  assert cos_img > 0.99, f"vision cosine {cos_img:.4f} <= 0.99"

  for i in range(len(labels)):
    cos_t = float(ane_txt[i] @ hf_txt[i] / (np.linalg.norm(ane_txt[i]) * np.linalg.norm(hf_txt[i])))
    assert cos_t > 0.99, f"text cosine {cos_t:.4f} <= 0.99"

  # Zero-shot classification test
  classified = clip.classify(dummy_px, labels)
  assert len(classified) == 2
  assert sum(p for _, p in classified) == pytest.approx(1.0, abs=1e-4)
