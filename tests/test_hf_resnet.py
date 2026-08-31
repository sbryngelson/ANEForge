"""load_resnet from a Hugging Face repo id (e.g. microsoft/resnet-50): key remap + HF-logit match."""
import numpy as np

from aneforge.models import _hf_resnet_to_tv
from _helpers import requires_ane


def _fake_hf(depths, layer_type="bottleneck"):
  """A minimal HF-named ResNet state_dict (zeros) for testing the pure key remap without a download."""
  hf: dict = {}

  def bn(p):
    for s in ("weight", "bias", "running_mean", "running_var"):
      hf[f"{p}.{s}"] = np.zeros(4, np.float32)

  hf["resnet.embedder.embedder.convolution.weight"] = np.zeros((4, 3, 7, 7), np.float32)
  bn("resnet.embedder.embedder.normalization")
  nconv = 3 if layer_type == "bottleneck" else 2
  for s, n in enumerate(depths):
    for li in range(n):
      hp = f"resnet.encoder.stages.{s}.layers.{li}"
      for hi in range(nconv):
        hf[f"{hp}.layer.{hi}.convolution.weight"] = np.zeros((4, 4, 1, 1), np.float32)
        bn(f"{hp}.layer.{hi}.normalization")
      if li == 0:                                    # first block of each stage projects the residual
        hf[f"{hp}.shortcut.convolution.weight"] = np.zeros((4, 4, 1, 1), np.float32)
        bn(f"{hp}.shortcut.normalization")
  hf["classifier.1.weight"] = np.zeros((1000, 4), np.float32)
  hf["classifier.1.bias"] = np.zeros(1000, np.float32)
  return hf


def test_hf_resnet_remap_produces_torchvision_keys():
  """A bottleneck HF checkpoint must remap to the exact torchvision names Vision._build reads."""
  sd, block, stages = _hf_resnet_to_tv(_fake_hf([2, 1]), "bottleneck", [2, 1])
  assert block == "bottleneck" and stages == [2, 1]
  expected = {"conv1.weight", "bn1.running_var",
              "layer1.0.conv1.weight", "layer1.0.conv2.weight", "layer1.0.conv3.weight",
              "layer1.0.bn3.weight", "layer1.0.downsample.0.weight", "layer1.0.downsample.1.running_mean",
              "layer1.1.conv1.weight", "layer2.0.conv3.weight", "fc.weight", "fc.bias"}
  assert expected <= set(sd), expected - set(sd)
  assert "layer1.1.downsample.0.weight" not in sd    # non-first blocks have no projection
  assert sd["fc.weight"].shape == (1000, 4)


def test_hf_resnet_remap_basic_has_two_convs():
  """A basic-block checkpoint maps to conv1/conv2 only (no conv3)."""
  sd, block, _ = _hf_resnet_to_tv(_fake_hf([1], "basic"), "basic", [1])
  assert block == "basic"
  assert "layer1.0.conv2.weight" in sd and "layer1.0.conv3.weight" not in sd


@requires_ane
def test_hf_resnet50_matches_hf():
  """load_resnet('microsoft/resnet-50') logits match HF on the same preprocessed image."""
  import torch
  from transformers import AutoModelForImageClassification
  import aneforge as af
  x = (np.random.default_rng(0).standard_normal((1, 3, 224, 224)) * 0.5).astype(np.float32)
  ane = np.asarray(af.load_resnet("microsoft/resnet-50")(x)).ravel()
  hf = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").eval()
  with torch.no_grad():
    ref = hf(torch.tensor(x)).logits[0].numpy()
  cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref) + 1e-9))
  assert cos > 0.99 and int(ane.argmax()) == int(ref.argmax()), f"cosine {cos:.4f}"
