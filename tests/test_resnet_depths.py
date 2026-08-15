"""Stage layout and shortcut rules for the ResNet loader.

The parts that generalize `load_resnet18` to 18/34/50/101 are host-side and hardware-free: which
stage layout each depth uses, and whether a block's residual branch is projected. The end-to-end
check against torchvision needs an ANE and pretrained weights, so it lives in the PR body.

`[dev]` alone does not install torchvision (it sits in the `models` extra), so the tests that need it
skip rather than fail. The ones that pin the constants run everywhere.
"""
import numpy as np
import pytest

from aneforge.models import _RESNETS, _resnet_depth, Vision

EXPECTED = {                       # mirrors the torchvision factories
  18: ("basic", (2, 2, 2, 2)),
  34: ("basic", (3, 4, 6, 3)),
  50: ("bottleneck", (3, 4, 6, 3)),
  101: ("bottleneck", (3, 4, 23, 3)),
}


def test_layout_table():
  assert _RESNETS == EXPECTED


@pytest.mark.parametrize("given,depth", [(18, 18), ("18", 18), ("resnet50", 50), ("ResNet101", 101), (34, 34)])
def test_depth_accepts_int_str_and_name(given, depth):
  assert _resnet_depth(given) == depth


@pytest.mark.parametrize("bad", [19, "resnet19", "152", "resnext50", "", "resnet"])
def test_depth_rejects_unsupported(bad):
  with pytest.raises(ValueError, match="unsupported ResNet"):
    _resnet_depth(bad)


def test_shortcut_is_the_identity_when_the_block_has_no_projection():
  """No `downsample.0.weight` in the weights means the residual branch passes x through untouched."""
  v = object.__new__(Vision)
  v.sd = {}
  x = object()                     # the identity path must not look at it
  assert v._shortcut(x, "layer1.0", 1) is x


def test_shortcut_reads_the_weights_rather_than_a_stage_rule():
  """A block with a projection must take the conv branch even at stride 1, which is exactly the
  Bottleneck stage-1 case (64 -> 256). Keyed off the weights, stride carries no meaning here."""
  v = object.__new__(Vision)
  v.sd = {
    "layer1.0.downsample.0.weight": np.zeros((256, 64, 1, 1), np.float32),
    "layer1.0.downsample.1.weight": np.ones(256, np.float32),
    "layer1.0.downsample.1.bias": np.zeros(256, np.float32),
    "layer1.0.downsample.1.running_mean": np.zeros(256, np.float32),
    "layer1.0.downsample.1.running_var": np.ones(256, np.float32),
  }
  from aneforge.graph import input as _input
  out = v._shortcut(_input((1, 64, 56, 56)), "layer1.0", 1)
  assert out.shape == (1, 256, 56, 56), "stage-1 Bottleneck must project 64 -> 256 at stride 1"


def test_layout_table_matches_torchvision():
  tv = pytest.importorskip("torchvision")
  from torchvision.models.resnet import BasicBlock, Bottleneck
  for depth, (block, stages) in _RESNETS.items():
    m = getattr(tv.models, f"resnet{depth}")(weights=None)     # no download: architecture only
    got = tuple(len(getattr(m, f"layer{i}")) for i in (1, 2, 3, 4))
    kind = Bottleneck if block == "bottleneck" else BasicBlock
    assert got == stages, f"resnet{depth} stage layout"
    assert isinstance(m.layer1[0], kind), f"resnet{depth} block type"


def test_bottleneck_projects_stage_one_and_basicblock_does_not():
  """The reason `_shortcut` reads the weights. A "stage 1 never downsamples" rule is true for
  BasicBlock and false for Bottleneck, where layer1 changes 64 channels into 256 at stride 1."""
  tv = pytest.importorskip("torchvision")
  for depth, projected in [(18, False), (34, False), (50, True), (101, True)]:
    sd = getattr(tv.models, f"resnet{depth}")(weights=None).state_dict()
    assert ("layer1.0.downsample.0.weight" in sd) is projected, f"resnet{depth} layer1 projection"
