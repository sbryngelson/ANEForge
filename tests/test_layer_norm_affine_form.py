"""layer_norm's affine layout, asserted off-device so CI runs it (#162).

A non-uniform affine const on the CHANNEL axis fails ANECCompile from D=1024 up, so the affine must
be applied after the reshape back, at rank 2 on the last axis. That is a property of the emitted MIL,
and `_lower_fused_to_dir` produces it without compiling or dispatching, so CI can check it even
though CI cannot reach an ANE.
"""
import re

import numpy as np

import aneforge as af
from aneforge._compile import _lower_fused_to_dir

R, D = 8, 1024


def _mil(tmp_path, gamma, beta):
  x = af.input((R, D))
  d = _lower_fused_to_dir(x.layer_norm(gamma, beta), build_dir=str(tmp_path / "prog"))
  return (d / "model.mil").read_text()


def _affine(tmp_path):
  """MIL for a non-uniform affine, the case that regressed."""
  rng = np.random.default_rng(13)
  g = (rng.standard_normal(D).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
  b = (rng.standard_normal(D).astype(np.float32) * 0.1).astype(np.float16)
  return _mil(tmp_path, g, b)


def test_affine_consts_are_rank2_on_the_last_axis(tmp_path):
  # [1,D,1,1] puts D on the channel axis, which is what ANECCompile rejects above D=1024.
  mil = _affine(tmp_path)
  for slot in ("_g", "_b"):
    decl = re.search(rf"tensor<fp16, \[([^\]]+)\]> \w+{slot} = const\(\)", mil)
    assert decl, f"no const declaration for the {slot} affine slot"
    assert [s.strip() for s in decl.group(1).split(",")] == ["1", str(D)], \
        f"{slot} affine const is [{decl.group(1)}], expected [1, {D}]"


def test_affine_is_applied_after_the_reshape_back(tmp_path):
  # Order matters as much as shape: a [1,D] const multiplied while still rank-4 would broadcast
  # differently and lose the property this guards.
  mil = _affine(tmp_path)
  assert mil.index("_rb = reshape(") < mil.index("_gg = mul("), \
      "gamma is applied before the reshape back to [M,D]"
  assert mil.index("_gg = mul(") < mil.rindex("= add("), "beta is applied before gamma"


def test_normalized_body_stays_rank4(tmp_path):
  # The reduce still needs the [M,D,1,1] form; only the affine moved. Guards an over-correction
  # that would also move the reduce and change what is measured.
  mil = _affine(tmp_path)
  assert f"tensor<fp16,[{R},{D},1,1]> " in mil, "the normalization body is no longer rank-4"
  assert "reduce_mean(" in mil


def test_layout_does_not_depend_on_the_affine_values(tmp_path):
  # The emitted MIL is value-independent: consts are BLOBFILE references. A uniform affine must not
  # take a different path from a non-uniform one, or the fix would only cover the case it was
  # written against.
  uniform = _mil(tmp_path / "u", np.full(D, 1.1, np.float16), np.full(D, 0.1, np.float16))
  varied = _affine(tmp_path / "v")
  assert uniform == varied
