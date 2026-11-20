"""A last-axis-offset slice_by_size is lowered off the last axis (aneforge/_compile.py, #310):
transpose, slice with a zero last-axis begin, transpose back. Build-level, off-device."""
import re
from pathlib import Path

import aneforge as af
from aneforge._compile import _lower_fused_to_dir


def _mil(t, tmp_path, name):
  d = _lower_fused_to_dir(t, str(tmp_path / name))
  return Path(d, "model.mil").read_text()


def _slices(mil):   # (begin, size) of every emitted slice_by_size, in order
  vals = dict(re.findall(r"(\w+) = const\(\)\[name = string\(\"\w+\"\), val = tensor<int32, \[\d+\]>\((\[[^\]]*\])\)\]", mil))
  return [(vals[b], vals[s]) for b, s in re.findall(r"slice_by_size\(begin = (\w+), size = (\w+),", mil)]


def test_width_offset_slice_is_routed(tmp_path, monkeypatch):
  monkeypatch.delenv("ANEFORGE_ROUTE_WIDTH_SLICES", raising=False)
  mil = _mil(af.input((2, 8, 6)).slice_by_size([0, 1, 4], [2, 3, 2]).relu(), tmp_path, "a")
  # sliced with the last two axes swapped: begin [0,4,1], size [2,2,3]; the last-axis begin is not 4
  assert _slices(mil) == [("[0, 4, 1]", "[2, 2, 3]")]
  assert mil.count("transpose(") == 2 and "tensor<fp16, [2, 3, 2]>" in mil      # back to the asked shape


def test_other_slices_are_emitted_as_written(tmp_path, monkeypatch):
  monkeypatch.delenv("ANEFORGE_ROUTE_WIDTH_SLICES", raising=False)
  # zero last-axis begin: nothing to route
  mil = _mil(af.input((8, 6)).slice_by_size([3, 0], [2, 6]).relu(), tmp_path, "a")
  assert _slices(mil) == [("[3, 0]", "[2, 6]")] and "transpose(" not in mil
  # the switch emits a width-offset slice as written (what bench/slice_offset_probe.py calls `raw`)
  monkeypatch.setenv("ANEFORGE_ROUTE_WIDTH_SLICES", "0")
  mil = _mil(af.input((8, 6)).slice_by_size([0, 4], [8, 2]).relu(), tmp_path, "b")
  assert _slices(mil) == [("[0, 4]", "[8, 2]")] and "transpose(" not in mil
