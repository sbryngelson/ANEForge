"""A last-axis-offset slice_by_size is lowered off the last axis (aneforge/_compile.py, #310):
transpose, slice with a zero last-axis begin, transpose back. The MIL checks are build-level and
off-device; the value check runs on the ANE."""
import re
from pathlib import Path

import numpy as np
import pytest
from _helpers import requires_ane

import aneforge as af
from aneforge._compile import _lower_fused_to_dir


def _mil(t, tmp_path, name):
  d = _lower_fused_to_dir(t, str(tmp_path / name))
  return Path(d, "model.mil").read_text()


def _slices(mil):   # (begin, size) of every emitted slice_by_size, in order
  vals = dict(re.findall(r"(\w+) = const\(\)\[name = string\(\"\w+\"\), val = tensor<int32, \[\d+\]>\((\[[^\]]*\])\)\]", mil))
  return [(vals[b], vals[s]) for b, s in re.findall(r"slice_by_size\(begin = (\w+), size = (\w+),", mil)]


def _last_begins(mil):
  return [int(b.strip("[]").split(",")[-1]) for b, _ in _slices(mil)]


def test_width_offset_slice_is_routed(tmp_path, monkeypatch):
  monkeypatch.delenv("ANEFORGE_ROUTE_WIDTH_SLICES", raising=False)
  # only the last axis is offset: one slice, taken with the last two axes swapped
  mil = _mil(af.input((2, 8, 6)).slice_by_size([0, 0, 4], [2, 8, 2]).relu(), tmp_path, "a")
  assert _slices(mil) == [("[0, 4, 0]", "[2, 2, 8]")]
  assert mil.count("transpose(") == 2 and "tensor<fp16, [2, 8, 2]>" in mil      # back to the asked shape


def test_offsets_on_both_trailing_axes_leave_no_last_axis_begin(tmp_path, monkeypatch):
  monkeypatch.delenv("ANEFORGE_ROUTE_WIDTH_SLICES", raising=False)
  # swapping alone would turn begin [0,1,4] into [0,4,1]: still a nonzero last-axis begin. The
  # other axes are taken first, then the last-axis window.
  mil = _mil(af.input((2, 8, 6)).slice_by_size([0, 1, 4], [2, 3, 2]).relu(), tmp_path, "a")
  assert _slices(mil) == [("[0, 1, 0]", "[2, 3, 6]"), ("[0, 4, 0]", "[2, 2, 3]")]
  assert "tensor<fp16, [2, 3, 2]>" in mil
  # the conv-gradient patch extraction: begin [0, 0, u, v] with u and v both positive
  mil = _mil(af.input((1, 4, 9, 9)).slice_by_size([0, 0, 2, 1], [1, 4, 7, 7]).relu(), tmp_path, "b")
  assert _last_begins(mil) == [0, 0] and len(_slices(mil)) == 2


def test_other_slices_are_emitted_as_written(tmp_path, monkeypatch):
  monkeypatch.delenv("ANEFORGE_ROUTE_WIDTH_SLICES", raising=False)
  # zero last-axis begin: nothing to route
  mil = _mil(af.input((8, 6)).slice_by_size([3, 0], [2, 6]).relu(), tmp_path, "a")
  assert _slices(mil) == [("[3, 0]", "[2, 6]")] and "transpose(" not in mil
  # the switch emits a width-offset slice as written (what bench/slice_offset_probe.py calls `raw`)
  monkeypatch.setenv("ANEFORGE_ROUTE_WIDTH_SLICES", "0")
  mil = _mil(af.input((8, 6)).slice_by_size([0, 4], [8, 2]).relu(), tmp_path, "b")
  assert _slices(mil) == [("[0, 4]", "[8, 2]")] and "transpose(" not in mil


@requires_ane
@pytest.mark.parametrize("shape,begin,size", [
  ((8, 6), [0, 4], [8, 2]),                       # last axis only
  ((8, 6), [3, 2], [4, 3]),                       # both axes
  ((2, 8, 6), [0, 1, 4], [2, 3, 2]),              # both trailing axes
  ((2, 8, 6), [1, 0, 5], [1, 8, 1]),              # leading + last, second-to-last full
  ((1, 4, 9, 9), [0, 0, 2, 1], [1, 4, 7, 7]),     # conv-gradient patch
  ((1, 4, 9, 9), [0, 1, 0, 3], [1, 2, 9, 5]),     # channel + last
])
def test_routed_slice_values_match_numpy(shape, begin, size):
  x = np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape) / 8
  out = np.asarray(af.compile(af.input(shape).slice_by_size(begin, size) * 2.0)(x), np.float32)
  ref = x[tuple(slice(b, b + s) for b, s in zip(begin, size))].astype(np.float32) * 2.0
  assert out.shape == ref.shape and np.array_equal(out, ref)
