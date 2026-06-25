"""Native ANE `InputView` (contiguous offset view) on Path A.
See docs/developer/bridges.md."""

from __future__ import annotations

import plistlib
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def input_view_fused(x: np.ndarray, offset: int, size: int, *, dimension: str = "Width") -> np.ndarray:
  """Return the ANE-computed view `x[offset:offset+size]` along `dimension`."""
  from . import _netplist as g

  x = np.asarray(x, dtype=np.float16).reshape(-1)
  W = x.size
  if offset < 0 or offset + size > W: raise ValueError("view window out of range")

  unit = {
    "Bottom": ["x"], "InputType": ["Float16"], "Name": "input_view-1",
    "OutputType": "Float16",
    "Params": {"Dimension": dimension, "Offset": int(offset), "Size": int(size)},
    "Type": "InputView",
  }
  pl = g.build_plist(
    "network_input_view-1", "input_view-1",
    [g.input_entry("x", width=W, height=1, channels=1, entry_name="input_view-1")],
    [g.output_entry("y", "input_view-1")], {"input_view-1": unit},
  )
  d = Path(tempfile.mkdtemp(prefix="ane_inputview_"))
  with (d / "net.plist").open("wb") as f:
    plistlib.dump(pl, f, sort_keys=True)
  (d / "weights.0").write_bytes(b"")
  (d / "in_x.f16").write_bytes(x.tobytes())
  g.invoke_netplist(
    g.ensure_invoker("sdpa_invoker"), d / "net.plist",
    weights=[d / "weights.0"], inputs=[("x", d / "in_x.f16")],
    outputs=[("y", d / "out.f16")], warmup=0,
  )
  return np.frombuffer((d / "out.f16").read_bytes(), dtype=np.float16)


def numpy_reference(x: np.ndarray, offset: int, size: int) -> np.ndarray:
  x = np.asarray(x, dtype=np.float16).reshape(-1)
  return x[offset:offset + size]


if __name__ == "__main__":
  x = np.arange(4, dtype=np.float16)
  for off, sz in [(1, 2), (0, 3), (2, 2)]:
    y = input_view_fused(x, off, sz)
    ref = numpy_reference(x, off, sz)
    ok = np.array_equal(y, ref)
    print(f"offset={off} size={sz}: ane={y.tolist()} ref={ref.tolist()} match={ok}")
    assert ok
  print("InputView CRACKED: view verified vs numpy.")
