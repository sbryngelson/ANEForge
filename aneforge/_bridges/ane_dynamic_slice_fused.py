"""Native ANE `DynamicSlice` (runtime/parametric slice) on Path A.
See docs/developer/bridges.md."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def dynamic_slice_fused(x: np.ndarray, start: int, *, slice_size: int = 2) -> np.ndarray:
  """Slice fp16 x to x[start:start+slice_size] on the ANE; start goes in the index constant (weights.1) for a runtime-selectable window. Accepted variant fixes W=4, SliceSize=2."""
  from . import _netplist as g

  x = np.asarray(x, dtype=np.float16).reshape(-1)
  if x.size != 4 or slice_size != 2:
    raise ValueError("the accepted netplist variant requires W=4, SliceSize=2")
  if start < 0 or start + slice_size > x.size: raise ValueError("slice window out of range")

  d = Path(tempfile.mkdtemp(prefix="ane_dynslice_"))
  g.write_model("dynamic_slice_const_u16", d, width=4, height=1, channels=1)
  (d / "weights.1").write_bytes(struct.pack("<H", int(start)))
  (d / "in_x.f16").write_bytes(x.tobytes())
  g.invoke_netplist(
    g.ensure_invoker("sdpa_invoker"), d / "net.plist",
    weights=[d / "weights.0", d / "weights.1"],
    inputs=[("x", d / "in_x.f16")], outputs=[("y", d / "out.f16")], warmup=0,
  )
  return np.frombuffer((d / "out.f16").read_bytes(), dtype=np.float16)


def numpy_reference(x: np.ndarray, start: int, slice_size: int = 2) -> np.ndarray:
  x = np.asarray(x, dtype=np.float16).reshape(-1)
  return x[start:start + slice_size]


if __name__ == "__main__":
  x = np.array([10, 20, 30, 40], dtype=np.float16)
  for start in (0, 1, 2):
    y = dynamic_slice_fused(x, start)
    ref = numpy_reference(x, start)
    ok = np.array_equal(y, ref)
    print(f"start={start}: ane={y.tolist()} ref={ref.tolist()} match={ok}")
    assert ok
  print("DynamicSlice CRACKED: parametric slice verified vs numpy.")
