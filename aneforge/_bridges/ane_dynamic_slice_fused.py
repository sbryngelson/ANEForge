"""Native ANE `DynamicSlice` (runtime/parametric slice) on Path A.

:func:`dynamic_slice_fused` extracts a window `x[start : start + slice_size]`
along a tensor axis on the ANE, with `start` bound at runtime through a netplist
constant rather than baked into the op.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def dynamic_slice_fused(x: np.ndarray, start: int, *, slice_size: int = 2) -> np.ndarray:
    """Slice `x` (a length-W fp16 vector) to `x[start:start+slice_size]` on the ANE.

    Uses the `dynamic_slice_const_u16` generator and overwrites the index constant
    (weights.1) with `start` so the window is runtime-selectable.  The generator's
    accepted variant fixes `SliceSize=2` and W=4; this function validates
    `slice_size`/shape accordingly.
    """
    from . import _netplist as g

    x = np.asarray(x, dtype=np.float16).reshape(-1)
    if x.size != 4 or slice_size != 2:
        raise ValueError("the accepted netplist variant requires W=4, SliceSize=2")
    if start < 0 or start + slice_size > x.size:
        raise ValueError("slice window out of range")

    d = Path(tempfile.mkdtemp(prefix="ane_dynslice_"))
    g.write_model("dynamic_slice_const_u16", d, width=4, height=1, channels=1)
    (d / "weights.1").write_bytes(struct.pack("<H", int(start)))
    (d / "in_x.f16").write_bytes(x.tobytes())
    cmd = [
        str(g.ensure_invoker("sdpa_invoker")), "--net-plist", str(d / "net.plist"),
        "--weights", str(d / "weights.0"), "--weights", str(d / "weights.1"),
        "--input", f"x={d / 'in_x.f16'}", "--output", f"y={d / 'out.f16'}",
        "--repeats", "1", "--warmup", "0",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"dynamic_slice invoker failed:\n{p.stderr}")
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
