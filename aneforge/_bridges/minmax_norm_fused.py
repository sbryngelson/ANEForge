"""Native ANE `MinMaxNormalization` on Path A: y = (x-min)/(max-min+eps) over
`Params.Dimension` (Width|Height; Channel is ARCH-GATED). `Params.Epsilon` is a
fp16 bit pattern. See docs/developer/bridges.md."""

from __future__ import annotations

import plistlib
import tempfile
from pathlib import Path

import numpy as np

from ._netplist import build_plist, ensure_invoker, input_entry, invoke_netplist, output_entry

_AXIS = {"Width": 3, "Height": 2, "Channel": 1}


def fp16_bits(value: float) -> int:
  return int(np.array(value, dtype=np.float16).view(np.uint16).item())


def _plist(channels: int, height: int, width: int, dimension: str, eps_bits: int) -> dict:
  unit = {"Bottom": ["x"], "InputType": ["Float16"], "Name": "op-1",
          "OutputChannels": channels, "OutputType": "Float16",
          "Type": "MinMaxNormalization",
          "Params": {"Dimension": dimension, "Epsilon": eps_bits}}
  return build_plist(
    "network_minmax-1", "op-1",
    [input_entry("x", width=width, height=height, channels=channels, entry_name="op-1")],
    [output_entry("y", "op-1")], {"op-1": unit})


def minmax_norm_fused(x: np.ndarray, *, dimension: str = "Width",
                      epsilon: float = 1e-4) -> np.ndarray:
  """Min-max normalize fp16 x (B=1,C,H,W) over `dimension` (Width|Height; Channel ARCH-GATED) on the ANE; eps stabilizes the (max-min) denominator."""
  x = np.asarray(x, dtype=np.float16)
  if x.ndim != 4: raise ValueError("x must be (B,C,H,W)")
  B, C, H, W = x.shape
  if B != 1: raise ValueError("B=1 only")
  invoker = ensure_invoker("layer_invoker")
  x5 = x.reshape(1, C, 1, H, W)
  with tempfile.TemporaryDirectory(prefix="ane_mm_") as d:
    cd = Path(d)
    with (cd / "net.plist").open("wb") as f:
      plistlib.dump(_plist(C, H, W, dimension, fp16_bits(epsilon)), f,
                    fmt=plistlib.FMT_BINARY)
    (cd / "weights.0").write_bytes(b"\x00" * 1024)
    (cd / "in_x.bin").write_bytes(x5.astype(np.float16).tobytes())
    invoke_netplist(
      invoker, cd / "net.plist",
      weights=[cd / "weights.0"], inputs=[("x", cd / "in_x.bin")],
      outputs=[("y", cd / "out_y.bin")], warmup=0,
      extra=["--raw-output", f"y={cd / 'out_raw.bin'}"])
    raw = (cd / "out_y.bin").read_bytes()
    return np.frombuffer(raw[:C * H * W * 2], dtype=np.float16).reshape(C, H, W)


def numpy_reference(x: np.ndarray, *, dimension: str = "Width",
                    epsilon: float = 1e-4) -> np.ndarray:
  xf = x.astype(np.float32)
  ax = _AXIS[dimension]
  mn = xf.min(axis=ax, keepdims=True); mx = xf.max(axis=ax, keepdims=True)
  return ((xf - mn) / (mx - mn + epsilon)).astype(np.float16)


if __name__ == "__main__":
  x = np.arange(1, 17, dtype=np.float16).reshape(1, 2, 2, 4)
  for dim in ("Width", "Height"):
    y = minmax_norm_fused(x, dimension=dim, epsilon=1e-4)
    ref = numpy_reference(x, dimension=dim, epsilon=1e-4).reshape(2, 2, 4)
    err = float(np.abs(y.astype(np.float32) - ref.astype(np.float32)).max())
    print(f"ANE MinMaxNorm({dim}) max_abs_err={err:.4g}")
    assert err < 0.05, f"numerics mismatch on {dim}"
  print("CRACKED: MinMaxNormalization verified (Width, Height). Channel ARCH-GATED.")
