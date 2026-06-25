"""Native ANE `ScaledElementWise` on Path A: y = scale * (x OP z). `Params.Type`
selects the op; `Params.Scale` is a fp16 bit pattern. See docs/developer/bridges.md."""

from __future__ import annotations

import plistlib
import tempfile
from pathlib import Path

import numpy as np

from ._netplist import build_plist, ensure_invoker, input_entry, invoke_netplist, output_entry

_OP = {"Add": np.add, "Mult": np.multiply, "Sub": np.subtract,
       "Min": np.minimum, "Max": np.maximum}


def fp16_bits(value: float) -> int:
  return int(np.array(value, dtype=np.float16).view(np.uint16).item())


def _plist(width: int, op_type: str, scale_bits: int) -> dict:
  unit = {"Bottom": ["x", "z"], "InputType": ["Float16", "Float16"],
          "Name": "op-1", "OutputChannels": 1, "OutputType": "Float16",
          "Type": "ScaledElementWise",
          "Params": {"Type": op_type, "Scale": scale_bits}}
  return build_plist(
    "network_scaled_ew-1", "op-1",
    [input_entry("x", width=width, entry_name="op-1"),
     input_entry("z", width=width, entry_name="op-1")],
    [output_entry("y", "op-1")], {"op-1": unit})


def scaled_elementwise_fused(x: np.ndarray, z: np.ndarray, *,
                             op: str = "Add", scale: float = 1.0) -> np.ndarray:
  """Compute `scale * (x OP z)` on the ANE; x/z are fp16 1-D, op in Add|Mult|Sub|Min|Max."""
  x = np.asarray(x, dtype=np.float16).reshape(-1)
  z = np.asarray(z, dtype=np.float16).reshape(-1)
  if x.shape != z.shape: raise ValueError("x and z must share shape")
  W = x.shape[0]
  invoker = ensure_invoker("layer_invoker")
  with tempfile.TemporaryDirectory(prefix="ane_se_") as d:
    cd = Path(d)
    with (cd / "net.plist").open("wb") as f:
      plistlib.dump(_plist(W, op, fp16_bits(scale)), f, fmt=plistlib.FMT_BINARY)
    (cd / "weights.0").write_bytes(b"\x00" * 1024)
    (cd / "in_x.bin").write_bytes(x.reshape(1, 1, 1, 1, W).tobytes())
    (cd / "in_z.bin").write_bytes(z.reshape(1, 1, 1, 1, W).tobytes())
    invoke_netplist(
      invoker, cd / "net.plist",
      weights=[cd / "weights.0"],
      inputs=[("x", cd / "in_x.bin"), ("z", cd / "in_z.bin")],
      outputs=[("y", cd / "out_y.bin")], warmup=0,
      extra=["--raw-output", f"y={cd / 'out_raw.bin'}"])
    raw = (cd / "out_y.bin").read_bytes()
    return np.frombuffer(raw[:W * 2], dtype=np.float16)


def numpy_reference(x, z, *, op="Add", scale=1.0) -> np.ndarray:
  xf = np.asarray(x, dtype=np.float32).reshape(-1)
  zf = np.asarray(z, dtype=np.float32).reshape(-1)
  return (float(scale) * _OP[op](xf, zf)).astype(np.float16)


if __name__ == "__main__":
  x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
  z = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float16)
  y = scaled_elementwise_fused(x, z, op="Add", scale=2.0)
  ref = numpy_reference(x, z, op="Add", scale=2.0)
  err = float(np.abs(y.astype(np.float32) - ref.astype(np.float32)).max())
  print(f"ANE ScaledElementWise(Add, scale=2) out={y.tolist()} ref={ref.tolist()} max_abs_err={err:.4g}")
  assert err < 0.05
  print("CRACKED: ScaledElementWise verified.")
