"""Native ANE scaled-elementwise op via a hand-authored ANECIR netplist.

`ScaledElementWise` (ANECIR Type) fuses a binary elementwise op with a
scalar scale factor:   y = scale * (x OP z).  `Params.Type` selects the
elementwise op (Add / Mult / Sub / ...) and `Params.Scale` is a fp16 bit
pattern (use `fp16_bits`).
"""

from __future__ import annotations

import json
import plistlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ._netplist import bin_dir

_INVOKER = bin_dir() / "layer_invoker"
_INVOKER_SRC = Path(__file__).resolve().parents[1] / "_invokers" / "layer_invoker.mm"

_OP = {"Add": np.add, "Mult": np.multiply, "Sub": np.subtract,
       "Min": np.minimum, "Max": np.maximum}


def fp16_bits(value: float) -> int:
  return int(np.array(value, dtype=np.float16).view(np.uint16).item())


def _ensure_invoker() -> Path:
  if _INVOKER.exists() and _INVOKER.stat().st_mtime >= _INVOKER_SRC.stat().st_mtime:
    return _INVOKER
  _INVOKER.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["xcrun", "clang++", "-O2", "-fobjc-arc", "-std=gnu++17",
         "-framework", "Foundation", "-framework", "IOSurface",
         str(_INVOKER_SRC), "-o", str(_INVOKER)]
  p = subprocess.run(cmd, capture_output=True, text=True)
  if p.returncode != 0: raise RuntimeError(p.stderr)
  return _INVOKER


def _plist(width: int, op_type: str, scale_bits: int) -> dict:
  net = "network_scaled_ew-1"
  unit = {"Bottom": ["x", "z"], "InputType": ["Float16", "Float16"],
          "Name": "op-1", "OutputChannels": 1, "OutputType": "Float16",
          "Type": "ScaledElementWise",
          "Params": {"Type": op_type, "Scale": scale_bits}}

  def _in(sym):
    return {"BatchSize": 1, "InputChannels": 1, "InputDepth": 1,
            "InputHeight": 1, "InputInterleave": 1, "InputName": sym,
            "InputType": "Float16", "InputWidth": width, "Name": "op-1",
            "OperationName": "op0"}
  return {"Version": "1.0.10", "Networks": [net],
          "ProcedureList": [{"Name": "procedure_scaled_ew-1",
              "InputList": [_in("x"), _in("z")],
              "OperationList": [{"NetworkName": net, "OperationName": "op0"}],
              "OutputList": [{"Name": "op-1", "OperationName": "op0",
                              "OutputInterleave": 1, "OutputName": "y",
                              "OutputType": "Float16"}]}],
          net: {"op-1": unit, "Units": ["op-1"], "Weights": ["weights.0"],
                "y": {"Bottom": "op-1", "OutputInterleave": 1,
                      "OutputName": "y", "OutputType": "Float16"}}}


def scaled_elementwise_fused(x: np.ndarray, z: np.ndarray, *,
                             op: str = "Add", scale: float = 1.0) -> np.ndarray:
  """Compute `scale * (x OP z)` on the ANE.

    Args:
        x, z: fp16 1-D arrays (flattened along Width).
        op:   "Add" | "Mult" | "Sub" | "Min" | "Max".
        scale: scalar applied after the elementwise op.
    """
  x = np.asarray(x, dtype=np.float16).reshape(-1)
  z = np.asarray(z, dtype=np.float16).reshape(-1)
  if x.shape != z.shape: raise ValueError("x and z must share shape")
  W = x.shape[0]
  invoker = _ensure_invoker()
  with tempfile.TemporaryDirectory(prefix="ane_se_") as d:
    cd = Path(d)
    with (cd / "net.plist").open("wb") as f:
      plistlib.dump(_plist(W, op, fp16_bits(scale)), f, fmt=plistlib.FMT_BINARY)
    (cd / "weights.0").write_bytes(b"\x00" * 1024)
    (cd / "in_x.bin").write_bytes(x.reshape(1, 1, 1, 1, W).tobytes())
    (cd / "in_z.bin").write_bytes(z.reshape(1, 1, 1, 1, W).tobytes())
    cmd = [str(invoker), "--net-plist", str(cd / "net.plist"),
           "--weights", str(cd / "weights.0"),
           "--input", f"x={cd / 'in_x.bin'}", "--input", f"z={cd / 'in_z.bin'}",
           "--output", f"y={cd / 'out_y.bin'}",
           "--raw-output", f"y={cd / 'out_raw.bin'}", "--repeats", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
    if not lines: raise RuntimeError(f"no JSON from invoker:\n{p.stdout}\n{p.stderr}")
    info = json.loads(lines[-1])
    if info.get("status") != "ok": raise RuntimeError(f"scaled_elementwise invoker failed: {info}")
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
