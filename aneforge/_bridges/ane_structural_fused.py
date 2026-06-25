"""Native ANE structural layers (Flatten / Dropout / Broadcast) on Path A."""

from __future__ import annotations

import plistlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import _netplist as G


def _run_unit(unit: dict, name: str, x: np.ndarray) -> np.ndarray:
  """Author a one-unit netplist around `unit` and run x (C,H,W) through it."""
  if x.ndim != 3: raise ValueError("x must be (C, H, W)")
  C, H, W = x.shape
  x = np.ascontiguousarray(x.astype(np.float16))
  with tempfile.TemporaryDirectory(prefix="ane_struct_") as d:
    wd = Path(d)
    inp = G.input_entry("x", width=W, height=H, channels=C, entry_name=name)
    out = G.output_entry("y", name)
    plist = G.build_plist(f"network_{name}", name, [inp], [out], {name: unit})
    with (wd / "net.plist").open("wb") as f:
      plistlib.dump(plist, f, sort_keys=True)
    (wd / "weights.0").write_bytes(b"")
    (wd / "in_x.f16").write_bytes(x.tobytes())
    G.invoke_netplist(
      G.ensure_invoker("sdpa_invoker"), wd / "net.plist",
      weights=[wd / "weights.0"],
      inputs=[("x", wd / "in_x.f16")],
      outputs=[("y", wd / "out_y.f16")],
    )
    return np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16)


def flatten(x: np.ndarray) -> np.ndarray:
  """ANE `Type=Flatten` (NCHW). Returns x flattened to 1-D."""
  unit = {
    "Bottom": ["x"], "InputType": ["Float16"], "Name": "flatten-1",
    "OutputChannels": 1, "OutputType": "Float16",
    "Params": {"FlattenType": "NCHW", "Mode": "NCHW"}, "Type": "Flatten",
  }
  return _run_unit(unit, "flatten-1", x)


def dropout(x: np.ndarray, rate: float = 0.0) -> np.ndarray:
  """ANE `Type=Dropout`. Inference-only: rate must be 0 (identity)."""
  if rate != 0.0: raise ValueError("ANE Dropout is inference-only; rate must be 0.0")
  unit = {
    "Bottom": ["x"], "InputType": ["Float16"], "Name": "dropout-1",
    "OutputChannels": 1, "OutputType": "Float16",
    "Params": {"DropoutRate": 0, "Seed": 0}, "Type": "Dropout",
  }
  return _run_unit(unit, "dropout-1", x)


def _selftest() -> int:
  rng = np.random.default_rng(0)
  failures = 0

  x = np.arange(2 * 2 * 3, dtype=np.float16).reshape(2, 2, 3)
  y = flatten(x)
  ok = np.array_equal(y, x.reshape(-1))
  print(f"Flatten   {'OK ' if ok else 'FAIL'} shape {x.shape} -> {y.shape}")
  failures += not ok

  x = rng.standard_normal((1, 1, 8)).astype(np.float16)
  y = dropout(x)
  ok = np.array_equal(y, x.reshape(-1))
  print(f"Dropout   {'OK ' if ok else 'FAIL'} (identity at rate=0)")
  failures += not ok

  # Broadcast is exercised through broadcast_const_add (length-1 -> width).
  with tempfile.TemporaryDirectory(prefix="ane_bcast_") as d:
    wd = Path(d)
    G.write_model("broadcast_const_add", wd, width=4, const_value=5.0)
    xb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    (wd / "in_x.f16").write_bytes(xb.tobytes())
    weights = sorted(wd.glob("weights.*"))
    cmd = [str(G.ensure_invoker("sdpa_invoker")), "--net-plist", str(wd / "net.plist")]
    for w in weights:
      cmd += ["--weights", str(w)]
    cmd += ["--input", f"x={wd / 'in_x.f16'}", "--output", f"y={wd / 'out_y.f16'}", "--repeats", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    yb = np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16) if proc.returncode == 0 else None
  ok = yb is not None and np.array_equal(yb, xb + 5.0)
  print(f"Broadcast {'OK ' if ok else 'FAIL'} (length-1 const -> width-4, then add)")
  failures += not ok

  print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
  return failures


if __name__ == "__main__":
  raise SystemExit(_selftest())
