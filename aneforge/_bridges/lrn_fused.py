"""Native ANE `LocalResponseNormalization` (cross-channel LRN) on Path A. `Params.Alpha` is a fp16 bit pattern, divided by KernelChannel internally."""

from __future__ import annotations

import plistlib
import tempfile
from pathlib import Path

import numpy as np

from ._netplist import build_plist, ensure_invoker, input_entry, invoke_netplist, output_entry


def fp16_bits(value: float) -> int:
  return int(np.array(value, dtype=np.float16).view(np.uint16).item())


def _plist(channels, height, width, kernel_channel, alpha_bits, beta, k) -> dict:
  unit = {"Bottom": ["x"], "InputType": ["Float16"], "Name": "op-1",
          "OutputChannels": channels, "OutputType": "Float16",
          "Type": "LocalResponseNormalization",
          "Params": {"Type": "Channel", "KernelWidth": 1, "KernelHeight": 1,
                     "KernelChannel": kernel_channel, "Alpha": alpha_bits,
                     "Beta": beta, "K": k}}
  return build_plist(
    "network_lrn-1", "op-1",
    [input_entry("x", width=width, height=height, channels=channels, entry_name="op-1")],
    [output_entry("y", "op-1")], {"op-1": unit})


def lrn_fused(x: np.ndarray, *, alpha: float = 1.0, beta: float = 0.75,
              k: float = 1.0) -> np.ndarray:
  """Channel-mode LRN over the full channel window on the ANE; x is fp16 (B=1,C,H,W), KernelChannel=C, alpha is the desired value (fp16-bits + /C handled here)."""
  x = np.asarray(x, dtype=np.float16)
  if x.ndim != 4: raise ValueError("x must be (B,C,H,W)")
  B, C, H, W = x.shape
  if B != 1: raise ValueError("B=1 only")
  invoker = ensure_invoker("layer_invoker")
  # ANE divides parsed alpha by KernelChannel; pre-multiply to recover `alpha`.
  alpha_bits = fp16_bits(alpha * C)
  x5 = x.reshape(1, C, 1, H, W)
  with tempfile.TemporaryDirectory(prefix="ane_lrn_") as d:
    cd = Path(d)
    with (cd / "net.plist").open("wb") as f:
      plistlib.dump(_plist(C, H, W, C, alpha_bits, beta, k), f,
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


def numpy_reference(x, *, alpha=1.0, beta=0.75, k=1.0) -> np.ndarray:
  """Full-window channel LRN reference (window = C)."""
  xf = x.reshape(x.shape[1], x.shape[2], x.shape[3]).astype(np.float32)
  sq_sum = (xf ** 2).sum(axis=0, keepdims=True)  # window = all channels
  return (xf / (k + alpha * sq_sum) ** beta).astype(np.float16)


if __name__ == "__main__":
  C, H, W = 5, 4, 4
  x = np.arange(1, C * H * W + 1, dtype=np.float16).reshape(1, C, H, W)
  y = lrn_fused(x, alpha=1.0, beta=0.75, k=1.0)
  ref = numpy_reference(x, alpha=1.0, beta=0.75, k=1.0)
  err = float(np.abs(y.astype(np.float32) - ref.astype(np.float32)).max())
  print(f"ANE LRN(Channel, C=5) max_abs_err={err:.4g}")
  assert err < 0.05
  print("CRACKED: LocalResponseNormalization verified.")
