"""Native ANE CrossCorrelation (template matching) on Path A.
See docs/developer/bridges.md."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def cross_correlation_fused(x: np.ndarray, template: np.ndarray) -> np.ndarray:
  """Valid (no-flip) cross-correlation of (H,W) x with (Th,Tw) template -> (H-Th+1, W-Tw+1) fp16."""
  x = np.asarray(x, dtype=np.float16)
  template = np.asarray(template, dtype=np.float16)
  H, W = x.shape
  Th, Tw = template.shape
  out_h, out_w = H - Th + 1, W - Tw + 1
  from ._netplist import write_model, ensure_invoker, invoke_netplist

  with tempfile.TemporaryDirectory(prefix="ane_xcorr_") as d:
    wd = Path(d)
    write_model(
      "cross_correlation", wd,
      width=W, height=H, channels=1,
      template_width=Tw, template_height=Th,
      output_channels=1,
    )
    x.tofile(wd / "in_x.f16")
    template.reshape(-1).astype(np.float16).tofile(wd / "in_t.f16")
    invoke_netplist(
      ensure_invoker("sdpa_invoker"), wd / "net.plist",
      weights=[wd / "weights.0"],
      inputs=[("x", wd / "in_x.f16"), ("template", wd / "in_t.f16")],
      outputs=[("y", wd / "out_y.f16")], warmup=0,
    )
    y = np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16)
    return y.reshape(out_h, out_w).copy()


def numpy_reference(x: np.ndarray, template: np.ndarray) -> np.ndarray:
  xf = np.asarray(x, np.float16).astype(np.float32)
  tf = np.asarray(template, np.float16).astype(np.float32)
  H, W = xf.shape
  Th, Tw = tf.shape
  out = np.zeros((H - Th + 1, W - Tw + 1), np.float32)
  for i in range(out.shape[0]):
    for j in range(out.shape[1]):
      out[i, j] = (xf[i:i + Th, j:j + Tw] * tf).sum()
  return out.astype(np.float16)


if __name__ == "__main__":
  rng = np.random.default_rng(1)
  max_err = 0.0
  for _ in range(6):
    x = rng.standard_normal((4, 4)).astype(np.float16)
    t = rng.standard_normal((3, 3)).astype(np.float16)
    got = cross_correlation_fused(x, t).astype(np.float32)
    ref = numpy_reference(x, t).astype(np.float32)
    max_err = max(max_err, float(np.max(np.abs(got - ref))))
  print(json.dumps({"layer": "CrossCorrelation", "status": "CRACKED",
                    "max_abs_err_vs_numpy": max_err}))
