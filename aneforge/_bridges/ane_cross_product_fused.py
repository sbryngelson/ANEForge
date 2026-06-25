"""Native ANE 3-vector cross product on Path A.
See docs/developer/bridges.md."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def cross_product_fused(x: np.ndarray, z: np.ndarray) -> np.ndarray:
  """Return `cross(x, z)` (length-3 fp16) for two 3-vectors, on the ANE."""
  x = np.asarray(x, dtype=np.float16).reshape(3)
  z = np.asarray(z, dtype=np.float16).reshape(3)
  from ._netplist import write_model, ensure_invoker, invoke_netplist

  with tempfile.TemporaryDirectory(prefix="ane_xprod_") as d:
    wd = Path(d)
    write_model("cross_product", wd)
    (wd / "in_x.f16").write_bytes(x.tobytes())
    (wd / "in_z.f16").write_bytes(z.tobytes())
    invoke_netplist(
      ensure_invoker("sdpa_invoker"), wd / "net.plist",
      weights=[wd / "weights.0"],
      inputs=[("x", wd / "in_x.f16"), ("z", wd / "in_z.f16")],
      outputs=[("y", wd / "out_y.f16")], warmup=0,
    )
    return np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16).copy()


def numpy_reference(x: np.ndarray, z: np.ndarray) -> np.ndarray:
  return np.cross(
    np.asarray(x, np.float16).astype(np.float32),
    np.asarray(z, np.float16).astype(np.float32),
  ).astype(np.float16)


if __name__ == "__main__":
  rng = np.random.default_rng(0)
  max_err = 0.0
  for _ in range(8):
    x = rng.standard_normal(3).astype(np.float16)
    z = rng.standard_normal(3).astype(np.float16)
    got = cross_product_fused(x, z).astype(np.float32)
    ref = numpy_reference(x, z).astype(np.float32)
    max_err = max(max_err, float(np.max(np.abs(got - ref))))
  print(json.dumps({"layer": "CrossProduct", "status": "CRACKED",
                    "max_abs_err_vs_numpy": max_err}))
