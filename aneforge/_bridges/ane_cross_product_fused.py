"""Native ANE 3-vector cross product on Path A.

:func:`cross_product_fused` computes `cross(x, z)` for two length-3 fp16 vectors
on the ANE (matches `numpy.cross(x, z)`).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def cross_product_fused(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return `cross(x, z)` for two 3-vectors, computed on the ANE.

    Args:
        x, z: length-3 fp16-castable arrays.
    Returns:
        length-3 fp16 array equal to `numpy.cross(x, z)`.
    """
    x = np.asarray(x, dtype=np.float16).reshape(3)
    z = np.asarray(z, dtype=np.float16).reshape(3)
    from ._netplist import write_model, ensure_invoker  # type: ignore

    with tempfile.TemporaryDirectory(prefix="ane_xprod_") as d:
        wd = Path(d)
        write_model("cross_product", wd)
        (wd / "in_x.f16").write_bytes(x.tobytes())
        (wd / "in_z.f16").write_bytes(z.tobytes())
        cmd = [
            str(ensure_invoker("sdpa_invoker")),
            "--net-plist", str(wd / "net.plist"),
            "--weights", str(wd / "weights.0"),
            "--input", f"x={wd / 'in_x.f16'}",
            "--input", f"z={wd / 'in_z.f16'}",
            "--output", f"y={wd / 'out_y.f16'}",
            "--repeats", "1", "--warmup", "0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cross_product invoker failed:\n{proc.stderr}\n{proc.stdout}")
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        if info.get("status") != "ok":
            raise RuntimeError(f"cross_product non-ok: {info}")
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
