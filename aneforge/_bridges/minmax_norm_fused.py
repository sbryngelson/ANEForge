"""Native ANE min-max normalization via a hand-authored ANECIR netplist.

`MinMaxNormalization` running natively on the ANE via Path A.

Computes:   y = (x - min) / (max - min + eps)   over `Params.Dimension`.

Supported reduction axes:
    Dimension="Width"   -> per-row min/max over W
    Dimension="Height"  -> per-column min/max over H
    Dimension="Channel" -> ARCH-GATED (ANECCompile fails on this host).

Epsilon convention: `Params.Epsilon` is a fp16 bit pattern, NOT a float
real.  Pass `fp16_bits(desired_eps)`.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=the reverse-engineering corpus \
        python3 -m aneforge._bridges.minmax_norm_fused
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

_AXIS = {"Width": 3, "Height": 2, "Channel": 1}


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
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return _INVOKER


def _plist(channels: int, height: int, width: int, dimension: str, eps_bits: int) -> dict:
    net = "network_minmax-1"
    unit = {"Bottom": ["x"], "InputType": ["Float16"], "Name": "op-1",
            "OutputChannels": channels, "OutputType": "Float16",
            "Type": "MinMaxNormalization",
            "Params": {"Dimension": dimension, "Epsilon": eps_bits}}
    inp = {"BatchSize": 1, "InputChannels": channels, "InputDepth": 1,
           "InputHeight": height, "InputInterleave": 1, "InputName": "x",
           "InputType": "Float16", "InputWidth": width, "Name": "op-1",
           "OperationName": "op0"}
    return {"Version": "1.0.10", "Networks": [net],
            "ProcedureList": [{"Name": "procedure_minmax-1", "InputList": [inp],
                "OperationList": [{"NetworkName": net, "OperationName": "op0"}],
                "OutputList": [{"Name": "op-1", "OperationName": "op0",
                                "OutputInterleave": 1, "OutputName": "y",
                                "OutputType": "Float16"}]}],
            net: {"op-1": unit, "Units": ["op-1"], "Weights": ["weights.0"],
                  "y": {"Bottom": "op-1", "OutputInterleave": 1,
                        "OutputName": "y", "OutputType": "Float16"}}}


def minmax_norm_fused(x: np.ndarray, *, dimension: str = "Width",
                      epsilon: float = 1e-4) -> np.ndarray:
    """Min-max normalize `x` over `dimension` on the ANE.

    Args:
        x: fp16 array of logical shape (B=1, C, H, W).
        dimension: reduction axis — "Width" (default) or "Height".
                   "Channel" is ARCH-GATED (compile failure) on this host.
        epsilon: stability eps added to the (max-min) denominator.

    Returns:
        fp16 array, same shape as x.
    """
    x = np.asarray(x, dtype=np.float16)
    if x.ndim != 4:
        raise ValueError("x must be (B,C,H,W)")
    B, C, H, W = x.shape
    if B != 1:
        raise ValueError("B=1 only")
    invoker = _ensure_invoker()
    x5 = x.reshape(1, C, 1, H, W)
    with tempfile.TemporaryDirectory(prefix="ane_mm_") as d:
        cd = Path(d)
        with (cd / "net.plist").open("wb") as f:
            plistlib.dump(_plist(C, H, W, dimension, fp16_bits(epsilon)), f,
                          fmt=plistlib.FMT_BINARY)
        (cd / "weights.0").write_bytes(b"\x00" * 1024)
        (cd / "in_x.bin").write_bytes(x5.astype(np.float16).tobytes())
        cmd = [str(invoker), "--net-plist", str(cd / "net.plist"),
               "--weights", str(cd / "weights.0"), "--input", f"x={cd / 'in_x.bin'}",
               "--output", f"y={cd / 'out_y.bin'}",
               "--raw-output", f"y={cd / 'out_raw.bin'}", "--repeats", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
        if not lines:
            raise RuntimeError(f"no JSON from invoker:\n{p.stdout}\n{p.stderr}")
        info = json.loads(lines[-1])
        if info.get("status") != "ok":
            raise RuntimeError(f"minmax invoker failed (dimension={dimension}): {info}")
        raw = (cd / "out_y.bin").read_bytes()
        return np.frombuffer(raw[:C * H * W * 2], dtype=np.float16).reshape(C, H, W)


def numpy_reference(x: np.ndarray, *, dimension: str = "Width",
                    epsilon: float = 1e-4) -> np.ndarray:
    xf = x.astype(np.float32)
    ax = _AXIS[dimension]
    mn = xf.min(axis=ax, keepdims=True)
    mx = xf.max(axis=ax, keepdims=True)
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
