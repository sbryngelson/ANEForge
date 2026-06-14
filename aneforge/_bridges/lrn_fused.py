"""Native ANE local response normalization via a hand-authored ANECIR netplist.

`LocalResponseNormalization` (classic vision cross-channel LRN) running natively
on the ANE.

Computes (Channel mode):
    y[c] = x[c] / (K + alpha_eff * sum_{j in window} x[j]^2) ^ Beta

Non-obvious conventions:
    1. `Alpha` is a fp16 *bit pattern* (ZinParseFP16Token), NOT a float.
       A float real int-truncates to ~0 -> identity output.
    2. ANE divides the parsed alpha by KernelChannel internally, so the
       effective alpha is  fp16(Alpha_bits) / KernelChannel.  For a desired
       alpha, pass  Alpha = fp16_bits(desired_alpha * KernelChannel).
    3. Only the first KernelChannel output channels are normalized; the rest
       are identity-copied.  Use KernelChannel = C for full-tensor LRN; C must
       exceed KernelChannel-as-window only spatially - empirically C ==
       KernelChannel gives full coverage (C=5/KC=5).
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


def _plist(channels, height, width, kernel_channel, alpha_bits, beta, k) -> dict:
    net = "network_lrn-1"
    unit = {"Bottom": ["x"], "InputType": ["Float16"], "Name": "op-1",
            "OutputChannels": channels, "OutputType": "Float16",
            "Type": "LocalResponseNormalization",
            "Params": {"Type": "Channel", "KernelWidth": 1, "KernelHeight": 1,
                       "KernelChannel": kernel_channel, "Alpha": alpha_bits,
                       "Beta": beta, "K": k}}
    inp = {"BatchSize": 1, "InputChannels": channels, "InputDepth": 1,
           "InputHeight": height, "InputInterleave": 1, "InputName": "x",
           "InputType": "Float16", "InputWidth": width, "Name": "op-1",
           "OperationName": "op0"}
    return {"Version": "1.0.10", "Networks": [net],
            "ProcedureList": [{"Name": "procedure_lrn-1", "InputList": [inp],
                "OperationList": [{"NetworkName": net, "OperationName": "op0"}],
                "OutputList": [{"Name": "op-1", "OperationName": "op0",
                                "OutputInterleave": 1, "OutputName": "y",
                                "OutputType": "Float16"}]}],
            net: {"op-1": unit, "Units": ["op-1"], "Weights": ["weights.0"],
                  "y": {"Bottom": "op-1", "OutputInterleave": 1,
                        "OutputName": "y", "OutputType": "Float16"}}}


def lrn_fused(x: np.ndarray, *, alpha: float = 1.0, beta: float = 0.75,
              k: float = 1.0) -> np.ndarray:
    """Channel-mode LRN over the full channel window on the ANE.

    Args:
        x: fp16 array of logical shape (B=1, C, H, W). KernelChannel = C.
        alpha, beta, k: standard LRN coefficients (alpha is the *desired*
                        value; this routine handles the fp16-bits + /C scaling).
    """
    x = np.asarray(x, dtype=np.float16)
    if x.ndim != 4:
        raise ValueError("x must be (B,C,H,W)")
    B, C, H, W = x.shape
    if B != 1:
        raise ValueError("B=1 only")
    invoker = _ensure_invoker()
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
            raise RuntimeError(f"lrn invoker failed: {info}")
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
