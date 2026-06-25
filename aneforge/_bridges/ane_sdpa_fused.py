"""Native ANE fused-attention (`Type=SDPA`) on Path A, reaching the hardware
fused-attention layer instead of Apple's HWX decomposition. Drop-in for
`(B, heads, seq, d_head)` fp16 Q/K/V. See docs/developer/bridges.md."""

from __future__ import annotations

import math
import plistlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._netplist import bin_dir, invoke_netplist


_INVOKER_BIN = bin_dir() / "sdpa_invoker"
_INVOKER_SRC = Path(__file__).resolve().parents[1] / "_invokers" / "sdpa_invoker.mm"


def _ensure_invoker() -> Path:
  """Build the SDPA invoker once if missing/stale."""
  if (
    _INVOKER_BIN.exists()
    and _INVOKER_SRC.exists()
    and _INVOKER_BIN.stat().st_mtime >= _INVOKER_SRC.stat().st_mtime
  ):
    return _INVOKER_BIN
  _INVOKER_BIN.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    "xcrun", "clang++", "-O2", "-Wall", "-Wextra",
    "-fobjc-arc", "-std=gnu++17",
    "-framework", "Foundation",
    "-framework", "IOSurface",
    str(_INVOKER_SRC),
    "-o", str(_INVOKER_BIN),
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0: raise RuntimeError(f"failed to build sdpa invoker:\n{proc.stderr}")
  return _INVOKER_BIN


def _write_netplist(
  workdir: Path,
  *,
  channels: int,
  sequence: int,
  dim: int,
  scale: float,
  constant_flag_spelling: str,
  subtract_max: bool = True,
) -> tuple[Path, list[Path]]:
  """Generate a one-op SDPA netplist + weights at workdir."""
  from ._netplist import write_model
  write_model(
    "sdpa",
    workdir,
    sdpa_channels=channels,
    sdpa_sequence=sequence,
    sdpa_dim=dim,
    sdpa_scale=scale,
    sdpa_constant_flag_spelling=constant_flag_spelling,
    sdpa_subtract_max=subtract_max,
  )
  weights = sorted(workdir.glob("weights.*"))
  return workdir / "net.plist", weights


def _expected_shape(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[int, ...]:
  if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
    raise ValueError("Q, K, V must each have shape [B, H, S, D]")
  # K/V share shape; Q's seq may differ (KV-cache decode). See bridges.md.
  if K.shape != V.shape:
    raise ValueError(f"K and V must share shape (same cached sequence); got {K.shape}, {V.shape}")
  if Q.shape[0] != 1 or K.shape[0] != 1:
    raise ValueError(f"only B=1 supported on the netplist path; got B={Q.shape[0]}/{K.shape[0]}")
  if Q.shape[1] != K.shape[1] or Q.shape[3] != K.shape[3]:
    raise ValueError(f"Q and K/V must share H (heads) and D (embedding); got {Q.shape}, {K.shape}")
  return Q.shape


def _numpy_reference(
  Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> np.ndarray:
  """Standard fp16 SDPA reference: softmax(Q @ K^T * scale) @ V."""
  Qf = Q.astype(np.float32)
  Kf = K.astype(np.float32)
  Vf = V.astype(np.float32)
  attn = (Qf @ Kf.swapaxes(-1, -2)) * float(scale)
  attn = attn - attn.max(axis=-1, keepdims=True)
  attn = np.exp(attn)
  attn = attn / attn.sum(axis=-1, keepdims=True)
  return (attn @ Vf).astype(np.float16)


def sdpa_fused(
  Q: np.ndarray,
  K: np.ndarray,
  V: np.ndarray,
  scale: Optional[float] = None,
  *,
  mask: Optional[np.ndarray] = None,
  repeats: int = 1,
  warmup: int = 0,
  constant_flag_spelling: str = "Constants_array",
  subtract_max: bool = True,
  return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
  """Native ANE fused-attention over `[1, heads, seq, d_head]` fp16 Q/K/V;
    returns Y in the same layout (or `(Y, info)` with timings if
    `return_details`). `scale` defaults to `1/sqrt(d_head)`.

    `constant_flag_spelling` default `"Constants_array"` is the only spelling
    observed to compile + load on this host (others raise `ANECCompile() FAILED`).
    """
  Q = np.asarray(Q, dtype=np.float16)
  K = np.asarray(K, dtype=np.float16)
  V = np.asarray(V, dtype=np.float16)
  B, H, S, D = _expected_shape(Q, K, V)
  Sq, Skv = Q.shape[2], K.shape[2]              # query seq vs cached K/V seq (decode: Sq < Skv)
  if scale is None: scale = 1.0 / math.sqrt(D)

  # ANE-native layout is seq-in-C, heads-in-H; PyTorch/MIL is the opposite, so
  # pre-transpose here and post-transpose Y back. See bridges.md.
  Q_ane = np.ascontiguousarray(Q.transpose(0, 2, 1, 3))  # (B, S, H, D)
  K_ane = np.ascontiguousarray(K.transpose(0, 2, 1, 3))
  V_ane = np.ascontiguousarray(V.transpose(0, 2, 1, 3))

  invoker = _ensure_invoker()

  with tempfile.TemporaryDirectory(prefix="ane_sdpa_") as workdir_str:
    workdir = Path(workdir_str)
    # `channels`=ANE C dim (seq after transpose); `sequence`=ANE H dim (heads).
    netplist, weights = _write_netplist(
      workdir,
      channels=Skv,
      sequence=H,
      dim=D,
      scale=float(scale),
      constant_flag_spelling=constant_flag_spelling,
      subtract_max=subtract_max,
    )
    (workdir / "in_query.f16").write_bytes(Q_ane.tobytes())
    (workdir / "in_key.f16").write_bytes(K_ane.tobytes())
    (workdir / "in_value.f16").write_bytes(V_ane.tobytes())
    inputs = [("query", workdir / "in_query.f16"),
              ("key", workdir / "in_key.f16"),
              ("value", workdir / "in_value.f16")]

    if mask is not None or Sq != Skv:
      # Patch the netplist for (a) decode shape (query carries Sq, K/V carry Skv)
      # and/or (b) an additive mask bottom [C=Sq, H=1, W=Skv]. See bridges.md.
      pl = plistlib.loads(Path(netplist).read_bytes())
      il = pl["ProcedureList"][0]["InputList"]
      if Sq != Skv:
        for e in il:
          if e["InputName"] == "query": e["InputChannels"] = Sq
        for proc in pl["ProcedureList"]:
          for o in proc.get("OutputList", []):
            o["OutputChannels"] = Sq
      if mask is not None:
        def _units(o):
          if isinstance(o, dict):
            if o.get("Type") == "SDPA" and "Bottom" in o: yield o
            for v in o.values():
              yield from _units(v)
          elif isinstance(o, list):
            for v in o:
              yield from _units(v)
        unit = next(_units(pl))
        unit["Bottom"] = list(unit["Bottom"]) + ["mask"]
        unit["InputType"] = list(unit["InputType"]) + ["Float16"]
        mentry = dict(next(e for e in il if e["InputName"] == "query"))
        mentry["InputName"] = "mask"
        mentry["InputChannels"], mentry["InputHeight"], mentry["InputWidth"] = Sq, 1, Skv
        il.append(mentry)
        (workdir / "in_mask.f16").write_bytes(np.ascontiguousarray(mask, dtype=np.float16).tobytes())
        inputs.append(("mask", workdir / "in_mask.f16"))
      Path(netplist).write_bytes(plistlib.dumps(pl))

    info = invoke_netplist(
      invoker, netplist,
      weights=weights,
      inputs=inputs,
      outputs=[("y", workdir / "out_y.f16")],
      repeats=repeats, warmup=warmup,
    )

    # ANE output is (B, S, H, D); transpose back to (B, heads, seq, d_head).
    Y_ane = np.frombuffer(
      (workdir / "out_y.f16").read_bytes(),
      dtype=np.float16,
    )[: B * Sq * H * D].reshape(B, Sq, H, D)
    Y = np.ascontiguousarray(Y_ane.transpose(0, 2, 1, 3))

    if return_details: return Y, info
    return Y


def numpy_reference(
  Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
) -> np.ndarray:
  """Public-facing alias for the numpy SDPA reference (fp32 internal, fp16 out)."""
  Q = np.asarray(Q, dtype=np.float16)
  K = np.asarray(K, dtype=np.float16)
  V = np.asarray(V, dtype=np.float16)
  if scale is None: scale = 1.0 / math.sqrt(Q.shape[-1])
  return _numpy_reference(Q, K, V, float(scale))
