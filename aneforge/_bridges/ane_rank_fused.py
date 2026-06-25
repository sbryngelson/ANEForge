"""Native ANE rank-family layers (Sort / TopK / ArgMinMax / GlobalArgMinMax) on Path A. Integer index outputs are fp16-encoded but exact (ranges < 2048)."""

from __future__ import annotations

import tempfile
import plistlib
from pathlib import Path

import numpy as np

from ._netplist import build_plist, ensure_invoker, input_entry, invoke_netplist, output_entry


# Minimal netplist scaffolding: single op, single input x, output y.

def _build_plist(layer_type: str, params: dict, *, width, height, channels):
  unit = {
    "Bottom": ["x"], "InputType": ["Float16"], "Name": "op-1",
    "OutputChannels": channels, "OutputType": "Float16",
    "Type": layer_type, "Params": params,
  }
  return build_plist(
    f"network_{layer_type.lower()}-1", "op-1",
    [input_entry("x", width=width, height=height, channels=channels, entry_name="op-1")],
    [output_entry("y", "op-1")], {"op-1": unit})


def _run(layer_type: str, params: dict, x: np.ndarray, *,
         channels: int, width: int, height: int = 1,
         return_details: bool = False):
  """Run x ([C,H,W] fp16), decoding output with the runtime-reported strides."""
  invoker = ensure_invoker("rank_invoker")
  x = np.asarray(x, dtype=np.float16)
  with tempfile.TemporaryDirectory(prefix=f"ane_{layer_type.lower()}_") as ws:
    ws = Path(ws)
    plist = _build_plist(layer_type, params, width=width,
                         height=height, channels=channels)
    with (ws / "net.plist").open("wb") as f:
      plistlib.dump(plist, f, fmt=plistlib.FMT_BINARY)
    (ws / "weights.0").write_bytes(b"\x00" * 1024)
    (ws / "in_x.f16").write_bytes(x.tobytes())
    info = invoke_netplist(
      invoker, ws / "net.plist",
      weights=[ws / "weights.0"], inputs=[("x", ws / "in_x.f16")],
      repeats=None, warmup=None,
      extra=["--output-raw", f"y={ws / 'out_y.bin'}", "--output-bytes", "65536"])
    oi = info["live_outputs"][0]
    ps, rs = oi["PlaneStride"], oi["RowStride"]
    ch, hh, wd = oi["Channels"], oi["Height"], oi["Width"]
    raw = (ws / "out_y.bin").read_bytes()
    out = np.zeros((ch, hh, wd), dtype=np.float16)
    for c in range(ch):
      for hr in range(hh):
        off = c * ps + hr * rs
        out[c, hr] = np.frombuffer(raw[off:off + wd * 2], dtype=np.float16)
    out = np.squeeze(out)
    if return_details: return out, info
    return out


# Public per-layer entry points. x is fp16 [C, W] for Sort/TopK/GlobalArgMinMax,
# or [C, H, W] for ArgMinMax.

def sort(x: np.ndarray, *, descending: bool = False, key_lane: int = 0,
         return_indices: bool = False, **kw):
  """Sort x ([C,W]) along Width, permuting all channels by channel key_lane (SortDimension=Width, VectorDimension=Channel)."""
  x = np.atleast_2d(np.asarray(x, np.float16))
  ch, wd = x.shape
  params = {
    "Direction": "Descending" if descending else "Ascending",
    "SortDimension": "Width", "VectorDimension": "Channel",
    "SortIndices": [int(key_lane)],
  }
  if return_indices: params["Indices"] = True
  return _run("Sort", params, x, channels=ch, width=wd, **kw)


def topk(x: np.ndarray, k: int, *, largest: bool = True, key_lane: int = 0,
         return_indices: bool = False, **kw):
  """Top-k along Width of x ([C,W]) keyed by channel key_lane; k in {3,4} is ARCH-GATED."""
  x = np.atleast_2d(np.asarray(x, np.float16))
  ch, wd = x.shape
  params = {
    "Type": "Max" if largest else "Min", "K": int(k),
    "SortDimension": "Width", "VectorDimension": "Channel",
    "SortIndices": [int(key_lane)],
  }
  if return_indices: params["Indices"] = True
  return _run("TopK", params, x, channels=ch, width=wd, **kw)


def argminmax(x: np.ndarray, mode: str, **kw):
  """ArgMinMax over x ([C,H,W]), mode {Spatial,Channel}{ArgMax,ArgMin}; Spatial* -> one (H*W) index per channel, Channel* -> one channel index per (h,w)."""
  x = np.asarray(x, np.float16)
  if x.ndim == 2: x = x[:, None, :]   # [C, 1, W]
  ch, hh, wd = x.shape
  params = {"Mode": mode, "KernelWidth": wd, "KernelHeight": hh,
            "PadLeft": 0, "PadRight": 0, "PadTop": 0, "PadBot": 0}
  return _run("ArgMinMax", params, x, channels=ch, width=wd, height=hh, **kw)


def global_argminmax(x: np.ndarray, *, dimension: str = "Width",
                     largest: bool = True, **kw):
  """GlobalArgMinMax: arg index along `dimension` (Width|Height|Channel)."""
  x = np.atleast_2d(np.asarray(x, np.float16))
  ch, wd = x.shape
  params = {"Type": "Max" if largest else "Min", "Dimension": dimension}
  return _run("GlobalArgMinMax", params, x, channels=ch, width=wd, **kw)


# ---- numpy references (for parity checks) --------------------------------

def numpy_sort(x, descending=False, key_lane=0, return_indices=False):
  x = np.atleast_2d(np.asarray(x, np.float32))
  keyrow = x[key_lane]
  order = np.argsort(-keyrow if descending else keyrow)
  return order.astype(np.float16) if return_indices else x[:, order].astype(np.float16)


def numpy_topk(x, k, largest=True, key_lane=0, return_indices=False):
  x = np.atleast_2d(np.asarray(x, np.float32))
  keyrow = x[key_lane]
  order = np.argsort(-keyrow if largest else keyrow)[:k]
  return order.astype(np.float16) if return_indices else x[:, order].astype(np.float16)


def numpy_argminmax(x, mode):
  x = np.asarray(x, np.float32)
  if x.ndim == 2: x = x[:, None, :]
  ch, hh, wd = x.shape
  if mode.startswith("Spatial"):
    flat = x.reshape(ch, -1)
    return (flat.argmax if "Max" in mode else flat.argmin)(axis=1).astype(np.float16)
  return (x.argmax if "Max" in mode else x.argmin)(axis=0).astype(np.float16)


def numpy_global_argminmax(x, dimension="Width", largest=True):
  x = np.atleast_2d(np.asarray(x, np.float32))
  axis = {"Width": 1, "Channel": 0}[dimension]
  return (x.argmax if largest else x.argmin)(axis=axis).astype(np.float16)


if __name__ == "__main__":
  rng = np.random.default_rng(0)
  x = rng.standard_normal((4, 8)).astype(np.float16)
  print("Sort asc values exact :", np.array_equal(sort(x), numpy_sort(x)))
  print("Sort asc indices exact:", np.array_equal(sort(x, return_indices=True),
                                                   numpy_sort(x, return_indices=True)))
  print("TopK k=1 idx exact    :", np.array_equal(
    np.ravel(topk(x, 1, return_indices=True)),
    np.ravel(numpy_topk(x, 1, return_indices=True))))
  g = global_argminmax(x, dimension="Channel")
  print("GlobalArgMax(ch) exact:", np.array_equal(g, numpy_global_argminmax(x, "Channel")))
  x3 = rng.standard_normal((4, 4, 8)).astype(np.float16)
  print("ArgMinMax Channel exact:", np.array_equal(argminmax(x3, "ChannelArgMax"),
                                                    numpy_argminmax(x3, "ChannelArgMax")))
