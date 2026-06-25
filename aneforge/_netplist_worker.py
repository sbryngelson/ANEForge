"""A2: a persistent Path-A worker for aneforge's netplist-bridge ops (pays compile/load/map once, services many evals over a pipe)."""
from __future__ import annotations

import json
import math
import os
import select
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ._bridges._netplist import bin_dir

_INVOKER_BIN_NAME = "persistent_worker"


def _ensure_worker_built() -> Path:
  """Build the persistent-worker invoker once if missing/stale."""
  binp = bin_dir() / _INVOKER_BIN_NAME
  src = Path(__file__).resolve().parent / "_invokers" / "persistent_worker.mm"
  if binp.exists() and src.exists() and binp.stat().st_mtime >= src.stat().st_mtime: return binp
  binp.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    "xcrun", "clang++", "-O2", "-Wall", "-Wextra", "-fobjc-arc", "-std=gnu++17",
    "-framework", "Foundation", "-framework", "IOSurface",
    str(src), "-o", str(binp),
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError(f"failed to build persistent worker:\n{proc.stderr}")
  return binp


class _Worker:
  """Owns one spawned worker process holding ONE compiled+loaded netplist (batched fp16 wire protocol)."""

  def __init__(self, netplist: Path, weights: list[Path], in_syms: list[str],
               out_syms: list[str], workdir: tempfile.TemporaryDirectory,
               op: str = "netplist"):
    self._workdir = workdir  # keep the netplist + weights alive on disk
    self._op = op
    binp = _ensure_worker_built()
    cmd = [str(binp), "--net-plist", str(netplist)]
    for w in weights: cmd += ["--weights", str(w)]
    for s in in_syms: cmd += ["--input", s]
    for s in out_syms: cmd += ["--output", s]
    # bufsize=0: _read_exact select()s on the raw stdout fd, so no Python-level read buffering.
    self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
    assert self._proc.stdin is not None
    assert self._proc.stdout is not None
    assert self._proc.stderr is not None
    # non-Optional IO so pyright can narrow in other methods.
    self._stdin = self._proc.stdin
    self._stdout = self._proc.stdout
    self._stderr = self._proc.stderr
    ready = self._stdout.readline()
    if not ready:
      err = self._stderr.read().decode(errors="replace")
      raise RuntimeError(f"persistent worker failed to start: {err}")
    info = json.loads(ready.decode())
    if info.get("status") != "ready":
      raise RuntimeError(f"persistent worker not ready: {info}")
    self.compile_ms = info["compile_ms"]
    self.load_ms = info["load_ms"]
    self.in_elems = list(info["input_elems"])
    self.out_elems = list(info["output_elems"])
    self._out_bytes = 2 * sum(self.out_elems)

  def eval(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """One eval. fp16 arrays in spawn symbol order in and out."""
    return self.eval_batch([inputs])[0]

  def eval_batch(self, input_sets: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """Run N evals in ONE pipe round-trip (bit-identical to N `eval` calls)."""
    if self._proc.poll() is not None: raise RuntimeError("persistent worker has exited")
    n = len(input_sets)
    if n == 0: return []
    parts = [struct.pack("<I", n)]
    for inputs in input_sets:
      if len(inputs) != len(self.in_elems):
        raise ValueError(
          f"persistent worker ({self._op}) expects {len(self.in_elems)} "
          f"input arrays per set; got {len(inputs)}")
      for i, a in enumerate(inputs):
        raw_in = np.ascontiguousarray(a, np.float16).tobytes()
        if len(raw_in) != 2 * self.in_elems[i]:
          raise ValueError(
            f"persistent worker ({self._op}) input {i} is {len(raw_in)} "
            f"bytes; expected {2 * self.in_elems[i]} ({self.in_elems[i]} fp16 elems)")
        parts.append(raw_in)
    try:
      data = memoryview(b"".join(parts))
      while data:
        data = data[self._stdin.write(data):]
      self._stdin.flush()
    except (BrokenPipeError, OSError):
      err = self._stderr.read().decode(errors="replace")
      raise RuntimeError(f"persistent worker ({self._op}) died on request: {err}")
    raw = self._read_exact(self._out_bytes * n)
    out_sets, off = [], 0
    for _ in range(n):
      outs = []
      for ne in self.out_elems:
        outs.append(np.frombuffer(raw[off:off + 2 * ne], dtype=np.float16).copy())
        off += 2 * ne
      out_sets.append(outs)
    return out_sets

  def _read_exact(self, n: int) -> bytes:
    timeout = float(os.environ.get("ANEFORGE_WORKER_TIMEOUT_S", "120"))
    fd = self._stdout.fileno()
    buf = bytearray()
    while len(buf) < n:
      if timeout > 0:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
          self._proc.kill()
          self._proc.wait()
          err = self._stderr.read().decode(errors="replace")
          raise RuntimeError(
            f"persistent worker ({self._op}) unresponsive after "
            f"{timeout:g}s: {err}")
      chunk = os.read(fd, n - len(buf))
      if not chunk:
        err = self._stderr.read().decode(errors="replace")
        raise RuntimeError(f"persistent worker ({self._op}) died mid-eval: {err}")
      buf += chunk
    return bytes(buf)

  def release(self) -> None:
    if self._proc.poll() is None:
      try:
        self._stdin.close()  # EOF -> clean shutdown
        self._proc.wait(timeout=5)
      except Exception:
        self._proc.kill()
    try:
      self._workdir.cleanup()
    except Exception:
      pass


# per-op worker builders: author the netplist with the A1 bridge generator, then hand it to
# a persistent _Worker. Each returns a callable (src_arrays, attrs) -> fp16 output.

def _build_sdpa_worker(shape, attrs):
  """Persistent SDPA worker (shape = Q/K/V [1,H,S,D]); pre/post transpose heads<->seq."""
  from ._bridges import ane_sdpa_fused as af  # the A1 bridge = the netplist author

  B, H, S, D = shape
  if B != 1:
    raise ValueError(f"SDPA worker only supports B=1; got B={B}")
  scale = attrs.get("scale")
  if scale is None: scale = 1.0 / math.sqrt(D)

  workdir = tempfile.TemporaryDirectory(prefix="ane_sdpa_worker_")
  wd = Path(workdir.name)
  # channels=S (ANE C=seq after transpose), sequence=H (ANE H=heads), dim=D.
  netplist, weights = af._write_netplist(
    wd, channels=S, sequence=H, dim=D, scale=float(scale),
    constant_flag_spelling="Constants_array", subtract_max=True,
  )
  worker = _Worker(netplist, list(weights), ["query", "key", "value"], ["y"], workdir,
           op="sdpa")

  def run(srcs, attrs2):
    q, k, v = srcs
    # to ANE layout (B, S, H, D)
    qa = np.ascontiguousarray(np.asarray(q, np.float16).transpose(0, 2, 1, 3))
    ka = np.ascontiguousarray(np.asarray(k, np.float16).transpose(0, 2, 1, 3))
    va = np.ascontiguousarray(np.asarray(v, np.float16).transpose(0, 2, 1, 3))
    (y_flat,) = worker.eval([qa, ka, va])
    y_ane = y_flat.reshape(B, S, H, D)
    return np.ascontiguousarray(y_ane.transpose(0, 2, 1, 3))  # back to [B,H,S,D]

  return worker, run


def _write_rank_netplist(wd: Path, layer_type: str, params: dict, *,
             channels: int, width: int, height: int = 1):
  """Author a single-op rank netplist (Sort/TopK/ArgMinMax) into `wd`; returns (netplist_path, [weight])."""
  import plistlib
  from ._bridges import ane_rank_fused as rf  # the A1 bridge = the netplist author
  plist = rf._build_plist(layer_type, params, width=width, height=height,
              channels=channels)
  netplist = wd / "net.plist"
  with netplist.open("wb") as f:
    plistlib.dump(plist, f, fmt=plistlib.FMT_BINARY)
  weight = wd / "weights.0"
  weight.write_bytes(b"\x00" * 1024)
  return netplist, [weight]


def _build_argmax_worker(shape, attrs):
  """Persistent GlobalArgMinMax worker (shape = [C,W]); output keepdims [C,1] (axis=1) or [1,W] (axis=0)."""
  C, W = shape
  axis = attrs["axis"]
  dim = "Width" if axis == 1 else "Channel"
  params = {"Type": "Max", "Dimension": dim}

  workdir = tempfile.TemporaryDirectory(prefix="ane_argmax_worker_")
  wd = Path(workdir.name)
  netplist, weights = _write_rank_netplist(wd, "GlobalArgMinMax", params,
                       channels=C, width=W)
  worker = _Worker(netplist, list(weights), ["x"], ["y"], workdir, op="argmax")

  def run(srcs, attrs2):
    (x,) = srcs
    xa = np.ascontiguousarray(np.asarray(x, np.float16).reshape(C, W))
    (y_flat,) = worker.eval([xa])
    # GlobalArgMinMax(Width) -> one index per channel (C); Channel -> per width (W).
    return y_flat.reshape((C, 1) if axis == 1 else (1, W))

  return worker, run


def _build_topk_worker(shape, attrs):
  """Persistent per-row TopK worker (shape = [C,W]); one 1-channel program eval'd C times, output [C,k]."""
  C, W = shape
  k, largest = attrs["k"], attrs["largest"]
  params = {
    "Type": "Max" if largest else "Min", "K": int(k),
    "SortDimension": "Width", "VectorDimension": "Channel",
    "SortIndices": [0],
  }

  workdir = tempfile.TemporaryDirectory(prefix="ane_topk_worker_")
  wd = Path(workdir.name)
  netplist, weights = _write_rank_netplist(wd, "TopK", params,
                       channels=1, width=W)
  worker = _Worker(netplist, list(weights), ["x"], ["y"], workdir, op="topk")

  def run(srcs, attrs2):
    (x,) = srcs
    xa = np.ascontiguousarray(np.asarray(x, np.float16).reshape(C, W))
    # C separate eval() round-trips, NOT one eval_batch: batching regressed the median (loses pipe pacing).
    rows = [worker.eval([xa[c]])[0].reshape(k) for c in range(C)]
    return np.stack(rows, axis=0)

  return worker, run


# op name -> builder. Ops absent here fall back to the A1 bridge.
_WORKER_BUILDERS = {
  "sdpa": _build_sdpa_worker,
  "argmax": _build_argmax_worker,
  "topk": _build_topk_worker,
}


def has_worker(op: str) -> bool:
  return op in _WORKER_BUILDERS


def build_worker(op: str, shape, attrs):
  """Return `(worker, run_fn)` for `op`. Raises KeyError if no worker route exists."""
  return _WORKER_BUILDERS[op](shape, attrs)
