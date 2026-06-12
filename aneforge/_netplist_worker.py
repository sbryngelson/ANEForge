"""A2: a persistent Path-A worker for aneforge's netplist-bridge ops.

The correctness-first path (A1) spawns a one-shot ObjC probe per `af.sdpa` /
`af.argmax` / `af.topk` call: that probe pays the full
descriptor -> compileWithQoS -> loadWithQoS -> mapIOSurfaces tax (hundreds of
ms), evaluates once, and exits.  For repeated calls of a *fixed-shape* netplist
program (the common decode-loop case) that startup tax dominates.

A2 keeps a long-lived helper process (`~/.cache/aneforge/bin/persistent_worker`)
that pays compile/load/map ONCE at startup, then services many evals over a
pipe.  Steady-state per call is just host<->IOSurface memcpy + evaluateWithQoS.

This module is import-lazy: `aneforge` stays standalone-importable; the worker
is built/spawned only the first time a netplist-bridge op runs.  It authors the
netplist with the SAME in-package bridge generators A1 uses
(`ane_rank_fused._build_plist` / `ane_sdpa_fused._write_netplist`) — we never
duplicate the netplist layout, only swap the *dispatch* (load-once-eval-many vs
one-shot).

Supported ops: `sdpa` (GOAL #3 shape), `argmax` (GlobalArgMinMax over Width or
Channel) and `topk` (native TopK, per-row tiled).  `topk`'s per-row tiling (the
native op keys all channels by one lane, so per-row top-k runs each row as its
own 1-channel program) is reproduced by loading ONE 1-channel program and
eval-ing it C times, matching the A1 result bit-for-bit.
"""
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
    """Build the persistent-worker invoker once if missing/stale (UNIQUE path so
    it never clobbers the one-shot invokers)."""
    binp = bin_dir() / _INVOKER_BIN_NAME
    src = Path(__file__).resolve().parent / "_invokers" / "persistent_worker.mm"
    if binp.exists() and src.exists() and binp.stat().st_mtime >= src.stat().st_mtime:
        return binp
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
    """Owns one spawned worker process holding ONE compiled+loaded netplist.

    The wire protocol is batched: per request we write a 4-byte little-endian
    uint32 count `N` followed by `N` input sets packed back-to-back as raw fp16
    bytes (each set in the spawn symbol order), and read back `N` output sets
    packed the same way.  `self.in_elems` / `self.out_elems` (from the worker's
    "ready" line) give the per-set framing.  `eval` is the N=1 case; `eval_batch`
    packs many input sets into ONE round-trip (the per-row-tiled topk/sort win).
    """

    def __init__(self, netplist: Path, weights: list[Path], in_syms: list[str],
                 out_syms: list[str], workdir: tempfile.TemporaryDirectory,
                 op: str = "netplist"):
        self._workdir = workdir  # keep the netplist + weights alive on disk
        self._op = op
        binp = _ensure_worker_built()
        cmd = [str(binp), "--net-plist", str(netplist)]
        for w in weights:
            cmd += ["--weights", str(w)]
        for s in in_syms:
            cmd += ["--input", s]
        for s in out_syms:
            cmd += ["--output", s]
        # bufsize=0: the protocol is length-prefixed binary and `_read_exact`
        # select()s on the raw stdout fd, which is only correct with no
        # Python-level read buffering in front of it.
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, bufsize=0)
        ready = self._proc.stdout.readline()
        if not ready:
            err = self._proc.stderr.read().decode(errors="replace")
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
        """One eval. `inputs` are contiguous fp16 arrays in spawn symbol order;
        returns the output tensors as flat fp16 arrays in spawn symbol order."""
        return self.eval_batch([inputs])[0]

    def eval_batch(self, input_sets: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        """Run `N = len(input_sets)` evals in ONE pipe round-trip.  Each entry of
        `input_sets` is a full input set (one array per spawn symbol, in order).
        Returns N output sets (one flat fp16 array per output symbol).  Bit-identical
        to calling `eval` N times, but pays a single request/reply instead of N --
        the per-row-tiled topk/sort win."""
        if self._proc.poll() is not None:
            raise RuntimeError("persistent worker has exited")
        n = len(input_sets)
        if n == 0:
            return []
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
                data = data[self._proc.stdin.write(data):]
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            err = self._proc.stderr.read().decode(errors="replace")
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
        fd = self._proc.stdout.fileno()
        buf = bytearray()
        while len(buf) < n:
            if timeout > 0:
                ready, _, _ = select.select([fd], [], [], timeout)
                if not ready:
                    self._proc.kill()
                    self._proc.wait()
                    err = self._proc.stderr.read().decode(errors="replace")
                    raise RuntimeError(
                        f"persistent worker ({self._op}) unresponsive after "
                        f"{timeout:g}s: {err}")
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                err = self._proc.stderr.read().decode(errors="replace")
                raise RuntimeError(f"persistent worker ({self._op}) died mid-eval: {err}")
            buf += chunk
        return bytes(buf)

    def release(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()  # EOF -> clean shutdown
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        try:
            self._workdir.cleanup()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# per-op worker builders: author the netplist with the A1 bridge generator,    #
# then hand it to a persistent _Worker.  Each returns a callable               #
# (src_arrays, attrs) -> output fp16 array matching the A1 runner's contract.  #
# --------------------------------------------------------------------------- #

def _build_sdpa_worker(shape, attrs):
    """Persistent SDPA. `shape` is the Q/K/V shape [1, H, S, D]; `attrs` has `scale`.
    Mirrors ane_sdpa_fused.sdpa_fused's ANE layout: pre/post transpose Q,K,V
    (heads<->seq) so the netplist sees seq-in-C, heads-in-H."""
    from ._bridges import ane_sdpa_fused as af  # the A1 bridge = the netplist author

    B, H, S, D = shape
    if B != 1:
        raise ValueError(f"SDPA worker only supports B=1; got B={B}")
    scale = attrs.get("scale")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

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
    """Author a single-op rank netplist (Sort/TopK/ArgMinMax/GlobalArgMinMax) into
    `wd` via the A1 bridge's own generator (`ane_rank_fused._build_plist`) so the
    worker sees byte-identical layout.  Returns (netplist_path, [weight])."""
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
    """Persistent GlobalArgMinMax (argmax).  `shape` is the input `[C, W]`; `attrs`
    has `axis` (1=Width, 0=Channel).  One loaded program, one eval per call.  Output
    matches the A1 `_run_argmax` contract: keepdims `[C,1]` for axis=1 (Width) or
    `[1,W]` for axis=0 (Channel)."""
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
    """Persistent per-row TopK.  `shape` is the input `[C, W]`; `attrs` has `k` and
    `largest`.  The native TopK keys *all* channels by ONE lane's order, so
    PyTorch-style per-row top-k requires each row run as its own 1-channel program.
    We load ONE 1-channel (channels=1, width=W, K=k) program and eval it C times (one
    row per eval), reproducing the A1 stack exactly.  Output is `[C, k]` values."""
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
        # note: the C row-tiles dispatch as C separate eval() round-trips, not one
        # eval_batch(). Measured on M5 Pro, topk's per-call floor is set by the C
        # sequential ANE evaluateWithQoS dispatches (~0.08 ms each), not by pipe
        # round-trips (a single trip+eval is ~0.083 ms; the worker blocks on read and
        # replies immediately, so the trip is ~free). Packing all C rows into one
        # round-trip removes only the ~free trips and regresses the median (per-eval
        # cost rose 0.079->0.28 ms at N=16) because the worker's back-to-back eval
        # loop loses the pipe pacing that keeps the ANE warm. So we keep the per-call
        # dispatch; eval_batch stays available (and bit-identical) when the round-trip
        # truly dominates.
        rows = [worker.eval([xa[c]])[0].reshape(k) for c in range(C)]
        return np.stack(rows, axis=0)

    return worker, run


# op name -> builder. Ops absent here have no worker route yet and fall back to
# the A1 bridge (see _compile._netplist_runner).
_WORKER_BUILDERS = {
    "sdpa": _build_sdpa_worker,
    "argmax": _build_argmax_worker,
    "topk": _build_topk_worker,
}


def has_worker(op: str) -> bool:
    return op in _WORKER_BUILDERS


def build_worker(op: str, shape, attrs):
    """Return `(worker, run_fn)` for `op` at `shape`/`attrs`.  Raises
    KeyError if no worker route exists for `op`."""
    return _WORKER_BUILDERS[op](shape, attrs)
