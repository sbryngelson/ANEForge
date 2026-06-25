"""Device runtime: a ctypes wrapper over libane_e5rt_dispatch.dylib exposing compile / eval (bind fp16 in, run, read fp16 out) / release."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

# Device-mask bits. aneforge always targets the ANE; the others exist for parity.
BNNS_MASK = 0x1     # CPU
GPU_MASK = 0x2      # GPU (MPSGraph)
ANE_MASK = 0x4      # Apple Neural Engine


_DYLIB = "libane_e5rt_dispatch.dylib"


def _find_dylib() -> Path:
  """Locate libane_e5rt_dispatch.dylib, building on first use if needed. Order: ANEFORGE_DYLIB, packaged `_lib/`, legacy `ane_build/lib/`, build cache, else compile-and-cache. ANEFORGE_NO_AUTOBUILD=1 requires an explicit build."""
  override = os.environ.get("ANEFORGE_DYLIB")
  if override and Path(override).expanduser().exists(): return Path(override).expanduser()
  here = Path(__file__).resolve()
  bundled = here.parent / "_lib" / _DYLIB
  if bundled.exists(): return bundled
  for up in range(len(here.parents)):                      # legacy ane_build/lib
    candidate = here.parents[up] / "ane_build" / "lib" / _DYLIB
    if candidate.exists(): return candidate
  from .build import _cache_dir, build_dylib
  cached = _cache_dir() / _DYLIB
  if cached.exists(): return cached
  if os.environ.get("ANEFORGE_NO_AUTOBUILD"):
    raise FileNotFoundError(
      f"{_DYLIB} not built. Build it with:\n"
      "  python -m aneforge.build      # or: sh aneforge/_lib/build.sh"
    )
  return build_dylib()


class _Dylib:
  """Lazy ctypes proxy: the dylib is loaded and ABI-bound on first attribute access, so `import aneforge` works without it (only compile/dispatch needs the dylib)."""

  def __init__(self) -> None:
    self._lib = None

  def _ensure(self):
    if self._lib is not None: return self._lib
    lib = ctypes.CDLL(str(_find_dylib()))
    try:
      self._bind(lib)
    except AttributeError as e:
      raise RuntimeError(
        f"{_DYLIB} is stale (missing symbol: {e}). Rebuild it with:\n"
        "  python -m aneforge.build      # or: sh aneforge/_lib/build.sh"
      ) from e
    self._lib = lib
    return lib

  @staticmethod
  def _bind(lib) -> None:
    lib.ane_e5rt_program_compile.restype = ctypes.c_void_p
    lib.ane_e5rt_program_compile.argtypes = [
      ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
      ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ]
    lib.ane_e5rt_program_set_input_fp16.restype = ctypes.c_int
    lib.ane_e5rt_program_set_input_fp16.argtypes = [
      ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t]
    lib.ane_e5rt_program_get_output_fp16.restype = ctypes.c_int
    lib.ane_e5rt_program_get_output_fp16.argtypes = [
      ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t]
    lib.ane_e5rt_program_execute.restype = ctypes.c_int
    lib.ane_e5rt_program_execute.argtypes = [ctypes.c_void_p]
    # zero-copy: hand back the persistent CPU+ANE-visible buffer for a port
    for _fn in ("ane_e5rt_program_input_buffer", "ane_e5rt_program_output_buffer"):
      getattr(lib, _fn).restype = ctypes.c_int
      getattr(lib, _fn).argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    lib.ane_e5rt_program_release.restype = None
    lib.ane_e5rt_program_release.argtypes = [ctypes.c_void_p]
    # resident state: rebind an input port to read another port's output buffer
    # (aliasing own output->input keeps a tensor on-device across executes)
    lib.ane_e5rt_program_share_buffer.restype = ctypes.c_int
    lib.ane_e5rt_program_share_buffer.argtypes = [
      ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    # compile-only cross-target check: compile a MIL for another ANE family from this host
    lib.ane_e5rt_compile_check.restype = ctypes.c_int
    lib.ane_e5rt_compile_check.argtypes = [
      ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_char_p]

  def __getattr__(self, name):
    return getattr(self._ensure(), name)


_lib = _Dylib()


def compile_check(mil_path, cache_dir, custom_ane_options=None, device_mask=ANE_MASK) -> int:
  """Compile `mil_path` for `custom_ane_options` (e.g. `'TargetArchitecture=h13'`) without creating the device op. rc 0 = a library was produced."""
  opt = custom_ane_options.encode() if custom_ane_options else None
  return _lib.ane_e5rt_compile_check(
    str(mil_path).encode(), str(cache_dir).encode(), device_mask, opt)


def _numel(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape: n *= int(d)
  return n


def _fp16_bytes(shape: tuple[int, ...]) -> int:
  return _numel(shape) * 2


# bytes-per-element of supported input dtypes (compute is fp16; uint8 dequant is in-graph)
_DT_BYTES = {"fp16": 2, "uint8": 1}


def _port_bytes(shape: tuple[int, ...], dtype: str) -> int:
  """Byte size of an input port; uint8 rounds up to even (set_input copies in 2-byte units)."""
  n = _numel(shape) * _DT_BYTES[dtype]
  return n + (n % 2)


def _u16(arr: np.ndarray) -> np.ndarray:
  """Reinterpret a contiguous fp16 array as uint16 for ctypes."""
  if arr.dtype != np.float16: arr = arr.astype(np.float16, copy=False)
  return np.ascontiguousarray(arr).view(np.uint16)


def _ptr(view: np.ndarray):
  return view.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


class Program:
  """A compiled, dispatch-ready ANE program. Output buffers are reused across `eval` calls (copy to retain results)."""

  def __init__(self, handle: int, inputs: dict, outputs: dict, device_mask: int,
               input_dtypes: dict | None = None) -> None:
    self._handle = handle
    self._inputs = dict(inputs)
    self._outputs = dict(outputs)
    self._device_mask = device_mask
    # per-port input dtype ("fp16" default, "uint8" for image inputs): only the feed differs
    self._input_dtypes = dict(input_dtypes or {})
    self._out_buffers = {name: np.zeros(shape, np.float16) for name, shape in outputs.items()}

  def _feed(self, name: str, arr: np.ndarray) -> None:
    """Copy `arr` into input port `name` (fp16 view, or raw bytes via uint16 view for uint8 ports)."""
    if self._input_dtypes.get(name, "fp16") == "uint8":
      b = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8)).reshape(-1)
      if b.size % 2:                                   # pad odd byte count for the uint16 view
        b = np.concatenate([b, np.zeros(1, np.uint8)])
      view = b.view(np.uint16)
    else:
      view = _u16(arr)
    rc = _lib.ane_e5rt_program_set_input_fp16(self._handle, name.encode(), _ptr(view), view.size)
    if rc != 0: raise RuntimeError(f"set_input({name}) failed rc={rc}")

  def eval(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for name, arr in inputs.items():
      if name not in self._inputs: raise KeyError(f"unknown input port: {name!r}")
      if tuple(arr.shape) != self._inputs[name]:
        raise ValueError(f"input {name!r}: expected {self._inputs[name]}, got {tuple(arr.shape)}")
      self._feed(name, arr)
    rc = _lib.ane_e5rt_program_execute(self._handle)
    if rc != 0: raise RuntimeError(f"execute failed rc={rc}")
    for name, buf in self._out_buffers.items():
      view = buf.view(np.uint16)
      rc = _lib.ane_e5rt_program_get_output_fp16(self._handle, name.encode(), _ptr(view), view.size)
      if rc != 0: raise RuntimeError(f"get_output({name}) failed rc={rc}")
    return self._out_buffers

  # ---- resident-state primitives (granular set / execute / read + alias) ----
  # seed state ports once, alias each state output onto its input, then per step feed
  # only the data ports and execute - no state copy.

  def share_buffer(self, src_op_idx: int, src_out_port: str,
                   dst_op_idx: int, dst_in_port: str) -> None:
    """Rebind `dst`'s input port to read `src`'s output buffer (aliasing own output->input keeps state resident across `execute`)."""
    rc = _lib.ane_e5rt_program_share_buffer(
      self._handle, ctypes.c_size_t(src_op_idx), src_out_port.encode(),
      ctypes.c_size_t(dst_op_idx), dst_in_port.encode())
    if rc != 0:
      raise RuntimeError(
        f"share_buffer(op{src_op_idx}.{src_out_port}->op{dst_op_idx}.{dst_in_port}) "
        f"failed rc={rc}")

  def set_input(self, name: str, arr: np.ndarray) -> None:
    """Write one input port (feed a data port / seed resident state)."""
    if name not in self._inputs: raise KeyError(f"unknown input port: {name!r}")
    if tuple(arr.shape) != self._inputs[name]:
      raise ValueError(f"input {name!r}: expected {self._inputs[name]}, got {tuple(arr.shape)}")
    self._feed(name, arr)

  def execute(self) -> None:
    """Run the program once (no input binding, no output read)."""
    rc = _lib.ane_e5rt_program_execute(self._handle)
    if rc != 0: raise RuntimeError(f"execute failed rc={rc}")

  def read_output(self, name: str) -> np.ndarray:
    """Read one output port into a fresh fp16 array (a checkpoint read)."""
    if name not in self._outputs: raise KeyError(f"unknown output port: {name!r}")
    buf = self._out_buffers[name]
    view = buf.view(np.uint16)
    rc = _lib.ane_e5rt_program_get_output_fp16(self._handle, name.encode(), _ptr(view), view.size)
    if rc != 0: raise RuntimeError(f"get_output({name}) failed rc={rc}")
    return buf.copy()

  # ---- zero-copy buffer views (skip the per-call host<->device memcpy) ----
  def _buffer_view(self, name: str, shape: tuple[int, ...], getter) -> np.ndarray:
    ptr, nbytes = ctypes.c_void_p(), ctypes.c_size_t()
    rc = getter(self._handle, name.encode(), ctypes.byref(ptr), ctypes.byref(nbytes))
    if rc != 0 or not ptr.value: raise RuntimeError(f"no zero-copy buffer for port {name!r} (rc={rc})")
    n = _numel(shape)
    if nbytes.value < n * 2: raise RuntimeError(f"port {name!r} buffer too small ({nbytes.value} < {n*2})")
    raw = np.ctypeslib.as_array((ctypes.c_uint16 * n).from_address(ptr.value))
    return raw.view(np.float16).reshape(shape)            # writable view onto ANE memory

  def input_view(self, name: str) -> np.ndarray:
    """Writable fp16 view onto the persistent ANE-visible INPUT buffer; write in place then `execute()` (skips the host->device copy). Valid only while the Program is alive."""
    if name not in self._inputs: raise KeyError(f"unknown input port: {name!r}")
    return self._buffer_view(name, self._inputs[name], _lib.ane_e5rt_program_input_buffer)

  def output_view(self, name: str) -> np.ndarray:
    """fp16 view onto the persistent ANE-visible OUTPUT buffer (valid after `execute()`); `.copy()` to retain across the next `execute()`."""
    if name not in self._outputs: raise KeyError(f"unknown output port: {name!r}")
    return self._buffer_view(name, self._outputs[name], _lib.ane_e5rt_program_output_buffer)

  def release(self) -> None:
    if self._handle:
      _lib.ane_e5rt_program_release(self._handle)
      self._handle = 0

  def __enter__(self) -> "Program":
    return self

  def __exit__(self, *args: Any) -> None:
    self.release()

  def __del__(self) -> None:
    try:
      self.release()
    except Exception:
      pass


class E5RT:
  """Compiles a MIL program into a ready-to-run :class:`Program` (stateless)."""

  @staticmethod
  def compile(mil_path, *, cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "aneforge", "e5rt"),
              inputs: dict[str, tuple[int, ...]], outputs: dict[str, tuple[int, ...]],
              device_mask: int = ANE_MASK, input_dtypes: dict | None = None) -> Program:
    mil_path, cache_dir = os.fspath(mil_path), os.fspath(cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(mil_path): raise FileNotFoundError(mil_path)
    input_dtypes = dict(input_dtypes or {})
    in_names, out_names = list(inputs), list(outputs)
    in_n = (ctypes.c_char_p * len(in_names))(*[n.encode() for n in in_names])
    in_s = (ctypes.c_size_t * len(in_names))(
      *[_port_bytes(inputs[n], input_dtypes.get(n, "fp16")) for n in in_names])
    out_n = (ctypes.c_char_p * len(out_names))(*[n.encode() for n in out_names])
    out_s = (ctypes.c_size_t * len(out_names))(*[_fp16_bytes(outputs[n]) for n in out_names])
    # pace consecutive compile failures (backstop for the autotuner's variant burst)
    from . import _circuit
    _circuit.guard_before_compile()
    handle = _lib.ane_e5rt_program_compile(
      mil_path.encode(), cache_dir.encode(), ctypes.c_uint64(device_mask),
      in_n, in_s, ctypes.c_size_t(len(in_names)),
      out_n, out_s, ctypes.c_size_t(len(out_names)))
    _circuit.note_compile_result(bool(handle))
    if not handle:
      raise RuntimeError(
        f"ane_e5rt_program_compile failed (mil={mil_path}, mask=0x{device_mask:x}). "
        "See stderr for the actual Espresso error.")
    return Program(handle, inputs, outputs, device_mask, input_dtypes)
