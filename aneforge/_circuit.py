"""Compile-failure backoff: after a failure, pace the next compile by ~15s (backstop for the autotuner's variant sweep); a success clears it. Env: ANEFORGE_COMPILE_BACKOFF, ANEFORGE_DISABLE_COMPILE_BREAKER, ANEFORGE_COMPILE_BREAKER_STRICT."""
from __future__ import annotations

import os
import threading
import time
import warnings

# injectable for tests (a fake clock replaces these so no real time passes)
_monotonic = time.monotonic
_sleep = time.sleep

_BACKOFF_S = float(os.environ.get("ANEFORGE_COMPILE_BACKOFF", "15.0"))
_DISABLED = os.environ.get("ANEFORGE_DISABLE_COMPILE_BREAKER") == "1"
_STRICT = os.environ.get("ANEFORGE_COMPILE_BREAKER_STRICT") == "1"

_lock = threading.Lock()
_last_failure_ts: float | None = None   # monotonic time of the last compile failure


class CompileBackoffError(RuntimeError):
  """Raised (strict mode) when a compile is attempted within the backoff window."""


def reset() -> None:
  """Clear the backoff state."""
  global _last_failure_ts
  with _lock: _last_failure_ts = None


def note_compile_result(ok: bool) -> None:
  """Record a compile outcome: failure arms the backoff, success clears it."""
  global _last_failure_ts
  with _lock: _last_failure_ts = None if ok else _monotonic()


def guard_before_compile() -> None:
  """Call before a compile; if one failed within `_BACKOFF_S`, pace this one (sleep, or raise in strict mode)."""
  if _DISABLED or _BACKOFF_S <= 0.0: return
  with _lock:
    if _last_failure_ts is None: return
    wait = _BACKOFF_S - (_monotonic() - _last_failure_ts)
  if wait <= 0.0: return
  if _STRICT:
    raise CompileBackoffError(
      f"aneforge: a compile failed < {_BACKOFF_S:.0f}s ago; refusing another for "
      f"{wait:.1f}s (a defensive backstop for the autotuner's burst of variant "
      f"compiles). Set ANEFORGE_DISABLE_COMPILE_BREAKER=1 to override.")
  warnings.warn(
    f"aneforge: pacing this compile by {wait:.1f}s - a compile failed recently "
    f"(a defensive backstop for the autotuner's burst of variant compiles). "
    f"(ANEFORGE_DISABLE_COMPILE_BREAKER=1 to override.)",
    stacklevel=3)
  _sleep(wait)
