"""Compile-failure backoff rate-limiter.

After a compile FAILURE, ANEForge paces the next compile by a short interval (~15 s),
a backstop for the one path that compiles many programs in a burst (the autotuner's
variant sweep) should a compile fail mid-sweep. A successful compile clears the
back-off; no failure-count cap is needed, pacing consecutive failures apart suffices.

It rate-limits *consecutive compile failures*: a clean op-rejection and a harder
failure both surface as a null handle, so the plain "failures" rate limiter is
enough. Tunable / disablable via env:

    ANEFORGE_COMPILE_BACKOFF             seconds to pace (default 15.0; 0 disables pacing)
    ANEFORGE_DISABLE_COMPILE_BREAKER=1   turn the guard off entirely
    ANEFORGE_COMPILE_BREAKER_STRICT=1    raise CompileBackoffError instead of sleeping
"""
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
    """Raised (in strict mode) when a compile is attempted within the backoff window
    after a recent compile failure."""


def reset() -> None:
    """Clear the backoff state (forget the last failure)."""
    global _last_failure_ts
    with _lock:
        _last_failure_ts = None


def note_compile_result(ok: bool) -> None:
    """Record the outcome of a compile. A failure arms the backoff; a success clears it."""
    global _last_failure_ts
    with _lock:
        _last_failure_ts = None if ok else _monotonic()


def guard_before_compile() -> None:
    """Call immediately before a compile. If a compile failed within the last
    `_BACKOFF_S` seconds, pace this one to keep consecutive failures a short interval
    apart (default: sleep the remainder; strict mode: raise)."""
    if _DISABLED or _BACKOFF_S <= 0.0:
        return
    with _lock:
        if _last_failure_ts is None:
            return
        wait = _BACKOFF_S - (_monotonic() - _last_failure_ts)
    if wait <= 0.0:
        return
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
