"""The compile-failure backoff rate-limiter (aneforge/_circuit.py).

A defensive backstop for the autotuner's burst of variant compiles: after a
compile FAILURE, pace the next compile by a short interval (~15 s) so consecutive
failures stay apart. The guard is a per-process rate limiter on consecutive
failures; no failure-count cap is needed.

These are pure unit tests (no ANE): a fake clock replaces _monotonic/_sleep so no
real time passes.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 -m pytest tests/test_compile_breaker.py -q
"""
import warnings

import pytest

from aneforge import _circuit


class _Clock:
    """Deterministic clock: _sleep advances time instead of blocking."""
    def __init__(self, t0=1000.0):
        self.t = t0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # fresh breaker state + a fake clock for every test
    clock = _Clock()
    monkeypatch.setattr(_circuit, "_monotonic", clock.now)
    monkeypatch.setattr(_circuit, "_sleep", clock.sleep)
    monkeypatch.setattr(_circuit, "_BACKOFF_S", 15.0)
    monkeypatch.setattr(_circuit, "_DISABLED", False)
    monkeypatch.setattr(_circuit, "_STRICT", False)
    _circuit.reset()
    yield clock
    _circuit.reset()


def test_no_failure_no_backoff(_isolate):
    _circuit.guard_before_compile()
    _circuit.guard_before_compile()
    assert _isolate.slept == []          # nothing ever failed -> never paced


def test_backoff_after_failure(_isolate):
    _circuit.note_compile_result(False)  # a compile failed
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _circuit.guard_before_compile()  # next compile must wait the backoff interval
    assert len(_isolate.slept) == 1
    assert _isolate.slept[0] == pytest.approx(15.0)
    assert any("compile" in str(w.message).lower() for w in rec)


def test_success_clears_backoff(_isolate):
    _circuit.note_compile_result(False)
    _circuit.note_compile_result(True)   # a later success clears the backoff
    _circuit.guard_before_compile()
    assert _isolate.slept == []          # no pacing after a clean success


def test_backoff_expires_with_elapsed_time(_isolate):
    _circuit.note_compile_result(False)
    _isolate.t += 20.0                   # 20 s passed on its own (> the 15 s interval)
    _circuit.guard_before_compile()
    assert _isolate.slept == []          # already paced by wall time


def test_partial_backoff_sleeps_only_remainder(_isolate):
    _circuit.note_compile_result(False)
    _isolate.t += 9.0                    # 9 s already elapsed
    _circuit.guard_before_compile()
    assert _isolate.slept[0] == pytest.approx(6.0)   # sleep only the remaining 6 s


def test_consecutive_failures_each_paced(_isolate):
    # the failure pattern: failure -> (paced) -> failure -> (paced) ...
    _circuit.note_compile_result(False)
    _circuit.guard_before_compile()
    _circuit.note_compile_result(False)
    _circuit.guard_before_compile()
    assert len(_isolate.slept) == 2
    assert all(s == pytest.approx(15.0) for s in _isolate.slept)


def test_disabled_never_paces(_isolate, monkeypatch):
    monkeypatch.setattr(_circuit, "_DISABLED", True)
    _circuit.note_compile_result(False)
    _circuit.guard_before_compile()
    assert _isolate.slept == []


def test_strict_mode_raises_instead_of_sleeping(_isolate, monkeypatch):
    monkeypatch.setattr(_circuit, "_STRICT", True)
    _circuit.note_compile_result(False)
    with pytest.raises(_circuit.CompileBackoffError):
        _circuit.guard_before_compile()
    assert _isolate.slept == []          # strict refuses rather than waiting


def test_reset_clears_state(_isolate):
    _circuit.note_compile_result(False)
    _circuit.reset()
    _circuit.guard_before_compile()
    assert _isolate.slept == []
