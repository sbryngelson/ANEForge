"""Pytest config for the on-device suites: env vars set before tests fork/import aneforge."""
import os
import sys

import pytest

# Must be set before any fork/dylib load (Obj-C fork-safety; dup OpenMP).
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single OpenMP thread: --forked os.fork() can't copy worker-thread pools.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
  os.environ.setdefault(_thread_var, "1")

# The sibling helper modules (_helpers, _corpus) are imported bare by the suites; make
# sure this directory is importable from conftest too (needed by the hook below).
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
  config.addinivalue_line(
    "markers", "requires_ane: needs real ANE hardware; auto-skipped when the e5rt dylib is unavailable")


def pytest_collection_modifyitems(items):
  """Auto-skip requires_ane tests when there is no ANE, so a plain off-device `pytest` run
  skips them instead of erroring. CI additionally deselects them with -m "not requires_ane";
  this covers a bare run (no -m) on a machine without the engine. Probe runs once, in the
  main process before --forked forks the per-test subprocesses."""
  from _helpers import ane_available
  if ane_available():
    return
  skip = pytest.mark.skip(reason="ANE/e5rt dylib unavailable")
  for item in items:
    if "requires_ane" in item.keywords:
      item.add_marker(skip)
