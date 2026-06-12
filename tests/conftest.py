"""Pytest configuration for the on-device suites.

The tests run each in their own forked subprocess (``--forked`` in pyproject's
``addopts``). Process isolation matters here because every ``compile`` allocates an
e5rt program against the ANE, and a single process accumulates them across a whole
suite — the compile-heavy training and streaming tests alone build hundreds of
programs, which can approach the per-PID program limit and cause sporadic
late-suite failures. A fresh process per test resets that
state, and the fork overhead is negligible next to the ANE compile time itself.

These environment variables must be set before any test forks or loads the dispatch
dylib, so they are set here at collection time (before the test modules import
aneforge):
  - OBJC_DISABLE_INITIALIZE_FORK_SAFETY: the dispatch path loads Apple Obj-C
    frameworks; forking after they initialize trips Obj-C's fork-safety abort unless
    this is set.
  - KMP_DUPLICATE_LIB_OK: tolerate the duplicate OpenMP runtime that numpy/the dylib
    can both pull in.
"""
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
