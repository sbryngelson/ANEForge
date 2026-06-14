"""Pytest configuration for the on-device suites.

The tests run each in their own forked subprocess (``--forked`` in pyproject's
``addopts``). Process isolation matters here because every ``compile`` allocates an
e5rt program against the ANE, and a single process accumulates them across a whole
suite - the compile-heavy training and streaming tests alone build hundreds of
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
  - OMP/MKL/OPENBLAS/VECLIB thread counts = 1: --forked calls os.fork() in a process
    where numpy/torch may have started an OpenMP worker-thread pool. fork() does not
    copy those worker threads, so the child inherits a broken pool, and the next
    OpenMP op (e.g. torch CPU autograd in the grad-vs-torch CIFAR tests) faults inside
    libomp (__kmp_fork_barrier) - a SIGSEGV that surfaced only late in the forked
    suite. One OpenMP thread means there is no pool to break, so the fork is safe.
"""
import os

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
