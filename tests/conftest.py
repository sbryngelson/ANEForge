"""Pytest config for the on-device suites: env vars set before tests fork/import aneforge."""
import os

# Must be set before any fork/dylib load (Obj-C fork-safety; dup OpenMP).
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single OpenMP thread: --forked os.fork() can't copy worker-thread pools.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
  os.environ.setdefault(_thread_var, "1")
