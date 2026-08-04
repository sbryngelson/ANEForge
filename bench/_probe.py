"""Fresh-process compile/execute probe harness, factored out of the layer_norm probe (#163).

Two things make an op-frontier probe trustworthy, and both are easy to get wrong:

- **One compile per process.** After a compile failure the circuit breaker paces the next compile
  (`ANEFORGE_COMPILE_BACKOFF`), so cells measured later in a long-lived process inherit the earlier
  failure and look broken. `probe_cell` disables the breaker and is meant to run once per interpreter.
- **Staging the outcome.** A compile failure and a dispatch failure have different causes, so they are
  reported separately as `C-FAIL` and `D-FAIL` rather than collapsed into one "fail".

Writing a new probe is then a `build_graph`, a `feed`, and an argv contract; see the `rms_norm` scan in
`bench/rms_norm_compile_probe.py` for a minimal consumer.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:                                       # avoid importing aneforge/numpy at module load
  import numpy as np
  from aneforge.graph import Tensor

OK = "OK"
PREFIXES = (OK, "C-FAIL", "D-FAIL")


def probe_cell(build_graph: "Callable[[], Tensor]", feed: "Callable[[], np.ndarray]") -> str:
  """Compile and dispatch one graph in this process; returns 'OK', 'C-FAIL <Err>' or 'D-FAIL <Err>'.

  Call once per interpreter: the breaker is disabled here, so a second call in the same process would
  not be paced and its result would not be independent of the first. `build_graph` and `feed` are
  callables rather than values so nothing touches aneforge before the env is set."""
  os.environ["ANEFORGE_DISABLE_COMPILE_BREAKER"] = "1"
  os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
  import warnings

  import numpy as np
  warnings.filterwarnings("ignore")
  import aneforge as af

  try:
    net = af.compile(build_graph())
  except Exception as e:                                # noqa: BLE001 - any failure is the datum
    return f"C-FAIL {type(e).__name__}"
  try:
    out = np.asarray(net(feed()), np.float32)
    return OK if np.isfinite(out).all() else "D-FAIL nonfinite"
  except Exception as e:                                # noqa: BLE001
    return f"D-FAIL {type(e).__name__}"


def probe_isolated(argv: Sequence[str], script: str) -> str:
  """Run `script` with `argv` in a fresh interpreter and return its verdict line.

  The child is expected to print exactly one line starting with OK / C-FAIL / D-FAIL. stderr is
  ignored on purpose: E5RT writes compiler noise there, which would otherwise swamp the parse."""
  env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
  p = subprocess.run([sys.executable, script, *argv], capture_output=True, text=True, env=env)
  for line in reversed(p.stdout.splitlines()):
    if line.startswith(PREFIXES):
      return line.strip()
  return "C-FAIL nolines"


def short(res: str) -> str:
  """The verdict without its exception type, for table cells."""
  return res.split()[0]


def chip() -> str:
  """Chip name for the report header, so cells are comparable across generations (#115)."""
  try:
    from bench import _machine
    return _machine.fingerprint()["hardware"]["chip"]
  except Exception:                                     # noqa: BLE001 - the probe works without it
    return "unknown chip"
