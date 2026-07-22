"""compile() warns (DispatchFloorWarning) for dispatch-floor-bound programs, not compute-bound; filterable."""
import warnings

import numpy as np

import aneforge as af
from _helpers import requires_ane

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE


def _warned(build):
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    build()
    return any(issubclass(x.category, af.DispatchFloorWarning) for x in w)


def test_floor_bound_program_warns():
  rng = np.random.default_rng(0)
  W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
  assert _warned(lambda: af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1).relu().mean((2, 3)))), \
      "tiny program should be flagged dispatch-floor-bound"


def test_compute_bound_program_does_not_warn():
  rng = np.random.default_rng(1)
  Wt = (rng.standard_normal((4096, 4096)) * 0.02).astype(np.float32)
  assert not _warned(lambda: af.compile(af.input((2048, 4096)).linear(Wt))), \
      "large compute-bound matmul should not be flagged"


def test_warning_is_filterable():
  rng = np.random.default_rng(2)
  W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=af.DispatchFloorWarning)
    af.compile(af.conv(af.input((1, 8, 16, 16)), W, pad=1).relu().mean((2, 3)))
    assert not any(issubclass(x.category, af.DispatchFloorWarning) for x in w)
