"""Run the dsp module self-test battery under pytest (issue #127).

`aneforge.dsp._selftest()` validates the FIR / FFT-based DSP routines on the
engine (vs scipy / numpy) and returns 0 when the gate holds. iOS-like CI cannot
reach the ANE, so this is gated; it passed locally on M2 Pro.
"""
import aneforge.dsp as dsp
from _helpers import requires_ane

pytestmark = requires_ane  # the battery dispatches FIR/FFT ops to the engine


def test_dsp_selftest_battery_passes():
  assert dsp._selftest() == 0
