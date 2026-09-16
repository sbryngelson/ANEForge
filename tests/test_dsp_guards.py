"""Public dsp guards (windows, filters, spectral entry points).

Every one rejects its input before any ANE dispatch, so this module runs
off-device in CI. The numeric coverage of the same routines lives in the
``requires_ane`` modules next to it.
"""
import numpy as np
import pytest

import aneforge.dsp as dsp


def test_get_window_rejects_unknown_name():
  with pytest.raises(ValueError):
    dsp.get_window("nope", 16)


def test_get_window_rejects_wrong_array_length():
  with pytest.raises(ValueError):
    dsp.get_window(np.ones(5, np.float32), 16)


def test_fir_filter_rejects_taps_longer_than_signal():
  with pytest.raises(ValueError):
    dsp.fir_filter(np.ones(4, np.float32), np.ones(9, np.float32))


def test_fir_filter_rejects_unknown_mode():
  with pytest.raises(ValueError):
    dsp.fir_filter(np.ones(16, np.float32), np.ones(3, np.float32), mode="xx")


def test_freq_filter_rejects_unknown_kind():
  with pytest.raises(ValueError):
    dsp.freq_filter(np.ones(16, np.float32), "bandreject", 0.2)


def test_correlate_rejects_template_longer_than_signal():
  with pytest.raises(ValueError):
    dsp.correlate(np.ones(4, np.float32), np.ones(8, np.float32))


def test_correlate_rejects_unknown_mode():
  with pytest.raises(ValueError):
    dsp.correlate(np.ones(16, np.float32), np.ones(4, np.float32), mode="xx")


def test_autocorrelate_rejects_out_of_range_max_lag():
  with pytest.raises(ValueError):
    dsp.autocorrelate(np.ones(16, np.float32), max_lag=99)


def test_hilbert_rejects_too_short():
  with pytest.raises(ValueError):
    dsp.hilbert(np.zeros(1, np.float32))


def test_welch_rejects_non_pow2_nperseg():
  with pytest.raises(ValueError):
    dsp.welch(np.ones(16, np.float32), nperseg=12)
