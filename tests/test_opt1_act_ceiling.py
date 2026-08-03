"""opt=1 must decline a lossy variant whose activation-encoding ceiling the graph can exceed (#153).

Build-level: no dispatch, so these run in CI without an ANE. The on-device consequence (opt=1
returning inf on a bounded matmul) is reproduced by `bench/opt1_int8_saturation_repro.py`, committed
alongside this file.
"""
import numpy as np
import pytest

import aneforge as af
from aneforge import _optimize
from aneforge._compile import _act_max_abs
from aneforge._targets import FP16_SLICE_SAT, Q4_X16_SAT
from aneforge.graph import _const

W = (np.random.default_rng(0).standard_normal((15, 16)) / 4).astype(np.float16)


def _labels(cfgs):
  return {"int8" if c.get("int8") else "other" for c in cfgs}


def test_ceiling_is_pinned_at_4094():
  """The Q.4 x16 encoding range, measured identical on M1 Max / M2 Pro / M5 Pro and invariant to
  weight scale, K, N and seed. 4095 is not representable in fp16 (spacing is 2 there), so the gate
  sits exactly between the adjacent representables 4094 and 4096."""
  assert Q4_X16_SAT == 4094.0
  assert FP16_SLICE_SAT == Q4_X16_SAT, "the slice path and the int8 variant share one encoding"
  assert np.float16(4095) == np.float16(4096), "no fp16 activation lies between 4094 and 4096"


def test_ceiling_is_per_variant_not_global():
  """int4-LUT / sparse may encode differently, so the ceiling is keyed by variant, and a variant
  with no encoding limit reports None rather than inheriting int8's."""
  assert _optimize._variant_act_ceiling({"int8": True}) == Q4_X16_SAT
  assert _optimize._variant_act_ceiling({"int8": False}) is None
  assert _optimize._variant_act_ceiling({"int8": False, "constfold": [0]}) is None


def test_runtime_input_has_no_static_bound():
  assert _act_max_abs(af.input((31, 15)) @ W) is None

def test_const_fed_graph_is_bounded():
  assert _act_max_abs(_const((np.ones((31, 15)) * 10.0).astype(np.float16)) @ W) == pytest.approx(10.0)


def test_opt1_declines_int8_for_runtime_input():
  """Fail closed on an unknown bound, matching the slice-saturation rule in _targets.py."""
  out = af.input((31, 15)) @ W
  assert "int8" in _labels(_optimize._variants(out)), "int8 is offered without the gate"
  assert "int8" not in _labels(_optimize._variants(out, drop_unsafe=True))

def test_opt1_declines_int8_above_the_ceiling():
  out = _const((np.ones((31, 15)) * 5000.0).astype(np.float16)) @ W
  assert "int8" not in _labels(_optimize._variants(out, drop_unsafe=True))

def test_opt1_keeps_int8_below_the_ceiling():
  """The gate must not be a blanket ban: a bounded const-fed graph still gets the cheap variant."""
  out = _const((np.ones((31, 15)) * 10.0).astype(np.float16)) @ W
  assert "int8" in _labels(_optimize._variants(out, drop_unsafe=True))

@pytest.mark.parametrize("scale,expect_int8", [(4094.0, True), (4096.0, False)])
def test_gate_boundary_is_exact(scale, expect_int8):
  """At the encoding boundary: 4094 encodes, the next representable does not."""
  out = _const((np.ones((31, 15)) * scale).astype(np.float16)) @ W
  assert ("int8" in _labels(_optimize._variants(out, drop_unsafe=True))) is expect_int8


def test_opt2_still_offers_lossy_variants():
  """opt=2 measures and rejects on error, so it must keep int8 available (no drop_unsafe there)."""
  out = af.input((31, 15)) @ W
  assert "int8" in _labels(_optimize._variants(out))
