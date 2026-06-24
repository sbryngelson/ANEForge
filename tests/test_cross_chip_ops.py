"""Ops that M1 (A13) cannot run but newer Macs can. On M1 hardware we test the
TWO things that are testable: (1) the op catalog correctly declares each op's
cross-chip availability (walled/bridge on M1, native on the supporting family),
and (2) aneforge GUARDS them on M1 - it refuses (clear arch-gate error) rather
than silently miscompiling, and the same graph compiles when targeting a chip
that supports it. The op cannot actually EXECUTE here (that needs the newer silicon)."""
from __future__ import annotations
import pytest
import aneforge as af
from aneforge import _op_catalog as oc
import aneforge._targets as TG


def _ane():
  try:
    from aneforge._runtime import _find_dylib; _find_dylib(); return True
  except Exception:
    return False


requires_ane = pytest.mark.skipif(not _ane(), reason="ANE/e5rt dylib unavailable")


# --- (1) catalog correctly declares M1-can't / M5-can for the gated ops -----------------
# exposed ops whose native path needs a newer chip than M1 (A14/A15+)
GATED = ["affine", "topk", "sort", "sin", "cos", "resize_bilinear",
         "resize_nearest_neighbor", "upsample_bilinear", "crop_resize", "resample",
         "dropout", "random", "global_argmax", "global_argmin", "reduce_argmin"]


@pytest.mark.parametrize("op", [g for g in GATED if g in oc.OP_CATALOG])
def test_catalog_declares_cross_chip(op):
  d = oc.OP_CATALOG[op]
  # not natively runnable on M1 (walled, or bridge=host-decomposed - never plain native)
  assert d["m1"] != "native", f"{op} claims native on M1 but is gated"
  # a newer family runs it (native somewhere above M1, or it's a bridge op)
  higher_native = any(d[k] == "native" for k in ("m2", "m3", "m4_m5"))
  assert higher_native or d["m1"] == "bridge", f"{op} has no supporting chip"


def test_catalog_capability_grows_m1_to_m5():
  # M5 (family 5) runs at least as many native ops as M1 (family 2) - capability is a ladder.
  m1 = set(oc.ops_on("m1", "native"))
  m5 = set(oc.ops_on("m5", "native"))
  assert m1 <= m5, f"M1-native ops not all M5-native: {sorted(m1 - m5)}"
  assert len(m5) > len(m1), "M5 should expose strictly more native ops than M1"


# --- (2) aneforge guards M1-incapable ops (refuses, not miscompiles) --------------------
@requires_ane
@pytest.mark.parametrize("name,mk", [
    ("topk", lambda: af.topk(af.input((1, 16)), k=4)),                 # build-time arch guard
    ("sort", lambda: af.compile(af.sort(af.input((1, 16))))),          # compile-time family guard
])
def test_m1_refuses_gated_op(name, mk, monkeypatch):
  # Pin the M1 (h13) target so this asserts "M1 refuses" regardless of the host
  # chip: on an M5 host the same ops compile, so relying on host detection would
  # make the test pass only when run on M1.
  monkeypatch.setenv("ANEFORGE_TARGET", "h13"); TG._cpu_brand.cache_clear()
  try:
    with pytest.raises(Exception) as ei:
      mk()
    msg = str(ei.value).lower()
    assert ("arch-gated" in msg or "not runnable" in msg or "family" in msg), \
        f"{name} raised but not a clear arch-gate error: {msg[:80]}"
  finally:
    TG._cpu_brand.cache_clear()


@requires_ane
def test_sort_compiles_for_m5_not_m1(monkeypatch):
  """The exact graph M1 refuses compiles when targeting M5 - proof the op is M5-runnable."""
  g_m5 = af.sort(af.input((1, 16)))
  monkeypatch.setenv("ANEFORGE_TARGET", "h16s"); TG._cpu_brand.cache_clear()
  af.compile(g_m5, target="h16s")                 # must not raise (M5/A16 supports sort)
  # and it's refused for the M1 target
  monkeypatch.setenv("ANEFORGE_TARGET", "h13"); TG._cpu_brand.cache_clear()
  with pytest.raises(Exception):
    af.compile(af.sort(af.input((1, 16))), target="h13")


# --- regression: last-axis gather routes off the width axis (A13/A14 crop-DMA quirk) -----
# The composed gather (slice_by_size + concat) returns WRONG ELEMENTS for a nonzero WIDTH
# (last-axis) begin on A13/A14; gather transposes the axis off the last position so it is
# exact on every family. Guards the M2/A14 gather-axis-1 fix (master 41ca47b).
@requires_ane
def test_gather_last_axis_matches_numpy():
  import numpy as np
  # fp16-EXACT integer inputs (distinct values, well under 2048) so array_equal tests
  # element SELECTION precisely - a wrong element is an integer mismatch fp16 can't mask.
  x2 = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
  idx2 = [3, 0, 15, 7, 7, 1]
  got2 = af.compile(af.gather(af.input((8, 16)), idx2, axis=1))(x2)
  assert np.array_equal(got2, x2[:, idx2]), "2D last-axis gather not exact"
  # 3D last-axis (axis 2) - the transpose-routing must generalize past rank 2
  x3 = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
  idx3 = [5, 1, 0, 3]
  got3 = af.compile(af.gather(af.input((2, 4, 6)), idx3, axis=2))(x3)
  assert np.array_equal(got3, x3[:, :, idx3]), "3D last-axis gather not exact"
  # non-last axis is unchanged (axis 0) and also exact
  got0 = af.compile(af.gather(af.input((8, 16)), [5, 0, 2], axis=0))(x2)
  assert np.array_equal(got0, x2[[5, 0, 2]]), "axis-0 gather not exact"


# --- regression: the width-slice hazard guard targets the confirmed patterns -------------
# Two silicon-pinned modes (a2a323a refined the wrong-element mode; M2 silicon corrected the
# saturation family bound): (1) WRONG ELEMENTS only when >=2 width-offset slices are
# CONCATENATED (single slices select correctly - 180-config M2 sweep is exact); (2) magnitude
# SATURATION (|v|>4094 -> inf) on a single width slice, confirmed on A13 AND A14 (M2 probe);
# A15 pending M3. A16/M5 unaffected.
def test_width_slice_guard_modes():
  import warnings
  from aneforge import _compile
  xi = af.input((6, 6))
  single = xi.slice_by_size([0, 3], [6, 1])                                  # one width slice
  concat = af.concat([xi.slice_by_size([0, j], [6, 1]) for j in [3, 1, 0]], axis=1)  # gather pattern

  def fired(g, fam, key):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      _compile._warn_h13_slice_saturation(g, fam)
      return any(key in str(x.message) for x in w)

  A13, A14, A16 = int(TG.Family.A13), int(TG.Family.A14), int(TG.Family.A16)
  assert fired(concat, A14, "WRONG ELEMENTS")        # concat-of-slices warns on A14
  assert fired(concat, A13, "WRONG ELEMENTS")        # ...and A13
  assert fired(single, A14, "saturates")             # single slice SATURATES on A14 (M2-confirmed)
  assert fired(single, A13, "saturates")             # ...and A13 (conv weight-grad)
  assert not fired(single, A16, "aneforge:")         # A16/M5 unaffected
  assert not fired(concat, A16, "aneforge:")
