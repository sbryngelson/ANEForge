"""The native ANE op catalog (aneforge/_op_catalog.py)."""
from aneforge import _op_catalog as oc


def test_catalog_complete():
  assert len(oc.OP_CATALOG) >= 187      # grows as ops are added; the per-entry field check below is the real gate
  assert len(oc.categories()) == 14
  for n, d in oc.OP_CATALOG.items():
    for k in ("m1", "m2", "m3", "m4_m5"):
      assert d[k] in ("native", "bridge", "walled"), (n, k, d[k])


def test_query_api():
  assert oc.device_status("add", "m1") == "native"
  assert oc.is_native("conv", "m1")
  assert oc.device_status("sin", "m1") == "bridge"      # F4 trig -> host Horner on M1
  assert oc.device_status("sin", "m3") == "native"      # native A15+
  assert oc.min_native_family("crop_resize") == 3       # texture engine, A14/M2+
  assert oc.min_native_family("add") == 2                # all chips
  assert "mod" in oc.walled_everywhere()
  assert oc.op_info("nonexistent_op") is None


def test_monotone_capability():
  # capability is a strict ladder: if native on family k, native on all > k.
  order = ["m1", "m2", "m3", "m4_m5"]
  for n, d in oc.OP_CATALOG.items():
    seen_native = False
    for k in order:
      if d[k] == "native":
        seen_native = True
      elif seen_native:
        assert d[k] == "native", f"{n}: {k} regressed below an earlier native"


def test_floor_consistency():
  # ops with a known A14/A15 floor must not be native on M1
  for op in ("crop_resize", "resample", "affine"):
    if op in oc.OP_CATALOG:
      assert oc.device_status(op, "m1") != "native", op
