"""Autotuner accuracy-reference guards and cost-curve load errors, measurement seams monkeypatched."""
import warnings

import numpy as np
import pytest

import aneforge as af
from aneforge import _cost, _optimize


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
  # keep autotune cache reads/writes out of the repo (and out of other runs)
  monkeypatch.setenv("ANEFORGE_CACHE_DIR", str(tmp_path))


def _matmul_graph():
  rng = np.random.default_rng(7)
  x = af.input((8, 16))
  return x @ rng.standard_normal((16, 8)).astype(np.float16)


def _tanh_graph():
  # tanh has no _fp32_reference evaluator -> ref is None (the fallback case)
  rng = np.random.default_rng(7)
  x = af.input((1, 512))
  return (x @ rng.standard_normal((512, 8)).astype(np.float16)).tanh()


def _narrow_sum_tanh_graph():
  # adds a flagged narrow reduce_sum so a LOSSLESS rewrite variant is enumerated
  rng = np.random.default_rng(7)
  x = af.input((1, 512))
  y = x @ rng.standard_normal((512, 512)).astype(np.float16)
  return y.tanh().sum(1)


# 1. tune(): lossy variants need a lossless baseline
def test_tune_skips_lossy_when_no_lossless_baseline(monkeypatch):
  out = _matmul_graph()
  measured = []

  def fake_measure(out_, inputs, cfg, baseline_out=None, reps=20, warmup=5,
                   tol=_optimize._ACCURACY_TOL):
    measured.append(dict(cfg))
    if cfg.get("lossy"): return 10.0, np.zeros((8, 8), np.float32)
    return float("inf"), None          # every lossless variant fails to compile

  built = {}
  monkeypatch.setattr(_optimize, "measure", fake_measure)
  monkeypatch.setattr(_optimize, "build_variant",
                      lambda o, cfg: built.setdefault("cfg", cfg))

  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    _optimize.tune(out, inputs=[np.zeros((8, 16), np.float16)])

  assert not any(c.get("lossy") for c in measured)   # lossy never measured
  assert any("baseline" in str(w.message) for w in rec)
  assert not built["cfg"].get("lossy")               # safe lossless fallback chosen
  assert not built["cfg"].get("int8")


def test_tune_unchanged_when_lossless_baseline_exists(monkeypatch):
  out = _matmul_graph()
  measured = []

  def fake_measure(out_, inputs, cfg, baseline_out=None, reps=20, warmup=5,
                   tol=_optimize._ACCURACY_TOL):
    measured.append(dict(cfg))
    arr = np.zeros((8, 8), np.float32)
    return (50.0, arr) if cfg.get("lossy") else (100.0, arr)

  built = {}
  monkeypatch.setattr(_optimize, "measure", fake_measure)
  monkeypatch.setattr(_optimize, "build_variant",
                      lambda o, cfg: built.setdefault("cfg", cfg))

  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    _optimize.tune(out, inputs=[np.zeros((8, 16), np.float16)])

  assert any(c.get("lossy") for c in measured)       # lossy still measured
  assert not [w for w in rec if "baseline" in str(w.message)]
  assert built["cfg"].get("int8") is True            # 50us * 1.10 <= 100us -> int8 wins


# 2. tune_precision(): reference fallback and correctness
def test_tune_precision_falls_back_to_fp16_baseline_reference(monkeypatch):
  out = _tanh_graph()
  base = np.full((1, 8), 2.0, np.float32)
  refs_seen = []

  def fake_mwr(out_, inputs, cfg, ref, reps, warmup=5):
    refs_seen.append((cfg["label_kind"], None if ref is None else np.array(ref)))
    if cfg["label_kind"] == "fp16-baseline":
      return 100.0, float("nan"), base
    cur = base.astype(np.float64) + 0.5            # diverges 25% from the baseline
    relerr = float(np.abs(cur - ref).max() / (np.abs(ref).max() + 1e-9))
    return 80.0, relerr, cur.astype(np.float32)

  monkeypatch.setattr(_optimize, "_measure_with_ref", fake_mwr)
  monkeypatch.setattr(_optimize, "build_variant", lambda o, cfg: cfg)

  _, report = _optimize.tune_precision(out, target_error=0.01)

  assert report["ref_kind"] == "fp16-baseline"
  assert report["ref_available"] is True
  base_row = next(r for r in report["rows"] if r["label"] == "fp16-baseline")
  assert base_row["relerr"] == 0.0                   # the reference itself, not NaN
  other = next(r for r in report["rows"] if r["label"] != "fp16-baseline")
  assert other["relerr"] == pytest.approx(0.25)
  # 0.25 > target_error -> only the baseline meets the budget -> it is chosen,
  # and the reason names the reference that was actually used.
  assert report["chosen"]["label"] == "fp16-baseline"
  assert "fp16-baseline" in report["reason"]
  # the non-baseline variant was measured AGAINST the baseline's output
  assert refs_seen[0] == ("fp16-baseline", None)
  _, ref1 = refs_seen[1]
  assert ref1 is not None and np.allclose(ref1, 2.0)


def test_tune_precision_budget_enforced_vs_fp16_baseline(monkeypatch):
  out = _tanh_graph()
  base = np.full((1, 8), 2.0, np.float32)

  def fake_mwr(out_, inputs, cfg, ref, reps, warmup=5):
    if cfg["label_kind"] == "fp16-baseline":
      return 100.0, float("nan"), base
    cur = base.astype(np.float64) + 0.5
    relerr = float(np.abs(cur - ref).max() / (np.abs(ref).max() + 1e-9))
    return 80.0, relerr, cur.astype(np.float32)

  monkeypatch.setattr(_optimize, "_measure_with_ref", fake_mwr)
  monkeypatch.setattr(_optimize, "build_variant", lambda o, cfg: cfg)

  # a loose budget admits the 25%-divergent variant: the budget IS enforced as
  # divergence from the fp16 baseline, and the reason says so.
  _, report = _optimize.tune_precision(out, target_error=0.5)
  assert report["chosen"] is not None
  assert "min-cost meeting" in report["reason"]
  assert "fp16-baseline" in report["reason"]


def test_tune_precision_warns_when_no_reference_at_all(monkeypatch):
  out = _narrow_sum_tanh_graph()

  def fake_mwr(out_, inputs, cfg, ref, reps, warmup=5):
    if cfg["label_kind"] == "fp16-baseline":
      return float("inf"), 1.0, None             # the baseline fails to compile
    return 80.0, float("nan"), np.zeros((1, 1), np.float32)

  monkeypatch.setattr(_optimize, "_measure_with_ref", fake_mwr)
  monkeypatch.setattr(_optimize, "build_variant", lambda o, cfg: cfg)

  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    _, report = _optimize.tune_precision(out, target_error=0.01)

  assert report["ref_kind"] is None
  assert report["ref_available"] is False
  assert report["chosen"]["config"].get("lossy") is False  # never lossy w/o a reference
  assert "NOT enforced" in report["reason"]
  assert any("reference" in str(w.message) for w in rec)   # not a silent selection


def test_tune_precision_fp32_reference_path_unchanged(monkeypatch):
  out = _matmul_graph()                              # fully emulatable -> fp32 ref

  def fake_mwr(out_, inputs, cfg, ref, reps, warmup=5):
    assert ref is not None                         # the fp32 emulation exists here
    return 100.0, 1e-4, np.zeros((8, 8), np.float32)

  monkeypatch.setattr(_optimize, "_measure_with_ref", fake_mwr)
  monkeypatch.setattr(_optimize, "build_variant", lambda o, cfg: cfg)

  _, report = _optimize.tune_precision(out, target_error=1e-3)
  assert report["ref_kind"] == "fp32"
  assert report["rows"][0]["relerr"] == pytest.approx(1e-4)  # NOT forced to 0.0
  assert "fp32" in report["reason"]


# 3. _cost: broken-install error for the bundled cost curves
@pytest.fixture
def _fresh_curves():
  _cost._curves.cache_clear()
  yield
  _cost._curves.cache_clear()


def test_missing_curves_file_raises_install_error(monkeypatch, tmp_path, _fresh_curves):
  missing = tmp_path / "costmodel_curves.json"
  monkeypatch.setattr(_cost, "_curves_path", lambda: missing)
  with pytest.raises(RuntimeError, match="costmodel_curves.json"):
    _cost._curve_for_arch("h13")                   # NOT a KeyError blaming 'h13'
  with pytest.raises(RuntimeError, match="installation is broken"):
    _cost._curves()


def test_unparseable_curves_file_raises_install_error(monkeypatch, tmp_path, _fresh_curves):
  bad = tmp_path / "costmodel_curves.json"
  bad.write_text("{not json")
  monkeypatch.setattr(_cost, "_curves_path", lambda: bad)
  with pytest.raises(RuntimeError, match="installation is broken"):
    _cost._curve_for_arch("h17s")


def test_bundled_curves_still_load(_fresh_curves):
  c = _cost._curve_for_arch("h13")
  assert int(c["cores_0x238"]) >= 1
