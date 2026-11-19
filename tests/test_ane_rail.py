"""Off-device tests for the SMC-rail ANE power fallback (bench/_ane_rail.py, issue #289):
the estimator math, how measure_energy() picks the source and builds the package, and the
aggregator's marking, plus the busy-idle guard (#304) in both. powermetrics and the rail
are faked; no SMC, IOReport or ANE needed."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import _ane_rail as r   # noqa: E402
import aggregate_rooflines as agg   # noqa: E402


def _s(pp0b, pacc=1.4, eacc=0.3):
  return {"pp0b": pp0b, "pacc": pacc, "eacc": eacc}


def test_estimator():
  idle = [_s(0.8)] * 20                          # rail 0.8 W, cluster 1.4 W -> offset -0.6 W
  # ANE adds 4 W; the calling thread adds 2 W to the shared cluster and so to the rail;
  # three outlier readings must not move the median
  loaded = [_s(6.8, 3.4)] * 17 + [_s(40.0, 3.4), _s(0.0, 3.4), _s(30.0, 3.4)]
  out = r.summarize(loaded, idle)
  assert math.isclose(out["ane_active_mW"], 4000.0) and math.isclose(out["cpu_active_mW"], 2000.0)
  # the SMC holds a reading until its next update: 12 samples, 3 distinct readings -> flagged
  held = [_s(v) for v in [5.0] * 4 + [5.2] * 4 + [4.9] * 4]
  assert r.rail_updates([s["pp0b"] for s in held]) == 3
  assert any("PP0b updates" in f for f in r.summarize(held, idle)["flags"])
  # a rail that does not feed the cluster (Mac16,9, #304) stays flat while the cluster loads:
  # no ANE power from it, where a 1.5 W dip under its idle offset is still a reading
  off = r.summarize([_s(1.4, 14.0)] * 20, idle)
  assert math.isnan(off["ane_active_mW"]) and any("does not carry" in f for f in off["flags"])
  dip = r.summarize([_s(0.8, 2.9)] * 20, idle)
  assert dip["ane_active_mW"] == 0.0 and not any("does not carry" in f for f in dip["flags"])


class _FakePM:
  """Stands in for `sudo powermetrics`: writes a canned log, then exits after a few polls."""
  text = ""

  def __init__(self, args, stdout=None, stderr=None):
    stdout.write(_FakePM.text)
    stdout.flush()
    self._polls = 3

  def poll(self):
    self._polls -= 1
    return None if self._polls > 0 else 0

  def wait(self):
    return 0


class _FakeRail:
  pp0b, pacc = 0.0, 0.0

  def start(self):
    pass

  def stop(self):
    return [{"pp0b": self.pp0b + 0.1 * (i // 4), "pacc": self.pacc, "eacc": 0.3} for i in range(40)]


def _pm(ane, cpu):
  ane_line = f"ANE Power: {ane} mW\n" if ane is not None else ""
  return (ane_line + f"CPU Power: {cpu} mW\nGPU Power: 0 mW\n"
          f"Combined Power (CPU + GPU + ANE): {(ane or 0) + cpu} mW\n") * 30


def _measure(monkeypatch, pm_idle, pm_load, force=None):
  # imported here, not at module level: device_compare imports mlx.core, and with MLX
  # loaded in the pytest parent the --forked children of later tests segfault on macOS
  import device_compare_wattcomplete as wc
  monkeypatch.setattr(wc.subprocess, "Popen", _FakePM)
  monkeypatch.setattr(wc.time, "sleep", lambda s: None)
  monkeypatch.setattr(wc, "HAVE_SUDO", True)
  monkeypatch.setattr(wc, "RAIL", _FakeRail())
  if force:
    monkeypatch.setenv("ANEFORGE_ANE_POWER", force)
  else:
    monkeypatch.delenv("ANEFORGE_ANE_POWER", raising=False)
  _FakePM.text, wc.RAIL.pp0b, wc.RAIL.pacc = pm_idle, 0.8, 1.4
  wc.sample_idle(3)
  # under load the ANE adds 4 W (+0.15 W of drift) and the calling thread 1 W to the shared cluster
  _FakePM.text, wc.RAIL.pp0b, wc.RAIL.pacc = pm_load, 5.8, 2.4
  return wc.measure_energy(lambda: None, tag="test", window=2.5)


def test_measure_energy_source_and_package(monkeypatch):
  # powermetrics has the ANE (M4): it stays the source; the rail is recorded beside it
  e = _measure(monkeypatch, _pm(0, 10), _pm(3000, 2510))
  assert e["ane_power_source"] == "powermetrics" and math.isclose(e["active_pkg_W"], 5.5)
  assert math.isclose(e["rail"]["active_pkg_W"], 5.5 - 3.0 + e["rail"]["ane_active_mW"] / 1e3)
  # forced: the rail's ANE term replaces powermetrics' instead of adding to it
  e = _measure(monkeypatch, _pm(0, 10), _pm(3000, 2510), force="smc_pp0b")
  assert e["ane_power_source"] == "smc_pp0b"
  assert math.isclose(e["active_pkg_W"], 2.5 + e["ane_active_mW"] / 1e3)
  # no ANE and no CPU power (M6): ANE from the rail, CPU from the IOReport clusters
  e = _measure(monkeypatch, _pm(None, 0), _pm(None, 0))
  assert e["ane_power_source"] == "smc_pp0b" and e["ane_active_mW"] > 3900
  assert math.isclose(e["cpu_active_mW"], 1000.0)
  assert math.isclose(e["active_pkg_W"], (e["ane_active_mW"] + e["cpu_active_mW"]) / 1e3)


def test_busy_idle_is_flagged(monkeypatch):
  import device_compare_wattcomplete as wc
  # quiet baseline: no flag, and the package sits at or above its rails
  assert not _measure(monkeypatch, _pm(0, 400), _pm(3000, 2900))["flags"]
  assert not wc.IDLE_BUSY
  # 8 W of CPU at "idle" that is gone under load: the package (0.5 W) lands below the ANE (3 W)
  e = _measure(monkeypatch, _pm(0, 8000), _pm(3000, 5500))
  assert wc.IDLE_BUSY and math.isclose(e["active_pkg_W"], 0.5)
  assert any("idle baseline 8.0 W" in f for f in e["flags"]) and any("below its rails" in f for f in e["flags"])


def _report(source, idle_pkg_mW=300.0):
  sat = {"peaks": {"gemm": {"ANE": {"peak_gflops": 10000.0, "peak_perf_per_W": 1234.0}}},
         "meta": {"ane_power_source": source, "idle_pkg_mW": idle_pkg_mW}}
  return {"machine": {"hardware": {"chip": "Apple M6", "model_identifier": "Mac18,5"},
                      "hardware_hash": "x" + str(source), "environment": {"power": {"source": "ac"}}},
          "perf_rooflines": [{"script": "device_saturation_sweep.py", "returncode": 0, "summary": sat}]}


def test_aggregator_marks_rail_perf_per_w():
  assert "1234 (rail)" in agg.render([_report("smc_pp0b")])
  assert "1234 (rail)" not in agg.render([_report("powermetrics")])
  assert agg.build_headline_json([_report("smc_pp0b")])[0]["peak_perf_per_w_source"] == "smc_pp0b"


def test_aggregator_withholds_busy_idle_perf_per_w():
  busy = _report("powermetrics", idle_pkg_mW=7698.5)
  md = agg.render([busy])
  assert "| busy idle |" in md and "1234" not in md and "| 10.0 |" in md   # throughput stays
  row = agg.build_headline_json([busy])[0]
  assert row["peak_perf_per_w_gflops"] is None and row["peak_perf_per_w_withheld"] == "busy_idle"
  assert "busy idle" not in agg.render([_report("powermetrics")])
