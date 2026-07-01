"""The MLPerf-lite harness core (bench/mlperf/loadgen_lite.py): scenario semantics + stats, off-device."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench", "mlperf"))
import loadgen_lite as lg   # noqa: E402


class _CountingSUT(lg.SUT):
  """A fake SUT: records how many samples it was asked to run and returns argmax(features)."""
  name = "fake"
  def __init__(self):
    self.seen = 0
  def issue(self, qsl, indices):
    self.seen += len(indices)
    return [int(np.asarray(qsl.get(i)).argmax()) for i in indices]


class _Clock:
  """Deterministic clock: returns t, t+dt, t+2dt, ... so each measured query latency is exactly `dt`."""
  def __init__(self, dt=0.001):
    self.dt = dt; self.t = 0.0
  def __call__(self):
    v = self.t; self.t += self.dt; return v


def _qsl(n=8):
  data = np.arange(n * 4, dtype=np.float16).reshape(n, 4)   # sample i -> row; argmax is the last col
  return lg.QSL(n, lambda i: data[i])


def test_single_stream_counts_and_metric():
  sut = _CountingSUT()
  r = lg.run_single_stream(sut, _qsl(8), count=50, warmup=5, clock=_Clock(0.002))
  assert r.scenario == "SingleStream"
  assert r.queries == 50 and r.samples == 50
  assert len(r.latencies_ms) == 50
  assert sut.seen == 55                       # warmup(5) + counted(50)
  assert abs(r.p90_ms - 2.0) < 1e-9 and abs(r.mean_ms - 2.0) < 1e-9   # every latency == dt = 2 ms
  assert r.official is False                   # short run


def test_offline_reports_throughput_no_latency():
  sut = _CountingSUT()
  r = lg.run_offline(sut, _qsl(8), count=40, warmup=4, clock=_Clock(0.001))
  assert r.scenario == "Offline"
  assert r.samples == 40 and r.queries == 40   # batch=1 -> one query per sample
  assert r.latencies_ms == []
  assert sut.seen == 44                         # warmup(4) + 40
  assert r.wall_s > 0 and r.throughput_qps > 0
  import json
  d = r.to_dict()
  assert d["latency_ms"] is None                # no per-query latency for Offline
  json.dumps(d, allow_nan=False)                # must be valid JSON (no NaN)


def test_offline_batches_group_indices():
  sut = _CountingSUT()
  r = lg.run_offline(sut, _qsl(8), count=40, warmup=0, batch=8, clock=_Clock(0.001))
  assert r.samples == 40 and r.queries == 5     # 40 / 8
  assert sut.seen == 40


def test_qsl_cycles_when_count_exceeds_dataset():
  sut = _CountingSUT()
  r = lg.run_single_stream(sut, _qsl(4), count=10, warmup=0, clock=_Clock(0.001))
  assert r.samples == 10 and sut.seen == 10     # 4-sample QSL sampled with replacement to 10 queries


def test_result_to_dict_is_json_shaped():
  sut = _CountingSUT()
  r = lg.run_single_stream(sut, _qsl(8), count=8, warmup=0, clock=_Clock(0.001))
  d = r.to_dict()
  assert d["scenario"] == "SingleStream" and d["samples"] == 8
  assert set(d["latency_ms"]) == {"p50", "p90", "p99", "mean"}
  assert d["official"] is False
