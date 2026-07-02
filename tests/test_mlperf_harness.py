"""The MLPerf-lite harness core (bench/mlperf/loadgen_lite.py): scenario semantics + stats, off-device."""
import importlib.util
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench", "mlperf"))
import loadgen_lite as lg   # noqa: E402

_HAS_LOADGEN = importlib.util.find_spec("mlperf_loadgen") is not None


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


class _FakeLLMSUT(lg.SUT):
  """A fake decode SUT: uses the injected clock so TTFT and every per-token gap are exactly `dt`."""
  name = "fake-llm"
  def decode(self, prompt_ids, gen_len, clock):
    t0 = clock()
    times = [clock() for _ in range(gen_len)]
    ttft = times[0] - t0
    per = [times[i] - times[i - 1] for i in range(1, len(times))]
    return ttft, per, gen_len


def test_llm_decode_metrics():
  r = lg.run_llm_decode(_FakeLLMSUT(), [1, 2, 3, 4], gen_len=20, warmup=1, clock=_Clock(0.005))
  assert r.scenario == "LLMDecode" and r.samples == 20
  assert len(r.latencies_ms) == 19              # per-token gaps = gen_len - 1
  assert abs(r.extra["ttft_ms"] - 5.0) < 1e-9   # dt = 5 ms
  assert abs(r.p90_ms - 5.0) < 1e-9             # every TPOT == dt
  assert r.extra["prompt_len"] == 4 and r.extra["gen_len"] == 20
  import json
  json.dumps(r.to_dict(), allow_nan=False)
  assert "TTFT" in r.summary() and "tok/s" in r.summary()


@pytest.mark.skipif(not _HAS_LOADGEN, reason="mlperf_loadgen not installed")
def test_lite_tracks_real_loadgen():
  # differential: our lite SingleStream p90 must track real LoadGen's p90 on the same fixed-latency SUT.
  import tempfile
  import loadgen_official as lg_off   # noqa: E402

  class _BusySUT(lg.SUT):
    name = "busy"
    def issue(self, qsl, indices):
      for _ in indices:                # busy-wait a fixed 0.3 ms/query (precise, unlike time.sleep at this scale)
        t = time.perf_counter()
        while time.perf_counter() - t < 3.0e-4:
          pass
      return [0] * len(indices)

  sut, qsl = _BusySUT(), _qsl(8)
  # short runs are not MLPerf-VALID (that needs the 600s floor); the point here is that the two drivers agree.
  summ = lg_off.run(sut, qsl, scenario="SingleStream", min_query_count=256, outdir=tempfile.mkdtemp())
  loadgen_p90_ms = summ["p90_latency_ns"] / 1e6
  r = lg.run_single_stream(sut, qsl, count=256)
  ratio = r.p90_ms / loadgen_p90_ms
  assert 0.7 < ratio < 1.4, f"lite p90 {r.p90_ms:.4f} vs loadgen {loadgen_p90_ms:.4f} ms (ratio {ratio:.3f})"


def test_result_to_dict_is_json_shaped():
  sut = _CountingSUT()
  r = lg.run_single_stream(sut, _qsl(8), count=8, warmup=0, clock=_Clock(0.001))
  d = r.to_dict()
  assert d["scenario"] == "SingleStream" and d["samples"] == 8
  assert set(d["latency_ms"]) == {"p50", "p90", "p99", "mean"}
  assert d["official"] is False
