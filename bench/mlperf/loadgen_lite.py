"""A self-contained, LoadGen-shaped benchmark core for MLPerf-style measurement on the ANE.

It mirrors the MLCommons LoadGen SUT/QSL split and the SingleStream + Offline scenarios, so the numbers are
methodology-comparable and a workload written against this can be re-pointed at the real `mlperf_loadgen`
later with little change:
  - QSL (Query Sample Library): owns the dataset -- `count` samples and `get(index) -> features`.
  - SUT (System Under Test): a wrapped model -- `name` and `issue(qsl, indices) -> outputs`.
  - run_single_stream / run_offline: drive the SUT per the MLPerf scenario semantics, return a Result.

This is NOT an official LoadGen run: there is no MLCommons logging/audit trail and the default query counts
are short. It is the same measurement SHAPE, meant as the on-ramp to a real submission. See README.md.

Pure Python + numpy (no ANE, no external deps), so the scenario/stat logic is unit-tested off-device; the
clock is injectable for deterministic tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np


class QSL:
    """Query Sample Library: the dataset the harness draws samples from. `get(index)` returns the model-ready
    features for one sample (already preprocessed, e.g. a [1, 3, 224, 224] fp16 array)."""
    def __init__(self, count, get, name="qsl", cache=False):
        self.count = int(count)
        self._get = get
        self.name = name
        self._cache = {} if cache else None      # LoadGen load_query_samples preprocesses here (untimed)

    def get(self, index):
        index = int(index)
        if self._cache is None:
            return self._get(index)
        if index not in self._cache:
            self._cache[index] = self._get(index)
        return self._cache[index]

    def load(self, indices):
        """LoadGen load_query_samples: preprocess the requested samples into the cache (untimed), so a later
        issue() times inference only. No-op for an uncached QSL."""
        if self._cache is not None:
            for i in indices:
                self.get(i)

    def unload(self, indices):
        """LoadGen unload_query_samples: free those samples, bounding memory to the performance-sample window."""
        if self._cache is not None:
            for i in indices:
                self._cache.pop(int(i), None)


class SUT:
    """System Under Test base: wrap a model so a scenario can issue batches of sample indices. Subclasses
    implement `issue`. `name` identifies the system in the results."""
    name = "sut"

    def issue(self, qsl, indices):
        """Run inference for `indices` (into `qsl`); return a list of outputs, one per index."""
        raise NotImplementedError


@dataclass
class Result:
    """One scenario run. Latencies are per-query wall times in milliseconds (empty for Offline, which reports
    throughput rather than per-query latency)."""
    scenario: str
    sut: str
    queries: int
    samples: int
    latencies_ms: list
    wall_s: float
    official: bool = False
    extra: dict = field(default_factory=dict)

    def _pct(self, q):
        return float(np.percentile(self.latencies_ms, q)) if self.latencies_ms else float("nan")

    @property
    def p50_ms(self): return self._pct(50)
    @property
    def p90_ms(self): return self._pct(90)
    @property
    def p99_ms(self): return self._pct(99)
    @property
    def mean_ms(self): return float(np.mean(self.latencies_ms)) if self.latencies_ms else float("nan")
    @property
    def throughput_qps(self): return self.samples / self.wall_s if self.wall_s > 0 else float("nan")

    def to_dict(self):
        lat = ({"p50": self.p50_ms, "p90": self.p90_ms, "p99": self.p99_ms, "mean": self.mean_ms}
               if self.latencies_ms else None)          # Offline has no per-query latency -> null, not NaN
        return {"scenario": self.scenario, "sut": self.sut, "queries": self.queries, "samples": self.samples,
                "wall_s": self.wall_s, "official": self.official, "throughput_qps": self.throughput_qps,
                "latency_ms": lat, **({"extra": self.extra} if self.extra else {})}

    def summary(self):
        tag = "" if self.official else "  [methodology-only: short run, no MLCommons audit trail]"
        head = f"{self.scenario:12s} {self.sut}  ({self.queries} queries, {self.samples} samples){tag}"
        if self.scenario == "SingleStream":
            body = (f"  latency ms  p50={self.p50_ms:8.3f}  p90={self.p90_ms:8.3f}  p99={self.p99_ms:8.3f}"
                    f"  mean={self.mean_ms:8.3f}\n  metric (p90 latency) = {self.p90_ms:.3f} ms"
                    f"   [{self.throughput_qps:.1f} samples/s]")
        elif self.scenario == "LLMDecode":
            ttft = self.extra.get("ttft_ms", float("nan"))
            body = (f"  TTFT={ttft:8.1f} ms   TPOT p50={self.p50_ms:6.2f} p90={self.p90_ms:6.2f} ms/token"
                    f"   throughput = {self.throughput_qps:.1f} tok/s   ({self.samples} tokens)")
        else:
            body = f"  metric (throughput) = {self.throughput_qps:.1f} samples/s   [wall {self.wall_s:.2f} s]"
        return head + "\n" + body


# MLPerf run-rule minimums for an OFFICIAL SingleStream/Offline run. We default below these (a quick,
# methodology-only run) and flag a run as `official` only when both are met.
_OFFICIAL_MIN_QUERIES = 1024
_OFFICIAL_MIN_DURATION_S = 600.0


def _cycle(n, count):
    """Sample indices 0..n-1 cycled to length `count` (MLPerf samples with replacement over the QSL)."""
    return [i % n for i in range(count)]


def run_single_stream(sut, qsl, count=1024, warmup=16, min_duration_s=0.0, clock=time.perf_counter):
    """SingleStream: one sample at a time, record each latency; keep issuing until BOTH >= `count` queries and
    >= `min_duration_s` seconds are met (MLPerf metric: p90 latency). An official run needs 1024 queries and
    600 s; `min_duration_s` reaches the time floor regardless of per-query speed. Warmup queries are discarded."""
    for i in _cycle(qsl.count, warmup):
        sut.issue(qsl, [i])
    lat = []
    idx = 0
    t_start = clock()
    while True:
        t0 = clock()
        sut.issue(qsl, [idx % qsl.count])
        lat.append((clock() - t0) * 1e3)
        idx += 1
        if idx >= count and (clock() - t_start) >= min_duration_s:
            break
    wall = clock() - t_start
    official = idx >= _OFFICIAL_MIN_QUERIES and wall >= _OFFICIAL_MIN_DURATION_S
    return Result("SingleStream", sut.name, idx, idx, lat, wall, official)


def run_offline(sut, qsl, count=None, warmup=16, batch=1, min_duration_s=0.0, clock=time.perf_counter):
    """Offline: issue the query set and measure total wall time; MLPerf metric is throughput (samples/s). Keeps
    issuing until BOTH >= `count` samples and >= `min_duration_s` seconds. `batch` groups indices per `issue`
    (the ANE runs a fixed batch-1 program, so batch=1 -- a latency-bound engine's Offline rate then tracks its
    SingleStream rate, which is itself the honest finding)."""
    n = qsl.count if count is None else int(count)
    for i in _cycle(qsl.count, min(warmup, n)):
        sut.issue(qsl, [i])
    samples = queries = 0
    t0 = clock()
    while True:
        sut.issue(qsl, [(samples + j) % qsl.count for j in range(batch)])
        samples += batch
        queries += 1
        if samples >= n and (clock() - t0) >= min_duration_s:
            break
    wall = clock() - t0
    official = samples >= _OFFICIAL_MIN_QUERIES and wall >= _OFFICIAL_MIN_DURATION_S
    return Result("Offline", sut.name, queries, samples, [], wall, official)


def run_llm_decode(sut, prompt_ids, gen_len=64, warmup=1, clock=time.perf_counter):
    """LLM-decode scenario (the token-generation shape MLPerf uses for LLMs): greedily generate `gen_len`
    tokens and report TTFT (time to first token, prefill + first decode), TPOT (per-output-token latency, as
    the p50/p90 of `latencies_ms`), and output throughput (tokens/s). The SUT's `decode(prompt_ids, gen_len,
    clock) -> (ttft_s, per_token_s, n_tokens)` owns the timing (it has the per-token callback)."""
    for _ in range(warmup):
        sut.decode(list(prompt_ids), max(4, gen_len // 8), clock)     # discarded: triggers compile / cache reset
    ttft_s, per_token_s, ntok = sut.decode(list(prompt_ids), gen_len, clock)
    per_token_ms = [d * 1e3 for d in per_token_s]
    wall = ttft_s + sum(per_token_s)
    return Result("LLMDecode", sut.name, 1, ntok, per_token_ms, wall, official=False,
                  extra={"ttft_ms": ttft_s * 1e3, "prompt_len": len(prompt_ids), "gen_len": gen_len})
