"""Real MLCommons LoadGen driver, behind the SAME QSL/SUT the lite harness uses. This is the bridge from
methodology-only numbers to numbers produced by the official generator/logger: it runs the actual LoadGen
scenario state machines and writes the canonical `mlperf_log_summary.txt` / `mlperf_log_detail.txt`.

Import-gated: `available()` is False when `mlperf_loadgen` is not installed, so callers skip cleanly. Install
with `pip install mlcommons-loadgen` (a C++ extension; a compiler is needed if there is no wheel).

A workload written for `loadgen_lite` runs here unchanged -- only the driver differs:
  - the QSL's `get(index)` is reused (load/unload are no-ops: features materialize on demand);
  - the SUT's `issue(qsl, indices)` is called once per LoadGen query.
Performance mode reports empty responses (timing only); accuracy mode is a later addition.
"""
from __future__ import annotations
import os


def available() -> bool:
    try:
        import mlperf_loadgen  # noqa: F401
        return True
    except Exception:
        return False


def run(sut, qsl, scenario="SingleStream", mode="PerformanceOnly", outdir=None,
        min_query_count=1024, min_duration_ms=0, expected_latency_ns=0, perf_sample_count=None):
    """Run our (sut, qsl) under real LoadGen and return the fields parsed from mlperf_log_summary.txt
    (valid, p90 latency ns, samples/s, ...). `scenario` in {SingleStream, Offline, ...}; `mode` in
    {PerformanceOnly, AccuracyOnly, ...}."""
    import mlperf_loadgen as lg

    settings = lg.TestSettings()
    settings.scenario = getattr(lg.TestScenario, scenario)
    settings.mode = getattr(lg.TestMode, mode)
    settings.min_query_count = int(min_query_count)
    if min_duration_ms:
        settings.min_duration_ms = int(min_duration_ms)
    if expected_latency_ns:
        settings.single_stream_expected_latency_ns = int(expected_latency_ns)   # scheduler hint; loadgen refines it

    load = lambda samples: None          # our QSL.get materializes + caches on demand -> load/unload are no-ops
    unload = lambda samples: None
    perf_count = int(perf_sample_count or min(qsl.count, 1024))
    q = lg.ConstructQSL(qsl.count, perf_count, load, unload)

    def issue(query_samples):
        done = []
        for s in query_samples:
            sut.issue(qsl, [s.index])
            done.append(lg.QuerySampleResponse(s.id, 0, 0))     # PerformanceOnly: empty response (timing only)
        lg.QuerySamplesComplete(done)

    s = lg.ConstructSUT(issue, lambda: None)

    outdir = outdir or "."
    os.makedirs(outdir, exist_ok=True)
    log_out = lg.LogOutputSettings()
    log_out.outdir = outdir
    log_out.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.log_output = log_out

    lg.StartTestWithLogSettings(s, q, settings, log_settings)
    lg.DestroySUT(s)
    lg.DestroyQSL(q)
    return parse_summary(os.path.join(outdir, "mlperf_log_summary.txt"))


def _num(v):
    try:
        return float(v.split()[0].replace(",", ""))
    except Exception:
        return None


def parse_summary(path):
    """Pull the headline fields out of an mlperf_log_summary.txt into a dict (valid, metric, latencies)."""
    out = {"summary_path": path}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for raw in f:
            if ":" not in raw:
                continue
            k, v = raw.split(":", 1)
            low, v = k.strip().lower(), v.strip()
            if low.startswith("result is"):
                out["valid"] = v
            elif low.startswith("90.0th percentile latency"):     # SingleStream headline metric
                out["p90_latency_ns"] = _num(v)
            elif low.startswith("mean latency"):
                out["mean_latency_ns"] = _num(v)
            elif low.startswith("samples per second"):            # Offline headline metric
                out["samples_per_second"] = _num(v)
            elif low == "scenario":
                out["scenario"] = v
            elif low == "mode":
                out["mode"] = v
            elif low.startswith("min duration satisfied"):
                out["min_duration_satisfied"] = v
            elif low.startswith("min queries satisfied"):
                out["min_queries_satisfied"] = v
            elif low.startswith("early stopping satisfied"):
                out["early_stopping_satisfied"] = v
    return out
