# benchmarks

Throughput / dispatch benchmarks, not how-to demos. Run from the repo root, e.g.

```sh
PYTHONPATH=. python3 examples/benchmarks/bench_encoder_batched.py
```

- `bench_encoder_batched.py` - batched encoder serving throughput on the ANE.
- `bench_encoder_gpu.py` - the torch-MPS (GPU) baseline for comparison.
- `rank_worker_bench.py`, `sdpa_worker_bench.py`, `topk_worker_bench.py` - persistent
  native-worker dispatch latency for the argmax/topk - sdpa - sort bridge ops.
