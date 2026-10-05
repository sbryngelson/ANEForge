# Hardware results (ANE leaderboard)

Contributors run one benchmark suite on their own Macs and submit the results, so
the project keeps a growing cross-chip record of what the Neural Engine actually
does on each Apple Silicon machine. Each submission is a fingerprinted JSON file
(chip, model, cores, memory, macOS, power state, and the exact commit it ran on).

## Where to see it

- [**ANE leaderboard**](https://huggingface.co/spaces/aneforge/ane-leaderboard) on
  Hugging Face: an interactive, sortable table that reads the live data from `main`.
- [`bench/results/ROOFLINES.md`](https://github.com/sbryngelson/ANEForge/blob/main/bench/results/ROOFLINES.md):
  the full generated table on GitHub, with every machine, the numeric cliffs, and
  the headline performance numbers.
- [`aneforge/ane-rooflines`](https://huggingface.co/datasets/aneforge/ane-rooflines)
  dataset on Hugging Face, synced from `main` on every change, and the raw per-run
  files under [`bench/results/rooflines/`](https://github.com/sbryngelson/ANEForge/tree/main/bench/results/rooflines).

## What is measured

| Column | Meaning |
| --- | --- |
| Peak fp16 GEMM (TF/s) | the highest sustained fp16 matmul throughput on the engine |
| Bandwidth (GB/s) | the effective streaming bandwidth the engine sees |
| Ridge (FLOP/byte) | where the roofline turns from bandwidth-bound to compute-bound |
| Peak perf/W (GF/s/W) | throughput per watt of ANE power, from `powermetrics` |
| Decode @b1 (tok/s) | batch-1 decode of the canonical small LLM |
| Numeric cliffs | where fp16 on that silicon silently goes wrong; see [numeric cliffs](numeric-cliffs.md) |

Performance depends on power and thermal state, so every row records whether the
run was on AC or battery and in which energy mode.

## Add your machine

Chips we do not own are the most useful. From a clean checkout of `main`:

```bash
PYTHONPATH=. python3 bench/roofline_suite.py --perf   # a few minutes; sudo for watts
python3 bench/aggregate_rooflines.py                  # regenerate the table
```

Then open a PR with the JSON it wrote and the regenerated files. The full
instructions are in
[`bench/results/rooflines/README.md`](https://github.com/sbryngelson/ANEForge/blob/main/bench/results/rooflines/README.md),
and the pinned issue [#137](https://github.com/sbryngelson/ANEForge/issues/137)
lists the chips still missing.
