# Roofline submissions

One JSON file per benchmark run, from contributors on different Apple Silicon
machines. The aggregate view is [`../ROOFLINES.md`](../ROOFLINES.md) (generated).

## Submit your machine (PR)

1. Run the suite from a clean checkout of `main`:

   ```bash
   PYTHONPATH=. python3 bench/roofline_suite.py            # numeric cliffs (fast, no sudo)
   # or, on AC power, the full battery incl. per-rail watts:
   PYTHONPATH=. python3 bench/roofline_suite.py --perf
   ```

   It writes `roofline-<chip>-<model>-<hwhash>-<runid>.json` into this directory.

2. Regenerate the table and open a PR with both files:

   ```bash
   python3 bench/aggregate_rooflines.py                    # updates ../ROOFLINES.md
   git add bench/results/rooflines/roofline-*.json bench/results/ROOFLINES.md
   ```

CI runs `aggregate_rooflines.py --check` and fails the PR if `ROOFLINES.md` was
not regenerated, so the table can never drift from the data.

## What each file records

Every submission carries a full machine fingerprint (see `bench/_machine.py`):

- **hardware** (hashed into `hardware_hash`): chip, model identifier, CPU P/E
  cores, GPU cores, unified memory, ANE rated TOPS. The hash groups identical
  machines; the model identifier keeps an M2 Air, M2 16-inch MBP, and M2 Mac mini
  distinct because they sustain different clocks.
- **environment** (not hashed): macOS version/build, aneforge version, the git
  commit **and its merge-base on canonical `main`** with ahead/behind + dirty flag
  (so drift over time is explicit), AC/battery power state, thermal state.

Filenames include a `run_id`, so multiple runs of the same machine - and runs by
different people on the same machine model - never collide. Submissions are
immutable: add new ones, do not edit old ones.

Please submit from a **clean, non-dirty** checkout where possible; a `dirty` or
`battery` submission is accepted (the flags ride along) but is not comparable to
clean/AC data for the performance rooflines.
