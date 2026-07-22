# Contributing

Build, test, and "adding an operator" details are in
[`docs/development.md`](docs/development.md). This is the short version.

## Bug reports

Behavior on the ANE is per-chip and per-OS, so include:

- The chip (M1 through M5, or the A-series equivalent) and the macOS version.
- A minimal graph that reproduces it: ops, shapes, dtypes.
- Expected versus actual: numbers, traceback, or compile error.

Two things are expected rather than bugs: a graph that compiles for one family
but overflows a dimension cap on another (caps are per family, see
[`docs/capabilities.md`](docs/capabilities.md)), and small CPU/ANE divergence at
boundary fp16 values. A wrong result well inside the fp16 range is a real bug.

Report security issues privately, not in a public issue: see
[`SECURITY.md`](SECURITY.md).

## Changes

For a first contribution, take an issue labeled
[`good first issue`](https://github.com/sbryngelson/ANEForge/labels/good%20first%20issue):
each one names the file, the composition path, and the reference implementation
to test against, and any Apple Silicon Mac is hardware enough. CI cannot reach
an ANE, so note in the pull request that the on-device tests passed locally.

Beyond those, operator-coverage gaps are the place to start: anything in
[`docs/capabilities.md`](docs/capabilities.md) not yet covered, via the four-step
path in [`docs/development.md`](docs/development.md#adding-an-operator). Open an
issue first for larger or architectural changes.

## Setup

```sh
pip install -e ".[dev]"               # ruff + pytest
sh aneforge/_lib/build.sh             # build the dispatch dylib (needs the Mac)
git config core.hooksPath .githooks   # off-hardware pre-commit checks
```

The corpus is the gate and must pass before a change lands:

```sh
PYTHONPATH=. python3 tests/run_corpus.py
```

Most tests need a real ANE, so CI runs only the off-hardware checks. Run the
corpus and the pytest suite on your Mac before opening a pull request.

## Style

Python 3.10+, linted with `ruff`. Match the surrounding packed style; do not
reformat existing code.

## License

Contributions are licensed under the [MIT License](LICENSE).
