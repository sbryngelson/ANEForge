# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, and friends) working in this
repository. Humans: see [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/development.md`](docs/development.md) for the same material in prose.

## What this is

ANEForge is a CoreML-free Python frontend that compiles an op graph to one fused
e5rt program and dispatches it to the Apple Neural Engine (ANE). The core is
numpy-only; the sole native artifact is an Objective-C++ dispatch shim built on
demand. Activations are fp16.

## The one caveat that shapes everything: hardware

CI cannot reach an ANE. GitHub's hosted macOS runners are virtualized and the ANE
is not passed through, so **no automated check verifies on-device numerics**. This
splits every task into two kinds:

- **Off-device (CI can fully verify):** docs, pure-numpy helpers (e.g. window and
  waveform generators), build-level op registration, anything checkable against a
  numpy/scipy oracle without dispatching. An agent in a Linux/CI-only environment
  can complete these end to end.
- **On-device (needs an Apple Silicon Mac):** anything whose correctness depends on
  running the compiled program. You must run it on a real Mac and **state in the PR
  that the on-device tests passed locally**, with the chip and macOS version. Do not
  claim a numeric result you have not run.

If you cannot reach a Mac, prefer off-device issues (many `good first issue`s are
fully off-device) and say so in the PR rather than guessing at numbers.

## Setup

```sh
pip install -e ".[dev]"                 # ruff + pytest + pyright (lint/type/test)
pip install -e ".[dev,models]" scipy    # what the type-check job installs (optional deps)
python -m aneforge.build                # build the dispatch dylib (Mac only; lazy otherwise)
git config core.hooksPath .githooks     # off-hardware pre-commit checks (ruff + compileall)
```

`import aneforge` works without the dylib; only compiling/dispatching to the ANE needs it.

## Checks CI runs (run these before opening a PR)

Match these locally and your PR will be green on everything CI can see:

```sh
ruff check                                              # lint
python -m pylint --disable=all -e W0311 -e C0303 \
  --indent-string='  ' --recursive=y aneforge tests     # the 2-space-indent gate
pyright aneforge                                        # types: gate is 0 errors
python -m compileall -q aneforge                        # every module byte-compiles
pytest -m "not requires_ane" -q                         # off-device unit suite
mkdocs build --strict                                   # docs (if you touched docs/)
python bench/aggregate_rooflines.py --check             # only if you touched bench/results/
```

On-device (a Mac with an ANE), before a PR that changes numerics:

```sh
PYTHONPATH=. python3 tests/run_corpus.py   # the standing correctness gate; must be green
PYTHONPATH=. python3 -m pytest tests/ -q   # full suite (each test forks its own process)
```

## House style (do not reformat existing code)

- Python 3.10+. Linted with `ruff`.
- **Two-space indentation**, tinygrad-compact packed style (short bodies on one line
  where the surrounding code does). The pylint gate above enforces the indent and
  no-trailing-whitespace; nothing else is auto-enforced, so match the neighbors.
- **ASCII only.** The source is plain ASCII; do not introduce Unicode punctuation,
  arrows, or box-drawing (use `->`, `<=`, `x`, `-` etc.).
- Keep changes surgical. Do not reflow, rename, or "tidy" code you are not changing.

## Adding an operator (the four-step path)

Full detail in [`docs/development.md`](docs/development.md#adding-an-operator).

1. **Catalog it** in `aneforge/_op_catalog.py` (source of truth for per-device
   availability), then regenerate `python docs/gen_op_catalog.py > docs/op-catalog.md`.
2. **Lower it** with a fused-MIL emitter in `aneforge/_compile.py`, or a netplist
   bridge under `aneforge/_bridges/` if MIL rejects an op the hardware supports.
3. **Expose it** as a method on the graph type in `aneforge/graph.py`.
4. **Test it**: compile, run on the ANE, compare against a numpy reference at fp16,
   and add it to the corpus (`tests/run_corpus.py`).

## Adding a library function (most `good first issue`s)

The `linalg` / `dsp` / `special` / `fft` / `onnx` modules compose existing graph ops
rather than adding new lowerings. For these:

- Implement in the module (`aneforge/linalg.py`, `dsp.py`, `special.py`, `fft.py`,
  or an `@onnx_op("Name")` handler in `onnx.py`), composing methods that already exist.
- The **oracle** is named in the issue: `numpy`, `scipy.special` / `scipy.signal` /
  `scipy.linalg`, or `onnxruntime`. scipy is optional, so guard scipy-based tests with
  `pytest.importorskip`.
- Add a case to the module's `_selftest` (or `main`) battery and to `tests/test_<mod>.py`.
- Prefer at least one reference-free check (an identity, a residual `A x ~ b`, a closed
  form) alongside the oracle comparison, so a subtly wrong result cannot pass on an
  averaged norm.

## Picking work

Start from a [`good first issue`](https://github.com/sbryngelson/ANEForge/labels/good%20first%20issue):
each names the file, the composition path, and the reference to test against. Open an
issue first for larger or architectural changes. Report security issues privately
([`SECURITY.md`](SECURITY.md)), never in a public issue.

## PR checklist

- [ ] `ruff check`, the pylint 2-space gate, and `pyright aneforge` are clean.
- [ ] `pytest -m "not requires_ane" -q` passes; docs build if you touched `docs/`.
- [ ] Numeric changes: corpus + `pytest tests/` run **on a real ANE**, result and
      chip/macOS stated in the PR body.
- [ ] Changes are surgical, two-space, ASCII, and match the surrounding style.
- [ ] The PR says which checks you ran and where (CI-only vs on-device).
