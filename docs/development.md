# Development

How to build, test, and extend the `aneforge` package.

## Building

The package is an editable install with a numpy-only core. The one native
artifact it needs at runtime is the e5rt dispatch shim, compiled with
`xcrun clang++` from the source that ships in the package.

```sh
pip install -e .                 # core: numpy only
python -m aneforge.build         # build aneforge/_lib/libane_e5rt_dispatch.dylib (optional)
```

The shim compiles `aneforge/_lib/ane_e5rt_dispatch.mm` against Apple's frameworks,
so it must build on the target Mac. It builds automatically the first time you
dispatch to the ANE (cached next to the source, or in `~/.cache/aneforge/<version>/`
if the package tree is read-only); `python -m aneforge.build` does it ahead of time,
and `sh aneforge/_lib/build.sh` still works for an in-tree build. The resulting
`.dylib` is a build artifact (not tracked); it rebuilds after a pull that touches
`ane_e5rt_dispatch.mm`. The dylib loads lazily, so `import aneforge` works without
it; only compiling or dispatching to the ANE needs it. Set `ANEFORGE_NO_AUTOBUILD=1`
to require an explicit build.

## Python environment

Use a persistent virtualenv in the repo. Python 3.12 has the widest wheel
coverage for the optional extras.

```sh
python3.12 -m venv .venv                     # .venv/ is gitignored
.venv/bin/python -m pip install -e ".[dev]"      # ruff + pytest (lint/test)
# optional, for the pretrained-model and diffusion demos (lazy imports):
.venv/bin/python -m pip install -e ".[models]"   # torch, torchvision, transformers
```

### Pre-commit hook (enable once per clone)

A committed hook at `.githooks/pre-commit` runs the off-hardware checks
(`ruff check` + `compileall aneforge`) before every commit. It is dependency-free.
Git does not auto-trust committed hooks, so enable it once after cloning:

```sh
git config core.hooksPath .githooks
```

The hook does not run the corpus or pytest (those need ANE hardware and gate
separately). Bypass a single commit with `git commit --no-verify`.

## Tests

`tests/run_corpus.py` is the standing correctness gate: a corpus of operator
graphs compiled and run on the ANE and compared against numpy references at
fp16. It must be green before a release.

```sh
PYTHONPATH=. .venv/bin/python tests/run_corpus.py   # the corpus gate
PYTHONPATH=. .venv/bin/python tests/op_smoketest.py # per-op smoke test
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q   # full suite
```

The pytest suite runs each test in its own forked subprocess. Every `compile`
allocates an ANE program, and a single long-lived process accumulates them
across the whole suite, which approaches the per-process program limit and can
cause sporadic failures late in a run; a fresh process per test
avoids that. `tests/conftest.py` sets the environment the forked path needs, so
a plain `pytest tests/` works.

## Adding an operator

Operator coverage grows by mapping a MIL spelling to a Python op. The path:

1. **Catalog it.** Add the op to `aneforge/_op_catalog.py` (the single source of
   truth for per-device availability), then regenerate the reference table with
   `python docs/gen_op_catalog.py > docs/op-catalog.md`.
2. **Lower it.** Add the fused-MIL emitter in `aneforge/_compile.py`, or, for an
   op that MIL rejects but the hardware supports, add a netplist bridge under
   `aneforge/_bridges/`.
3. **Expose it.** Add the method on the graph type in `aneforge/graph.py`.
4. **Test it.** Mirror an existing case: compile, run on the ANE, and compare
   against a numpy reference at fp16. Add it to the corpus.

## Code style

The package targets Python 3.10+ and is linted with `ruff` (the pre-commit hook
runs `ruff check`); match the surrounding style.

## Contributing

Useful contributions:

- Operator coverage gaps (anything in [capabilities.md](capabilities.md) marked
  as not yet covered).
- Bug reports for shape combinations that compile but produce wrong output
  (CPU/ANE divergence at boundary fp16 values is real).

Propose larger architectural changes in an issue first.

## License

MIT. The framework symbols `aneforge` calls are Apple's private API and not under
any license; they may change without notice, and nothing here constitutes an API
contract.
