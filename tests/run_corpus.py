"""Run the full aneforge correctness corpus on the ANE.

This is the GATE for any future graph-optimizer pass: run it before and after a
rewrite; the pass-rate and per-case relerr must not regress. Each case builds a
graph, compiles it (``af.compile``), runs it on the M-series ANE, and asserts the
output against a numpy golden reference within a per-category tolerance.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 tests/run_corpus.py

Optimizer-diff reuse: import ALL_CASES and tests._corpus.run_case to run the same
builds with optimization on/off and diff the two ANE outputs directly.

Also pytest-compatible (the default suite skips it, so name the file):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. pytest tests/run_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/  -> _corpus
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # repo root -> aneforge

from _corpus import run_corpus, eval_case  # noqa: E402
import test_nn_blocks  # noqa: E402
import test_synthetic  # noqa: E402
import test_corners  # noqa: E402
import test_shapes  # noqa: E402
import test_broad  # noqa: E402

ALL_CASES = (test_nn_blocks.CASES + test_synthetic.CASES + test_corners.CASES
             + test_shapes.CASES + test_broad.CASES)


def main():
    n_int8 = sum(c.int8_ok for c in ALL_CASES)
    n_xfail = sum(bool(c.xfail) for c in ALL_CASES)
    cats = {}
    for c in ALL_CASES:
        cats[c.category] = cats.get(c.category, 0) + 1
    print(f"aneforge corpus: {len(ALL_CASES)} cases "
          f"({', '.join(f'{k}={v}' for k, v in sorted(cats.items()))}); "
          f"{n_int8} run fp16+int8; {n_xfail} xfail-marked\n")
    _, code = run_corpus(ALL_CASES)
    return code


# ---- pytest entry points (one test per case variant) --------------------- #
try:
    import pytest

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.name for c in ALL_CASES])
    def test_case(case):
        for rec in eval_case(case):
            assert rec["status"] in ("PASS", "XFAIL"), \
                f"{rec['name']}/{rec['variant']}: {rec['status']} ({rec['metric']}) {rec['err']}"
except ImportError:
    pass


if __name__ == "__main__":
    sys.exit(main())
