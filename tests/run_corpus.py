"""Run the full aneforge correctness corpus on the ANE (the graph-optimizer GATE)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/  -> _corpus
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # repo root -> aneforge

import aneforge  # noqa: F401  (importing the package sets KMP_DUPLICATE_LIB_OK before numpy loads)

from _corpus import run_corpus, eval_case
import test_nn_blocks
import test_synthetic
import test_corners
import test_shapes
import test_broad

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
