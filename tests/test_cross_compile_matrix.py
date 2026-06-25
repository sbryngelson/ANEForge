"""Cross-compile the fused corpus for every ANE family; gate that _targets agrees with the compiler."""
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))      # tests/ -> import corpus
import run_corpus
from _corpus import _build_graph

from aneforge import _targets as T
from aneforge._compile import _lower_fused_to_dir, cross_compile_check

ARCHS = [("h13", 2), ("h14", 3), ("h15", 4), ("h16s", 5)]


def _model_compiles(graph, family: int) -> bool:
  """Model prediction: raw graph compiles iff every op native and nothing oversize."""
  rep = T.preflight(graph, family)
  return not rep.reject and not rep.decompose and not rep.oversize


def _fused_cases():
  for c in run_corpus.ALL_CASES:
    try:
      g = _build_graph(c)
      _lower_fused_to_dir(g, None)          # raises on bridge/segmented/unreachable
    except Exception:
      continue
    yield c.name, g


@pytest.mark.parametrize("name,graph", list(_fused_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_model_matches_compiler(name, graph):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    host_family = T.detect_family()
    for arch, fam in ARCHS:
      actual = cross_compile_check(graph, arch)
      predicted = _model_compiles(graph, fam)
      # host artifact: host can't EMIT ops above its own family, so skip rather than flag the model
      if predicted and not actual and fam > host_family: continue
      assert actual == predicted, (
        f"{name} @ {arch} (family {fam}): compiler {'compiles' if actual else 'rejects'} "
        f"but model predicts {'native' if predicted else 'non-native'}")
