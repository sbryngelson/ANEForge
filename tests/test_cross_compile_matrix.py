"""Cross-compile the whole fused corpus for every ANE family, and gate that our static
target model (aneforge._targets) agrees with what the real compiler does.

This turns the cross-target lever into actual coverage: every fused corpus case (the op
surface) is compiled for h13/h14/h15/h16s on this one M5 box (h16s stands in for the
whole A16-ceiling capability tier — the h17*/h18 targets compile too but are
capability-identical, core-count scaling only), and the per-(case, family)
compile result is compared to the model's prediction (a raw graph compiles iff every op
is native and nothing is oversize). Any disagreement is a bug in _OP_FLOOR / _LIMITS /
preflight — this is how we keep the hand-curated model honest against the ground truth.

Compile-level only (cross-target programs can't execute on this host); numeric
correctness still needs the real silicon. Bridge/segmented cases are skipped (the
cross-target checker is fused-route only).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=.:tests python3 -m pytest tests/test_cross_compile_matrix.py -q
"""
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))      # tests/ -> import corpus
import run_corpus  # noqa: E402
from _corpus import _build_graph  # noqa: E402

from aneforge import _targets as T  # noqa: E402
from aneforge._compile import _lower_fused_to_dir, cross_compile_check  # noqa: E402

ARCHS = [("h13", 2), ("h14", 3), ("h15", 4), ("h16s", 5)]


def _model_compiles(graph, family: int) -> bool:
    """The model's prediction for whether the RAW graph compiles at this family: every op
    native (no decomposition needed) and nothing oversize."""
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
            # cross_compile_check uses the HOST compiler, which can only EMIT ops the host's
            # family supports. When the model says a higher-family TARGET runs the op natively
            # but the host is below that family, the host simply can't emit it — cross_compile
            # rejects for a host reason, not because the target lacks the op. That's a host
            # artifact (it would pass on an A15+/M5 host), so skip rather than flag the model.
            if predicted and not actual and fam > host_family:
                continue
            assert actual == predicted, (
                f"{name} @ {arch} (family {fam}): compiler {'compiles' if actual else 'rejects'} "
                f"but model predicts {'native' if predicted else 'non-native'}")
