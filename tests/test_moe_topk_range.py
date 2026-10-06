"""Off-device: the MoE top-k carries expert indices as fp16 integers (exact through 2048), so it rejects counts
outside 1 <= k <= E <= 2049 instead of silently selecting more than k experts. Builds graphs only; no dispatch."""
import pytest
import aneforge as af
from aneforge.moe import _topk_gate


@pytest.mark.parametrize("k,E", [(0, 8), (9, 8), (2, 2050)])
def test_topk_gate_rejects_unsupported_counts(k, E):
  with pytest.raises(ValueError):
    _topk_gate(af.input((1, E)), k, E, 1)


def test_topk_gate_builds_at_largest_supported_count():
  assert _topk_gate(af.input((1, 2049)), 2, 2049, 1).shape == (1, 2049)
