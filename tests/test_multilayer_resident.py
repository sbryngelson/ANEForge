"""Multi-layer GPT decode with resident per-layer KV-cache matches numpy token-for-token."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import requires_ane   # skips when the ANE dylib is absent (not only when ANEFORGE_NO_ANE is set)


@requires_ane
@pytest.mark.parametrize("L", [1, 2, 3])
def test_multilayer_resident_matches_numpy(L):
  from examples.gpt_multilayer_resident import TinyGPTResident
  m = TinyGPTResident(L=L, seed=0)
  assert m.generate([3, 7, 1], 5) == m.ref_generate([3, 7, 1], 5)
