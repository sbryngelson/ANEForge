"""Multi-layer GPT decode with every layer's KV-cache resident on the ANE matches numpy token-for-token."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
requires_ane = pytest.mark.skipif(os.environ.get("ANEFORGE_NO_ANE") == "1", reason="needs ANE")


@requires_ane
@pytest.mark.parametrize("L", [1, 2, 3])
def test_multilayer_resident_matches_numpy(L):
  from examples.gpt_multilayer_resident import TinyGPTResident
  m = TinyGPTResident(L=L, seed=0)
  assert m.generate([3, 7, 1], 5) == m.ref_generate([3, 7, 1], 5)
