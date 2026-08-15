"""_rope_theta reads the RoPE base from transformers-5.x's rope_parameters dict.

transformers 5.x moved rope_theta out of a top-level config attribute into the `rope_parameters`
dict, so the old `getattr(c, "rope_theta", 10000.0)` silently returned the 10000.0 fallback for every
model with a non-default base (Qwen3 uses 1e6, Llama-3 uses 5e5) -- wrong RoPE, silent wrong logits.
This asserts the base is read from either layout. Off-device: pure config parsing, no ANE.
"""
from types import SimpleNamespace

from aneforge.llm import _rope_theta


def test_reads_base_from_rope_parameters_dict():
  # transformers 5.x layout: base lives in rope_parameters, with no top-level rope_theta attribute.
  c = SimpleNamespace(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
  assert _rope_theta(c) == 1000000.0
  # guard the regression directly: the pre-fix code path returns the wrong value on this layout.
  assert float(getattr(c, "rope_theta", 10000.0)) == 10000.0


def test_reads_legacy_top_level_attribute():
  c = SimpleNamespace(rope_theta=500000.0)
  assert _rope_theta(c) == 500000.0


def test_falls_back_to_default_when_absent():
  assert _rope_theta(SimpleNamespace()) == 10000.0


def test_rope_parameters_without_theta_falls_back_to_attr():
  # a rope_parameters dict that lacks rope_theta must not shadow a legacy top-level attribute.
  c = SimpleNamespace(rope_parameters={"rope_type": "default"}, rope_theta=800000.0)
  assert _rope_theta(c) == 800000.0
