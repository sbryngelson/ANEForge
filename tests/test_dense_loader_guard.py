"""`from_pretrained` must reject an arch the dense fallback would silently mis-load.

The ADAPTERS registry matches by exact model_type and anything unmatched falls through to the dense
Llama path. A Gemma-2/3 checkpoint is Llama-shaped enough to load without error there, but the dense
graph drops its logit softcapping / GeGLU / (1+w)-norm / embed-scale -- measured cosine ~0 vs HF. The
guard turns that silent garbage into a clear error. Off-device: pure config inspection, no ANE.
"""
from types import SimpleNamespace

import pytest

from aneforge.llm import _assert_dense_compatible, ModelType


def test_softcapping_config_is_rejected():
  # real Gemma-2 sets attn_logit_softcapping=50.0, final_logit_softcapping=30.0
  c = SimpleNamespace(model_type=ModelType.GEMMA2, attn_logit_softcapping=50.0, final_logit_softcapping=30.0)
  with pytest.raises(ValueError, match="softcapping"):
    _assert_dense_compatible(c)


def test_incompatible_model_type_is_rejected_even_without_softcapping():
  # gemma3 dropped softcapping but is still not the dense arch; the name backstop catches it
  c = SimpleNamespace(model_type=ModelType.GEMMA3)
  with pytest.raises(ValueError):
    _assert_dense_compatible(c)


def test_plain_llama_config_passes():
  c = SimpleNamespace(model_type=ModelType.LLAMA, attn_logit_softcapping=None, final_logit_softcapping=None)
  _assert_dense_compatible(c)   # must not raise


def test_llama_shaped_finetune_with_unusual_model_type_passes():
  # a genuine Llama-shaped finetune must not be rejected just for a nonstandard model_type
  _assert_dense_compatible(SimpleNamespace(model_type="my_custom_llama"))   # must not raise
