"""Deep small-dim models: the decode program must chunk by layer count, not just weight bytes.

A model like SmolLM2-360M (32 small layers) stays under the per-program weight ceiling but hits an
op-count ceiling (~31 layers) and the ANE fails the execute with rc=-1. `_layer_chunks` caps layers
per chunk so these segment and decode correctly."""
from aneforge.llm import LlamaConfig
from _helpers import requires_ane, make_random_llama_model as _random_model


def _deep_cfg(n_layers):
  return LlamaConfig(dim=128, n_heads=4, n_kv_heads=2, ffn_dim=256, vocab=64, n_layers=n_layers, head_dim=32)


def test_layer_chunks_caps_layer_count():
  """A deep, small-dim model that fits the byte budget must still split on the layer-count cap."""
  m = _random_model(_deep_cfg(32))                 # 32 small layers: well under _chunk_bytes
  chunks = m._layer_chunks()
  assert max(len(c) for c in chunks) <= m._chunk_max_layers
  assert sum(len(c) for c in chunks) == 32         # every layer is covered exactly once
  assert [i for c in chunks for i in c] == list(range(32))


def test_layer_chunks_single_when_shallow():
  """A shallow model stays a single chunk (no needless segmentation / dispatch overhead)."""
  m = _random_model(_deep_cfg(8))
  assert m._layer_chunks() == [range(0, 8)]


@requires_ane
def test_deep_model_decodes_without_rc_error():
  """32 layers used to fail decode with `execute failed rc=-1`; with the layer cap it generates."""
  out = _random_model(_deep_cfg(32)).generate([1, 2, 3], max_new_tokens=3)
  assert len(out) == 3


@requires_ane
def test_segmented_decode_matches_single_program():
  """A model split into >1 decode chunk must produce the same greedy tokens as one that fits in a chunk."""
  cfg = _deep_cfg(28)
  m = _random_model(cfg)
  assert len(m._layer_chunks()) > 1                # 28 > cap -> segmented
  ref = _random_model(cfg)
  ref._chunk_max_layers = 999                      # force a single chunk on an identical model
  assert len(ref._layer_chunks()) == 1
  prompt = [1, 2, 3, 4]
  assert list(m.generate(prompt, max_new_tokens=5)) == list(ref.generate(prompt, max_new_tokens=5))
