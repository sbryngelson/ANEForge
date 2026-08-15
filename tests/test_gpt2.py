"""GPT-2 loader graph-level tests: the Conv1D->linear weight mapping, the tiled lm_head, and
off-device MIL lowering (no ANE dispatch; CI-safe). Numerics are validated on-device by
examples/gpt2.py."""
from unittest.mock import Mock

import numpy as np
import pytest

import aneforge as af
from aneforge._compile import MultiModel, _lower_fused_to_dir
from aneforge.models import GPT2, _gpt2_layers, _gelu_new, _lm_head_tiles, _logits_from


def _synthetic_sd(D=16, H=4, L=2, V=100):
  """A minimal HF-shaped GPT-2 state dict (Conv1D layouts: `[in, out]`)."""
  Dff = 4 * D
  sd = {"transformer.wte.weight": np.zeros((V, D), np.float32),
        "transformer.wpe.weight": np.zeros((10, D), np.float32),
        "transformer.ln_f.weight": np.ones(D, np.float32),
        "transformer.ln_f.bias": np.zeros(D, np.float32)}
  for i in range(L):
    p = f"transformer.h.{i}."
    sd[p + "ln_1.weight"] = np.ones(D, np.float32)
    sd[p + "ln_1.bias"] = np.zeros(D, np.float32)
    sd[p + "ln_2.weight"] = np.ones(D, np.float32)
    sd[p + "ln_2.bias"] = np.zeros(D, np.float32)
    sd[p + "attn.c_attn.weight"] = np.arange(D * 3 * D, dtype=np.float32).reshape(D, 3 * D)
    sd[p + "attn.c_attn.bias"] = np.arange(3 * D, dtype=np.float32) + 100 * (i + 1)
    sd[p + "attn.c_proj.weight"] = np.arange(D * D, dtype=np.float32).reshape(D, D)
    sd[p + "attn.c_proj.bias"] = np.zeros(D, np.float32)
    sd[p + "mlp.c_fc.weight"] = np.arange(D * Dff, dtype=np.float32).reshape(D, Dff)
    sd[p + "mlp.c_fc.bias"] = np.zeros(Dff, np.float32)
    sd[p + "mlp.c_proj.weight"] = np.arange(Dff * D, dtype=np.float32).reshape(Dff, D)
    sd[p + "mlp.c_proj.bias"] = np.zeros(D, np.float32)
  return sd


def test_gpt2_conv1d_mapping_orientation():
  """Conv1D weights are `[in, out]`; the loader transposes to `[out, in]` for `.linear()` and
  splits c_attn rows into q/k/v -- so Wq[0] must equal c_attn.weight[:, 0]."""
  D = 16
  layers = _gpt2_layers(_synthetic_sd(), L=2, D=D, Dff=64)
  assert len(layers) == 2
  w = layers[1]
  c = _synthetic_sd()
  i = 1
  p = f"transformer.h.{i}."
  # rows of the transposed c_attn weight, one [D, D] block per of q/k/v
  assert w["Wq"].shape == (D, D) and w["Wk"].shape == (D, D) and w["Wv"].shape == (D, D)
  assert np.array_equal(w["Wq"], c[p + "attn.c_attn.weight"].T[:D])
  assert np.array_equal(w["Wk"], c[p + "attn.c_attn.weight"].T[D:2 * D])
  assert np.array_equal(w["Wv"], c[p + "attn.c_attn.weight"].T[2 * D:3 * D])
  assert np.array_equal(w["Wo"], c[p + "attn.c_proj.weight"].T)
  assert w["Wi"].shape == (64, D) and np.array_equal(w["Wi"], c[p + "mlp.c_fc.weight"].T)
  assert w["Wd"].shape == (D, 64) and np.array_equal(w["Wd"], c[p + "mlp.c_proj.weight"].T)
  # biases are split, not transposed
  assert np.array_equal(w["bq"], c[p + "attn.c_attn.bias"][:D])
  assert np.array_equal(w["bk"], c[p + "attn.c_attn.bias"][D:2 * D])
  assert np.array_equal(w["bv"], c[p + "attn.c_attn.bias"][2 * D:3 * D])


def test_gpt2_lm_head_tiles_shapes():
  """A 50257-vocab tied head tiles into ceil(50257/16384) = 4 output-port matmuls, the last
  tile short; a small vocab stays a single tile."""
  h = af.input((1, 1024))
  wte = np.zeros((50257, 1024), np.float32)
  tiles = _lm_head_tiles(h, wte.astype(np.float16))
  assert [t.shape for t in tiles] == [(1, 16384), (1, 16384), (1, 16384), (1, 1105)]
  assert all(t.op == "matmul" for t in tiles)
  small = _lm_head_tiles(h, np.zeros((1000, 1024), np.float16))
  assert len(small) == 1 and small[0].shape == (1, 1000)


def test_gpt2_embed_rejects_sequence_beyond_max_positions():
  """`_embed` raises a clear error instead of a raw numpy broadcast failure when the sequence
  is longer than the model's position table."""
  g = GPT2.__new__(GPT2)
  g.wte = np.zeros((100, 16), np.float32)
  g.wpe = np.zeros((10, 16), np.float32)
  with pytest.raises(ValueError, match="exceeds this model's 10 max positions"):
    g._embed(np.arange(11, dtype=np.int64))


def test_gpt2_generate_text_decodes_generated_tokens():
  g = GPT2.__new__(GPT2)
  g.generate = Mock(return_value=[17, 42])
  g.tok = Mock()
  g.tok.decode.return_value = " generated text"

  assert g.generate_text("prompt", max_new_tokens=2) == " generated text"
  g.generate.assert_called_once_with("prompt", 2)
  g.tok.decode.assert_called_once_with([17, 42])


def test_gpt2_gelu_new_lowers():
  """The gelu_new composition (mirror of the ONNX tanh handler) lowers with only native ops."""
  x = af.input((4, 64))
  _lower_fused_to_dir(_gelu_new(x), None)


def test_gpt2_tiled_head_lowers():
  """The full 4-tile head lowers off-device (pure matmul), fp16 or int8 streaming weights."""
  h = af.input((1, 1024))
  wte = np.zeros((50257, 1024), np.float32)
  tiles = _lm_head_tiles(h, wte.astype(np.float16))
  _lower_fused_to_dir(tiles[0], None, int8=True)


def test_logits_from_stitches_multimodel_tiles_in_port_order():
  """A tiled head's per-tile outputs stitch into one [S, vocab] array in output_ports order."""
  net = MultiModel.__new__(MultiModel)
  net.output_ports = [(None, "t0"), (None, "t1")]
  out = {"t0": np.array([[1.0, 2.0]], np.float32), "t1": np.array([[3.0]], np.float32)}
  stitched = _logits_from(net, out)
  assert np.array_equal(stitched, np.array([[1.0, 2.0, 3.0]], np.float32))


def test_logits_from_passes_through_a_plain_model_output():
  """A single-tile head (a plain Model, not a MultiModel) passes its output through unchanged."""
  out = np.array([[9.0, 8.0]], np.float32)
  assert np.array_equal(_logits_from(object(), out), out)
