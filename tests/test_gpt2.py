"""GPT-2 loader graph-level tests: the Conv1D->linear weight mapping, the tiled lm_head, and
off-device MIL lowering (no ANE dispatch; CI-safe). Numerics are validated on-device by
examples/gpt2.py."""
from unittest.mock import Mock

import numpy as np
import pytest

import aneforge as af
import aneforge.models as models
from aneforge._compile import _lower_fused_to_dir
from aneforge.models import _gpt2_layers, _gelu_new, _lm_head_tiles, GPT2
from aneforge.llm import _gpt2_adapter, LlamaConfig, LlamaPrefill, ModelType
from _helpers import requires_ane


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


def test_gpt2_lm_head_tiles_shapes(monkeypatch):
  """A 50257-vocab tied head tiles to fit the target family's max tensor dim: four output-port
  matmuls at the A13-A15 cap (16384, last tile short), one tile at the A16+ cap (65536). Mock the
  cap so the result does not depend on which chip runs the test; a small vocab is always one tile."""
  from aneforge import _targets
  h = af.input((1, 1024))
  wte = np.zeros((50257, 1024), np.float16)

  monkeypatch.setattr(_targets, "limit", lambda *_a, **_k: 16384)   # A13-A15
  tiles = _lm_head_tiles(h, wte)
  assert [t.shape for t in tiles] == [(1, 16384), (1, 16384), (1, 16384), (1, 1105)]
  assert all(t.op == "matmul" for t in tiles)

  monkeypatch.setattr(_targets, "limit", lambda *_a, **_k: 65536)   # A16+: 50257 fits one tile
  assert [t.shape for t in _lm_head_tiles(h, wte)] == [(1, 50257)]

  small = _lm_head_tiles(h, np.zeros((1000, 1024), np.float16))
  assert len(small) == 1 and small[0].shape == (1, 1000)


def test_gpt2_lm_head_tiles_follow_target_family(monkeypatch):
  """A16+ can keep GPT-2's vocabulary in one output tile; earlier families retain four."""
  h = af.input((1, 1024))
  wte = np.zeros((50257, 1024), np.float16)
  monkeypatch.setattr(models._targets, "detect_family", lambda: 5)
  assert [t.shape for t in models._lm_head_tiles(h, wte)] == [(1, 50257)]
  monkeypatch.setattr(models._targets, "detect_family", lambda: 3)
  assert [t.shape for t in models._lm_head_tiles(h, wte)] == [(1, 16384), (1, 16384), (1, 16384), (1, 1105)]


def test_logits_caches_contiguous_fp32():
  """`_logits` builds a contiguous fp32 lm_head transpose once (skipping the per-call fp16->fp32
  conversion of a strided view). The result is bit-identical to the previous per-call matmul
  (`h @ np.asarray(lm_head).T`), and the cache is reused, not rebuilt, across calls."""
  rng = np.random.default_rng(0)
  cfg = LlamaConfig(dim=16, n_layers=2, n_heads=4, n_kv_heads=4, ffn_dim=64, vocab=100)
  w = {"lm_head": rng.standard_normal((100, 16)).astype(np.float16), "layers": []}
  model = LlamaPrefill(cfg, w)
  h = rng.standard_normal(16).astype(np.float32)
  expected = h @ np.asarray(w["lm_head"]).T          # the pre-cache matmul, unchanged numerics
  assert np.array_equal(model._logits(h), expected)
  first = model._lmT
  assert first is not None and first.dtype == np.float32 and first.flags["C_CONTIGUOUS"]
  model._logits(h)          # second call reuses the cached transpose
  assert model._lmT is first


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


def test_gpt2_adapter_rejects_unimplemented_attn_scaling():
  """`_gpt2_adapter` fails fast for checkpoints (e.g. Cerebras-GPT) that set
  scale_attn_by_inverse_layer_idx/reorder_and_upcast_attn -- semantics the adapter does not
  implement, so silently loading would produce wrong logits."""
  class FakeConfig:
    model_type = ModelType.GPT2
    n_embd, n_layer, n_head, n_inner, vocab_size = 16, 2, 4, 64, 100
    layer_norm_epsilon = 1e-5
    scale_attn_by_inverse_layer_idx = True
  with pytest.raises(ValueError, match="scale_attn_by_inverse_layer_idx"):
    _gpt2_adapter(FakeConfig(), _synthetic_sd(D=16, H=4, L=2, V=100))


def test_llama_prefill_check_positions_rejects_beyond_wpe():
  """`_check_positions` raises a clear error instead of an out-of-bounds `wpe` index/broadcast
  failure when a sequence length exceeds the model's position table -- exercised through
  `_hidden` (prompt too long) and `generate`'s `max_len` (decode cache too long)."""
  class FakeConfig:
    model_type = ModelType.GPT2
    n_embd, n_layer, n_head, n_inner, vocab_size = 16, 2, 4, 64, 100
    layer_norm_epsilon = 1e-5
  cfg, w = _gpt2_adapter(FakeConfig(), _synthetic_sd(D=16, H=4, L=2, V=100))  # wpe has 10 rows
  model = LlamaPrefill(cfg, w)
  with pytest.raises(ValueError, match="exceeds this model's 10 max positions"):
    model._hidden(np.arange(11, dtype=np.int64))
  with pytest.raises(ValueError, match="exceeds this model's 10 max positions"):
    model.generate([1, 2, 3], max_new_tokens=4, max_len=20)
  with pytest.raises(ValueError, match="exceeds this model's 10 max positions"):
    model.warmup(20)   # warmup goes straight to _decoder; must hit the same guard


def test_llama_prefill_release_clears_state():
  """`release()` nulls `_net`/`_dec`/`_pre` (not just releasing the underlying program), so a
  subsequent call recompiles instead of replaying a freed program."""
  class FakeConfig:
    model_type = ModelType.GPT2
    n_embd, n_layer, n_head, n_inner, vocab_size = 16, 2, 4, 64, 100
    layer_norm_epsilon = 1e-5
  cfg, w = _gpt2_adapter(FakeConfig(), _synthetic_sd(D=16, H=4, L=2, V=100))
  model = LlamaPrefill(cfg, w)

  class _FakeNet:
    def __init__(self): self.released = False
    def release(self): self.released = True

  net = _FakeNet()
  chunk_net = _FakeNet()
  model._net = net
  model._seq = 5
  model._dec = {"M": 20, "chunks": [{"net": chunk_net}]}
  model._pre = {"seq": 5, "chunks": [{"net": chunk_net}]}
  model._lmT = object()          # the cached host lm_head transpose must be dropped too
  model.release()
  assert net.released and chunk_net.released
  assert model._net is None and model._seq == 0 and model._dec is None and model._pre is None
  assert model._lmT is None


def test_gpt2_adapter_mapping():
  """The _gpt2_adapter correctly translates a GPT-2 config + state_dict into LlamaConfig and weights."""
  class FakeConfig:
    model_type = ModelType.GPT2
    n_embd = 16
    n_layer = 2
    n_head = 4
    n_inner = 64
    vocab_size = 100
    layer_norm_epsilon = 1e-5

  sd = _synthetic_sd(D=16, H=4, L=2, V=100)
  cfg, w = _gpt2_adapter(FakeConfig(), sd)
  assert cfg.dim == 16 and cfg.n_layers == 2 and cfg.n_heads == 4 and cfg.ffn_dim == 64
  assert cfg.norm_type == "layer" and not cfg.rope
  assert len(cfg.layers) == 2 and all(l.mixer == "attention" and l.mlp == "gelu_new" for l in cfg.layers)
  assert "wpe" in w and w["wpe"].shape == (10, 16)
  assert len(w["layers"]) == 2
  lw = w["layers"][0]
  assert lw["wq"].shape == (16, 16) and lw["wk"].shape == (16, 16) and lw["wv"].shape == (16, 16)
  assert lw["wfc"].shape == (64, 16) and lw["wproj"].shape == (16, 64)
  assert lw["q_bias"].shape == (16,) and lw["fc_bias"].shape == (64,)


@requires_ane
def test_gpt2_matches_huggingface():
  """On-device test: GPT-2 prefill cosine > 0.99 vs HF fp32 reference and greedy decode matches."""
  import torch
  from transformers import GPT2Config, GPT2LMHeadModel
  torch.manual_seed(0)
  hf_cfg = GPT2Config(n_embd=64, n_layer=2, n_head=4, n_inner=128, vocab_size=64, n_positions=64)
  hf = GPT2LMHeadModel(hf_cfg).eval()
  toks = np.random.default_rng(0).integers(0, 64, 8)
  with torch.no_grad():
    ref_logits = hf(torch.tensor(toks)[None]).logits[0, -1].numpy()
    ref_greedy = hf.generate(torch.tensor(toks)[None], max_new_tokens=4, do_sample=False)[0].numpy()[len(toks):]
  cfg, weights = _gpt2_adapter(hf.config, {k: v.detach().float().numpy() for k, v in hf.state_dict().items()})
  model = LlamaPrefill(cfg, weights)
  ane_logits = model.prefill(toks)[0]
  cos = float(ane_logits @ ref_logits / (np.linalg.norm(ane_logits) * np.linalg.norm(ref_logits) + 1e-9))
  assert cos > 0.99 and int(ane_logits.argmax()) == int(ref_logits.argmax())
  ane_greedy = model.generate(toks, max_new_tokens=4)
  assert list(ane_greedy) == list(ref_greedy)


@requires_ane
def test_gpt2_long_context_above_512():
  """On-device test: GPT-2 resident KV-cache decode operates with context length M > 512."""
  class FakeConfig:
    model_type = ModelType.GPT2
    n_embd = 64
    n_layer = 2
    n_head = 2
    n_inner = 128
    vocab_size = 48
    layer_norm_epsilon = 1e-5

  sd = _synthetic_sd(D=64, H=2, L=2, V=48)
  sd["transformer.wpe.weight"] = np.zeros((700, 64), np.float32)
  cfg, weights = _gpt2_adapter(FakeConfig(), sd)
  model = LlamaPrefill(cfg, weights)
  prompt = [1, 2, 3, 4]
  out = model.generate(prompt, max_new_tokens=4, max_len=600)
  assert len(out) == 4 and all(0 <= t < cfg.vocab for t in out)
