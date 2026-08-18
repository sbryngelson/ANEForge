"""Off-device tests for the ANE lm_head: tiled compile_multi path (CI-safe, no ANE dispatch)."""
import warnings
from unittest.mock import Mock, patch

import numpy as np

from aneforge.llm import LlamaConfig, LlamaPrefill, _lm_head_tile_size, _weights_from_state_dict
from _helpers import make_random_llama_model as _random_model


def _make_mock_net(outs, **kw):
  """Build a mock MultiModel with proper input/output ports by traversing the graph."""
  visited = set(); inputs = []
  def _walk(t):
    if id(t) in visited: return
    visited.add(id(t))
    if t.op == "input": inputs.append(t)
    for s in t.srcs: _walk(s)
  for o in outs: _walk(o)
  inputs.sort(key=lambda t: t.attrs.get("idx", 0))
  for i, t in enumerate(inputs): t._name = f"in{i}"
  for i, t in enumerate(outs): t._name = f"out{i}"
  m = Mock()
  m.input_ports = [(t, t._name) for t in inputs]
  m.output_ports = [(t, t._name) for t in outs]
  m.prog = Mock()
  m.release = Mock()
  return m


def test_lm_head_tiles_follow_target_family(monkeypatch):
  """Tile shapes match the target family: A13-A15 (cap 16384) -> 4 tiles for 50257 vocab;
  A16+ (cap 65536) -> 1 tile. Small vocab is always one tile."""
  from aneforge import _targets
  rng = np.random.default_rng(0); R = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=50257)
  sd = {"model.embed_tokens.weight": R(50257, 64), "model.norm.weight": np.ones(64, np.float32),
        "lm_head.weight": R(50257, 64)}
  sd["model.layers.0.self_attn.q_proj.weight"] = R(64, 64)
  sd["model.layers.0.self_attn.k_proj.weight"] = R(32, 64)
  sd["model.layers.0.self_attn.v_proj.weight"] = R(32, 64)
  sd["model.layers.0.self_attn.o_proj.weight"] = R(64, 64)
  sd["model.layers.0.mlp.gate_proj.weight"] = R(128, 64)
  sd["model.layers.0.mlp.up_proj.weight"] = R(128, 64)
  sd["model.layers.0.mlp.down_proj.weight"] = R(64, 128)
  sd["model.layers.0.input_layernorm.weight"] = np.ones(64, np.float32)
  sd["model.layers.0.post_attention_layernorm.weight"] = np.ones(64, np.float32)
  m = LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg), ane_lm_head=True)

  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
    monkeypatch.setattr(_targets, "detect_family", lambda: 3)   # A14 (M2)
    pr = m._lm_head_program()
    assert pr is not None
    assert pr["V"] == 50257
    tile = _lm_head_tile_size()
    assert tile == 16384
    assert len(pr["outs"]) == 4

    m._lmh = None; m._lmh_off = False
    monkeypatch.setattr(_targets, "detect_family", lambda: 5)   # A16 (M4/M5)
    pr16 = m._lm_head_program()
    assert pr16 is not None
    assert len(pr16["outs"]) == 1

    cfg2 = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
    sd2 = {"model.embed_tokens.weight": R(100, 64), "model.norm.weight": np.ones(64, np.float32),
           "lm_head.weight": R(100, 64)}
    sd2["model.layers.0.self_attn.q_proj.weight"] = R(64, 64)
    sd2["model.layers.0.self_attn.k_proj.weight"] = R(32, 64)
    sd2["model.layers.0.self_attn.v_proj.weight"] = R(32, 64)
    sd2["model.layers.0.self_attn.o_proj.weight"] = R(64, 64)
    sd2["model.layers.0.mlp.gate_proj.weight"] = R(128, 64)
    sd2["model.layers.0.mlp.up_proj.weight"] = R(128, 64)
    sd2["model.layers.0.mlp.down_proj.weight"] = R(64, 128)
    sd2["model.layers.0.input_layernorm.weight"] = np.ones(64, np.float32)
    sd2["model.layers.0.post_attention_layernorm.weight"] = np.ones(64, np.float32)
    m2 = LlamaPrefill(cfg2, _weights_from_state_dict(sd2, cfg2), ane_lm_head=True)
    pr_small = m2._lm_head_program()
    assert pr_small is not None
    assert len(pr_small["outs"]) == 1


def test_lm_head_program_declines_when_oversized():
  """When _lmhead_bytes is too low, _lm_head_program returns None and _logits falls back to host."""
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
  m = _random_model(cfg, seed=42)
  m.ane_lm_head = True
  m._lmhead_bytes = 1
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pr = m._lm_head_program()
    assert pr is None
    assert len(w) == 1 and "skipped" in str(w[0].message)
  rng = np.random.default_rng(99)
  h = rng.standard_normal(cfg.dim).astype(np.float32)
  host_logits = h @ np.ascontiguousarray(np.asarray(m.w["lm_head"]).T, np.float32)
  np.testing.assert_array_equal(m._logits(h), host_logits)


def test_release_drops_lm_head_program():
  """release() frees the ANE lm_head and resets _lmh/_lmh_off so the model recompiles cleanly."""
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
  m = _random_model(cfg, seed=42)
  m.ane_lm_head = True
  mock_net = Mock(); m._lmh = {"net": mock_net, "x": 0, "outs": [1], "V": 100}
  m._lmh_off = True
  m.release()
  assert m._lmh is None
  assert m._lmh_off is False
  mock_net.release.assert_called_once()


def test_lm_head_program_survives_decoder_rebuild():
  """_lmh is shape-independent of max_len; a new _decoder(M) does not invalidate it."""
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
  m = _random_model(cfg, seed=42)
  m.ane_lm_head = True
  sentinel = object()
  m._lmh = sentinel
  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
    m._decoder(16)
  assert m._lmh is sentinel, "decoder rebuild must not overwrite _lmh"
  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
    m._decoder(32)
  assert m._lmh is sentinel, "second decoder rebuild must not overwrite _lmh"


def test_logits_host_path_unchanged_when_flag_off():
  """With ane_lm_head=False, _logits never builds _lmh and returns h @ lm_head.T fp32 bit-identical."""
  cfg = LlamaConfig(dim=16, n_layers=2, n_heads=4, n_kv_heads=4, ffn_dim=64, vocab=100)
  m = _random_model(cfg, seed=0)
  assert m.ane_lm_head is False
  rng = np.random.default_rng(7)
  h = rng.standard_normal(cfg.dim).astype(np.float32)
  lm = np.ascontiguousarray(np.asarray(m.w["lm_head"]).T, np.float32)
  ref = h @ lm
  got = m._logits(h)
  np.testing.assert_array_equal(got, ref)
  assert m._lmh is None, "_lmh must not be built when flag is off"


def test_lm_head_compile_failure_falls_back():
  """If compile_multi raises, _lm_head_program declines and _logits returns host result."""
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
  m = _random_model(cfg, seed=42)
  m.ane_lm_head = True
  with patch("aneforge._compile.compile_multi", side_effect=RuntimeError("ANE error")):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      pr = m._lm_head_program()
      assert pr is None
      assert m._lmh_off is True
      fallback = [x for x in w if "failed" in str(x.message)]
      assert len(fallback) == 1
  rng = np.random.default_rng(88)
  h = rng.standard_normal(cfg.dim).astype(np.float32)
  host_logits = h @ np.ascontiguousarray(np.asarray(m.w["lm_head"]).T, np.float32)
  np.testing.assert_array_equal(m._logits(h), host_logits)


def test_lm_head_program_dim_mismatch_asserts():
  """If lm_head dim != model dim, _lm_head_program catches the assertion and falls back to host."""
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=100)
  m = _random_model(cfg, seed=42)
  m.ane_lm_head = True
  m.w["lm_head"] = np.zeros((100, 32), np.float16)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pr = m._lm_head_program()
    assert pr is None
    assert m._lmh_off is True
    fallback = [x for x in w if "failed" in str(x.message)]
    assert len(fallback) == 1
