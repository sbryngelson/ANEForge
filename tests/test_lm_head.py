"""Off-device tests for the ANE lm_head: tiled compile_multi path (CI-safe, no ANE dispatch)."""
import warnings
from unittest.mock import Mock, patch

import numpy as np

from aneforge.llm import LlamaConfig, _lm_head_tile_size
from _helpers import make_random_llama_model as _random_model


def _cfg(vocab, dim=64, n_layers=1):
  return LlamaConfig(dim=dim, n_layers=n_layers, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=vocab)


def _make_mock_net(outs, **kw):
  """Mock a MultiModel: name the graph's input/output tensors as ports, and keep the output tensors
  on `.tiles` so a test can assert the tiling the caller actually built."""
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
  m.tiles = list(outs)
  m.prog = Mock()
  shapes = {t._name: t.shape for t in outs}   # so `_ane_logits` can run without a device
  m.prog.read_output = Mock(side_effect=lambda name: np.zeros(shapes[name], np.float16))
  m.release = Mock()
  return m


def _host_logits(m, h):
  return h @ np.ascontiguousarray(np.asarray(m.w["lm_head"]).T, np.float32)


def test_lm_head_tiles_follow_target_family(monkeypatch):
  """Tile widths, not just the tile count, follow the target family's max tensor dim: cap 16384 splits
  50257 into 16384/16384/16384/1105; cap 65536 emits one tile. A small vocab is always one tile."""
  from aneforge import _targets
  m = _random_model(_cfg(50257), seed=0); m.ane_lm_head = True

  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
    monkeypatch.setattr(_targets, "detect_family", lambda: 3)      # A14 (M2)
    assert _lm_head_tile_size() == 16384
    pr = m._lm_head_program()
    assert pr is not None and pr["V"] == 50257
    assert [t.shape for t in pr["net"].tiles] == [(1, 16384), (1, 16384), (1, 16384), (1, 1105)]

    m._lmh = None; m._lmh_off = False
    monkeypatch.setattr(_targets, "detect_family", lambda: 5)      # A16 (M4/M5)
    assert _lm_head_tile_size() == 65536
    pr16 = m._lm_head_program()
    assert pr16 is not None
    assert [t.shape for t in pr16["net"].tiles] == [(1, 50257)]

    small = _random_model(_cfg(100), seed=0); small.ane_lm_head = True
    assert [t.shape for t in small._lm_head_program()["net"].tiles] == [(1, 100)]


def test_lm_head_program_declines_when_oversized():
  """When _lmhead_bytes is too low, _lm_head_program returns None and _logits falls back to host."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  m._lmhead_bytes = 1
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    assert m._lm_head_program() is None
    assert len(w) == 1 and "over the" in str(w[0].message)
  h = np.random.default_rng(99).standard_normal(m.cfg.dim).astype(np.float32)
  np.testing.assert_array_equal(m._logits(h), _host_logits(m, h))


def test_release_drops_lm_head_program():
  """release() frees the ANE lm_head and resets _lmh/_lmh_off so the model recompiles cleanly."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  mock_net = Mock(); m._lmh = {"net": mock_net, "x": 0, "outs": [1], "V": 100}
  m._lmh_off = True
  m.release()
  assert m._lmh is None
  assert m._lmh_off is False
  mock_net.release.assert_called_once()


def test_lm_head_program_survives_decoder_rebuild():
  """_lmh is shape-independent of max_len; a new _decoder(M) does not invalidate it."""
  m = _random_model(_cfg(100, n_layers=2), seed=42)
  m.ane_lm_head = True
  sentinel = object()
  m._lmh = sentinel
  for M in (16, 32):
    with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
      m._decoder(M)
    assert m._lmh is sentinel, f"_decoder({M}) must not overwrite _lmh"


def test_lm_head_program_is_built_once():
  """Repeated calls reuse the same compiled program instead of recompiling per token."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net) as cm:
    first = m._lm_head_program()
    assert m._lm_head_program() is first
    assert cm.call_count == 1


def test_logits_host_path_unchanged_when_flag_off():
  """With ane_lm_head=False, _logits never builds _lmh and returns h @ lm_head.T fp32 bit-identical."""
  m = _random_model(_cfg(100, dim=16), seed=0)
  assert m.ane_lm_head is False
  h = np.random.default_rng(7).standard_normal(m.cfg.dim).astype(np.float32)
  np.testing.assert_array_equal(m._logits(h), _host_logits(m, h))
  assert m._lmh is None, "_lmh must not be built when flag is off"


def test_logits_per_call_override_does_not_touch_the_instance():
  """`_logits(h, ane=...)` is the per-call override `generate` threads down: it must not write back
  to `self.ane_lm_head`, and `ane=False` must keep the host path on an ANE-head model."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  h = np.random.default_rng(5).standard_normal(m.cfg.dim).astype(np.float32)
  np.testing.assert_array_equal(m._logits(h, ane=False), _host_logits(m, h))
  assert m._lmh is None and m.ane_lm_head is True

  m.ane_lm_head = False
  with patch("aneforge._compile.compile_multi", side_effect=_make_mock_net):
    got = m._logits(h, ane=True)
  assert got.shape == (m.cfg.vocab,)          # tiles concatenated back to one row of logits
  assert m._lmh is not None and m.ane_lm_head is False


def test_generate_override_never_leaks_into_the_instance():
  """An `ane_lm_head=` override on generate() must not survive the call, including when generate
  raises or returns early -- it is threaded through _logits, not written to the instance."""
  m = _random_model(_cfg(100), seed=42)
  m._decoder = lambda M: (_ for _ in ()).throw(RuntimeError("boom"))
  try:
    m.generate([1, 2, 3], max_new_tokens=2, max_len=8, ane_lm_head=True)
  except RuntimeError:
    pass
  assert m.ane_lm_head is False


def test_lm_head_compile_failure_falls_back():
  """If compile_multi raises, _lm_head_program declines and _logits returns host result."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  with patch("aneforge._compile.compile_multi", side_effect=RuntimeError("ANE error")):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      assert m._lm_head_program() is None
      assert m._lmh_off is True
      assert len([x for x in w if "compile failed" in str(x.message)]) == 1
  h = np.random.default_rng(88).standard_normal(m.cfg.dim).astype(np.float32)
  np.testing.assert_array_equal(m._logits(h), _host_logits(m, h))


def test_decode_logits_tie_margin_fallback():
  """_decode_logits recomputes a greedy step on the host fp32 head only when the ANE top-2 gap is under
  lmhead_tie_margin * max|logit| -- so greedy matches the host on ties, keeps ANE speed otherwise, and
  never falls back while sampling."""
  m = _random_model(_cfg(100), seed=1)
  m.ane_lm_head = True
  m._lmh = object()                                  # pretend the ANE head is live
  m.lmhead_tie_margin = 3e-3
  near_tie = np.zeros(100, np.float32); near_tie[0] = 10.0; near_tie[1] = 10.0 - 0.001   # gap 0.001 < 0.03
  clear = np.zeros(100, np.float32); clear[0] = 10.0                                       # gap 10 >> margin
  host = np.zeros(100, np.float32); host[1] = 10.0; host[0] = 9.99                         # host prefers idx 1
  calls = []

  def fake(which):
    def _f(h, ane=None):
      calls.append(ane)
      return which if ane else host
    return _f

  m._logits = fake(near_tie)                          # near-tie -> ANE then host recompute
  out = m._decode_logits(np.zeros(m.cfg.dim, np.float32), True, greedy=True)
  assert calls == [True, False] and int(out.argmax()) == 1     # host's pick wins the tie

  calls.clear(); m._logits = fake(clear)              # clear winner -> ANE only, no host recompute
  out = m._decode_logits(np.zeros(m.cfg.dim, np.float32), True, greedy=True)
  assert calls == [True] and int(out.argmax()) == 0

  calls.clear(); m._logits = fake(near_tie)           # sampling -> never falls back, even on a tie
  m._decode_logits(np.zeros(m.cfg.dim, np.float32), True, greedy=False)
  assert calls == [True]


def test_lm_head_program_dim_mismatch_falls_back():
  """A lm_head whose in-dim disagrees with cfg.dim declines with a clear reason (a raise, not an
  assert, so `python -O` keeps the check) and leaves the host path serving logits."""
  m = _random_model(_cfg(100), seed=42)
  m.ane_lm_head = True
  m.w["lm_head"] = np.zeros((100, 32), np.float16)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    assert m._lm_head_program() is None
    assert m._lmh_off is True
    assert len([x for x in w if "lm_head dim 32 != model dim 64" in str(x.message)]) == 1
