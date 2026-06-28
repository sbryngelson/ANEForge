"""Speculative decoding: must emit the same tokens as greedy (it's exact), for any draft."""
from _helpers import requires_ane, make_random_llama_model
from aneforge.llm import LlamaConfig


@requires_ane
def test_spec_generate_matches_greedy():
  from aneforge.speculative import spec_generate
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  tgt = make_random_llama_model(cfg, seed=0); drf = make_random_llama_model(cfg, seed=1)   # different draft -> exercises rejection
  prompt = [1, 2, 3, 4]
  g = tgt.generate(list(prompt), max_new_tokens=12, max_len=40)
  s = spec_generate(tgt, drf, prompt, max_new_tokens=12, max_len=40, n_draft=3)
  assert g == s, f"spec != greedy:\n  greedy {g}\n  spec   {s}"


@requires_ane
def test_spec_accepts_drafts_when_draft_equals_target(monkeypatch):
  # identical draft+target -> every draft is accepted, so the multi-token verify runs far fewer times than the
  # tokens it emits. If acceptance were broken (always rejected) it would verify once per token instead.
  from aneforge import speculative as sp
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  tgt = make_random_llama_model(cfg, seed=0); drf = make_random_llama_model(cfg, seed=0)   # identical models
  verifies = {"n": 0}; real_verify = sp._verify
  def counting_verify(*a, **k):
    verifies["n"] += 1
    return real_verify(*a, **k)
  monkeypatch.setattr(sp, "_verify", counting_verify)
  out = sp.spec_generate(tgt, drf, [1, 2, 3, 4], max_new_tokens=16, max_len=48, n_draft=3)
  g = tgt.generate([1, 2, 3, 4], max_new_tokens=16, max_len=48)
  assert out == g                                                   # still exact
  assert verifies["n"] < len(out), f"no speculative batching: {verifies['n']} verifies for {len(out)} tokens"
