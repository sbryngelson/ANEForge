"""Speculative decoding: must emit the same tokens as greedy (it's exact), for any draft."""
import numpy as np
from _helpers import requires_ane
from aneforge.llm import LlamaConfig, LlamaPrefill, _weights_from_state_dict


def _rand_model(cfg, seed):
  rng = np.random.default_rng(seed); dh = cfg.dh
  R = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32)
  sd = {"model.embed_tokens.weight": R(cfg.vocab, cfg.dim), "model.norm.weight": np.ones(cfg.dim, np.float32),
        "lm_head.weight": R(cfg.vocab, cfg.dim)}
  for L in range(cfg.n_layers):
    p = f"model.layers.{L}."
    for nm, sh in [("self_attn.q_proj", (cfg.n_heads * dh, cfg.dim)), ("self_attn.k_proj", (cfg.n_kv_heads * dh, cfg.dim)),
                   ("self_attn.v_proj", (cfg.n_kv_heads * dh, cfg.dim)), ("self_attn.o_proj", (cfg.dim, cfg.n_heads * dh)),
                   ("mlp.gate_proj", (cfg.ffn_dim, cfg.dim)), ("mlp.up_proj", (cfg.ffn_dim, cfg.dim)), ("mlp.down_proj", (cfg.dim, cfg.ffn_dim))]:
      sd[p + nm + ".weight"] = R(*sh)
    sd[p + "input_layernorm.weight"] = np.ones(cfg.dim, np.float32); sd[p + "post_attention_layernorm.weight"] = np.ones(cfg.dim, np.float32)
  return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg))


@requires_ane
def test_spec_generate_matches_greedy():
  from aneforge.speculative import spec_generate
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  tgt = _rand_model(cfg, 0); drf = _rand_model(cfg, 1)        # different draft -> exercises the rejection path
  prompt = [1, 2, 3, 4]
  g = tgt.generate(list(prompt), max_new_tokens=12, max_len=40)
  s = spec_generate(tgt, drf, prompt, max_new_tokens=12, max_len=40, n_draft=3)
  assert g == s, f"spec != greedy:\n  greedy {g}\n  spec   {s}"
