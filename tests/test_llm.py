import numpy as np
import aneforge as af
from aneforge.llm import LlamaConfig, LlamaPrefill, prefill_block, rope, rope_tables, _weights_from_state_dict, _cfg_from_hf
from _helpers import requires_ane


def test_rope_tables_shape():
  cos, sin = rope_tables(8, 16)
  assert cos.shape == (8, 16) and sin.shape == (8, 16)

def test_rope_builds():
  cos, sin = rope_tables(8, 16)
  out = rope(af.input((1, 2, 8, 16)), cos, sin)
  assert out.shape == (1, 2, 8, 16) and out.op == "add"

def test_prefill_block_builds():  # GQA 4 heads / 2 kv; output keeps [S, dim]
  cfg = LlamaConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=32)
  S, dh = 12, cfg.dim // cfg.n_heads
  rng = np.random.default_rng(0); R = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
  w = {"wq": R(cfg.n_heads * dh, cfg.dim), "wk": R(cfg.n_kv_heads * dh, cfg.dim), "wv": R(cfg.n_kv_heads * dh, cfg.dim),
       "wo": R(cfg.dim, cfg.n_heads * dh), "wgate": R(cfg.ffn_dim, cfg.dim), "wup": R(cfg.ffn_dim, cfg.dim),
       "wdown": R(cfg.dim, cfg.ffn_dim), "attn_norm": np.ones(cfg.dim, np.float32), "mlp_norm": np.ones(cfg.dim, np.float32)}
  cos, sin = rope_tables(S, dh, cfg.rope_base)
  out = prefill_block(af.input((S, cfg.dim)), w, cfg, cos, sin)
  assert out.shape == (S, cfg.dim) and out.op == "add"

@requires_ane
def test_rope_matches_numpy():
  cos, sin = rope_tables(8, 16); rng = np.random.default_rng(0); x = rng.standard_normal((1, 2, 8, 16)).astype(np.float16)
  got = np.asarray(af.compile(rope(af.input((1, 2, 8, 16)), cos, sin))(x)).astype(np.float32)
  c, s = cos.astype(np.float32), sin.astype(np.float32); h = 8
  ref = x.astype(np.float32) * c + np.concatenate([-x[..., h:].astype(np.float32), x[..., :h].astype(np.float32)], -1) * s
  cos_sim = float(got.ravel() @ ref.ravel() / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9))
  assert cos_sim > 0.999, f"RoPE vs reference cosine={cos_sim}"

def _random_model(cfg, compress=None):
  rng = np.random.default_rng(0); dh = cfg.dh
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
  return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg), compress=compress)

@requires_ane
def test_generate_runs():  # the decode loop produces the requested number of valid tokens
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  out = _random_model(cfg).generate([1, 2, 3], max_new_tokens=5)
  assert len(out) == 5 and all(0 <= t < cfg.vocab for t in out)

@requires_ane
def test_int8_decode_matches_fp16():  # int8-quantized ANE weights stay faithful to fp16 (prefill + decode both run)
  cfg = LlamaConfig(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=256, vocab=64)
  prompt = [1, 2, 3, 4]
  ref = _random_model(cfg).prefill(prompt)[0]
  q8 = _random_model(cfg, compress="int8")                   # same weights, ANE matmuls quantized per-channel int8
  lo = q8.prefill(prompt)[0]
  cos = float(ref @ lo / (np.linalg.norm(ref) * np.linalg.norm(lo)))
  assert cos > 0.99, f"int8 prefill logits diverged from fp16 (cos {cos:.4f})"
  out = q8.generate(prompt, max_new_tokens=5)                # the int8 decode program (compile_multi compress=) runs
  assert len(out) == 5 and all(0 <= t < cfg.vocab for t in out)

@requires_ane
def test_prefill_matches_huggingface_llama():  # the definitive check: ANE prefill == HF LlamaForCausalLM
  import torch
  from transformers import LlamaConfig as HF, LlamaForCausalLM
  hf_cfg = HF(**{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                 "intermediate_size": 128, "vocab_size": 64, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                 "max_position_embeddings": 64})
  torch.manual_seed(0); m = LlamaForCausalLM(hf_cfg).eval()
  toks = np.random.default_rng(0).integers(0, 64, 10)
  with torch.no_grad(): ref = m(torch.tensor(toks)[None]).logits[0, -1].numpy()
  cfg = _cfg_from_hf(m.config); sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
  ane = np.asarray(LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg)).prefill(toks)).ravel().astype(np.float32)
  cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref) + 1e-9))
  assert cos > 0.99 and int(ane.argmax()) == int(ref.argmax()), f"ANE prefill vs HF cosine={cos}, argmax {ane.argmax()} vs {ref.argmax()}"
