import numpy as np
import aneforge as af
from aneforge.llm import LlamaConfig, LlamaPrefill, prefill_block, rope, rope_tables, _weights_from_state_dict, _cfg_from_hf, _gemma_adapter, _mistral_adapter
from _helpers import requires_ane, make_random_llama_model as _random_model

pytestmark = requires_ane  # every test in this module compiles/dispatches to the ANE


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

@requires_ane
def test_generate_matches_prefill_argmax():  # decode step 0 must equal the prefill argmax (catches a broken KV-cache step)
  cfg = LlamaConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  m = _random_model(cfg); prompt = [1, 2, 3]
  first = m.generate(prompt, max_new_tokens=1)[0]
  ref = int(np.asarray(m.prefill(prompt)).ravel().argmax())   # P(next | prompt) via the independent prefill program
  assert first == ref, f"decode first token {first} != prefill argmax {ref}"
  out = m.generate(prompt, max_new_tokens=5)
  assert len(out) == 5 and all(0 <= t < cfg.vocab for t in out)
  assert m.generate(prompt, max_new_tokens=5) == out          # greedy is deterministic

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
def test_segmented_decode_matches_single():  # splitting the decode across chunks must not change the output
  cfg = LlamaConfig(dim=64, n_layers=6, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  out1 = _random_model(cfg).generate([1, 2, 3], max_new_tokens=6)    # fits in one chunk
  m2 = _random_model(cfg); m2._chunk_bytes = 1                       # force one layer per chunk -> 6 chunks chained
  assert len(m2._layer_chunks()) == cfg.n_layers
  assert m2.generate([1, 2, 3], max_new_tokens=6) == out1, "segmented decode diverged from single-program"

def test_llama3_rope_scaling():  # Llama-3.1 "llama3" rope_scaling rescales low freqs; must match HF and be off by default
  from aneforge.llm import rope_tables, _llama3_rope_scale
  sc = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192, "rope_type": "llama3"}
  inv = 1.0 / (500000.0 ** (np.arange(0, 128, 2) / 128))
  scaled = _llama3_rope_scale(inv, sc)
  # HF _compute_llama3_parameters math, inline:
  f, lo, hi, old = 8.0, 1.0, 4.0, 8192.0
  wl = 2 * np.pi / inv
  il = np.where(wl > old / lo, inv / f, inv)
  smd = (1 - (old / wl - lo) / (hi - lo)) * il / f + (old / wl - lo) / (hi - lo) * il
  ref = np.where(~(wl < old / hi) * ~(wl > old / lo), smd, il)
  assert np.max(np.abs(scaled - ref)) < 1e-12, "llama3 rope scaling diverges from HF"
  assert scaled[-1] < inv[-1] and abs(scaled[0] - inv[0]) < 1e-9   # low freq divided down; high freq unchanged
  c_off, _ = rope_tables(16, 128, 500000.0)                        # scaling off by default -> unchanged tables
  c_on, _ = rope_tables(16, 128, 500000.0, scaling=sc)
  assert not np.allclose(c_off, c_on), "scaling did not change the rope tables"

@requires_ane
def test_attention_context_above_512():  # M*dh > 65536 formerly tripped the ANE per-axis limit in the GQA repeat
  cfg = LlamaConfig(dim=256, n_layers=2, n_heads=2, n_kv_heads=1, ffn_dim=128, vocab=48)   # dh=128 -> old cap M=512
  out = _random_model(cfg).generate([1, 2, 3, 4], max_new_tokens=4, max_len=600)           # M=600 -> M*dh=76800
  assert len(out) == 4 and all(0 <= t < cfg.vocab for t in out)

@requires_ane
def test_batched_prefill_matches_token_by_token():  # batched prefill (KV-bridge -> seed -> decode) == token-by-token
  cfg = LlamaConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=128, vocab=48)
  prompt = list(range(1, 9))
  oracle = _random_model(cfg).generate(prompt, max_new_tokens=8, batched_prefill=False)   # token-by-token prefill
  assert _random_model(cfg).generate(prompt, max_new_tokens=8, batched_prefill=True) == oracle
  assert _random_model(cfg).generate(prompt, max_new_tokens=8, batched_prefill=True, prefill_pad=16) == oracle  # bucketed
  m = _random_model(cfg); m._chunk_bytes = 1                       # force one layer per prefill/decode chunk
  assert m.generate(prompt, max_new_tokens=8, batched_prefill=True) == oracle, "chunked batched prefill diverged"

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


@requires_ane
def test_prefill_matches_huggingface_gemma():  # the Gemma adapter: embed scale, (1+w) norms, GeGLU
  import torch
  from transformers import GemmaConfig as HF, GemmaForCausalLM
  hf_cfg = HF(**{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1,
                 "head_dim": 16, "intermediate_size": 128, "vocab_size": 64, "rms_norm_eps": 1e-6,
                 "max_position_embeddings": 64, "tie_word_embeddings": True,
                 "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}})
  torch.manual_seed(0); m = GemmaForCausalLM(hf_cfg).eval()
  toks = np.random.default_rng(0).integers(0, 64, 10)
  with torch.no_grad(): ref = m(torch.tensor(toks)[None]).logits[0, -1].numpy()
  sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
  cfg, w = _gemma_adapter(m.config, sd)
  ane = np.asarray(LlamaPrefill(cfg, w).prefill(toks)).ravel().astype(np.float32)
  cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref) + 1e-9))
  assert cos > 0.99 and int(ane.argmax()) == int(ref.argmax()), f"ANE Gemma prefill vs HF cosine={cos}, argmax {ane.argmax()} vs {ref.argmax()}"


@requires_ane
def test_prefill_matches_huggingface_mistral():  # the Mistral adapter: GQA + sliding window surfaced in extra
  import torch
  from transformers import MistralConfig as HF, MistralForCausalLM
  hf_cfg = HF(**{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                 "intermediate_size": 128, "vocab_size": 64, "rms_norm_eps": 1e-5, "sliding_window": 32,
                 "max_position_embeddings": 64, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}})
  torch.manual_seed(0); m = MistralForCausalLM(hf_cfg).eval()
  toks = np.random.default_rng(0).integers(0, 64, 10)
  with torch.no_grad(): ref = m(torch.tensor(toks)[None]).logits[0, -1].numpy()
  sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
  cfg, w = _mistral_adapter(m.config, sd)
  assert cfg.extra["sliding_window"] == 32, "sliding window must be surfaced in cfg.extra"
  ane = np.asarray(LlamaPrefill(cfg, w).prefill(toks)).ravel().astype(np.float32)
  cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref) + 1e-9))
  assert cos > 0.99 and int(ane.argmax()) == int(ref.argmax()), f"ANE Mistral prefill vs HF cosine={cos}, argmax {ane.argmax()} vs {ref.argmax()}"
