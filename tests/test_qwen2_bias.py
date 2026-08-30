"""Qwen2/2.5 QKV attention biases: the dense adapter must load them (dropping them wrecks attention)."""
import numpy as np

from aneforge.llm import LlamaConfig, LlamaPrefill, _weights_from_state_dict, _dense_adapter
from _helpers import requires_ane


def _dense_sd(D=8, H=2, KV=2, L=1, V=16, ffn=16, biases=True):
  """A minimal Llama/Qwen state_dict; with biases=True it carries q/k/v proj biases (Qwen2 style, no o bias)."""
  rng = np.random.default_rng(0)
  R = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
  sd = {"model.embed_tokens.weight": R(V, D), "model.norm.weight": np.ones(D, np.float32),
        "lm_head.weight": R(V, D)}
  for i in range(L):
    p = f"model.layers.{i}."
    sd.update({p + "self_attn.q_proj.weight": R(H * (D // H), D), p + "self_attn.k_proj.weight": R(KV * (D // H), D),
               p + "self_attn.v_proj.weight": R(KV * (D // H), D), p + "self_attn.o_proj.weight": R(D, H * (D // H)),
               p + "mlp.gate_proj.weight": R(ffn, D), p + "mlp.up_proj.weight": R(ffn, D),
               p + "mlp.down_proj.weight": R(D, ffn), p + "input_layernorm.weight": np.ones(D, np.float32),
               p + "post_attention_layernorm.weight": np.ones(D, np.float32)})
    if biases:
      sd[p + "self_attn.q_proj.bias"] = R(H * (D // H))
      sd[p + "self_attn.k_proj.bias"] = R(KV * (D // H))
      sd[p + "self_attn.v_proj.bias"] = R(KV * (D // H))
  return sd


def test_dense_adapter_loads_qkv_bias():
  """Qwen2 has q/k/v proj biases (no o bias); the dense adapter must carry them into the weight dict."""
  cfg = LlamaConfig(dim=8, n_layers=1, n_heads=2, n_kv_heads=2, ffn_dim=16, vocab=16)
  sd = _dense_sd(biases=True)
  w = _weights_from_state_dict(sd, cfg)
  lw = w["layers"][0]
  assert np.array_equal(lw["q_bias"], sd["model.layers.0.self_attn.q_proj.bias"])
  assert np.array_equal(lw["k_bias"], sd["model.layers.0.self_attn.k_proj.bias"])
  assert np.array_equal(lw["v_bias"], sd["model.layers.0.self_attn.v_proj.bias"])
  assert "o_bias" not in lw            # Qwen2 o_proj has no bias


def test_dense_adapter_no_bias_when_absent():
  """A Llama-style checkpoint (no attention bias) must not gain bias keys -- keeps the graph bias-free."""
  cfg = LlamaConfig(dim=8, n_layers=1, n_heads=2, n_kv_heads=2, ffn_dim=16, vocab=16)
  w = _weights_from_state_dict(_dense_sd(biases=False), cfg)
  lw = w["layers"][0]
  assert not any(k in lw for k in ("q_bias", "k_bias", "v_bias", "o_bias"))


@requires_ane
def test_qwen2_matches_huggingface():
  """On-device: Qwen2 prefill cosine > 0.99 vs HF fp32 and greedy decode matches. Fails if q/k/v biases are dropped."""
  import torch
  from transformers import Qwen2Config, Qwen2ForCausalLM
  torch.manual_seed(0)
  hf_cfg = Qwen2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       intermediate_size=128, vocab_size=64, max_position_embeddings=64, tie_word_embeddings=True)
  hf = Qwen2ForCausalLM(hf_cfg).eval()
  toks = np.random.default_rng(0).integers(0, 64, 8)
  with torch.no_grad():
    ref_logits = hf(torch.tensor(toks)[None]).logits[0, -1].numpy()
    ref_greedy = hf.generate(torch.tensor(toks)[None], max_new_tokens=4, do_sample=False)[0].numpy()[len(toks):]
  cfg, weights = _dense_adapter(hf.config, {k: v.detach().float().numpy() for k, v in hf.state_dict().items()})
  model = LlamaPrefill(cfg, weights)
  ane_logits = model.prefill(toks)[0]
  cos = float(ane_logits @ ref_logits / (np.linalg.norm(ane_logits) * np.linalg.norm(ref_logits) + 1e-9))
  assert cos > 0.99 and int(ane_logits.argmax()) == int(ref_logits.argmax()), f"cosine {cos:.4f}"
  assert list(model.generate(toks, max_new_tokens=4)) == list(ref_greedy)
