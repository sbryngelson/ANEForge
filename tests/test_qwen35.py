"""Qwen3.5 hybrid mixer validation against the transformers reference."""
import numpy as np
from _helpers import requires_ane


@requires_ane
def test_gated_deltanet_decode_matches_transformers():
  import torch
  from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
  from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
  import aneforge.qwen35  # noqa: F401 - registers the gated_deltanet mixer
  from aneforge.llm import LlamaConfig, LayerSpec, DECODE_MIXERS
  from aneforge.graph import input as _input
  from aneforge import _compile
  torch.manual_seed(0)
  nk, nv, dk, dv, K, D = 2, 6, 16, 16, 4, 64
  ct = Qwen3_5TextConfig(hidden_size=D, linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=dk,
                     linear_value_head_dim=dv, linear_conv_kernel_dim=K, hidden_act="silu", rms_norm_eps=1e-6)
  m = Qwen3_5GatedDeltaNet(ct, layer_idx=0).eval()
  T = 6; x = torch.randn(1, T, D); x = x / x.pow(2).mean(-1, keepdim=True).sqrt()   # unit-RMS rows
  with torch.no_grad():
    o = m(x); ref = (o[0] if isinstance(o, tuple) else o)[0].numpy()   # [T, D] = out_proj(deltanet)
  W = {n: p.detach().numpy().astype(np.float32) for n, p in m.named_parameters()}
  w = {"in_norm": np.ones(D, np.float32), "in_proj_qkv": W["in_proj_qkv.weight"], "in_proj_z": W["in_proj_z.weight"],
       "in_proj_a": W["in_proj_a.weight"], "in_proj_b": W["in_proj_b.weight"], "conv1d": np.squeeze(W["conv1d.weight"]),
       "neg_exp_A": -np.exp(W["A_log"]), "dt_bias": W["dt_bias"], "ssm_norm": W["norm.weight"], "out_proj": W["out_proj.weight"]}
  cfg = LlamaConfig(dim=D, n_layers=1, n_heads=1, n_kv_heads=1, ffn_dim=1, vocab=1, norm_eps=1e-6,
                    extra={"nk": nk, "nv": nv, "dk": dk, "dv": dv, "conv_k": K})
  xin = _input((1, D))
  h, pairs = DECODE_MIXERS["gated_deltanet"](xin, w, cfg, LayerSpec(mixer="gated_deltanet"), {}, T)
  net = _compile.compile_multi([h] + [o for o, _ in pairs])
  inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
  for o, i in pairs: net.prog.share_buffer(0, om[o], 0, inm[id(i)])
  for o, i in pairs: net.prog.set_input(inm[id(i)], np.zeros(i.shape, np.float16))
  xs = x[0].numpy(); outs = np.zeros((T, D), np.float32)
  for t in range(T):
    net.prog.set_input(inm[id(xin)], xs[t].reshape(1, D).astype(np.float16)); net.prog.execute()
    outs[t] = np.asarray(net.prog.read_output(om[h])).reshape(D)
  delta = outs - xs                                                 # strip the residual -> out_proj(deltanet)
  cos = float(delta.ravel() @ ref.ravel() / (np.linalg.norm(delta) * np.linalg.norm(ref)))
  assert cos > 0.95, f"gated_deltanet decode vs transformers: cosine {cos}"


@requires_ane
def test_gated_attention_decode_matches_transformers():
  import torch
  from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
  from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding
  import aneforge.qwen35  # noqa: F401 - registers gated_attention
  from aneforge.llm import LlamaConfig, LayerSpec, DECODE_MIXERS
  from aneforge.graph import input as _input
  from aneforge import _compile
  torch.manual_seed(0)
  H, KV, dh, D = 4, 2, 32, 128
  ct = Qwen3_5TextConfig(hidden_size=D, num_attention_heads=H, num_key_value_heads=KV, head_dim=dh,
                     partial_rotary_factor=0.5, attn_output_gate=True, rms_norm_eps=1e-6, rope_theta=1e4,
                     max_position_embeddings=64)
  att = Qwen3_5Attention(ct, layer_idx=0).eval(); rot = Qwen3_5TextRotaryEmbedding(ct)
  T = 6; x = torch.randn(1, T, D); x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
  pos = torch.arange(T)[None]; cos, sin = rot(x, pos)
  rd = cos.shape[-1]
  mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
  with torch.no_grad(): ref = att(x, (cos, sin), mask)[0][0].numpy()        # out_proj(gate*attn), no residual
  W = {n: p.detach().numpy().astype(np.float32) for n, p in att.named_parameters()}
  w = {"in_norm": np.ones(D, np.float32), "wq": W["q_proj.weight"], "wk": W["k_proj.weight"],
       "wv": W["v_proj.weight"], "wo": W["o_proj.weight"], "q_norm": 1.0 + W["q_norm.weight"], "k_norm": 1.0 + W["k_norm.weight"]}
  cfg = LlamaConfig(dim=D, n_layers=1, n_heads=H, n_kv_heads=KV, ffn_dim=1, vocab=1, head_dim=dh,
                    rotary_dim=rd, rope_interleaved=False, norm_eps=1e-6)
  xin = _input((1, D)); ctx = {k: _input(s) for k, s in
                               [("oh", (1, T, 1)), ("inv", (1, T, 1)), ("mask", (1, 1, T)), ("cosp", (1, dh)), ("sinp", (1, dh))]}
  h, pairs = DECODE_MIXERS["gated_attention"](xin, w, cfg, LayerSpec(mixer="gated_attention"), ctx, T)
  net = _compile.compile_multi([h] + [o for o, _ in pairs])
  inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
  for o, i in pairs: net.prog.share_buffer(0, om[o], 0, inm[id(i)])
  for o, i in pairs: net.prog.set_input(inm[id(i)], np.zeros(i.shape, np.float16))
  f16 = np.float16
  pad = lambda a, fl: np.concatenate([a, np.full((a.shape[0], dh - a.shape[1]), fl, a.dtype)], 1)
  cosF = pad(cos[0].numpy(), 1.0); sinF = pad(sin[0].numpy(), 0.0); xs = x[0].numpy(); outs = np.zeros((T, D), np.float32)
  for t in range(T):
    oh = np.zeros((1, T, 1), f16); oh[0, t, 0] = 1; iv = np.ones((1, T, 1), f16); iv[0, t, 0] = 0
    mv = np.full((1, 1, T), -1e4, f16); mv[..., :t + 1] = 0
    for nm, val in [("oh", oh), ("inv", iv), ("mask", mv), ("cosp", cosF[t][None]), ("sinp", sinF[t][None])]:
      net.prog.set_input(inm[id(ctx[nm])], val.astype(f16))
    net.prog.set_input(inm[id(xin)], xs[t].reshape(1, D).astype(f16)); net.prog.execute()
    outs[t] = np.asarray(net.prog.read_output(om[h])).reshape(D)
  delta = outs - xs
  cos_ = float(delta.ravel() @ ref.ravel() / (np.linalg.norm(delta) * np.linalg.norm(ref)))
  assert cos_ > 0.95, f"gated_attention decode vs transformers: cosine {cos_}"


@requires_ane
def test_qwen35_hybrid_model_matches_transformers():
  import torch
  from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
  from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
  import aneforge.qwen35 as q35
  from aneforge.llm import LlamaPrefill
  torch.manual_seed(0)
  ct = Qwen3_5TextConfig(hidden_size=64, num_hidden_layers=4, full_attention_interval=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.5, linear_num_key_heads=2, linear_num_value_heads=6,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6,
    vocab_size=32, rope_theta=1e4, intermediate_size=128, mlp_only_layers=[0, 1, 2, 3], max_position_embeddings=32)
  m = Qwen3_5TextModel(ct).eval()
  T = 8; ids = torch.randint(0, 32, (1, T))
  with torch.no_grad(): ref = m(ids).last_hidden_state[0].numpy()
  sd = {k: v.detach().float().numpy() for k, v in m.named_parameters()}
  cfg, w = q35.adapt(ct, sd)
  assert [ls.mixer for ls in cfg.layers] == ["gated_deltanet", "gated_deltanet", "gated_deltanet", "gated_attention"]
  model = LlamaPrefill(cfg, w); M = T; d = model._decoder(M); chunks = d["chunks"]; f16 = np.float16
  for c in chunks:
    for name, shape in c["p"]["states"].items(): c["net"].prog.set_input(name, np.zeros(shape, f16))
  outs = np.zeros((T, cfg.dim), np.float32); idl = ids[0].numpy()
  for pos in range(T):
    oh = np.zeros((1, M, 1), f16); oh[0, pos, 0] = 1; iv = np.ones((1, M, 1), f16); iv[0, pos, 0] = 0
    mv = np.full((1, 1, M), -1e4, f16); mv[..., :pos + 1] = 0
    vals = {"oh": oh, "inv": iv, "mask": mv, "cosp": d["cos"][pos][None], "sinp": d["sin"][pos][None]}
    h = np.asarray(w["embed"][idl[pos]])[None].astype(f16)
    for c in chunks:
      p = c["p"]; pr = c["net"].prog; pr.set_input(p["x"], h)
      for k in ("oh", "inv", "mask", "cosp", "sinp"):
        if k in p: pr.set_input(p[k], vals[k].astype(f16))
      pr.execute(); h = np.asarray(pr.read_output(p["h"])).astype(f16)
    outs[pos] = h.reshape(cfg.dim).astype(np.float32)
  cos = float((outs.ravel() @ ref.ravel()) / (np.linalg.norm(outs) * np.linalg.norm(ref)))
  assert cos > 0.95, f"qwen3.5 hybrid model vs transformers: cosine {cos}"
