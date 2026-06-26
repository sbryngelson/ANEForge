"""Qwen3.5 hybrid mixer validation against the transformers reference."""
import numpy as np
from _helpers import requires_ane


@requires_ane
def test_gated_deltanet_decode_matches_transformers():
  import torch
  from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
  from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
  import aneforge.qwen35  # noqa: F401 - registers the gated_deltanet mixer
  from aneforge.llm import LlamaConfig, LayerSpec, DECODE_MIXERS
  from aneforge.graph import input as _input
  from aneforge import _compile
  torch.manual_seed(0)
  nk, nv, dk, dv, K, D = 2, 6, 16, 16, 4, 64
  ct = Qwen3_5Config(hidden_size=D, linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=dk,
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
