import numpy as np
import aneforge as af
import aneforge.moe  # noqa: F401 - registers the "moe" MLP into the llm registries
from aneforge.llm import LlamaConfig, LayerSpec
from aneforge.moe import _moe
from _helpers import requires_ane


def _cfg(dim=32, E=8, k=2):
  return LlamaConfig(dim=dim, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=16, vocab=16,
                     extra={"n_experts": E, "n_experts_per_tok": k, "norm_topk_prob": True})


def _weights(cfg, ffn, seed=0):
  rng = np.random.default_rng(seed); dim, E = cfg.dim, cfg.extra["n_experts"]
  R = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32)
  return {"mlp_norm": np.ones(dim, np.float32), "router": R(E, dim),
          "wgate_exps": R(E, ffn, dim), "wup_exps": R(E, ffn, dim), "wdown_exps": R(E, ffn, dim)}


def _ref(h, w, cfg, ffn):
  """Numpy reference: HF Qwen3-MoE routing (softmax-all -> top-k -> renorm) over SwiGLU experts, + residual."""
  E, k, eps = cfg.extra["n_experts"], cfg.extra["n_experts_per_tok"], cfg.norm_eps
  hn = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps) * w["mlp_norm"]
  logits = hn @ w["router"].T
  probs = np.exp(logits - logits.max(-1, keepdims=True)); probs /= probs.sum(-1, keepdims=True)
  idx = np.argsort(-probs, axis=1)[:, :k]
  gw = np.zeros_like(probs)
  for t in range(h.shape[0]): gw[t, idx[t]] = probs[t, idx[t]]
  gw /= gw.sum(1, keepdims=True)
  silu = lambda x: x / (1.0 + np.exp(-x))
  out = np.zeros((h.shape[0], cfg.dim))
  for ex in range(E):
    g = silu(hn @ w["wgate_exps"][ex].T) * (hn @ w["wup_exps"][ex].T)   # [T, ffn]
    out += gw[:, ex:ex + 1] * (g @ w["wdown_exps"][ex])                  # wdown stored [ffn, dim]
  return h + out


def test_moe_builds():  # the moe MLP builds with the right output shape and is registered for prefill+decode
  cfg = _cfg(); ffn = 16
  out = _moe(af.input((5, cfg.dim)), _weights(cfg, ffn), cfg, LayerSpec(mlp="moe"))
  assert out.shape == (5, cfg.dim)
  assert "moe" in af.llm.PREFILL_MLPS and "moe" in af.llm.DECODE_MLPS


@requires_ane
def test_moe_matches_numpy_reference():  # dense ANE MoE == exact top-k routing reference (prefill + decode shapes)
  cfg = _cfg(dim=32, E=8, k=2); ffn = 16; w = _weights(cfg, ffn)
  for T in (1, 5):
    h = (np.random.default_rng(T).standard_normal((T, cfg.dim)) * 0.5).astype(np.float32)
    got = np.asarray(af.compile(_moe(af.input((T, cfg.dim)), w, cfg, LayerSpec(mlp="moe")))(h.astype(np.float16))).astype(np.float32)
    ref = _ref(h, w, cfg, ffn)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9))
    assert cos > 0.99, f"T={T}: dense MoE diverged from top-k reference (cos {cos:.4f})"
