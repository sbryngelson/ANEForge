"""Shared test helpers (plain importable module, NOT a test file)."""
from __future__ import annotations

import numpy as np
import pytest

import aneforge as af


def f16(rng, *shape, scale=1.0, pos=False):
  """Random fp16 tensor of the given shape; scale multiplies a normal, pos makes values positive via abs(.)+0.5."""
  a = rng.standard_normal(shape).astype(np.float32) * scale
  if pos:
    a = np.abs(a) + 0.5
  return a.astype(np.float16)


def onehot_select(t: af.Tensor, i: int, w: int | None = None) -> af.Tensor:
  """Select element i of a [1, W] tensor as [1, 1] via a folded one-hot column matmul (stays fused)."""
  W = w or t.shape[-1]
  sel = np.zeros((W, 1), np.float16)
  sel[i, 0] = 1.0
  return t @ sel.astype(np.float16)


def make_random_llama_model(cfg, compress=None, seed=0):
  """A dense Llama-style LlamaPrefill with reproducible random weights (1/sqrt(fan-in) scaled). Shared by the
  decode and speculative tests so the weight layout stays in one place."""
  from aneforge.llm import LlamaPrefill, _weights_from_state_dict
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
  # host lm_head for a deterministic fp32 baseline; ANE-head tests opt in with m.ane_lm_head = True
  return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg), compress=compress, ane_lm_head=False)


def ane_available() -> bool:
  """True iff the ANE/e5rt dispatch dylib can be located (device tests can run)."""
  try:
    from aneforge._runtime import _find_dylib
    _find_dylib()
    return True
  except Exception:
    return False


# A real, selectable marker (not a bare skipif): CI runs `pytest -m "not requires_ane"`
# to collect only the hardware-free suites, and conftest's collection hook auto-skips any
# requires_ane test that is still collected on a machine without the ANE (so a plain
# `pytest` run off-device skips rather than errors). Applied per test (@requires_ane) or
# per module (pytestmark = requires_ane). The ane_available() probe runs once in the hook,
# not at import, so importing this helper never triggers a dylib build.
requires_ane = pytest.mark.requires_ane
