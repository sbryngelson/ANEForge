"""Speculative decoding on the ANE.

A small draft model proposes K tokens; the target verifies all K in ONE forward against its resident KV-cache.
On the ANE this is a strong win: decode is latency/dispatch-bound at one token, so `verify(K) ~= verify(1)`
(~1.05x for K=5 on Qwen3-8B) — the draft's proposals are checked almost for free. With a tiled ANE lm_head
(which also amortizes, ~1.02x), measured 2.28x end-to-end on Qwen3-8B with a Qwen3-0.6B draft (7.4 -> 16.8
tok/s); the draft is then the remaining cost. Greedy speculative is exact: it emits the same tokens as greedy.

Target: a dense Llama/Qwen `LlamaPrefill` (the registered `attention` mixer). Draft: a smaller `LlamaPrefill`
sharing the tokenizer/vocab. Use via `spec_generate(target, draft, ids, ...)`."""
from __future__ import annotations
import numpy as np

from .graph import input as _input, concat as _cat, _const
from . import _compile
from .llm import rope_tables, _repeat_kv3, _DECODE_CTX

f16 = np.float16


def _verify_block(x, w, cfg, Kq, Kin, Vin, place, invc, mask, cosp, sinp):
  """One dense attention+SwiGLU decoder block over `Kq` query tokens against the resident cache. `place`
  [KV,M,Kq] scatters the new K/V into the cache, `invc` [1,M,1] zeroes those slots, `mask` [1,Kq,M] is the
  per-query causal mask. Returns (hidden+residuals, Kout, Vout)."""
  H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.dh
  xn = x.rms_norm(w["attn_norm"], cfg.norm_eps)
  q = xn.linear(w["wq"]).reshape((Kq, H, dh)); k = xn.linear(w["wk"]).reshape((Kq, KV, dh)); v = xn.linear(w["wv"]).reshape((Kq, KV, dh))
  if "q_norm" in w:
    q = q.reshape((Kq * H, dh)).rms_norm(w["q_norm"], cfg.norm_eps).reshape((Kq, H, dh))
    k = k.reshape((Kq * KV, dh)).rms_norm(w["k_norm"], cfg.norm_eps).reshape((Kq, KV, dh))
  hf = dh // 2
  def rope(t, nh):
    a = t.slice_by_size([0, 0, 0], [Kq, nh, hf]); b = t.slice_by_size([0, 0, hf], [Kq, nh, hf])
    return t * cosp.reshape((Kq, 1, dh)) + _cat([b * -1.0, a], axis=2) * sinp.reshape((Kq, 1, dh))
  q = rope(q, H).transpose([1, 0, 2]); k = rope(k, KV).transpose([1, 0, 2]); v = v.transpose([1, 0, 2])
  Kout = Kin * invc + place @ k; Vout = Vin * invc + place @ v
  Kr = _repeat_kv3(Kout, H // KV); Vr = _repeat_kv3(Vout, H // KV)
  sc = ((q @ Kr.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
  a = (sc @ Vr).transpose([1, 0, 2]).reshape((Kq, H * dh))
  h = x + a.linear(w["wo"]); hn = h.rms_norm(w["mlp_norm"], cfg.norm_eps)
  return h + (hn.linear(w["wgate"]).silu() * hn.linear(w["wup"])).linear(w["wdown"]), Kout, Vout


def _build_verifier(m, M, Kq):
  """Build (cached on `m`) the segmented Kq-token verify program for context length `M`, mirroring the decode
  chunking so each chunk's KV cache stays resident via `share_buffer` and the hidden chains chunk -> chunk."""
  cfg = m.cfg; KV, dh, D = cfg.n_kv_heads, cfg.dh, cfg.dim
  if cfg.layers:                                          # the verify block is hardcoded for dense attention + SwiGLU
    assert all(ls.mixer == "attention" and ls.mlp == "swiglu" for ls in cfg.layers), \
      "spec_generate target must be a dense attention + SwiGLU model (got a non-dense layer plan)"
  assert not cfg.rope_interleaved and cfg.rotary_dim in (0, dh), \
    "spec_generate verifier assumes full neox RoPE (rotary_dim == head_dim)"
  groups = m._layer_chunks(); chunks = []
  for gi, grp in enumerate(groups):
    x = _input((Kq, D)); place = _input((KV, M, Kq)); invc = _input((1, M, 1)); mask = _input((1, Kq, M))
    cosp = _input((Kq, dh)); sinp = _input((Kq, dh))
    Kin = [_input((KV, M, dh)) for _ in grp]; Vin = [_input((KV, M, dh)) for _ in grp]
    h = x; Ko = []; Vo = []
    for li, ki, vi in zip(grp, Kin, Vin):
      h, ko, vo = _verify_block(h, m.w["layers"][li], cfg, Kq, ki, vi, place, invc, mask, cosp, sinp); Ko.append(ko); Vo.append(vo)
    if gi == len(groups) - 1: h = h.rms_norm(m.w["final_norm"], cfg.norm_eps)
    net = _compile.compile_multi([h] + Ko + Vo, compress=m.compress)
    inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
    for ki, vi, ko, vo in zip(Kin, Vin, Ko, Vo):
      net.prog.share_buffer(0, om[ko], 0, inm[id(ki)]); net.prog.share_buffer(0, om[vo], 0, inm[id(vi)])
    p = {"x": inm[id(x)], "place": inm[id(place)], "invc": inm[id(invc)], "mask": inm[id(mask)],
         "cosp": inm[id(cosp)], "sinp": inm[id(sinp)], "h": om[h], "states": {inm[id(t)]: t.shape for t in Kin + Vin}}
    chunks.append({"net": net, "p": p})
  cos_t, sin_t = rope_tables(M, dh, cfg.rope_base, cfg.rotary_dim, cfg.rope_interleaved)  # match the decode tables
  return {"M": M, "Kq": Kq, "chunks": chunks, "cos": cos_t, "sin": sin_t, "lmhead": _build_lm_head(m, Kq)}


def _build_lm_head(m, Kq):
  """Tiled lm_head on the ANE: [Kq, dim] -> [Kq, vocab] in `ceil(vocab/65536)` matmul tiles (the per-op dim
  limit). Amortizes over Kq like the verify (~1.02x for K=5), so the per-round logit cost drops from Kq host
  matmuls to one ANE forward."""
  D = m.cfg.dim; lm = np.asarray(m.w["lm_head"]).astype(np.float32); V = lm.shape[0]
  nt = (V + 65535) // 65536; ts = (V + nt - 1) // nt
  x = _input((Kq, D)); outs = [x @ _const(np.ascontiguousarray(lm[i * ts:min((i + 1) * ts, V)].T)) for i in range(nt)]
  net = _compile.compile_multi(outs, compress=m.compress)
  inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
  return {"net": net, "x": inm[id(x)], "outs": [om[o] for o in outs], "V": V}


def _logits(lmh, hidden):
  """Run the tiled ANE lm_head on `hidden` [Kq, dim] -> logits [Kq, vocab]."""
  pr = lmh["net"].prog; pr.set_input(lmh["x"], hidden.astype(f16)); pr.execute()
  tiles = [np.asarray(pr.read_output(op)) for op in lmh["outs"]]
  return np.concatenate(tiles, axis=1)[:, :lmh["V"]].astype(np.float32)


def _verify(m, vf, toks, positions, attends):
  """Run the Kq-token verify: write `toks` to cache `positions`, each attending to cache 0..`attends[i]`.
  Returns the final hidden states [Kq, dim] (pass to `_logits` for next-token logits)."""
  M, Kq, chunks = vf["M"], vf["Kq"], vf["chunks"]
  plc = np.zeros((m.cfg.n_kv_heads, M, Kq), f16); iv = np.ones((1, M, 1), f16); mk = np.full((1, Kq, M), -1e4, f16)
  for i in range(Kq):
    plc[:, positions[i], i] = 1.0; iv[0, positions[i], 0] = 0.0; mk[0, i, :attends[i] + 1] = 0.0
  cs = np.stack([vf["cos"][p] for p in positions]); ss = np.stack([vf["sin"][p] for p in positions])
  h = np.asarray(m.w["embed"][np.asarray(toks)]).astype(f16)
  for c in chunks:
    p = c["p"]; pr = c["net"].prog; pr.set_input(p["x"], h)
    for nm, val in (("place", plc), ("invc", iv), ("mask", mk), ("cosp", cs), ("sinp", ss)): pr.set_input(p[nm], val.astype(f16))
    pr.execute(); h = np.asarray(pr.read_output(p["h"])).astype(f16)
  return _logits(vf["lmhead"], h.reshape(Kq, m.cfg.dim).astype(np.float32))


def _draft_step(draft, d, tok, pos):
  """Feed `tok` at `pos` through the draft's resident decoder (writing its K/V at `pos`); return the
  next-token argmax. The draft cache is maintained incrementally — re-feeding `pos` overwrites it, so rejected
  draft positions are simply overwritten next round (no rollback bookkeeping)."""
  M = d["M"]; cfg = draft.cfg
  ohv = np.zeros((1, M, 1), f16); ohv[0, pos, 0] = 1.0
  invv = np.ones((1, M, 1), f16); invv[0, pos, 0] = 0.0
  mv = np.full((1, 1, M), -1e4, f16); mv[..., :pos + 1] = 0.0
  h = np.asarray(draft.w["embed"][tok])[None].astype(f16)
  vals = {"oh": ohv, "inv": invv, "mask": mv, "cosp": d["cos"][pos][None], "sinp": d["sin"][pos][None]}
  for c in d["chunks"]:
    p = c["p"]; pr = c["net"].prog; pr.set_input(p["x"], h)
    for kk in _DECODE_CTX:
      if kk in p: pr.set_input(p[kk], vals[kk])
    pr.execute(); h = np.asarray(pr.read_output(p["h"])).astype(f16)
  return int((h.reshape(cfg.dim).astype(np.float32) @ np.asarray(draft.w["lm_head"]).T).argmax())


def _reset(prog_chunks):
  for c in prog_chunks:
    for nm, sh in c["p"]["states"].items(): c["net"].prog.set_input(nm, np.zeros(sh, f16))


def spec_generate(target, draft, token_ids, max_new_tokens=40, eos_id=None, max_len=None, n_draft=4, on_token=None):
  """Greedy speculative decoding: `draft` proposes `n_draft` tokens, `target` verifies them in one forward,
  accept the matching prefix + 1 correction. Emits the same tokens as `target.generate(...)`, faster. Both
  models share the tokenizer/vocab; `target` must be a dense attention model."""
  prompt = [int(t) for t in token_ids]; M = int(max_len or len(prompt) + max_new_tokens); Kq = n_draft + 1
  if getattr(target, "_verifier", None) is None or target._verifier["M"] != M or target._verifier["Kq"] != Kq:
    target._verifier = _build_verifier(target, M, Kq)
  vf = target._verifier; draft.warmup(M); dd = draft._dec
  _reset(vf["chunks"]); _reset(dd["chunks"])
  P = len(prompt)
  for i in range(P): _draft_step(draft, dd, prompt[i], i)             # prefill draft cache (0..P-1)
  lg = None; last = 0
  for s in range(0, P, Kq):                                           # prefill target cache in Kq-chunks
    ch = prompt[s:s + Kq]; pad = ch + [0] * (Kq - len(ch))
    lg = _verify(target, vf, pad, [min(s + i, M - 1) for i in range(Kq)], [s + i for i in range(Kq)])
    last = len(ch) - 1
  assert lg is not None                                              # P >= 1, so the prefill loop ran
  cur = int(lg[last].argmax())                                        # first emitted token (target's argmax at P-1)
  out = [cur]
  if on_token is not None and cur != eos_id: on_token(cur)
  p = P                                                               # position of `cur` (next slot to verify)
  while len(out) < max_new_tokens and p + Kq < M:
    drafts = []                                                       # draft proposes n_draft from `cur` (incremental)
    for i in range(n_draft): drafts.append(_draft_step(draft, dd, cur if i == 0 else drafts[-1], p + i))
    lg = _verify(target, vf, [cur] + drafts, [min(p + i, M - 1) for i in range(Kq)], [p + i for i in range(Kq)])
    a = 0
    while a < n_draft and int(lg[a].argmax()) == drafts[a]: a += 1    # accept matching prefix
    emit = drafts[:a] + [int(lg[a].argmax())]                        # + 1 correction/bonus
    for t in emit:
      if len(out) >= max_new_tokens: break
      out.append(t)
      if on_token is not None and t != eos_id: on_token(t)
      if t == eos_id: return out
    cur = emit[-1]; p += a + 1
  return out[:max_new_tokens]
