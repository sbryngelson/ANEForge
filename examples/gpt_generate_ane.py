"""End-to-end autoregressive (GPT-style) text generation on the Apple Neural Engine.

This runs the full decoder inference loop - prefill, KV-cache, per-step decode, greedy
sampling - natively on the ANE. The attention is the ANE's native fused-attention layer:
prefill uses CAUSAL sdpa, and each decode step uses the **KV-cache DECODE shape**
(``af.sdpa`` with seq_q=1 query attending to the cached K/V of length seq_kv), which the
native SDPA layer supports (the validator constrains only "K,V same seq" + "Q,K same embed").

A single decoder block (RMSNorm -> multi-head attention -> residual -> RMSNorm -> SwiGLU ->
residual). Two compiled graphs are reused every step:
  - ``proj``  : x -> (k, v) projections, to grow the host-side KV cache.
  - ``decode``: x + cached K/V -> next hidden state (the decode-shape attention + FFN).
The decode graph is compiled per cache length (aneforge programs are fixed-shape); the e5rt
compile cache makes repeat lengths cheap. Greedy ``argmax`` over an unembedding gives the next
token. Validated to produce token-for-token identical output to a numpy reference (see
tests/test_decoder_block.py::test_ane_generate_matches_numpy).

``generate_resident`` goes one step further: the fixed-max KV-cache lives ON-DEVICE across
steps via ``share_buffer`` (the masked positional write is in-graph and the cache output is
aliased onto its own input), so the cache never round-trips through the host - re-stream-free
decode, the host feeding only the token embedding + a tiny position one-hot/mask each step.
The decode attention is decomposed there (matmul->softmax->matmul) so the whole step is one
fused program (the resident-cache ``compile_multi`` path cannot take the native-SDPA graph cut).
"""
from __future__ import annotations
import numpy as np
import aneforge as af


class TinyDecoderANE:
    def __init__(self, vocab=20, D=16, H=2, Dff=32, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.H, self.dh, self.Dff = vocab, D, H, D // H, Dff
        self.emb = (rng.standard_normal((vocab, D)) * 0.3).astype(np.float32)
        self.uemb = (rng.standard_normal((D, vocab)) * 0.3).astype(np.float32)
        self.W = {k: (rng.standard_normal(s) * 0.1).astype(np.float32) for k, s in
                  {"Wq": (D, D), "Wk": (D, D), "Wv": (D, D), "Wo": (D, D),
                   "Wg": (D, Dff), "Wu": (D, Dff), "Wd": (Dff, D)}.items()}
        self.rn1, self.rn2 = np.ones(D, np.float32), np.ones(D, np.float32)
        self._proj = af.compile(self._proj_graph(af.input((1, D))))

    def _heads1(self, t):                       # [1,D] -> [H,1,dh]
        return t.reshape(1, self.H, self.dh).transpose([1, 0, 2])

    def _proj_graph(self, x):                   # x[1,D] -> [H,2,dh] (k|v stacked)
        xn = x.rms_norm(self.rn1)
        return af.concat([self._heads1(xn @ self.W["Wk"]), self._heads1(xn @ self.W["Wv"])], axis=1)

    def _decode_graph(self, x, Kc, Vc, Sc):     # block result [1,D]
        H, dh, D = self.H, self.dh, self.D
        xn = x.rms_norm(self.rn1)
        qn = self._heads1(xn @ self.W["Wq"])
        a = af.sdpa(qn.reshape(1, H, 1, dh), Kc.reshape(1, H, Sc, dh),
                    Vc.reshape(1, H, Sc, dh)).reshape(H, 1, dh)        # decode-shape attention
        h = x + a.transpose([1, 0, 2]).reshape(1, D) @ self.W["Wo"]
        hn = h.rms_norm(self.rn2)
        return h + ((hn @ self.W["Wg"]).silu() * (hn @ self.W["Wu"])) @ self.W["Wd"]

    def _kv(self, x16):
        kv = np.asarray(self._proj(x16)).reshape(self.H, 2, self.dh)
        return kv[:, :1], kv[:, 1:]

    def _decode_fixed_graph(self, x, Kc, Vc, mask, M):    # fixed-max cache; mask gates unfilled slots
        H, dh, D = self.H, self.dh, self.D
        xn = x.rms_norm(self.rn1)
        qn = self._heads1(xn @ self.W["Wq"])
        a = af.sdpa(qn.reshape(1, H, 1, dh), Kc.reshape(1, H, M, dh), Vc.reshape(1, H, M, dh),
                    attn_mask=mask).reshape(H, 1, dh)     # runtime position mask -> 5th SDPA bottom
        h = x + a.transpose([1, 0, 2]).reshape(1, D) @ self.W["Wo"]
        hn = h.rms_norm(self.rn2)
        return h + ((hn @ self.W["Wg"]).silu() * (hn @ self.W["Wu"])) @ self.W["Wd"]

    def generate_fixed(self, prompt, n_new, max_len):
        """Like generate(), but ONE decode compile for a fixed-max cache: the new token attends to
        the full [.,max_len,.] cache and a runtime position mask (-1e4) gates the unfilled slots."""
        H, dh, D = self.H, self.dh, self.D
        Kc = np.zeros((H, max_len, dh), np.float16); Vc = np.zeros((H, max_len, dh), np.float16)
        net = af.compile(self._decode_fixed_graph(
            af.input((1, D)), af.input((H, max_len, dh)), af.input((H, max_len, dh)),
            af.input((1, 1, 1, max_len)), max_len))       # compiled ONCE, reused every step
        t = 0
        for tok in prompt:                                # prefill: fill the cache
            k, v = self._kv(self.emb[tok][None].astype(np.float16)); Kc[:, t] = k[:, 0]; Vc[:, t] = v[:, 0]; t += 1
        out, x = [], self.emb[prompt[-1]][None].astype(np.float16)
        for _ in range(n_new):
            mask = np.full((1, 1, 1, max_len), -1e4, np.float32); mask[..., :t] = 0.0   # valid 0..t-1
            h = np.asarray(net(x, Kc, Vc, mask.astype(np.float16))).reshape(1, D).astype(np.float32)
            tok = int(np.argmax(h @ self.uemb)); out.append(tok)
            x = self.emb[tok][None].astype(np.float16)
            k, v = self._kv(x); Kc[:, t] = k[:, 0]; Vc[:, t] = v[:, 0]; t += 1
        return out

    def generate_resident(self, prompt, n_new, max_len):
        """Re-stream-free decode: the KV-cache lives ON-DEVICE across steps via share_buffer.
        The masked positional write (cache_out = cache*(1-onehot) + new_kv*onehot) happens IN
        the graph and its output buffer is aliased onto its own input, so the cache never
        round-trips through the host - each step feeds only the token embedding + a tiny
        position one-hot/mask. Same mechanism as the resident-optimizer Trainer path; the
        zero-copy KV-cache, productized."""
        from aneforge._compile import compile_multi
        H, dh, D, M, f16 = self.H, self.dh, self.D, max_len, np.float16
        scale = 1.0 / dh ** 0.5
        x = af.input((1, D)); oh = af.input((1, M, 1)); inv = af.input((1, M, 1))
        mask = af.input((1, 1, M)); Kin = af.input((H, M, dh)); Vin = af.input((H, M, dh))
        xn = x.rms_norm(self.rn1)
        k, v, q = self._heads1(xn @ self.W["Wk"]), self._heads1(xn @ self.W["Wv"]), self._heads1(xn @ self.W["Wq"])
        Kout = Kin * inv + k * oh                  # masked write of this token's K/V at position p
        Vout = Vin * inv + v * oh
        # decode-shape attention DECOMPOSED (matmul->softmax->matmul) so the whole step is ONE
        # fused program: compile_multi (the resident-cache share_buffer path) cannot take the
        # native-SDPA graph cut. seq_q=1 makes the decomposed attention cheap.
        scores = ((q @ Kout.transpose([0, 2, 1])) * scale + mask).softmax(-1)   # [H,1,M]
        a = (scores @ Vout).reshape(H, 1, dh)                                        # [H,1,dh]
        h = x + a.transpose([1, 0, 2]).reshape(1, D) @ self.W["Wo"]
        hn = h.rms_norm(self.rn2)
        hout = h + ((hn @ self.W["Wg"]).silu() * (hn @ self.W["Wu"])) @ self.W["Wd"]
        net = compile_multi([hout, Kout, Vout])
        inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
        net.prog.share_buffer(0, om[Kout], 0, inm[id(Kin)])   # cache stays resident across execute()
        net.prog.share_buffer(0, om[Vout], 0, inm[id(Vin)])
        net.prog.set_input(inm[id(Kin)], np.zeros((H, M, dh), f16))   # seed once
        net.prog.set_input(inm[id(Vin)], np.zeros((H, M, dh), f16))

        def step(tok, p):
            ohv = np.zeros((1, M, 1), f16); ohv[0, p, 0] = 1.0
            invv = np.ones((1, M, 1), f16); invv[0, p, 0] = 0.0
            mv = np.full((1, 1, M), -1e4, f16); mv[..., :p + 1] = 0.0
            net.prog.set_input(inm[id(x)], self.emb[tok][None].astype(f16))
            net.prog.set_input(inm[id(oh)], ohv); net.prog.set_input(inm[id(inv)], invv)
            net.prog.set_input(inm[id(mask)], mv)
            net.prog.execute()
            return np.asarray(net.prog.read_output(om[hout])).reshape(1, D).astype(np.float32)

        p, out = 0, []
        for tok in prompt[:-1]:                    # prefill 0..len-2 (write only; h discarded)
            step(tok, p); p += 1
        cur = prompt[-1]
        for _ in range(n_new):                     # decode: query attends to the resident cache
            h = step(cur, p); p += 1
            cur = int(np.argmax(h @ self.uemb)); out.append(cur)
        net.release()
        return out

    def generate(self, prompt, n_new):
        ck, cv = [], []
        for tok in prompt:                      # prefill: build the cache
            k, v = self._kv(self.emb[tok][None].astype(np.float16)); ck.append(k); cv.append(v)
        out, x = [], self.emb[prompt[-1]][None].astype(np.float16)
        for _ in range(n_new):                  # decode loop
            Sc = len(ck)
            Kc, Vc = np.concatenate(ck, 1).astype(np.float16), np.concatenate(cv, 1).astype(np.float16)
            net = af.compile(self._decode_graph(af.input((1, self.D)),
                                                af.input((self.H, Sc, self.dh)),
                                                af.input((self.H, Sc, self.dh)), Sc))
            h = np.asarray(net(x, Kc, Vc)).reshape(1, self.D).astype(np.float32)
            tok = int(np.argmax(h @ self.uemb)); out.append(tok)
            x = self.emb[tok][None].astype(np.float16)
            k, v = self._kv(x); ck.append(k); cv.append(v)
        return out


if __name__ == "__main__":
    m = TinyDecoderANE()
    prompt = [3, 7, 1]
    print(f"prompt {prompt} -> generated {m.generate(prompt, 6)}  (all on the ANE: native "
          f"prefill + KV-cache decode-shape attention + greedy sampling)")
    print(f"resident KV-cache (cache stays on-device, no host re-feed): "
          f"{m.generate_resident(prompt, 6, max_len=16)}")
