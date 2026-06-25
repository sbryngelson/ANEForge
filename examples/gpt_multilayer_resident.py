"""Multi-layer GPT decode on the ANE with every layer's KV-cache resident on-device via share_buffer."""
from __future__ import annotations
import numpy as np
import aneforge as af
from aneforge._compile import compile_multi


class TinyGPTResident:
    def __init__(self, vocab=20, D=16, H=2, Dff=32, L=3, M=12, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.H, self.dh, self.Dff, self.L, self.M = vocab, D, H, D // H, Dff, L, M
        self.emb  = (rng.standard_normal((vocab, D)) * 0.3).astype(np.float32)
        self.uemb = (rng.standard_normal((D, vocab)) * 0.3).astype(np.float32)
        self.lyr = [{k: (rng.standard_normal(s) * 0.1).astype(np.float32) for k, s in
                     {"Wq":(D,D),"Wk":(D,D),"Wv":(D,D),"Wo":(D,D),"Wg":(D,Dff),"Wu":(D,Dff),"Wd":(Dff,D)}.items()}
                    for _ in range(L)]
        for w in self.lyr:
            w["rn1"], w["rn2"] = np.ones(D, np.float32), np.ones(D, np.float32)
        self.scale = 1.0 / self.dh ** 0.5
        self._build()

    def _h1(self, t):                            # [1,D] -> [H,1,dh]
        return t.reshape(1, self.H, self.dh).transpose([1, 0, 2])

    def _block(self, x, w, Kin, Vin, oh, inv, mask):
        H, dh, D, s = self.H, self.dh, self.D, self.scale
        xn = x.rms_norm(w["rn1"])
        k, v, q = self._h1(xn @ w["Wk"]), self._h1(xn @ w["Wv"]), self._h1(xn @ w["Wq"])
        Kout = Kin * inv + k * oh                # masked positional write of this token's K/V
        Vout = Vin * inv + v * oh
        sc = ((q @ Kout.transpose([0, 2, 1])) * s + mask).softmax(-1)   # [H,1,M]
        a = (sc @ Vout).reshape(H, 1, dh)
        h = x + a.transpose([1, 0, 2]).reshape(1, D) @ w["Wo"]
        hn = h.rms_norm(w["rn2"])
        return h + ((hn @ w["Wg"]).silu() * (hn @ w["Wu"])) @ w["Wd"], Kout, Vout

    def _build(self):
        H, dh, D, M, L = self.H, self.dh, self.D, self.M, self.L
        x = af.input((1, D)); oh = af.input((1, M, 1)); inv = af.input((1, M, 1)); mask = af.input((1, 1, M))
        Kin = [af.input((H, M, dh)) for _ in range(L)]; Vin = [af.input((H, M, dh)) for _ in range(L)]
        h = x; Kout, Vout = [], []
        for l in range(L):
            h, Ko, Vo = self._block(h, self.lyr[l], Kin[l], Vin[l], oh, inv, mask)
            Kout.append(Ko); Vout.append(Vo)
        net = compile_multi([h] + Kout + Vout)
        inm = {id(t): n for t, n in net.input_ports}; om = dict(net.output_ports)
        for l in range(L):                       # every layer's cache resident, seeded once
            net.prog.share_buffer(0, om[Kout[l]], 0, inm[id(Kin[l])])
            net.prog.share_buffer(0, om[Vout[l]], 0, inm[id(Vin[l])])
            net.prog.set_input(inm[id(Kin[l])], np.zeros((H, M, dh), np.float16))
            net.prog.set_input(inm[id(Vin[l])], np.zeros((H, M, dh), np.float16))
        self.net, self.inm, self.om = net, inm, om
        self.ports = {"x": inm[id(x)], "oh": inm[id(oh)], "inv": inm[id(inv)], "mask": inm[id(mask)], "h": om[h]}

    def _step(self, tok, p):
        M, f16 = self.M, np.float16
        ohv = np.zeros((1, M, 1), f16); ohv[0, p, 0] = 1.0
        invv = np.ones((1, M, 1), f16); invv[0, p, 0] = 0.0
        mv = np.full((1, 1, M), -1e4, f16); mv[..., :p + 1] = 0.0
        po = self.ports
        self.net.prog.set_input(po["x"], self.emb[tok][None].astype(f16))
        self.net.prog.set_input(po["oh"], ohv); self.net.prog.set_input(po["inv"], invv)
        self.net.prog.set_input(po["mask"], mv); self.net.prog.execute()
        return np.asarray(self.net.prog.read_output(po["h"])).reshape(1, self.D).astype(np.float32)

    def generate(self, prompt, n_new):
        out = []
        for p, tok in enumerate(prompt[:-1]):    # prefill (write caches; discard h)
            self._step(tok, p)
        tok = prompt[-1]; p = len(prompt) - 1
        for _ in range(n_new):
            h = self._step(tok, p); tok = int(np.argmax(h @ self.uemb)); out.append(tok); p += 1
        return out

    def ref_generate(self, prompt, n_new):
        H, dh, D, M, L, s = self.H, self.dh, self.D, self.M, self.L, self.scale
        rms = lambda z, g: (z / np.sqrt((z * z).mean(-1, keepdims=True) + 1e-5)) * g
        silu = lambda z: z / (1 + np.exp(-z))
        Kc = [np.zeros((H, M, dh), np.float32) for _ in range(L)]; Vc = [np.zeros((H, M, dh), np.float32) for _ in range(L)]
        def step(tok, p):
            x = self.emb[tok][None].astype(np.float16).astype(np.float32)
            for l in range(L):
                w = self.lyr[l]; xn = rms(x, w["rn1"])
                hh = lambda t: t.reshape(1, H, dh).transpose(1, 0, 2)
                q, k, v = hh(xn @ w["Wq"]), hh(xn @ w["Wk"]), hh(xn @ w["Wv"])
                Kc[l][:, p, :] = k[:, 0, :]; Vc[l][:, p, :] = v[:, 0, :]
                sc = (q @ Kc[l].transpose(0, 2, 1)) * s
                m = np.full((1, M), -1e4); m[0, :p + 1] = 0.0; sc = sc + m
                a = (np.exp(sc - sc.max(-1, keepdims=True)) / np.exp(sc - sc.max(-1, keepdims=True)).sum(-1, keepdims=True) @ Vc[l]).reshape(H, 1, dh)
                h = x + a.transpose(1, 0, 2).reshape(1, D) @ w["Wo"]; hn = rms(h, w["rn2"])
                x = h + (silu(hn @ w["Wg"]) * (hn @ w["Wu"])) @ w["Wd"]
            return x
        out = []
        for p, tok in enumerate(prompt[:-1]): step(tok, p)
        tok = prompt[-1]; p = len(prompt) - 1
        for _ in range(n_new):
            h = step(tok, p); tok = int(np.argmax(h @ self.uemb)); out.append(tok); p += 1
        return out


if __name__ == "__main__":
    m = TinyGPTResident(L=3, seed=0); prompt = [3, 7, 1]
    res, ref = m.generate(prompt, 5), m.ref_generate(prompt, 5)
    print(f"{m.L}-layer resident-cache GPT decode:", res)
    print("numpy reference                  :", ref)
    print("MATCH:", res == ref)
