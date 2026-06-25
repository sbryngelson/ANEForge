"""Train a multi-layer character-level language model on the Apple Neural Engine - a 4-layer causal LLaMA-style transformer, forward AND backward AND the optimizer step
all on the engine, then generate from it.

This is the "at scale" training demonstration: not a single block but a stacked,
causally-masked transformer with token and positional embeddings, RMSNorm + SwiGLU
blocks, a tied-shape output projection, and a next-token cross-entropy objective. Every
parameter trains on the ANE, including the RMSNorm gains.

Two notes on what makes this fit the engine's reachable surface:
  - Token embedding is a one-hot matmul (`onehot @ W_emb`), not a `gather`: the
    embedding gradient is then an ordinary matmul gradient, sidestepping the gather/
    scatter wall while being mathematically identical to a lookup.
  - Causal masking is an additive bias (a fixed upper-triangular `-1e4` matrix) added to
    the attention scores before softmax, so position i attends only to positions <= i.

    python3 examples/train_charlm.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad

f16 = np.float16
S, D, HEADS, dh, NLAYERS, FF = 48, 64, 4, 16, 4, 256
STEPS, LR = 150, 0.003

TEXT = "the apple neural engine trains a language model on itself. " * 3


def main():
    chars = sorted(set(TEXT)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in TEXT], np.int64)

    def onehot(idx):
        o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
        return o

    Xv = onehot(ids[:S])              # input window
    Tv = onehot(ids[1:S + 1])        # next-char targets (teacher forcing over all S positions)
    cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)   # causal: position i sees <= i

    rng = np.random.default_rng(0)
    P = lambda sh, s=0.08: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    W_emb, W_pos, W_out = P((V, D)), P((S, D)), P((D, V))
    fin = agrad.parameter(np.ones((1, D), np.float32))       # final RMSNorm gain
    blocks = [{"Wq": P((D, D)), "Wk": P((D, D)), "Wv": P((D, D)), "Wo": P((D, D)),
                   "Wg": P((D, FF)), "Wu": P((D, FF)), "Wd": P((FF, D)),
                   "rn1": agrad.parameter(np.ones((1, D), np.float32)),
                   "rn2": agrad.parameter(np.ones((1, D), np.float32))}
              for _ in range(NLAYERS)]
    params = [W_emb, W_pos, W_out, fin] + [p for b in blocks for p in b.values()]
    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])

    def forward(x, mask):
        h = x @ W_emb + W_pos                                # one-hot embedding + learned position
        for b in blocks:
            xn = h.rms_norm(b["rn1"])                        # trainable RMSNorm gain
            q, k, v = heads(xn @ b["Wq"]), heads(xn @ b["Wk"]), heads(xn @ b["Wv"])
            sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
            h = h + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ b["Wo"]
            hn = h.rms_norm(b["rn2"])
            h = h + ((hn @ b["Wg"]).silu() * (hn @ b["Wu"])) @ b["Wd"]   # SwiGLU
        return h.rms_norm(fin) @ W_out                       # logits [S, V]

    x = af.input((S, V)); y = af.input((S, V))
    mask = af.input((S, S)); mask.attrs["value"] = cmask
    logits = forward(x, mask)

    tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, y), params, lr=LR,
                       loss_scale=1024.0, optimizer="adam", data_inputs={x: Xv, y: Tv})
    print(f"char-LM: {NLAYERS} causal LLaMA-style layers, D={D}, vocab={V}, "
          f"{len(params)} parameter tensors; forward + backward + optimizer on the ANE")
    print(f"\n{'step':>6} | {'cross-entropy':>13}")
    l0 = tr.loss()
    for it in range(STEPS):
        tr.step()
        if it % 50 == 0 or it == STEPS - 1:
            print(f"{it+1:>6} | {tr.loss():>13.4f}")
    lf = tr.loss()
    print(f"\ncross-entropy {l0:.4f} -> {lf:.4f}; the model trained end to end on the engine")

    # Next-token accuracy on the training window (the trained logits, argmax per position).
    logit = np.asarray(tr._fwd(*tr._feed(tr._fwd)))
    acc = float((logit.argmax(-1) == ids[1:S + 1]).mean())
    print(f"next-char accuracy over the {S} positions: {acc*100:.1f}%")

    # Greedy autoregressive generation. The model uses absolute positional embeddings
    # over a fixed S-position context, so generation reconstructs the sequence at its
    # true positions: place the prefix at positions 0..k-1, right-pad, and read the
    # logit at the last real position (which predicts the next character). Reuses the
    # trained forward program by feeding each window through its `x` input.
    def generate(seed):
        buf = [stoi[c] for c in seed]
        while len(buf) < S:
            w = buf + [0] * (S - len(buf))
            tr.data[x] = onehot(np.array(w, np.int64))
            out = np.asarray(tr._fwd(*tr._feed(tr._fwd)))
            buf.append(int(out[len(buf) - 1].argmax()))
        return "".join(itos[i] for i in buf)

    print("\nseed 'the apple' ->")
    print("  " + repr(generate("the apple")))
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
