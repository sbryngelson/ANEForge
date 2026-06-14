"""Train a character-level language model on a corpus on the Apple Neural Engine, and
show it GENERALIZES - held-out next-character accuracy well above the unigram baseline,
not memorization of a single window.

This is the data-scale companion to `train_charlm.py` (which overfits one window as a
sanity check). Here the 4-layer causal LLaMA-style model trains on random windows drawn
from the training split of a structured corpus and is evaluated on a disjoint validation
split it never trains on. Forward, backward, and the optimizer step all run on the engine.

Same reachable-surface choices as `train_charlm.py`: one-hot matmul embedding (the
embedding gradient is then a matmul gradient, sidestepping the gather/scatter wall), and
an additive causal mask before softmax. Each window is re-based to absolute positions
0..S-1, so the learned positional embeddings apply to every sampled window.

    python3 examples/train_charlm_corpus.py
"""
import sys
from collections import Counter
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad

S, D, HEADS, dh, NLAYERS, FF = 64, 64, 4, 16, 4, 256
STEPS, LR = 600, 0.003


def make_corpus(rng):
    # a small structured grammar: "the <animal> <verb> <adverb>." The word choices are
    # random, so the model cannot memorize a single sequence; it must learn the grammar
    # to predict the deterministic parts (articles, spaces, word completions, period).
    animals, verbs, advs = ["cat", "dog", "bird"], ["runs", "jumps", "sleeps"], ["fast", "slow"]
    out = []
    for _ in range(120):
        out.append(f"the {rng.choice(animals)} {rng.choice(verbs)} {rng.choice(advs)}. ")
    return "".join(out)


def main():
    rng = np.random.default_rng(0)
    corpus = make_corpus(rng)
    chars = sorted(set(corpus)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in corpus], np.int64)
    n = len(ids); ntr = int(n * 0.85)
    print(f"corpus {n} chars, vocab {V}; train {ntr} / val {n - ntr}")

    def onehot(idx):
        o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
        return o
    cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)

    P = lambda sh, s=0.08: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    W_emb, W_pos, W_out = P((V, D)), P((S, D)), P((D, V))
    fin = agrad.parameter(np.ones((1, D), np.float32))
    blocks = [dict(Wq=P((D, D)), Wk=P((D, D)), Wv=P((D, D)), Wo=P((D, D)),
                   Wg=P((D, FF)), Wu=P((D, FF)), Wd=P((FF, D)),
                   rn1=agrad.parameter(np.ones((1, D), np.float32)),
                   rn2=agrad.parameter(np.ones((1, D), np.float32)))
              for _ in range(NLAYERS)]
    params = [W_emb, W_pos, W_out, fin] + [p for b in blocks for p in b.values()]
    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])

    x = af.input((S, V)); y = af.input((S, V))
    mask = af.input((S, S)); mask.attrs["value"] = cmask
    h = x @ W_emb + W_pos
    for b in blocks:
        xn = h.rms_norm(b["rn1"])
        q, k, v = heads(xn @ b["Wq"]), heads(xn @ b["Wk"]), heads(xn @ b["Wv"])
        sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
        h = h + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ b["Wo"]
        hn = h.rms_norm(b["rn2"])
        h = h + ((hn @ b["Wg"]).silu() * (hn @ b["Wu"])) @ b["Wd"]
    logits = h.rms_norm(fin) @ W_out

    tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, y), params, lr=LR,
                       loss_scale=1024.0, optimizer="adam",
                       data_inputs={x: onehot(ids[:S]), y: onehot(ids[1:S + 1])})

    def logits_for(window):
        tr.data[x] = onehot(window)
        return np.asarray(tr._fwd(*tr._feed(tr._fwd)))

    def val_accuracy():
        correct = total = 0
        for i in range(ntr, n - S - 1, S):
            pred = logits_for(ids[i:i + S]).argmax(-1)
            correct += int((pred == ids[i + 1:i + S + 1]).sum()); total += S
        return correct / total

    mfc = Counter(ids[1:ntr].tolist()).most_common(1)[0][0]
    baseline = float((ids[ntr + 1:n] == mfc).mean())

    print(f"\n{'step':>6} | {'train CE':>9} | {'val acc':>8}")
    for it in range(STEPS):
        i = int(rng.integers(0, ntr - S - 1))
        tr.data[x] = onehot(ids[i:i + S]); tr.data[y] = onehot(ids[i + 1:i + S + 1])
        tr.step()
        if it % 150 == 0 or it == STEPS - 1:
            print(f"{it+1:>6} | {tr.loss():>9.3f} | {val_accuracy()*100:>7.1f}%")
    print(f"\nheld-out next-char accuracy {val_accuracy()*100:.1f}% vs unigram baseline "
          f"{baseline*100:.1f}% - the model generalized, trained entirely on the engine")

    # greedy generation from a seed: one sentence of the learned grammar
    def generate(seed, n_new=20):
        buf = [stoi[c] for c in seed]
        for _ in range(n_new):
            if len(buf) >= S:
                break
            w = buf + [0] * (S - len(buf))
            buf.append(int(logits_for(np.array(w, np.int64))[len(buf) - 1].argmax()))
        return "".join(itos[i] for i in buf)
    print("\nseed 'the ' ->")
    print("  " + repr(generate("the ")))
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
