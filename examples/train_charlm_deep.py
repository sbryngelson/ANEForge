"""Train a deep (16-layer) char-LM on the ANE with a layer-streamed (gradient-checkpointed) compile, so depth doesn't bound compile size. Run: python3 examples/train_charlm_deep.py"""
import sys
import time
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad
from aneforge._compile import compile_multi
from aneforge.streaming import CheckpointedStack

S, D, HEADS, dh, NLAYERS, FF = 32, 64, 4, 16, 16, 256
STEPS, LR = 120, 0.05
TEXT = "the apple neural engine trains a deep language model on itself. " * 2


def _val(t, vals):                          # fed value, else a baked constant's own value
    return vals[id(t)] if id(t) in vals else np.asarray(t.attrs["value"], np.float16)


def run(model, vals):                       # single-output Model, fed by tensor identity
    return np.asarray(model(*[_val(t, vals) for t in model._input_tensors]), np.float32)


def run_multi(mm, vals, outs):              # MultiModel -> list of named outputs
    for t, n in mm.input_ports:
        mm.prog.set_input(n, np.asarray(_val(t, vals), np.float16))
    mm.prog.execute()
    om = dict(mm.output_ports)
    return [np.asarray(mm.prog.read_output(om[o]), np.float32) for o in outs]


def main():
    chars = sorted(set(TEXT)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in TEXT], np.int64)

    def onehot(idx):
        o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
        return o
    Xv, Tgt = onehot(ids[:S]), ids[1:S + 1]
    cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)
    rng = np.random.default_rng(0)
    rnd = lambda sh, s=0.08: (rng.standard_normal(sh) * s).astype(np.float32)
    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])

    t0 = time.time()
    # embedding stage (compiled once): a0 = onehot @ W_emb + W_pos
    tok = af.input((S, V)); We = agrad.parameter(rnd((V, D))); Wp = agrad.parameter(rnd((S, D)))
    a0 = tok @ We + Wp
    embed_fwd = af.compile(a0)
    ga0 = af.input((S, D)); eg = agrad.backward_from(ga0, a0, [We, Wp])
    embed_bwd = compile_multi([eg[We], eg[Wp]])

    # the repeated transformer layer, compiled ONCE and reused for all NLAYERS
    mask = af.input((S, S)); mask.attrs["value"] = cmask

    def layer(p, x):
        Wq, Wk, Wv, Wo, Wg, Wu, Wd, rn1, rn2 = p
        xn = x.rms_norm(rn1)
        q, k, v = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
        sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
        h = x + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
        hn = h.rms_norm(rn2)
        return h + ((hn @ Wg).silu() * (hn @ Wu)) @ Wd

    ex = [rnd((D, D)), rnd((D, D)), rnd((D, D)), rnd((D, D)), rnd((D, FF)), rnd((D, FF)),
          rnd((FF, D)), np.ones((1, D), np.float32), np.ones((1, D), np.float32)]
    stack = CheckpointedStack(layer, ex, (S, D))
    layers = [[a.copy() for a in (rnd((D, D)), rnd((D, D)), rnd((D, D)), rnd((D, D)),
               rnd((D, FF)), rnd((D, FF)), rnd((FF, D)),
               np.ones((1, D), np.float32), np.ones((1, D), np.float32))] for _ in range(NLAYERS)]

    # output stage (compiled once): logits = rms_norm(aN, fin) @ W_out
    aN = af.input((S, D)); fin = agrad.parameter(np.ones((1, D), np.float32)); Wo_h = agrad.parameter(rnd((D, V)))
    logits = aN.rms_norm(fin) @ Wo_h
    head_fwd = af.compile(logits)
    glog = af.input((S, V)); hg = agrad.backward_from(glog, logits, [fin, Wo_h, aN])
    head_bwd = compile_multi([hg[fin], hg[Wo_h], hg[aN]])
    setup = time.time() - t0
    print(f"deep char-LM: {NLAYERS} layers, D={D}, vocab={V}; layer compiled ONCE and reused")
    print(f"setup compiled 6 programs (embed/layer/head x fwd+bwd) in {setup:.1f}s "
          f" - independent of the {NLAYERS} layers")

    # host parameters: embedding, head (per-layer params live in `layers`)
    Wev, Wpv, finv, Wohv = We.attrs["value"].copy(), Wp.attrs["value"].copy(), \
        fin.attrs["value"].copy(), Wo_h.attrs["value"].copy()

    print(f"\n{'step':>6} | {'cross-entropy':>13}")
    for it in range(STEPS):
        # forward
        a0v = run(embed_fwd, {id(tok): Xv.astype(np.float16), id(We): Wev.astype(np.float16),
                              id(Wp): Wpv.astype(np.float16)})
        aNv, ckpts = stack.forward(layers, a0v)
        lg = run(head_fwd, {id(aN): aNv.astype(np.float16), id(fin): finv.astype(np.float16),
                            id(Wo_h): Wohv.astype(np.float16)})
        # host-side softmax cross-entropy over the next-char targets
        m = lg - lg.max(-1, keepdims=True); e = np.exp(m); p = e / e.sum(-1, keepdims=True)
        ce = float(-np.log(p[np.arange(S), Tgt] + 1e-9).mean())
        g_logits = p.copy(); g_logits[np.arange(S), Tgt] -= 1.0; g_logits /= S
        # backward: head -> stack -> embed
        gfin, gWoh, gaN = run_multi(head_bwd, {id(aN): aNv.astype(np.float16),
                                    id(fin): finv.astype(np.float16), id(Wo_h): Wohv.astype(np.float16),
                                    id(glog): g_logits.astype(np.float16)}, [hg[fin], hg[Wo_h], hg[aN]])
        pgrads, ga0v = stack.backward(layers, ckpts, gaN)
        gWe, gWp = run_multi(embed_bwd, {id(tok): Xv.astype(np.float16), id(We): Wev.astype(np.float16),
                             id(Wp): Wpv.astype(np.float16), id(ga0): ga0v.astype(np.float16)}, [eg[We], eg[Wp]])
        # SGD over every parameter (compute on the engine; this update is host-side)
        Wev -= LR * gWe; Wpv -= LR * gWp; finv -= LR * gfin; Wohv -= LR * gWoh
        for li in range(NLAYERS):
            for j in range(len(layers[li])):
                layers[li][j] = layers[li][j] - LR * pgrads[li][j]
        if it % 20 == 0 or it == STEPS - 1:
            print(f"{it+1:>6} | {ce:>13.4f}")

    stack.release(); embed_fwd.release(); embed_bwd.release(); head_fwd.release(); head_bwd.release()
    print(f"\na {NLAYERS}-layer model trained on the engine with a depth-independent compile")


if __name__ == "__main__":
    sys.exit(main())
