"""Train a transformer block (multi-head self-attention + residual + MLP) FULLY on the
Apple Neural Engine, with K Adam steps UNROLLED into ONE fused program.

Attention is differentiable end to end on the engine - the gradient flows through
softmax, the batched score/value matmuls, and the head split/merge transposes, all of
which have vjps. The projections are trainable `af.parameter` weights (not `af.mha`,
which bakes const weights). The task is a fixed full-batch sequence regression, so this
uses the `af.adam_step` helper to unroll the steps directly (the minibatch
`af.UnrolledTrainer` fits the classification demos; this one trains one sequence).

Because the data is full-batch (fixed), this goes one step further than the classifier
demos: the data, the learning rate, AND the optimizer state all live on-device, so after
seeding once the training loop is just `prog.execute()` - the host feeds NOTHING per
dispatch (non-aliased input buffers persist across execute; state is share_buffer-aliased
output->input). The host's only per-dispatch action is issuing the dispatch itself.

    python3 examples/train_transformer.py
"""
import sys
from _common import f16   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge.autograd import backward, mse, adam_step
from aneforge._compile import compile_multi

S, D, HEADS, dh = 8, 16, 4, 4
K, DISPATCHES, LR = 6, 50, 0.01      # K steps unrolled per dispatch
B1, B2, EPS = 0.9, 0.999, 1e-8


def main():
    rng = np.random.default_rng(0)
    Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
    Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3)).astype(np.float32)

    Pp = lambda sh, s=0.2: af.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    Wq, Wk, Wv, Wo = Pp((D, D)), Pp((D, D)), Pp((D, D)), Pp((D, D))
    W1, b1 = Pp((D, 4 * D)), af.parameter(np.zeros((1, 4 * D), np.float32))
    W2, b2 = Pp((4 * D, D)), af.parameter(np.zeros((1, D), np.float32))
    P0 = [Wq, Wk, Wv, Wo, W1, b1, W2, b2]

    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])      # [H, S, dh]

    def block(W, inp):
        Wq, Wk, Wv, Wo, W1, b1, W2, b2 = W
        q, k, v = heads(inp @ Wq), heads(inp @ Wk), heads(inp @ Wv)
        scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)   # [H,S,S]
        attn = (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
        h = inp + attn                                                  # residual 1
        ff = (((h @ W1) + b1).gelu() @ W2) + b2
        return h + ff                                                   # residual 2

    # UNROLL K steps into one program; one fixed (x, y) feeds every step (full batch).
    x, y = af.input((S, D)), af.input((S, D))
    m_in = [af.input(p.shape) for p in P0]
    v_in = [af.input(p.shape) for p in P0]
    lr_ins = [af.input((1, 1)) for _ in range(K)]
    P, M, V, losses = list(P0), list(m_in), list(v_in), []
    for kk in range(K):
        loss = mse(block(P, x), y)
        losses.append(loss)
        g = backward(loss, P, loss_scale=1024.0)
        P, M, V = adam_step(P, M, V, g, lr_ins[kk], (B1, B2), EPS)
    loss_row = af.concat([l.reshape(1, 1) for l in losses], axis=1)
    net = compile_multi([loss_row, *P, *M, *V])
    prog = net.prog
    inm = {id(t): n for t, n in net.input_ports}
    om = dict(net.output_ports)

    # SEED ONCE, then the host does nothing but press "go". Everything the program
    # needs lives on-device across dispatches:
    #   - optimizer state (params, m, v): each updated output is share_buffer-aliased
    #     onto its own input port (resident across dispatches), seeded once;
    #   - the fixed full-batch data (x, y) and the learning rate: plain inputs SET ONCE
    # - non-aliased input buffers persist across execute(), so they need no re-feed.
    # (Full-batch is what makes this host-free: a shuffled MINIBATCH would change each
    # step, so its data would be the one thing the host must still feed.)
    for out_t, in_t in (list(zip(P, P0)) + list(zip(M, m_in)) + list(zip(V, v_in))):
        prog.share_buffer(0, om[out_t], 0, inm[id(in_t)])
    for p, mi, vi in zip(P0, m_in, v_in):
        prog.set_input(inm[id(p)], p.attrs["value"].astype(f16))
        prog.set_input(inm[id(mi)], np.zeros(p.shape, f16))
        prog.set_input(inm[id(vi)], np.zeros(p.shape, f16))
    prog.set_input(inm[id(x)], Xv.astype(f16))
    prog.set_input(inm[id(y)], Tgt.astype(f16))
    for lt in lr_ins:
        prog.set_input(inm[id(lt)], np.full((1, 1), LR, f16))     # constant lr, set once

    print(f"transformer block (mha + residual + MLP), {K} steps unrolled into ONE ANE "
          f"program; data + lr + optimizer state ALL resident on-device")
    print("the training loop is just `prog.execute()` - the host feeds NOTHING per dispatch")
    print(f"\n{'dispatch':>9} | {'steps':>6} | {'loss':>10}")
    for it in range(DISPATCHES):
        prog.execute()                                            # no set_input - host-free
        if it % 10 == 0 or it == DISPATCHES - 1:
            loss = prog.read_output(om[loss_row]).ravel()[0]      # a read, only for display
            print(f"{it+1:>9} | {(it+1)*K:>6} | {loss:>10.4f}")
    print(f"\ntrained entirely on the ANE - {DISPATCHES*K} steps in {DISPATCHES} dispatches, "
          f"host issued only execute() (no per-dispatch feed)")
    net.release()


if __name__ == "__main__":
    sys.exit(main())
