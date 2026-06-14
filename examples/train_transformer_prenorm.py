"""Train a PRE-NORM transformer block on the Apple Neural Engine, with LayerNorm
applied before attention and before the MLP - forward AND backward on the engine.

This is the end-to-end demonstration that LayerNorm no longer blocks training: the
gradient flows through `layer_norm` to the trainable attention projections and MLP
weights, and the block converges. The LayerNorm affine here is also trainable: passing
parameter Tensors for gamma/beta composes the normalize op with a learnable scale and
shift, so the norm's own affine learns alongside the projections.

Compare with `train_transformer.py`, which trains the same shape with plain residual
additions and no normalization.

    python3 examples/train_transformer_prenorm.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad

S, D, HEADS, dh = 8, 16, 4, 4
STEPS, LR = 300, 0.01


def main():
    rng = np.random.default_rng(0)
    Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
    Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)

    P = lambda sh, s=0.2: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
    W1, b1 = P((D, 4 * D)), agrad.parameter(np.zeros((1, 4 * D), np.float32))
    W2, b2 = P((4 * D, D)), agrad.parameter(np.zeros((1, D), np.float32))
    # trainable LayerNorm affine (gamma/beta as parameters, one per norm site)
    ln1g = agrad.parameter(np.ones((1, D), np.float32)); ln1b = agrad.parameter(np.zeros((1, D), np.float32))
    ln2g = agrad.parameter(np.ones((1, D), np.float32)); ln2b = agrad.parameter(np.zeros((1, D), np.float32))
    params = [Wq, Wk, Wv, Wo, W1, b1, W2, b2, ln1g, ln1b, ln2g, ln2b]

    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])   # [H, S, dh]
    x, y = af.input((S, D)), af.input((S, D))

    xn = x.layer_norm(ln1g, ln1b)                                   # PRE-NORM (attention)
    q, k, v = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
    scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
    h = x + (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo     # residual 1
    hn = h.layer_norm(ln2g, ln2b)                                   # PRE-NORM (MLP)
    out = h + ((((hn @ W1) + b1).gelu() @ W2) + b2)                  # residual 2

    tr = agrad.Trainer(agrad.mse(out, y), params, lr=LR, loss_scale=1024.0,
                       optimizer="adam", data_inputs={x: Xv, y: Tgt})

    print("pre-norm transformer block (layer_norm -> mha -> residual -> layer_norm -> "
          "MLP -> residual); forward + backward on the ANE, LayerNorm affine trained too")
    print(f"\n{'step':>6} | {'loss':>10}")
    l0 = tr.loss()
    for it in range(STEPS):
        tr.step()
        if it % 50 == 0 or it == STEPS - 1:
            print(f"{it+1:>6} | {tr.loss():>10.5f}")
    print(f"\nloss {l0:.4f} -> {tr.loss():.5f}; LayerNorm trains end-to-end on the engine")
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
