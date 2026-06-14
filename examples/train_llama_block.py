"""Train a LLaMA-style transformer block on the Apple Neural Engine - RMSNorm
pre-normalization and a SwiGLU feed-forward, forward AND backward on the engine.

This exercises the two gradients that modern LLM blocks depend on and that previously
had no backward rule: `rms_norm` and `silu` (the SwiGLU gate). The gradient flows
through both to the trainable attention projections and the three SwiGLU weights, and
the block converges. The RMSNorm affine (gamma) is held at its unit initialization - the norm VJP differentiates the input, not the affine.

The block is the standard LLaMA layer: ``h = x + attn(rms_norm(x))`` then
``out = h + swiglu(rms_norm(h))`` where ``swiglu(z) = (silu(z @ Wg) * (z @ Wu)) @ Wd``.
The RMSNorm gains are trainable parameters: passing a parameter Tensor for gamma
composes the normalize op with a learnable scale, so the gains learn too.

    python3 examples/train_llama_block.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad

S, D, HEADS, dh, FF = 8, 16, 4, 4, 64
STEPS, LR = 300, 0.01


def main():
    rng = np.random.default_rng(1)
    Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
    Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)
    P = lambda sh, s=0.2: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
    Wg, Wu, Wd = P((D, FF)), P((D, FF)), P((FF, D))                  # SwiGLU gate / up / down
    rn1 = agrad.parameter(np.ones((1, D), np.float32))              # trainable RMSNorm gains
    rn2 = agrad.parameter(np.ones((1, D), np.float32))
    params = [Wq, Wk, Wv, Wo, Wg, Wu, Wd, rn1, rn2]

    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])   # [H, S, dh]
    x, y = af.input((S, D)), af.input((S, D))

    xn = x.rms_norm(rn1)                                             # RMSNorm pre-attention
    q, k, v = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
    scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
    h = x + (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo     # residual 1
    hn = h.rms_norm(rn2)                                             # RMSNorm pre-FFN
    out = h + ((hn @ Wg).silu() * (hn @ Wu)) @ Wd                    # SwiGLU + residual 2

    tr = agrad.Trainer(agrad.mse(out, y), params, lr=LR, loss_scale=1024.0,
                       optimizer="adam", data_inputs={x: Xv, y: Tgt})

    print("LLaMA block (rms_norm -> mha -> residual -> rms_norm -> SwiGLU -> residual); "
          "forward + backward on the ANE, RMSNorm gains + SiLU trained end-to-end")
    print(f"\n{'step':>6} | {'loss':>10}")
    l0 = tr.loss()
    for it in range(STEPS):
        tr.step()
        if it % 50 == 0 or it == STEPS - 1:
            print(f"{it+1:>6} | {tr.loss():>10.5f}")
    print(f"\nloss {l0:.4f} -> {tr.loss():.5f}; RMSNorm + SwiGLU train end-to-end on the engine")
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
