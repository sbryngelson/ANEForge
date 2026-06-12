"""Finite-difference VJP sweep over unary/elementwise ops.

Checks aneforge's on-ANE autograd gradients against numerical (central-difference)
gradients for a spread of ops (exp, sqrt, inverse, erf, clip, l2_norm, ...). Each op is
its own parametrized case. Moved here from the standalone op_validation/vjp_sweep.py script.
"""
import math

import numpy as np
import pytest

from aneforge import autograd as agrad
from test_autograd import _cos, _eval_grad_tensor

SH = (4, 8)


def _agrad_of(build_loss, xv, ls=256.0):
    x = agrad.parameter(xv)
    g = agrad.backward(build_loss(x), [x], loss_scale=ls)
    return _eval_grad_tensor(g[x], [x]).reshape(x.shape) / ls


def _fd(fwd, xv, wv, e=1e-2):
    G = np.zeros_like(xv)
    it = np.nditer(xv, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp = xv.copy(); xp[i] += e
        xm = xv.copy(); xm[i] -= e
        G[i] = ((fwd(xp) * wv).sum() - (fwd(xm) * wv).sum()) / (2 * e)
    return G


# (name, ane-build(x, w_param), host-forward, positive-domain?)
CASES = [
    ("exp", lambda x, w: (x.exp() * w).sum((0, 1)), np.exp, False),
    ("sqrt", lambda x, w: (x.sqrt() * w).sum((0, 1)), np.sqrt, True),
    ("rsqrt", lambda x, w: (x.rsqrt() * w).sum((0, 1)), lambda z: 1 / np.sqrt(z), True),
    ("inverse", lambda x, w: (x.inverse() * w).sum((0, 1)), lambda z: 1 / z, True),
    ("log", lambda x, w: (x.log() * w).sum((0, 1)), np.log, True),
    ("erf", lambda x, w: (x.erf() * w).sum((0, 1)), lambda z: np.vectorize(math.erf)(z), False),
    ("cos", lambda x, w: (x.cos() * w).sum((0, 1)), np.cos, False),
    ("abs", lambda x, w: (x.abs() * w).sum((0, 1)), np.abs, False),
    ("leaky_relu", lambda x, w: (x.leaky_relu(0.1) * w).sum((0, 1)),
     lambda z: np.where(z > 0, z, 0.1 * z), False),
    ("elu", lambda x, w: (x.elu(1.0) * w).sum((0, 1)),
     lambda z: np.where(z > 0, z, 1.0 * (np.exp(z) - 1)), False),
    ("relu6", lambda x, w: (x.relu6() * w).sum((0, 1)), lambda z: np.clip(z, 0, 6), False),
    ("clip", lambda x, w: (x.clip(-1.0, 1.0) * w).sum((0, 1)), lambda z: np.clip(z, -1, 1), False),
    ("scaled_tanh", lambda x, w: (x.scaled_tanh(1.5, 0.7) * w).sum((0, 1)),
     lambda z: 1.5 * np.tanh(0.7 * z), False),
    ("sigmoid_hard", lambda x, w: (x.sigmoid_hard(0.2, 0.5) * w).sum((0, 1)),
     lambda z: np.clip(0.2 * z + 0.5, 0, 1), False),
    ("l2_norm", lambda x, w: (x.l2_norm() * w).sum((0, 1)),
     lambda z: z / np.sqrt((z ** 2).sum(-1, keepdims=True) + 1e-12), False),
]


@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c[0] for c in CASES])
def test_vjp(idx):
    name, build, fwd, pos = CASES[idx]
    rng = np.random.default_rng(21 + idx)
    xv = (np.abs(rng.standard_normal(SH)) + 0.5 if pos else rng.standard_normal(SH)).astype(np.float32)
    wv = rng.standard_normal(SH).astype(np.float32)
    gx = _agrad_of(lambda x: build(x, agrad.parameter(wv)), xv)
    ref = _fd(fwd, xv, wv)
    c = _cos(gx, ref)
    assert c > 0.97, (name, c)
