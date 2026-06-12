"""Synthetic cases for the aneforge corpus: random elementwise+activation chains,
numpy-style broadcasts, reshape/transpose chains that cancel, reductions along
various axes, and matmul/bmm rectangles. These stress the algebraic identities a
graph optimizer is most likely to touch.
"""
from __future__ import annotations

import numpy as np

from _corpus import Case, run_corpus  # noqa: E402
import aneforge as af  # noqa: E402

rng = np.random.default_rng(11)


def f16(*shape, scale=1.0, pos=False):
    a = rng.standard_normal(shape).astype(np.float32) * scale
    if pos:
        a = np.abs(a) + 0.5
    return a.astype(np.float16)


# random op-chain (elementwise + activations), broadcasts
def _elementwise_chain():
    a = f16(4, 8); b = f16(4, 8); c = f16(4, 8, pos=True)

    def build(at, bt, ct):
        h = (at + bt) * ct
        h = af.maximum(h, bt)
        h = af.minimum(h, ct)
        h = (h - at) / ct
        return h.tanh()

    def ref(aa, ba, ca):
        h = (aa + ba) * ca
        h = np.maximum(h, ba)
        h = np.minimum(h, ca)
        h = (h - aa) / ca
        return np.tanh(h)

    return Case("elementwise_chain", "synthetic", build, ref, [a, b, c], tol=0.02)


def _activation_zoo():
    x = f16(4, 8)

    def build(xt):
        h = xt.silu() + xt.gelu()
        h = h.sigmoid() * xt.relu6()
        h = h.elu(0.5) + xt.leaky_relu(0.1)
        return h.clip(-2, 2).square()

    def ref(xa):
        silu = xa / (1 + np.exp(-xa))
        import math
        gelu = 0.5 * xa * (1 + np.vectorize(math.erf)(xa / np.sqrt(2)))
        h = silu + gelu
        h = (1 / (1 + np.exp(-h))) * np.clip(xa, 0, 6)
        h = np.where(h > 0, h, 0.5 * (np.exp(h) - 1)) + np.where(xa > 0, xa, 0.1 * xa)
        return np.clip(h, -2, 2) ** 2

    return Case("activation_zoo", "synthetic", build, ref, [x], tol=0.02)


def _broadcast_rowvec():
    # [M, D] + [1, D] and * [1, D]: classic numpy broadcast the optimizer must keep.
    x = f16(6, 12); row = f16(1, 12); col_scale = 2.0

    def build(xt, rt):
        h = xt + rt
        h = h * rt
        return (h * col_scale).relu()

    def ref(xa, ra):
        h = xa + ra
        h = h * ra
        return np_relu(h * col_scale)

    return Case("broadcast_rowvec", "synthetic", build, ref, [x, row], tol=0.02)


def _broadcast_3d():
    # [B, M, D] + [1, 1, D]
    x = f16(2, 5, 8); bias = f16(1, 1, 8)

    def build(xt, bt):
        return (xt + bt).silu()

    def ref(xa, ba):
        h = xa + ba
        return h / (1 + np.exp(-h))

    return Case("broadcast_3d_bias", "synthetic", build, ref, [x, bias], tol=0.02)


def np_relu(x):
    return np.maximum(x, 0.0)


# reshape/transpose chains that cancel back to identity
def _reshape_cancel():
    x = f16(2, 3, 4)

    def build(xt):
        h = xt.reshape(6, 4).reshape(2, 3, 4)
        h = h.transpose([2, 0, 1]).transpose([1, 2, 0])  # back to original layout
        return h + xt

    def ref(xa):
        h = xa.reshape(6, 4).reshape(2, 3, 4)
        h = h.transpose(2, 0, 1).transpose(1, 2, 0)
        return h + xa

    return Case("reshape_transpose_cancel", "synthetic", build, ref, [x], tol=0.01)


def _transpose_matmul():
    # transpose then matmul (column-major-ish feed)
    x = f16(4, 6); W = f16(4, 5, scale=0.3)  # x.T is [6,4] @ [4,5]

    def build(xt):
        return xt.transpose([1, 0]) @ W

    def ref(xa):
        return xa.T @ W.astype(np.float32)

    return Case("transpose_then_matmul", "synthetic", build, ref, [x], tol=0.02)


# reductions along various axes
def _reduce_axes():
    x = f16(3, 4, 5)

    def build(xt):
        s = xt.sum(2)                 # [3,4,1]
        m = xt.mean((1,))             # [3,1,5]
        return s + m                  # broadcast [3,4,5]

    def ref(xa):
        s = xa.sum(2, keepdims=True)
        m = xa.mean(1, keepdims=True)
        return s + m

    return Case("reduce_sum_mean_axes", "synthetic", build, ref, [x], tol=0.02)


def _reduce_minmax():
    x = f16(4, 16)

    def build(xt):
        return af.maximum(xt.amax(1), xt.amin(1) * -1.0)

    def ref(xa):
        return np.maximum(xa.max(1, keepdims=True), xa.min(1, keepdims=True) * -1.0)

    return Case("reduce_amax_amin", "synthetic", build, ref, [x], tol=0.02)


def _reduce_then_normalize():
    x = f16(5, 10)

    def build(xt):
        mu = xt.mean(1)            # [5,1]
        centered = xt - mu        # broadcast
        return centered.softmax(-1)

    def ref(xa):
        mu = xa.mean(1, keepdims=True)
        c = xa - mu
        c = c - c.max(-1, keepdims=True)
        e = np.exp(c)
        return e / e.sum(-1, keepdims=True)

    return Case("center_softmax", "synthetic", build, ref, [x], tol=0.02)


# matmul / bmm rectangles
def _matmul_rect():
    x = f16(7, 13); W = f16(13, 5, scale=0.3)

    def build(xt):
        return xt @ W

    def ref(xa):
        return xa @ W.astype(np.float32)

    return Case("matmul_rect_7x13x5", "synthetic", build, ref, [x], tol=0.02, int8_ok=True)


def _bmm_rect():
    a = f16(3, 6, 9, scale=0.5); b = f16(3, 9, 4, scale=0.5)

    def build(at, bt):
        return at @ bt

    def ref(aa, ba):
        return aa @ ba

    return Case("bmm_rect_3x6x9x4", "synthetic", build, ref, [a, b], tol=0.02)


def _linear_chain():
    D0, D1, D2 = 12, 20, 7
    W1 = f16(D1, D0, scale=0.2); b1 = f16(D1, scale=0.1)
    W2 = f16(D2, D1, scale=0.2); b2 = f16(D2, scale=0.1)
    x = f16(5, D0)

    def build(xt):
        return xt.linear(W1, b1).relu().linear(W2, b2)

    def ref(xa):
        h = np_relu(xa @ W1.astype(np.float32).T + b1.astype(np.float32))
        return h @ W2.astype(np.float32).T + b2.astype(np.float32)

    return Case("linear_relu_linear", "synthetic", build, ref, [x], tol=0.03, int8_ok=True)


# new primitives: extra activations / reductions / structural / select
def _extra_activation_zoo():
    x = f16(4, 8)

    def build(xt):
        h = xt.softsign() + xt.atan()
        h = h.scaled_tanh(1.5, 0.5) + xt.threshold(0.0)
        h = h.thresholded_relu(0.1) + xt.linear_activation(0.5, 0.25)
        h = h.sigmoid_hard(0.2, 0.5) + xt.clamped_relu(0.1, 4.0)
        return h.exp2()

    def ref(xa):
        h = xa / (1 + np.abs(xa)) + np.arctan(xa)
        h = 1.5 * np.tanh(0.5 * h) + np.maximum(xa, 0.0)
        h = np.where(h >= 0.1, h, 0.0) + (0.5 * xa + 0.25)
        a02 = float(np.float16(0.2))
        sh = np.minimum(np.maximum(a02 * h + 0.5, 0.0), 1.0)
        a01 = float(np.float16(0.1))
        cr = np.where(xa >= 0.0, np.minimum(xa, 4.0), np.minimum(4.0, a01 * xa))
        return np.exp2(sh + cr)

    return Case("extra_activation_zoo", "synthetic", build, ref, [x], tol=0.03)


def _inverse_recip():
    x = f16(4, 8, pos=True)  # strictly positive -> reciprocal is well-defined

    def build(xt):
        return xt.inverse() + xt.sqrt().inverse()

    def ref(xa):
        return 1.0 / xa + 1.0 / np.sqrt(xa)

    return Case("inverse_recip", "synthetic", build, ref, [x], tol=0.02)


def _extra_reductions():
    x = f16(4, 8)

    def build(xt):
        return xt.l1_norm(1) + xt.sum_square(1)

    def ref(xa):
        return np.abs(xa).sum(1, keepdims=True) + (xa ** 2).sum(1, keepdims=True)

    return Case("reduce_l1_sumsq", "synthetic", build, ref, [x], tol=0.02)


def _log_sum_reduce():
    x = f16(4, 8, pos=True)

    def build(xt):
        return xt.log_sum(1)

    def ref(xa):
        return np.log(xa.sum(1, keepdims=True))

    return Case("reduce_log_sum", "synthetic", build, ref, [x], tol=0.02)


def _squeeze_expand_roundtrip():
    x = f16(1, 4, 1, 8)

    def build(xt):
        h = xt.squeeze([0, 2])            # (4, 8)
        h = h.expand_dims(1)              # (4, 1, 8)
        return h.flatten2d(2) * 2.0       # (4, 8)

    def ref(xa):
        h = xa.squeeze((0, 2))
        h = np.expand_dims(h, 1)
        return h.reshape(4, 8) * 2.0

    return Case("squeeze_expand_flatten2d", "synthetic", build, ref, [x], tol=0.01)


def _slice_by_size_window():
    x = f16(4, 8)

    def build(xt):
        return xt.slice_by_size([1, 2], [2, 5]).relu()

    def ref(xa):
        return np_relu(xa[1:3, 2:7])

    return Case("slice_by_size_window", "synthetic", build, ref, [x], tol=0.01)


def _stack_split():
    a = f16(4, 8); b = f16(4, 8)

    def build(at, bt):
        st = af.stack([at, bt], 0)        # (2, 4, 8)
        lo, hi = af.split(st, 2, 0)       # each (1, 4, 8)
        return (lo + hi).reshape(4, 8)

    def ref(aa, ba):
        st = np.stack([aa, ba], 0)
        return (st[:1] + st[1:]).reshape(4, 8)

    return Case("stack_split", "synthetic", build, ref, [a, b], tol=0.01)


def _select_greater():
    a = f16(4, 8); b = f16(4, 8)

    def build(at, bt):
        return af.select(at.greater(bt), at, bt)   # elementwise max via select

    def ref(aa, ba):
        return np.where(aa > ba, aa, ba)

    # select copies input values verbatim (no arithmetic) -> bit-exact vs the fp16 ref
    return Case("select_greater_max", "synthetic", build, ref, [a, b], exact=True)


CASES = [
    _elementwise_chain(),
    _activation_zoo(),
    _broadcast_rowvec(),
    _broadcast_3d(),
    _reshape_cancel(),
    _transpose_matmul(),
    _reduce_axes(),
    _reduce_minmax(),
    _reduce_then_normalize(),
    _matmul_rect(),
    _bmm_rect(),
    _linear_chain(),
    _extra_activation_zoo(),
    _inverse_recip(),
    _extra_reductions(),
    _log_sum_reduce(),
    _squeeze_expand_roundtrip(),
    _slice_by_size_window(),
    _stack_split(),
    _select_greater(),
]


if __name__ == "__main__":
    import sys
    _, code = run_corpus(CASES)
    sys.exit(code)
