"""Broad shape / numerical-edge corpus - fp32-golden cases that sweep the
shape-sensitive, numerically-risky ops across a wide range (including large and
edge shapes), so shape-dependent and fp16-cancellation bugs surface in the gate.

Motivation: the native-SDPA accuracy cliff at large sequence length (relerr ~1.0
vs fp32 at S=4096) slipped through because every existing case used small,
moderate shapes. Each case here validates the ANE output against a numpy fp32
golden (the TRUE answer), not a sibling ANE implementation, so a native-layer or
accumulation breakage shows up as a relerr failure rather than hiding.

Documented hardware walls are marked ``xfail=<reason>`` so they record the limit
without turning the gate red; an unexpected pass becomes XPASS (a limit to revisit)
and an unexpected fail is a newly-caught bug. Folded into ALL_CASES (run_corpus.py).
"""
from __future__ import annotations

import numpy as np

import aneforge as af
from _corpus import Case

rng = np.random.default_rng(7)


def f16(*shape, scale=1.0, pos=False):
    a = rng.standard_normal(shape).astype(np.float32) * scale
    if pos:
        a = np.abs(a) + 0.5
    return a.astype(np.float16)


# ---- numpy fp32 golden references --------------------------------------------
def np_sdpa(q, k, v, scale):
    q, k, v = (a.astype(np.float32) for a in (q, k, v))
    s = (q @ k.transpose(0, 1, 3, 2)) * scale
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(-1, keepdims=True)) @ v


def np_conv(x, w, pad=1, stride=1):
    x = x.astype(np.float32); w = w.astype(np.float32)
    Cout, _, kH, kW = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Hout = (x.shape[2] - kH) // stride + 1
    Wout = (x.shape[3] - kW) // stride + 1
    out = np.zeros((x.shape[0], Cout, Hout, Wout), np.float32)
    for i in range(Hout):
        for j in range(Wout):
            patch = x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def np_group_norm(x, g, b, groups, eps=1e-5):
    x = x.astype(np.float32); N, C, H, W = x.shape
    xg = x.reshape(N, groups, C // groups, H, W)
    xg = (xg - xg.mean((2, 3, 4), keepdims=True)) / np.sqrt(xg.var((2, 3, 4), keepdims=True) + eps)
    x = xg.reshape(N, C, H, W)
    return x * g.astype(np.float32).reshape(1, C, 1, 1) + b.astype(np.float32).reshape(1, C, 1, 1)


def np_layer_norm(x, g, b, eps=1e-5):
    x = x.astype(np.float32)
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g.astype(np.float32) + b.astype(np.float32)


def np_softmax(x, axis=-1):
    x = x.astype(np.float32); x = x - x.max(axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis, keepdims=True)


# ---- A. SDPA across sequence length (the native-accuracy envelope) -----------
def _sdpa_case(S, H=4, D=64):
    sc = 1.0 / D ** 0.5
    q, k, v = f16(1, H, S, D), f16(1, H, S, D), f16(1, H, S, D)
    return Case(f"sdpa_S{S}", "shape",
                lambda qt, kt, vt: af.sdpa(qt, kt, vt),
                lambda qa, ka, va: np_sdpa(qa, ka, va, sc),
                [q, k, v], tol=0.03)


# ---- B. matmul contraction across K (wide accumulator should hold) -----------
def _matmul_case(K, N=256):
    x = f16(1, K, scale=1.0); W = f16(N, K, scale=1.0 / np.sqrt(K))
    return Case(f"matmul_K{K}", "shape",
                lambda xt: xt.linear(W, None),
                lambda xa: xa.astype(np.float32) @ W.astype(np.float32).T,
                [x], tol=0.03)


# ---- C. conv across channels x spatial ---------------------------------------
def _conv_case(C, HW, Cout=None):
    Cout = Cout or C
    x = f16(1, C, HW, HW, scale=1.0); W = f16(Cout, C, 3, 3, scale=1.0 / np.sqrt(C * 9))
    return Case(f"conv_C{C}_{HW}x{HW}", "shape",
                lambda xt: af.conv(xt, W, pad=1),
                lambda xa: np_conv(xa, W, pad=1),
                [x], tol=0.04)


# ---- D. group_norm across feature-map (documented large-map wall) ------------
def _gn_case(C, HW, groups, xfail=""):
    g = f16(C, pos=True); b = f16(C, scale=0.1); x = f16(1, C, HW, HW)
    return Case(f"group_norm_C{C}_{HW}x{HW}", "shape",
                lambda xt: xt.group_norm(g, b, num_groups=groups),
                lambda xa: np_group_norm(xa, g, b, groups),
                [x], tol=0.03, xfail=xfail)


# ---- E. layer_norm across width ----------------------------------------------
def _ln_case(D):
    g = f16(D, pos=True); b = f16(D, scale=0.1); x = f16(8, D)
    return Case(f"layer_norm_D{D}", "shape",
                lambda xt: xt.layer_norm(g, b),
                lambda xa: np_layer_norm(xa, g, b),
                [x], tol=0.02)


# ---- F. softmax / reduction over a long axis (fp16 accumulation) -------------
def _softmax_case(N):
    x = f16(4, N, scale=2.0)
    return Case(f"softmax_N{N}", "shape",
                lambda xt: xt.softmax(-1),
                lambda xa: np_softmax(xa, -1),
                [x], tol=0.02)


def _reduce_case(N):
    # pos=True offsets the input away from 0 so the reference mean isn't ~0; a
    # zero-centered mean of N samples is a knife-edge for *relative* error (tiny
    # denominator), which made this case sensitive to the module rng stream position.
    x = f16(1, N, scale=1.0, pos=True)
    return Case(f"reduce_mean_N{N}", "shape",
                lambda xt: xt.mean((1,)),
                lambda xa: xa.astype(np.float32).mean(1, keepdims=True),  # ANE keeps the reduced dim
                [x], tol=0.02)


CASES = [
    _sdpa_case(256), _sdpa_case(1024), _sdpa_case(2048), _sdpa_case(4096),
    _matmul_case(512), _matmul_case(2048), _matmul_case(5632),
    _conv_case(64, 32), _conv_case(256, 32), _conv_case(512, 64),
    _gn_case(32, 32, 8), _gn_case(256, 32, 32), _gn_case(512, 64, 32),
    _gn_case(512, 128, 32),   # rank-4 tiling: D=16, H*W=16384 both fit the cap (SD-1.5 512ch@128)
    _gn_case(640, 64, 32),    # rank-4 tiling: D=20, H*W=4096 both fit (SD-1.5 640ch@64)
    _ln_case(256), _ln_case(1024), _ln_case(4096),
    _softmax_case(1024), _softmax_case(4096),
    _reduce_case(1024), _reduce_case(8192),
]
