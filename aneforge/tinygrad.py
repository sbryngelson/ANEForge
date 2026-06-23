"""Run an unmodified tinygrad model on the Apple Neural Engine.

`aneforge.tinygrad.trace(model, input_shape)` runs the model's forward once with a tracer
in place of the input. tinygrad and ANEForge are both lazy tensor graphs, so the tracer
maps the model's ops onto an ANEForge graph instead of computing; that graph compiles to
ONE e5rt program (Espresso, no CoreML) and runs on the engine. tinygrad's own ResNet-18/34/50
and the ViT-Ti/16 encoder trace this way - cosine 1.0000 vs tinygrad, 8-10x faster than its
JIT'd METAL on an M5 Pro (the same ANE program ANEForge benchmarks, so ~16x on energy too).

Sound by construction: any op the tracer does not map, or any compile failure, falls the
WHOLE model back to tinygrad, so correctness never depends on coverage. `run.on_ane` and
`run.reason` report which path ran. What falls back is a hardware limit (e.g. the ANE's
kernel width <= 15, so a 16x16 patch conv), an op the tracer has not mapped yet, or
something that cannot be a static graph at all (host reads, dynamic shapes).

tinygrad is an optional dependency; importing this module without it is fine, you just
cannot trace until it is installed.

    pip install "aneforge>=0.1.3" tinygrad
    from aneforge.tinygrad import trace
    run = trace(my_tinygrad_model, (1, 3, 224, 224))   # model is plain tinygrad
    y = run(x)                                         # x: a tinygrad Tensor; one ANE dispatch
"""
from __future__ import annotations

import math
import warnings

import numpy as np

import aneforge as af
from aneforge.graph import Tensor as _Node, concat

try:
    from tinygrad import Tensor as _TG, dtypes as _dtypes
except ImportError:                                   # tinygrad optional at import time
    _TG = _dtypes = None

TESTED_TINYGRAD = "0.11"

# no-arg ops dispatched in __getattr__: unary math maps to the matching ANEForge op, no-ops to identity.
_UNARY = set("relu gelu silu sigmoid tanh exp sqrt cos sin log abs square erf softplus sign floor ceil round".split())
_NOOP = set("cast float half contiguous realize detach dropout".split())


class ANEUnsupported(Exception):
    """Raised when the tracer meets an op it does not map; caught by `trace` to fall back."""


def _np(x) -> np.ndarray:
    if _TG is not None and isinstance(x, _TG):
        return x.numpy().astype(np.float32)
    return np.asarray(x, np.float32)


def _const(v, shape=None) -> _Node:
    """A baked constant (a `const_array` node, not an input port). scalar+shape broadcasts."""
    arr = (np.full((1,) * len(shape), float(v), np.float32) if shape is not None and isinstance(v, (int, float))
           else np.ascontiguousarray(v, np.float32))
    return _Node(arr.shape, "const_array", [], {"value": arr})


def _zeros(node, d, w) -> _Node:
    return _const(np.zeros([w if i == d else s for i, s in enumerate(node.shape)], np.float32))


def _u(v) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, (tuple, list)) and len(set(v)) == 1:
        return int(v[0])
    raise ANEUnsupported(f"non-uniform parameter {v}")


class ANETracer:
    """Wraps one ANEForge graph node and forwards tinygrad-style ops onto it."""

    def __init__(self, node: _Node):
        self.node = node

    @property
    def shape(self):
        return self.node.shape

    @property
    def ndim(self) -> int:
        return len(self.node.shape)

    @property
    def dtype(self):
        return _dtypes.float32 if _dtypes is not None else None   # the engine runs fp16/fp32

    def __getattr__(self, name):
        if name in _UNARY:
            return lambda: ANETracer(getattr(self.node, name)())
        if name in _NOOP:
            return lambda *a, **k: self
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ANEUnsupported(f"tinygrad op '{name}' not mapped by the ANE tracer")

    # contractions
    def conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None) -> "ANETracer":
        if groups != 1 or _u(dilation) != 1:
            raise ANEUnsupported("conv2d groups != 1 / dilation != 1")
        b = _np(bias) if bias is not None else None
        return ANETracer(af.conv(self.node, _np(weight), stride=_u(stride), pad=_u(padding), bias=b))

    def linear(self, weight, bias=None) -> "ANETracer":
        W = _np(weight)
        b = _np(bias) if bias is not None else None
        if W.ndim == 1:                                   # tinygrad linear with a 1-D weight is an affine
            return ANETracer((self.node * _const(W)) + _const(b) if b is not None else self.node * _const(W))
        return ANETracer(self.node.linear(np.ascontiguousarray(W.T), b))   # 2-D: x @ W (+ b)

    def dot(self, w, dtype=None) -> "ANETracer":
        return ANETracer(self.node @ (w.node if isinstance(w, ANETracer) else _np(w)))

    def matmul(self, w, **kw) -> "ANETracer":
        return self.dot(w)

    def __matmul__(self, w) -> "ANETracer":
        return self.dot(w)

    def scaled_dot_product_attention(self, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, enable_gqa=False) -> "ANETracer":
        if is_causal or enable_gqa or dropout_p:
            raise ANEUnsupported("sdpa causal / gqa / dropout")
        q, k, v = self.node, key.node, value.node
        *batch, T, d = q.shape
        Tk = k.shape[-2]
        B = math.prod(batch) if batch else 1              # collapse to a 3-D batched matmul
        scores = (q.reshape(B, T, d) @ k.reshape(B, Tk, d).transpose([0, 2, 1])) * (1.0 / d ** 0.5)
        if attn_mask is not None:
            m = attn_mask.node if isinstance(attn_mask, ANETracer) else _const(_np(attn_mask))
            scores = scores + m.reshape(math.prod(m.shape[:-2]) or 1, T, Tk)
        return ANETracer((scores.softmax(-1) @ v.reshape(B, Tk, d)).reshape(*batch, T, d))

    # unary math with args (the no-arg set relu/gelu/cos/sin/log/... is dispatched in __getattr__)
    def elu(self, alpha=1.0) -> "ANETracer":
        return ANETracer(self.node.elu(alpha))

    def clip(self, lo, hi) -> "ANETracer":
        return ANETracer(self.node.clip(lo, hi))

    def clamp(self, lo, hi) -> "ANETracer":
        return ANETracer(self.node.clip(lo, hi))

    def rsqrt(self, eps=0.0) -> "ANETracer":
        return ANETracer(self.node.adds(eps).rsqrt() if eps else self.node.rsqrt())

    # pooling
    def max_pool2d(self, kernel_size=2, stride=None, dilation=1, padding=0, **k) -> "ANETracer":
        return ANETracer(self.node.max_pool(_u(kernel_size), stride=None if stride is None else _u(stride), pad=_u(padding)))

    def avg_pool2d(self, kernel_size=2, stride=None, padding=0, **k) -> "ANETracer":
        return ANETracer(self.node.avg_pool(_u(kernel_size), stride=None if stride is None else _u(stride), pad=_u(padding)))

    # movement
    def flatten(self, start_dim=1) -> "ANETracer":
        return ANETracer(self.node.reshape(*self.node.shape[:start_dim], math.prod(self.node.shape[start_dim:])))

    def permute(self, order, *args) -> "ANETracer":
        return ANETracer(self.node.transpose(list((order, *args) if args else order)))

    def transpose(self, dim0=1, dim1=0) -> "ANETracer":
        perm = list(range(self.ndim))
        perm[dim0 % self.ndim], perm[dim1 % self.ndim] = dim1 % self.ndim, dim0 % self.ndim
        return ANETracer(self.node.transpose(perm))

    def reshape(self, *shape, **kw) -> "ANETracer":
        shape = list(kw["shape"] if "shape" in kw else
                     shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
        if -1 in shape:                                   # infer the free dim
            shape[shape.index(-1)] = math.prod(self.node.shape) // -math.prod(shape)
        return ANETracer(self.node.reshape(*shape))

    def cat(self, *others, dim=0) -> "ANETracer":
        nodes = [self.node] + [o.node if isinstance(o, ANETracer) else _const(_np(o)) for o in others]
        return ANETracer(concat(nodes, axis=dim))

    def chunk(self, n, dim=0) -> "list[ANETracer]":
        dim %= self.ndim
        step = self.node.shape[dim] // n
        out = []
        for i in range(n):
            begin = [i * step if j == dim else 0 for j in range(self.ndim)]
            size = [step if j == dim else s for j, s in enumerate(self.node.shape)]
            out.append(ANETracer(self.node.slice_by_size(begin, size)))
        return out

    def __getitem__(self, idx) -> "ANETracer":
        idx = idx if isinstance(idx, tuple) else (idx,)
        shp = self.node.shape
        begin, size, drop = [0] * self.ndim, list(shp), []
        for d, sl in enumerate(idx):
            if isinstance(sl, slice):
                if sl.step not in (None, 1):
                    raise ANEUnsupported("strided slice")
                start = (sl.start or 0) + (shp[d] if (sl.start or 0) < 0 else 0)
                stop = (shp[d] if sl.stop is None else sl.stop) + (shp[d] if (sl.stop or 0) < 0 else 0)
                begin[d], size[d] = start, stop - start
            elif isinstance(sl, int):                     # int index: select and drop the dim
                begin[d], size[d] = (sl + shp[d] if sl < 0 else sl), 1
                drop.append(d)
            else:
                raise ANEUnsupported("only slice / int indexing")
        node = self.node.slice_by_size(begin, size)
        return ANETracer(node.squeeze(drop) if drop else node)

    # reductions / normalization
    def _axes(self, axis):
        if axis is None:
            return tuple(range(self.ndim))
        return (axis,) if isinstance(axis, int) else tuple(axis)

    def mean(self, axis=None, keepdim=False) -> "ANETracer":
        return self._reduce(self.node.mean, axis, keepdim)

    def sum(self, axis=None, keepdim=False) -> "ANETracer":
        return self._reduce(self.node.sum, axis, keepdim)

    def _reduce(self, fn, axis, keepdim) -> "ANETracer":
        axes = self._axes(axis)
        r = fn(axes)                                      # ANEForge reductions keep dims
        if not keepdim:
            pos = {a % self.ndim for a in axes}
            r = r.reshape(*(tuple(s for i, s in enumerate(r.shape) if i not in pos) or (1,)))
        return ANETracer(r)

    def softmax(self, axis=-1, dtype=None) -> "ANETracer":
        return ANETracer(self.node.softmax(axis))

    def layernorm(self, axis=-1, eps=1e-5) -> "ANETracer":
        axes = self._axes(axis)
        xc = self.node - self.node.mean(axes)             # keepdim mean broadcasts
        return ANETracer(xc * (xc * xc).mean(axes).adds(eps).rsqrt())

    def batchnorm(self, weight, bias, mean, invstd, axis=1) -> "ANETracer":
        C = self.node.shape[axis]

        def bc(t):
            if isinstance(t, ANETracer):
                return t.node
            a = _np(t)
            return _const(a.reshape([C if i == axis else 1 for i in range(self.ndim)]) if a.ndim == 1 else a)
        x = (self.node - bc(mean)) * bc(invstd)
        if weight is not None:
            x = x * bc(weight)
        return ANETracer(x + bc(bias) if bias is not None else x)

    # pad / sequential / elementwise
    def pad(self, padding, mode="constant", value=0.0) -> "ANETracer":
        if mode != "constant" or value not in (0, 0.0):
            raise ANEUnsupported(f"pad mode={mode} value={value}")
        pairs = [[0, 0] for _ in range(self.ndim)]
        if padding and isinstance(padding[0], (tuple, list)):
            for d, p in enumerate(padding):
                pairs[d] = [(p or (0, 0))[0] or 0, (p or (0, 0))[1] or 0]
        else:                                             # flat torch-style, last dim first
            for i in range(0, len(padding), 2):
                pairs[self.ndim - 1 - i // 2] = [padding[i] or 0, padding[i + 1] or 0]
        node = self.node
        for d, (lo, hi) in enumerate(pairs):
            if lo:
                node = concat([_zeros(node, d, lo), node], axis=d)
            if hi:
                node = concat([node, _zeros(node, d, hi)], axis=d)
        return ANETracer(node)

    def sequential(self, layers) -> "ANETracer":
        x = self
        for layer in layers:
            x = layer(x) if layer is not None else x
        return x

    def _bin(self, o, op) -> "ANETracer":
        rhs = (o.node if isinstance(o, ANETracer)
               else _const(o, self.node.shape) if isinstance(o, (int, float)) else _const(_np(o)))
        return ANETracer(getattr(self.node, op)(rhs))

    def __add__(self, o): return self._bin(o, "__add__")
    def __mul__(self, o): return self._bin(o, "__mul__")
    def __sub__(self, o): return self._bin(o, "__sub__")
    def __truediv__(self, o): return self._bin(o, "__truediv__")
    def __neg__(self): return ANETracer(self.node * -1.0)
    __radd__ = __add__
    __rmul__ = __mul__
    def __rsub__(self, o): return ANETracer(_const(o, self.node.shape) - self.node)


def trace(model, input_shape, fallback: bool = True):
    """Trace `model` (a callable of one tensor) into one fused ANE program. On any unmapped op
    or compile failure (with `fallback`), the returned runner runs the model on tinygrad instead
    (`run.on_ane` is False, `run.reason` says why)."""
    prog = reason = None
    try:
        out = model(ANETracer(af.input(tuple(input_shape))))
        if not isinstance(out, ANETracer):
            raise ANEUnsupported("model did not return a single traced tensor")
        prog = af.compile(out.node)
    except Exception as e:                                # any failure -> fall back, never wrong
        if not fallback:
            raise
        reason = f"{type(e).__name__}: {e}"
        warnings.warn(f"aneforge.tinygrad: falling back to tinygrad ({reason})", RuntimeWarning, stacklevel=2)

    def run(x):
        try:
            y = np.asarray(prog(_np(x)), np.float32)
            return _TG(y) if _TG is not None else y
        except Exception as e:                            # runtime compile/shape issue -> fall back
            warnings.warn(f"aneforge.tinygrad: run fell back ({e})", RuntimeWarning, stacklevel=2)
            run.on_ane = False
            return model(x)
    if prog is None:
        def run(x): return model(x)
    run.on_ane, run.reason, run.program = prog is not None, reason, prog
    return run
