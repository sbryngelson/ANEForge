"""The aneforge compute graph: a lazy `Tensor` whose methods/operators record
structure (op + sources + attrs), plus the op constructors and the higher-level
neural-net helpers (conv, multi-head / cross attention, GEGLU). Nothing here
touches the device - `compile` (_compile.py) lowers the graph to one program.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _check_dtype(a: np.ndarray, where: str) -> None:
    if a.dtype not in (np.float16, np.float32, np.float64):
        raise TypeError(
            f"aneforge: {where} has dtype {a.dtype}; the ANE computes in fp16 only "
            f"(fp32/int32/bf16 are not implemented on the backend). Pass float weights."
        )


def _broadcast(sa: tuple, sb: tuple) -> tuple:
    """numpy-style broadcast shape (MIL add/sub/mul broadcast on the device)."""
    out = []
    for x, y in zip(reversed(sa), reversed(sb)):
        if not (x == y or x == 1 or y == 1):
            raise ValueError(f"shapes {sa} and {sb} are not broadcastable")
        out.append(max(x, y))
    longer = sa if len(sa) >= len(sb) else sb
    out += list(reversed(longer[: abs(len(sa) - len(sb))]))
    return tuple(reversed(out))


class Tensor:
    """A node in the compute graph."""

    __slots__ = ("shape", "op", "srcs", "attrs", "_name")

    def __init__(self, shape: Sequence[int], op: str, srcs: Sequence["Tensor"] = (),
                 attrs: dict | None = None) -> None:
        self.shape = tuple(int(d) for d in shape)
        # The ANE dimension model is rank <= 5 ("Rank ... must be between 0 and 5");
        # a rank-6+ tensor (reshape/stack/expand_dims past 5D) fails ANECCompile.
        # Guard at construction with a clear error, not a cryptic compile crash.
        if len(self.shape) > 5:
            raise ValueError(
                f"aneforge: tensor rank {len(self.shape)} exceeds the ANE maximum of 5 "
                f"(op={op!r}, shape={self.shape}); the ANE dimension model is rank<=5.")
        self.op = op
        self.srcs = list(srcs)
        self.attrs = attrs or {}
        self._name = ""

    # -- elementwise ------------------------------------------------------- #
    # param-free unary ops: graph op name == MIL op name (see _compile._e_unary)
    def relu(self) -> "Tensor": return Tensor(self.shape, "relu", [self])
    def gelu(self) -> "Tensor": return Tensor(self.shape, "gelu", [self])
    def silu(self) -> "Tensor": return Tensor(self.shape, "silu", [self])
    def sigmoid(self) -> "Tensor": return Tensor(self.shape, "sigmoid", [self])
    def tanh(self) -> "Tensor": return Tensor(self.shape, "tanh", [self])
    def exp(self) -> "Tensor": return Tensor(self.shape, "exp", [self])
    def sqrt(self) -> "Tensor": return Tensor(self.shape, "sqrt", [self])
    def abs(self) -> "Tensor": return Tensor(self.shape, "abs", [self])
    def square(self) -> "Tensor": return Tensor(self.shape, "square", [self])
    def sin(self) -> "Tensor": return Tensor(self.shape, "sin", [self])
    def cos(self) -> "Tensor": return Tensor(self.shape, "cos", [self])
    def erf(self) -> "Tensor": return Tensor(self.shape, "erf", [self])
    def softplus(self) -> "Tensor": return Tensor(self.shape, "softplus", [self])
    def relu6(self) -> "Tensor": return Tensor(self.shape, "relu6", [self])
    def softsign(self) -> "Tensor": return Tensor(self.shape, "softsign", [self])
    def atan(self) -> "Tensor": return Tensor(self.shape, "atan", [self])
    def exp2(self) -> "Tensor": return Tensor(self.shape, "exp2", [self])

    # unary ops carrying a parameter
    def log(self, eps: float = 0.0) -> "Tensor": return Tensor(self.shape, "log", [self], {"eps": eps})
    def rsqrt(self, eps: float = 0.0) -> "Tensor": return Tensor(self.shape, "rsqrt", [self], {"eps": eps})
    def inverse(self, eps: float = 0.0) -> "Tensor":
        """`1 / x` (elementwise reciprocal). `eps` is the MIL `inverse` epsilon
        floor under the divide (0.0 = plain reciprocal)."""
        return Tensor(self.shape, "inverse", [self], {"eps": eps})
    def elu(self, alpha: float = 1.0) -> "Tensor": return Tensor(self.shape, "elu", [self], {"alpha": alpha})
    def leaky_relu(self, alpha: float = 0.01) -> "Tensor": return Tensor(self.shape, "leaky_relu", [self], {"alpha": alpha})
    def clip(self, lo: float, hi: float) -> "Tensor": return Tensor(self.shape, "clip", [self], {"lo": lo, "hi": hi})

    # parametric activations (scalar fp16 params, like clip/elu)
    def scaled_tanh(self, alpha: float = 1.0, beta: float = 1.0) -> "Tensor":
        """`alpha * tanh(beta * x)`."""
        return Tensor(self.shape, "scaled_tanh", [self], {"alpha": alpha, "beta": beta})
    def threshold(self, alpha: float = 0.0) -> "Tensor":
        """`max(x, alpha)`."""
        return Tensor(self.shape, "threshold", [self], {"alpha": alpha})
    def thresholded_relu(self, alpha: float = 1.0) -> "Tensor":
        """`x if x >= alpha else 0`."""
        return Tensor(self.shape, "thresholded_relu", [self], {"alpha": alpha})
    def clamped_relu(self, alpha: float = 0.0, beta: float = 6.0) -> "Tensor":
        """`min(beta, x)` for `x >= 0` else `min(beta, alpha * x)` (a leaky relu6)."""
        return Tensor(self.shape, "clamped_relu", [self], {"alpha": alpha, "beta": beta})
    def sigmoid_hard(self, alpha: float = 0.2, beta: float = 0.5) -> "Tensor":
        """Hard sigmoid: `min(max(alpha * x + beta, 0), 1)`."""
        return Tensor(self.shape, "sigmoid_hard", [self], {"alpha": alpha, "beta": beta})
    def linear_activation(self, alpha: float = 1.0, beta: float = 0.0) -> "Tensor":
        """`alpha * x + beta` (scalar affine, fused as one op)."""
        return Tensor(self.shape, "linear_activation", [self], {"alpha": alpha, "beta": beta})

    def greater(self, o) -> "Tensor":
        """Elementwise `x > o` -> a BOOL tensor (use as the `cond` of `af.select`).
        Comparison ops output bool, not a numeric value on their own."""
        if not isinstance(o, Tensor):
            raise TypeError("greater expects a graph Tensor (compare against a streamed weight via select)")
        return Tensor(_broadcast(self.shape, o.shape), "greater", [self, o])

    def _cmp(self, o, op: str) -> "Tensor":
        if not isinstance(o, Tensor):
            raise TypeError(f"{op} expects a graph Tensor (compare against a streamed weight via select)")
        return Tensor(_broadcast(self.shape, o.shape), op, [self, o])

    def less(self, o) -> "Tensor": return self._cmp(o, "less")               # x < o  -> bool
    def equal(self, o) -> "Tensor": return self._cmp(o, "equal")             # x == o -> bool
    def not_equal(self, o) -> "Tensor": return self._cmp(o, "not_equal")     # x != o -> bool
    def less_equal(self, o) -> "Tensor": return self._cmp(o, "less_equal")   # x <= o -> bool
    def greater_equal(self, o) -> "Tensor": return self._cmp(o, "greater_equal")  # x >= o -> bool
    def logical_not(self) -> "Tensor": return Tensor(self.shape, "logical_not", [self])  # ~bool

    def floor(self) -> "Tensor": return Tensor(self.shape, "floor", [self])
    def ceil(self) -> "Tensor": return Tensor(self.shape, "ceil", [self])
    def round(self) -> "Tensor": return Tensor(self.shape, "round", [self])
    def sign(self) -> "Tensor": return Tensor(self.shape, "sign", [self])

    def prelu(self, alpha) -> "Tensor":
        """Per-channel PReLU: `x if x>0 else alpha[c]*x`. `alpha`: [C]; input rank>=3 [N,C,...]."""
        alpha = np.asarray(alpha)
        if len(self.shape) < 3:
            raise ValueError(f"prelu needs rank>=3 [N,C,...]; got {self.shape}")
        if alpha.shape != (self.shape[1],):
            raise ValueError(f"prelu alpha {alpha.shape} != channels {self.shape[1]}")
        return Tensor(self.shape, "prelu", [self], {"alpha": alpha})

    def __pow__(self, o) -> "Tensor": return _binary(self, o, "pow")
    def pow(self, o) -> "Tensor": return _binary(self, o, "pow")

    def reverse(self, axes) -> "Tensor":
        """Reverse along `axes` (native `reverse`)."""
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = tuple(a % len(self.shape) for a in axes)
        return Tensor(self.shape, "reverse", [self], {"axes": axes})

    def tile(self, reps) -> "Tensor":
        """Repeat `reps[i]` times along each axis (native `tile`; factors of {2,3,4,8})."""
        reps = tuple(int(r) for r in reps)
        out = tuple(d * r for d, r in zip(self.shape, reps))
        return Tensor(out, "tile", [self], {"reps": reps})

    def reduce_log_sum_exp(self, axes) -> "Tensor":
        """log(sum(exp(x))) over `axes` (native, stable softmax denominator)."""
        return self._reduce("reduce_log_sum_exp", axes)

    def clamp(self, lo: float, hi: float) -> "Tensor": return self.clip(lo, hi)  # alias

    def __add__(self, o) -> "Tensor": return _binary(self, o, "add")
    def __sub__(self, o) -> "Tensor": return _binary(self, o, "sub")
    def __truediv__(self, o) -> "Tensor": return _binary(self, o, "real_div")

    def __mul__(self, o) -> "Tensor":
        if isinstance(o, (int, float)):
            return Tensor(self.shape, "muls", [self], {"k": float(o)})
        return _binary(self, o, "mul")
    __rmul__ = __mul__

    def adds(self, k: float) -> "Tensor":
        """`x + scalar` as a fused scalar-add (additive sibling of `x * scalar`).
        The only fused way to inject a scalar offset (e.g. a normalization eps);
        `+` requires two graph Tensors."""
        return Tensor(self.shape, "adds", [self], {"k": float(k)})

    # -- linear algebra ---------------------------------------------------- #
    def __matmul__(self, W) -> "Tensor":
        """`x @ W`. `W` is a weight array (streamed), or another Tensor for an
        activationxactivation product (e.g. attention scores)."""
        if isinstance(W, Tensor):
            if self.shape[-1] != W.shape[-2]:
                raise ValueError(f"matmul: {self.shape} @ {W.shape} shape mismatch")
            return Tensor(self.shape[:-1] + (W.shape[-1],), "bmm", [self, W])
        W = np.asarray(W); _check_dtype(W, "matmul weight")
        if W.ndim != 2 or self.shape[-1] != W.shape[0]:
            raise ValueError(f"matmul: x{self.shape} @ W{W.shape} shape mismatch")
        # store as [N,K] and consume with transpose_y=true (the proven int8 layout)
        return Tensor(self.shape[:-1] + (W.shape[1],), "matmul", [self],
                      {"wt": np.ascontiguousarray(W.T)})

    def linear(self, W, bias=None) -> "Tensor":
        """`x @ W.T (+ bias)`. `W` is [out, in] (PyTorch convention)."""
        W = np.asarray(W); _check_dtype(W, "linear weight")
        if W.ndim != 2 or self.shape[-1] != W.shape[1]:
            raise ValueError(f"linear: x{self.shape} W{W.shape} shape mismatch")
        attrs: dict[str, Any] = {"wt": np.ascontiguousarray(W)}
        if bias is not None:
            attrs["bias"] = np.asarray(bias).astype(np.float32)
        return Tensor(self.shape[:-1] + (W.shape[0],), "matmul", [self], attrs)

    def transpose(self, perm) -> "Tensor":
        perm = tuple(p % len(self.shape) for p in perm)
        return Tensor(tuple(self.shape[p] for p in perm), "transpose", [self], {"perm": perm})

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(tuple(shape), "reshape", [self])

    def squeeze(self, axes) -> "Tensor":
        """Remove size-1 dims at `axes` (native `squeeze`). Each named axis must
        have size 1."""
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = tuple(a % len(self.shape) for a in axes)
        for a in axes:
            if self.shape[a] != 1:
                raise ValueError(f"squeeze: axis {a} has size {self.shape[a]} != 1")
        out = tuple(d for i, d in enumerate(self.shape) if i not in axes)
        return Tensor(out, "squeeze", [self], {"axes": axes})

    def expand_dims(self, axes) -> "Tensor":
        """Insert size-1 dims at `axes` (native `expand_dims`), the inverse of
        `squeeze`. Axes index into the OUTPUT rank."""
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        out_rank = len(self.shape) + len(axes)
        axes = sorted(a % out_rank for a in axes)
        out, src = [], iter(self.shape)
        for i in range(out_rank):
            out.append(1 if i in axes else next(src))
        return Tensor(tuple(out), "expand_dims", [self], {"axes": tuple(axes)})

    def flatten2d(self, axis: int = 1) -> "Tensor":
        """Collapse to 2-D about `axis`: dims `[:axis]` -> rows, `[axis:]` ->
        cols (native `flatten2d`)."""
        ax = axis % len(self.shape)
        rows = int(np.prod(self.shape[:ax])) if ax > 0 else 1
        cols = int(np.prod(self.shape[ax:]))
        return Tensor((rows, cols), "flatten2d", [self], {"axis": ax})

    def slice_by_size(self, begin, size) -> "Tensor":
        """Static slice `x[begin[i] : begin[i]+size[i]]` per axis (native
        `slice_by_size`). `begin`/`size` are per-axis lists matching the rank."""
        begin = [int(b) for b in begin]; size = [int(s) for s in size]
        if len(begin) != len(self.shape) or len(size) != len(self.shape):
            raise ValueError(f"slice_by_size: begin/size must have rank {len(self.shape)}")
        for i, (b, s) in enumerate(zip(begin, size)):
            if b < 0 or s <= 0 or b + s > self.shape[i]:
                raise ValueError(f"slice_by_size: axis {i} window [{b}:{b+s}] out of range for {self.shape[i]}")
        return Tensor(tuple(size), "slice_by_size", [self], {"begin": begin, "size": size})

    # -- reductions / normalisation --------------------------------------- #
    def _reduce(self, op: str, axes) -> "Tensor":
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = tuple(a % len(self.shape) for a in axes)
        out = tuple(1 if i in axes else d for i, d in enumerate(self.shape))
        return Tensor(out, op, [self], {"axes": axes})

    def mean(self, axes) -> "Tensor": return self._reduce("reduce_mean", axes)
    def sum(self, axes) -> "Tensor": return self._reduce("reduce_sum", axes)
    def amax(self, axes) -> "Tensor": return self._reduce("reduce_max", axes)
    def amin(self, axes) -> "Tensor": return self._reduce("reduce_min", axes)

    def cumsum(self, axis: int = -1) -> "Tensor":
        """Cumulative sum along `axis` (last axis only). The ANE has no native
        cumsum, but a last-axis cumsum is exactly `x @ triu_ones` -- a matmul with
        a baked upper-triangular-ones weight, made exact by the wide accumulator.
        A composition, not a native op (native cumsum is arch-gated). For other
        axes, transpose the target axis to last first."""
        ax = axis % len(self.shape)
        if ax != len(self.shape) - 1:
            raise ValueError(f"cumsum: only the last axis is supported (got axis={axis}, "
                             f"rank {len(self.shape)}); transpose the axis to last first")
        N = self.shape[-1]
        W = np.tril(np.ones((N, N), dtype=np.float32)).astype(np.float16)  # linear(W)=x@W.T=x@triu
        return self.linear(W, None)
    def l1_norm(self, axes) -> "Tensor":
        """`sum(|x|, axes)` (keepdims) via the native `reduce_l1_norm` op."""
        return self._reduce("reduce_l1_norm", axes)
    def log_sum(self, axes) -> "Tensor":
        """`log(sum(x, axes))` (keepdims). Expects a positive input."""
        return self._reduce("reduce_log_sum", axes)
    def sum_square(self, axes) -> "Tensor":
        """`sum(x**2, axes)` (keepdims) via the native `reduce_sum_square` op."""
        return self._reduce("reduce_sum_square", axes)

    def softmax(self, axis: int = -1) -> "Tensor":
        return Tensor(self.shape, "softmax", [self], {"axis": axis % len(self.shape)})

    def l2_norm(self, axis: int = -1, eps: float = 1e-12) -> "Tensor":
        """L2-normalize over `axis`: `x / sqrt(sum(x**2, axis) + eps)`.

        Runs as fused e5rt MIL (`reduce_l2_norm` over the axis, then `real_div`)
        - no graph cut. The MIL `l2_norm` op normalises over all non-batch dims,
        so we build the per-axis form explicitly."""
        ax = axis % len(self.shape)
        return Tensor(self.shape, "l2_norm", [self], {"axis": ax, "eps": float(eps)})

    def argmax(self, axis: int = -1) -> "Tensor":
        """Index of the maximum along `axis` (keepdims). Runs as a native-ANE
        GlobalArgMinMax sub-program (netplist bridge, like af.sdpa) - a graph cut.

        2D inputs [C, W] only, over the last axis (axis=-1/1, Width) or axis 0
        (Channel); indices are returned fp16-encoded (exact for index<2048)."""
        if len(self.shape) != 2:
            raise ValueError(f"argmax: only 2D [C,W] inputs are supported; got {self.shape}")
        ax = axis % 2
        out = tuple(1 if i == ax else d for i, d in enumerate(self.shape))
        return Tensor(out, "argmax", [self], {"axis": ax})

    def rms_norm(self, gamma, eps: float = 1e-5) -> "Tensor":
        """RMSNorm over the last dim. `gamma`: a [D] array for a fixed (baked)
        scale, or a broadcastable parameter `Tensor` ([1, D]) for a TRAINABLE
        scale (normalized with a unit-scale op, then scaled by the Tensor so its
        gradient flows via the mul VJP)."""
        if isinstance(gamma, Tensor):
            xn = self.rms_norm(np.ones(self.shape[-1], np.float32), eps)
            return xn * gamma
        gamma = np.asarray(gamma); _check_dtype(gamma, "rms_norm gamma")
        if len(self.shape) != 2 or gamma.shape != (self.shape[-1],):
            raise ValueError(f"rms_norm expects 2D [M,D] with gamma [D]; got {self.shape}, {gamma.shape}")
        return Tensor(self.shape, "rms_norm", [self], {"gamma": gamma, "eps": float(eps)})

    def layer_norm(self, gamma, beta, eps: float = 1e-5) -> "Tensor":
        """LayerNorm over the last dim (2D inputs [M, D]). `gamma`/`beta`: [D]
        arrays for a fixed (baked) affine, or broadcastable parameter `Tensor`s
        ([1, D]) for a TRAINABLE affine (normalized with a unit affine, then scaled
        and shifted by the Tensors so their gradients flow via the mul/add VJPs).
        Pass both as Tensors for the trainable form."""
        if isinstance(gamma, Tensor) or isinstance(beta, Tensor):
            if not (isinstance(gamma, Tensor) and isinstance(beta, Tensor)):
                raise TypeError("layer_norm: a trainable affine needs both gamma and beta as Tensors")
            D = self.shape[-1]
            xn = self.layer_norm(np.ones(D, np.float32), np.zeros(D, np.float32), eps)
            return xn * gamma + beta
        gamma = np.asarray(gamma); beta = np.asarray(beta)
        _check_dtype(gamma, "layer_norm gamma"); _check_dtype(beta, "layer_norm beta")
        if len(self.shape) != 2 or gamma.shape != (self.shape[-1],):
            raise ValueError(f"layer_norm expects 2D [M,D] with gamma/beta [D]; got {self.shape}, {gamma.shape}")
        return Tensor(self.shape, "layer_norm", [self], {"gamma": gamma, "beta": beta, "eps": float(eps)})

    def channel_layer_norm(self, gamma, beta, eps: float = 1e-5) -> "Tensor":
        """LayerNorm over the CHANNEL axis of a channels-first [N, C, 1, S] tensor -
        the ANE-native transformer layout. Each [N, :, 1, s] vector is normalized over
        C; `gamma`/`beta` are [C]. Same result as `layer_norm` on the [N*S, C] view, but
        with no transpose into and out of [seq, d], which is what makes the attention/MLP
        stack cheap on the ANE (projections stay 1x1 convs over [N, C, 1, S])."""
        gamma = np.asarray(gamma); beta = np.asarray(beta)
        _check_dtype(gamma, "channel_layer_norm gamma"); _check_dtype(beta, "channel_layer_norm beta")
        if (len(self.shape) != 4 or self.shape[2] != 1
                or gamma.shape != (self.shape[1],) or beta.shape != (self.shape[1],)):
            raise ValueError(f"channel_layer_norm expects [N,C,1,S] with gamma/beta [C]; "
                             f"got {self.shape}, {gamma.shape}")
        return Tensor(self.shape, "channel_layer_norm", [self], {"gamma": gamma, "beta": beta, "eps": float(eps)})

    def group_norm(self, gamma, beta, num_groups: int, eps: float = 1e-5) -> "Tensor":
        """GroupNorm over [1,C,H,W]. `gamma`/`beta`: [C] arrays for a fixed
        (baked) affine, or broadcastable parameter `Tensor`s ([1, C, 1, 1]) for a
        TRAINABLE affine (normalized with a unit affine, then scaled and shifted by
        the Tensors). Pass both as Tensors for the trainable form."""
        if isinstance(gamma, Tensor) or isinstance(beta, Tensor):
            if not (isinstance(gamma, Tensor) and isinstance(beta, Tensor)):
                raise TypeError("group_norm: a trainable affine needs both gamma and beta as Tensors")
            C = self.shape[1]
            xn = self.group_norm(np.ones(C, np.float32), np.zeros(C, np.float32), num_groups, eps)
            return xn * gamma + beta
        gamma = np.asarray(gamma); beta = np.asarray(beta)
        _check_dtype(gamma, "group_norm gamma"); _check_dtype(beta, "group_norm beta")
        if len(self.shape) != 4 or self.shape[0] != 1 or self.shape[1] % num_groups:
            raise ValueError(f"group_norm expects [1,C,H,W] with C%groups==0; got {self.shape}, G={num_groups}")
        # The rank-4 tiled lowering reshapes to [1,G,C/groups,H*W] and reduces the
        # trailing two axes, so the bound is the largest single axis, max(C/groups, H*W),
        # against the ANE's hard per-axis cap of 65536 - not the flattened (C/groups)*H*W
        # product (which overflowed for SD-UNet's 512ch@128 and 640ch@64; finding_sd15).
        _, C, H, W = self.shape
        axis = max(C // num_groups, H * W)
        if axis > 65536:
            raise ValueError(
                f"group_norm: largest tiled axis max(C/groups, H*W) = {axis} exceeds "
                f"the ANE per-axis bound (65536) for {self.shape} with groups={num_groups}; "
                f"reduce the feature map or channels, or use more groups.")
        return Tensor(self.shape, "group_norm", [self],
                      {"gamma": gamma, "beta": beta, "groups": num_groups, "eps": float(eps)})

    # -- spatial ----------------------------------------------------------- #
    def max_pool(self, k: int, stride: int | None = None, pad: int = 0) -> "Tensor":
        stride = stride or k
        N, C, H, W = self.shape
        out = (N, C, (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1)
        return Tensor(out, "max_pool", [self], {"k": k, "stride": stride, "pad": pad})

    def avg_pool(self, k: int, stride: int | None = None, pad: int = 0) -> "Tensor":
        stride = stride or k
        N, C, H, W = self.shape
        out = (N, C, (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1)
        return Tensor(out, "avg_pool", [self], {"k": k, "stride": stride, "pad": pad})

    def upsample(self, scale: int = 2) -> "Tensor":
        """Nearest-neighbour upsample [N,C,H,W] -> [N,C,scale*H,scale*W]."""
        N, C, H, W = self.shape
        return Tensor((N, C, H * scale, W * scale), "upsample", [self], {"scale": scale})

    def __repr__(self) -> str:
        return f"Tensor({self.op}, shape={self.shape})"


# --------------------------------------------------------------------------- #
# op constructors                                                             #
# --------------------------------------------------------------------------- #

_input_counter = [0]


def input(shape: Sequence[int], dtype: str = "fp16") -> Tensor:
    """A graph input placeholder. Inputs are fed to the compiled Model in the
    order they were created.

    `dtype` is the wire dtype of the input port: "fp16" (default, the ANE compute
    type) or "uint8" (raw camera/decoded-video bytes, dequantised in-graph - see
    `af.image_input`). A uint8 input only feeds an in-graph `cast`; not a compute
    tensor on its own."""
    if dtype not in ("fp16", "uint8"):
        raise ValueError(f"input: dtype must be 'fp16' or 'uint8'; got {dtype!r}")
    t = Tensor(tuple(shape), "input")
    t.attrs["idx"] = _input_counter[0]
    if dtype != "fp16":
        t.attrs["dtype"] = dtype
    _input_counter[0] += 1
    return t


def image_input(shape: Sequence[int], scale: float = 1.0 / 255.0, bias=0.0) -> Tensor:
    """A uint8 image input that is dequantised to fp16 ON the engine.

    Feed raw 8-bit pixels (camera / decoded-video bytes) straight to the compiled
    model - the uint8->fp16 conversion and the `scale*x + bias` normalisation run
    as in-graph ANE ops, so the host skips the float-convert + repack. Returns a
    normal fp16 Tensor for the rest of the graph.

    `scale`/`bias` are scalars by default (the usual `x/255` ImageNet-style
    normalisation is `scale=1/255`). For per-channel normalisation on an NCHW image
    pass length-C sequences; they broadcast as `[1,C,1,1]` constants. The dequant is
    `cast(uint8->fp16) -> mul(scale) -> add(bias)`; identity add/mul are dropped, so
    the common `scale=1/255, bias=0` case is a cast + one mul."""
    shape = tuple(int(d) for d in shape)
    x = input(shape, dtype="uint8")
    y = Tensor(shape, "cast", [x], {"dtype": "fp16"})

    def _coef(v, what: str):
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 0:
            return float(arr), None
        if len(shape) != 4 or arr.shape != (shape[1],):
            raise ValueError(f"image_input: per-channel {what} must be length-C={shape[1]} "
                             f"for an NCHW image; got {arr.shape} with shape {shape}")
        return None, arr.reshape(1, shape[1], 1, 1).astype(np.float16)

    sc_s, sc_v = _coef(scale, "scale")
    if sc_v is not None:                                    # per-channel scale: broadcast mul
        y = y * Tensor(sc_v.shape, "const_array", [], {"value": sc_v})
    elif sc_s != 1.0:
        y = y * sc_s
    bi_s, bi_v = _coef(bias, "bias")
    if bi_v is not None:                                    # per-channel bias: broadcast add
        y = y + Tensor(bi_v.shape, "const_array", [], {"value": bi_v})
    elif bi_s != 0.0:
        y = y.adds(bi_s)
    return y


def _binary(a: Tensor, b, kind: str) -> Tensor:
    if not isinstance(b, Tensor):
        raise TypeError("aneforge add/sub/mul expect two graph Tensors (use weights via matmul/linear)")
    return Tensor(_broadcast(a.shape, b.shape), kind, [a, b])


def conv(x: Tensor, weight, stride: int = 1, pad: int = 0, dilation: int = 1,
         groups: int = 1, bias=None) -> Tensor:
    """2D conv. `x`: [N,Cin,H,W]; `weight`: [Cout, Cin/groups, kH, kW]; `bias`: [Cout]."""
    weight = np.asarray(weight); _check_dtype(weight, "conv weight")
    if len(x.shape) != 4:
        raise ValueError(f"conv expects 4D input [N,Cin,H,W], got {x.shape}")
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = weight.shape
    # The ANE conv tiles along the kernel WIDTH: kW>=16 is unsupported by ANECCompile (kH
    # unconstrained; verified 16x3 compiles, 3x16 does not). Guard with a clear error.
    if kW > 15:
        raise ValueError(
            f"conv: kernel width kW={kW} exceeds the ANE limit (kW must be <=15); "
            f"got weight {weight.shape}. Kernel height kH is unconstrained.")
    Hout = (H + 2 * pad - dilation * (kH - 1) - 1) // stride + 1
    Wout = (W + 2 * pad - dilation * (kW - 1) - 1) // stride + 1
    attrs: dict[str, Any] = {"weight": weight, "stride": stride, "pad": pad,
                             "dilation": dilation, "groups": groups}
    if bias is not None:
        attrs["bias"] = np.asarray(bias).astype(np.float32)
    return Tensor((N, Cout, Hout, Wout), "conv", [x], attrs)


def dynamic_conv(x: Tensor, weight: Tensor, stride: int = 1, pad: int = 0,
                 dilation: int = 1, groups: int = 1) -> Tensor:
    """2D conv with a DYNAMIC (runtime-tensor) weight - the kernel is a graph value, not a
    baked constant. Lowers to the ANE's native dynamic-kernel path (`CreateDynamicKernel` /
    DynamicGOC), so the weight can be produced at runtime by an earlier op or fed as an input.
    Enables hypernetworks / per-sample (per-image) kernels - a capability no other ANE frontend
    exposes, since Apple's MIL/CoreML conv bakes the weight.

    `x`: [1, Cin, H, W]; `weight`: a Tensor [Cout, Cin/groups, kH, kW]. Returns
    [1, Cout, Hout, Wout].

    BATCH MUST BE 1. The ANE dynamic-kernel path does not support a dynamic-weight conv
    with batch >= 2, so it is rejected at build time. For batched convolution use `af.conv`
    (constant weight) or the im2col-based trainable `conv2d`."""
    if not isinstance(weight, Tensor):
        raise TypeError("dynamic_conv: weight must be a Tensor (a graph value); for a constant "
                        "weight use af.conv")
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise ValueError(f"dynamic_conv: x and weight must be 4D [N,Cin,H,W]/[Cout,Cin/g,kH,kW]; "
                         f"got {x.shape}, {weight.shape}")
    N, Cin, H, W = x.shape
    Cout, Cin_g, kH, kW = weight.shape
    if N != 1:
        raise ValueError(
            f"dynamic_conv requires batch N=1 (got N={N}): a dynamic-weight conv with batch>=2 "
            f"is unsupported on the ANE dynamic-kernel path. Use af.conv / conv2d for batched convolution.")
    if Cin_g * groups != Cin:
        raise ValueError(f"dynamic_conv: weight Cin/groups={Cin_g} x groups={groups} != input Cin={Cin}")
    if kW > 15:
        raise ValueError(f"dynamic_conv: kernel width kW={kW} exceeds the ANE limit (<=15); "
                         f"kernel height kH is unconstrained.")
    Hout = (H + 2 * pad - dilation * (kH - 1) - 1) // stride + 1
    Wout = (W + 2 * pad - dilation * (kW - 1) - 1) // stride + 1
    return Tensor((N, Cout, Hout, Wout), "dynamic_conv", [x, weight],
                  {"stride": stride, "pad": pad, "dilation": dilation, "groups": groups})


def conv_transpose(x: Tensor, weight, stride: int = 1, pad: int = 0, dilation: int = 1,
                   groups: int = 1, bias=None) -> Tensor:
    """2D transposed conv (deconv) - upsampling conv for VAE/segmentation decoders.
    `x`: [N,Cin,H,W]; `weight`: [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d layout);
    `bias`: [Cout]."""
    weight = np.asarray(weight); _check_dtype(weight, "conv_transpose weight")
    if len(x.shape) != 4:
        raise ValueError(f"conv_transpose expects 4D [N,Cin,H,W], got {x.shape}")
    N, Cin, H, W = x.shape
    _, Cout, kH, kW = weight.shape
    # Same kernel-WIDTH tiling limit as conv: kW>=16 fails ANECCompile (kH
    # unconstrained). Guard with a clear error.
    if kW > 15:
        raise ValueError(
            f"conv_transpose: kernel width kW={kW} exceeds the ANE limit (kW must be <=15); "
            f"got weight {weight.shape}. Kernel height kH is unconstrained.")
    Hout = (H - 1) * stride - 2 * pad + dilation * (kH - 1) + 1
    Wout = (W - 1) * stride - 2 * pad + dilation * (kW - 1) + 1
    attrs: dict[str, Any] = {"weight": weight, "stride": stride, "pad": pad,
                             "dilation": dilation, "groups": groups}
    if bias is not None:
        attrs["bias"] = np.asarray(bias).astype(np.float32)
    return Tensor((N, Cout, Hout, Wout), "conv_transpose", [x], attrs)


def batch_norm(x: Tensor, gamma, beta, mean, var, eps: float = 1e-5) -> Tensor:
    """BatchNorm in inference mode over [1,C,H,W] (or [1,C,...]); per-channel
    affine from precomputed running `mean`/`var`. gamma/beta/mean/var: [C]."""
    g, b, m, v = (np.asarray(a) for a in (gamma, beta, mean, var))
    for a, nm in ((g, "gamma"), (b, "beta"), (m, "mean"), (v, "var")):
        _check_dtype(a, f"batch_norm {nm}")
    if len(x.shape) < 3 or g.shape != (x.shape[1],):
        raise ValueError(f"batch_norm expects rank>=3 [1,C,...] with params [C]; got {x.shape}, {g.shape}")
    return Tensor(x.shape, "batch_norm", [x],
                  {"gamma": g, "beta": b, "mean": m, "var": v, "eps": float(eps)})


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise max of two graph tensors."""
    return _binary(a, b, "maximum")


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise min of two graph tensors."""
    return _binary(a, b, "minimum")


def concat(tensors: Sequence[Tensor], axis: int = 1) -> Tensor:
    """Concatenate tensors along `axis` (e.g. UNet skip connections)."""
    tensors = list(tensors)
    ax = axis % len(tensors[0].shape)
    out = list(tensors[0].shape)
    out[ax] = sum(t.shape[ax] for t in tensors)
    return Tensor(tuple(out), "concat", tensors, {"axis": ax})


def gather(x: Tensor, indices, axis: int = 0) -> Tensor:
    """Gather slices along `axis` by STATIC (build-time) integer `indices`. The
    ANE has no native gather, but for constant indices a gather is exact via
    `slice_by_size` + `concat` (a composition, not a native op -- native gather
    is arch-gated). Dynamic (data-dependent) indices are not reachable on the ANE."""
    idx = [int(i) for i in indices]
    rank = len(x.shape)
    ax = axis % rank
    # A last-axis (width) gather lowers to slice_by_size with a nonzero WIDTH begin-offset,
    # which routes through the A13/A14 x16 fixed-point crop-DMA path and returns the wrong
    # elements there (correct on A16+). Gather a NON-last axis instead: for rank>=2 transpose
    # the gathered axis off the last position and transpose back; for rank 1 gather a [N,1]
    # view. Both are identity-preserving and correct on every family (the same width-axis-slice
    # avoidance the conv im2col backward uses).
    if ax == rank - 1:
        if rank == 1:
            return gather(x.reshape(x.shape[0], 1), idx, axis=0).reshape(len(idx))
        perm = list(range(rank)); perm[ax], perm[-2] = perm[-2], perm[ax]
        return gather(x.transpose(perm), idx, axis=rank - 2).transpose(perm)
    rows = []
    for i in idx:
        if not (0 <= i < x.shape[ax]):
            raise ValueError(f"gather: index {i} out of range for axis {ax} (size {x.shape[ax]})")
        begin = [0] * rank; begin[ax] = i
        size = list(x.shape); size[ax] = 1
        rows.append(x.slice_by_size(begin, size))
    return concat(rows, axis=ax)


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Stack equal-shaped tensors along a NEW axis (native `stack`):
    N x [shape] -> [..., N, ...] with N inserted at `axis`."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError("stack: empty tensor list")
    base = tensors[0].shape
    for t in tensors:
        if t.shape != base:
            raise ValueError(f"stack: all tensors must share a shape; got {base} and {t.shape}")
    ax = axis % (len(base) + 1)
    out = base[:ax] + (len(tensors),) + base[ax:]
    return Tensor(out, "stack", tensors, {"axis": ax})


def split(x: Tensor, num_splits: int, axis: int = 0) -> list[Tensor]:
    """Split `x` into `num_splits` equal parts along `axis` (native `split`).
    Returns the list of output Tensors; the axis size must divide evenly."""
    ax = axis % len(x.shape)
    if x.shape[ax] % num_splits:
        raise ValueError(f"split: axis {ax} size {x.shape[ax]} not divisible by {num_splits}")
    part = x.shape[ax] // num_splits
    out_shape = x.shape[:ax] + (part,) + x.shape[ax + 1:]
    return [Tensor(out_shape, "split", [x], {"axis": ax, "num_splits": num_splits, "which": i})
            for i in range(num_splits)]


def select(cond: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Elementwise `cond ? a : b` (native `select`). `cond` is a BOOL tensor
    (e.g. from `x.greater(y)`); `a`/`b` are fp16 tensors."""
    if not (isinstance(cond, Tensor) and isinstance(a, Tensor) and isinstance(b, Tensor)):
        raise TypeError("select expects three graph Tensors (cond, a, b)")
    return Tensor(_broadcast(a.shape, b.shape), "select", [cond, a, b])


where = select  # alias


def instance_norm(x: Tensor, gamma, beta, eps: float = 1e-5) -> Tensor:
    """InstanceNorm over [N,C,H,W]: normalize each (N,C) slice over its spatial dims,
    then a per-channel affine. `gamma`/`beta`: [C]. Native `instance_norm` op."""
    g, b = np.asarray(gamma), np.asarray(beta)
    _check_dtype(g, "instance_norm gamma"); _check_dtype(b, "instance_norm beta")
    if len(x.shape) != 4 or g.shape != (x.shape[1],) or b.shape != (x.shape[1],):
        raise ValueError(f"instance_norm expects [N,C,H,W] with gamma/beta [C]; got {x.shape}, {g.shape}")
    return Tensor(x.shape, "instance_norm", [x], {"gamma": g, "beta": b, "eps": float(eps)})


def local_response_norm(x: Tensor, size: int = 5, alpha: float = 1e-4,
                        beta: float = 0.75, k: float = 1.0) -> Tensor:
    """Cross-channel LRN over [N,C,H,W] via the native MIL `local_response_norm` op
    (fused, no graph cut - distinct from the netplist `af.lrn` bridge): each output
    is `x / (k + alpha/size * sum_{window} x**2) ** beta` over a window of `size`
    neighbouring channels. `gamma`-free; `alpha`/`beta`/`k` in natural units."""
    if len(x.shape) != 4:
        raise ValueError(f"local_response_norm expects [N,C,H,W]; got {x.shape}")
    return Tensor(x.shape, "local_response_norm", [x],
                  {"size": int(size), "alpha": float(alpha), "beta": float(beta), "k": float(k)})


def einsum_native(equation: str, a: Tensor, b) -> Tensor:
    """Restricted batched contraction via the native MIL `einsum` op (distinct from
    the general `af.einsum` decomposer: this is the single hardware `einsum` layer).
    The only on-ANE-verified equation is `'nchw,nwhu->nchu'` (a batched matmul over
    the W/U dims sharing N,H): `a`=[N,C,H,W], `b`=[N,W,H,U] (streamed weight) ->
    [N,C,H,U]. `b` is a weight array (streamed), not a graph Tensor."""
    if equation.replace(" ", "") != "nchw,nwhu->nchu":
        raise NotImplementedError(
            f"einsum: only 'nchw,nwhu->nchu' is verified reachable on the ANE; got {equation!r}")
    b = np.asarray(b); _check_dtype(b, "einsum operand b")
    if len(a.shape) != 4 or b.ndim != 4:
        raise ValueError(f"einsum nchw,nwhu->nchu expects rank-4 a and b; got {a.shape}, {b.shape}")
    N, C, H, W = a.shape
    Nb, Wb, Hb, U = b.shape
    if (Nb, Wb, Hb) != (N, W, H):
        raise ValueError(f"einsum: b must be [N,W,H,U]=[{N},{W},{H},U]; got {b.shape}")
    return Tensor((N, C, H, U), "einsum", [a], {"b": b, "equation": "nchw,nwhu->nchu"})


def space_to_depth(x: Tensor, block_size: int = 2) -> Tensor:
    """Space-to-depth (TensorFlow `space_to_depth` / native MIL `space_to_depth`):
    `[N,C,H,W] -> [N, C*bs*bs, H/bs, W/bs]`. Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"space_to_depth expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    bs = int(block_size)
    if H % bs or W % bs:
        raise ValueError(f"space_to_depth: H={H},W={W} not divisible by block_size={bs}")
    return Tensor((N, C * bs * bs, H // bs, W // bs), "space_to_depth", [x], {"block_size": bs})


def depth_to_space(x: Tensor, block_size: int = 2) -> Tensor:
    """Depth-to-space (TensorFlow `depth_to_space` / native MIL `depth_to_space`):
    `[N, C*bs*bs, H, W] -> [N, C, H*bs, W*bs]` (inverse of `space_to_depth`).
    Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"depth_to_space expects 4D [N,C,H,W], got {x.shape}")
    N, C2, H, W = x.shape
    bs = int(block_size)
    if C2 % (bs * bs):
        raise ValueError(f"depth_to_space: channels {C2} not divisible by block_size^2={bs * bs}")
    return Tensor((N, C2 // (bs * bs), H * bs, W * bs), "depth_to_space", [x], {"block_size": bs})


def crop(x: Tensor, top: int, bottom: int, left: int, right: int) -> Tensor:
    """Spatial crop of [N,C,H,W]: drop `top`/`bottom` rows and `left`/`right`
    columns (native MIL `crop`). Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"crop expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    Hout, Wout = H - top - bottom, W - left - right
    if Hout <= 0 or Wout <= 0:
        raise ValueError(f"crop: result {Hout}x{Wout} is empty for input {H}x{W}")
    return Tensor((N, C, Hout, Wout), "crop", [x],
                  {"crop_h": (int(top), int(bottom)), "crop_w": (int(left), int(right))})


def resize_nearest_neighbor(x: Tensor, target_h: int, target_w: int) -> Tensor:
    """Nearest-neighbour resize of [N,C,H,W] to `(target_h, target_w)` (native MIL
    `resize_nearest_neighbor`, arbitrary target size). Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"resize_nearest_neighbor expects 4D [N,C,H,W], got {x.shape}")
    N, C, _, _ = x.shape
    return Tensor((N, C, int(target_h), int(target_w)), "resize_nearest_neighbor", [x],
                  {"target_h": int(target_h), "target_w": int(target_w)})


def resize_bilinear(x: Tensor, target_h: int, target_w: int,
                    align_corners: bool = False) -> Tensor:
    """Bilinear resize of [N,C,H,W] to an explicit `(target_h, target_w)` (native
    MIL `resize_bilinear`). Half-pixel sampling by default (`align_corners=False`).
    Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"resize_bilinear expects 4D [N,C,H,W], got {x.shape}")
    N, C, _, _ = x.shape
    return Tensor((N, C, int(target_h), int(target_w)), "resize_bilinear", [x],
                  {"target_h": int(target_h), "target_w": int(target_w),
                   "align_corners": bool(align_corners)})


def upsample_bilinear(x: Tensor, scale: int = 2, align_corners: bool = False) -> Tensor:
    """Bilinear upsample of [N,C,H,W] by an integer `scale` (native MIL
    `upsample_bilinear`, scale-factor form). Half-pixel sampling by default.
    Fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"upsample_bilinear expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    s = int(scale)
    return Tensor((N, C, H * s, W * s), "upsample_bilinear", [x],
                  {"scale": s, "align_corners": bool(align_corners)})


def affine(x: Tensor, transform, output_h: int, output_w: int,
           align_corners: bool = False) -> Tensor:
    """2-D affine warp of [N,C,H,W] to `(output_h, output_w)` via the native MIL
    `affine` op (`AffineTransform` hardware layer). `transform` is the [N,6]
    (or [1,6], broadcast) affine matrix `[a0,a1,a2, b0,b1,b2]` in
    normalized [-1,1] coordinates; bilinear sampling with zero padding. Fused MIL."""
    if len(x.shape) != 4:
        raise ValueError(f"affine expects 4D [N,C,H,W], got {x.shape}")
    T = np.asarray(transform); _check_dtype(T, "affine transform")
    if T.ndim != 2 or T.shape[1] != 6:
        raise ValueError(f"affine: transform must be [N,6] (or [1,6]); got {T.shape}")
    N, C, _, _ = x.shape
    return Tensor((N, C, int(output_h), int(output_w)), "affine", [x],
                  {"transform": T, "output_h": int(output_h), "output_w": int(output_w),
                   "align_corners": bool(align_corners)})


def pixel_shuffle(x: Tensor, r: int) -> Tensor:
    """Depth-to-space upscale (PyTorch `nn.PixelShuffle`):
    `[N, C*r*r, H, W] -> [N, C, H*r, W*r]`. Runs as fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"pixel_shuffle expects 4D [N,C,H,W], got {x.shape}")
    N, C2, H, W = x.shape
    if C2 % (r * r):
        raise ValueError(f"pixel_shuffle: channels {C2} not divisible by r*r={r*r}")
    return Tensor((N, C2 // (r * r), H * r, W * r), "pixel_shuffle", [x], {"r": int(r)})


def pixel_unshuffle(x: Tensor, r: int) -> Tensor:
    """Space-to-depth (PyTorch `nn.PixelUnshuffle`):
    `[N, C, H*r, W*r] -> [N, C*r*r, H, W]`. Runs as fused e5rt MIL (no cut)."""
    if len(x.shape) != 4:
        raise ValueError(f"pixel_unshuffle expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    if H % r or W % r:
        raise ValueError(f"pixel_unshuffle: H={H},W={W} not divisible by r={r}")
    return Tensor((N, C * r * r, H // r, W // r), "pixel_unshuffle", [x], {"r": int(r)})


def space_to_channel(x: Tensor, r: int) -> Tensor:
    """Space-to-depth on the ANE's native `SpaceToChannel` layer (TensorFlow
    `space_to_depth`, block-major channels): `[N,C,H*r,W*r] -> [N,C*r*r,H,W]`.
    Same shape law as PixelUnshuffle but the TF channel ordering. Graph cut."""
    if len(x.shape) != 4:
        raise ValueError(f"space_to_channel expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    if N != 1:
        raise ValueError(f"space_to_channel: the native layer supports batch N=1 only; got N={N}")
    if H % r or W % r:
        raise ValueError(f"space_to_channel: H={H},W={W} not divisible by r={r}")
    return Tensor((N, C * r * r, H // r, W // r), "space_to_channel", [x], {"r": int(r)})


def channel_to_space(x: Tensor, r: int) -> Tensor:
    """Depth-to-space on the ANE's native `ChannelToSpace` layer (TensorFlow
    `depth_to_space`, block-major channels): `[N,C*r*r,H,W] -> [N,C,H*r,W*r]`.
    Same shape law as PixelShuffle but the TF channel ordering. Graph cut."""
    if len(x.shape) != 4:
        raise ValueError(f"channel_to_space expects 4D [N,C,H,W], got {x.shape}")
    N, C2, H, W = x.shape
    if N != 1:
        raise ValueError(f"channel_to_space: the native layer supports batch N=1 only; got N={N}")
    if C2 % (r * r):
        raise ValueError(f"channel_to_space: channels {C2} not divisible by r*r={r * r}")
    return Tensor((N, C2 // (r * r), H * r, W * r), "channel_to_space", [x], {"r": int(r)})


def space_to_batch(x: Tensor, bh: int, bw: int) -> Tensor:
    """Move spatial blocks into the batch dim on the ANE's native `SpaceToBatch`
    layer: `[N,C,H,W] -> [N*bh*bw, C, H/bh, W/bw]`. Output batch slice
    `(n*bh+i)*bw+j` == `x[n, :, i::bh, j::bw]`. Graph cut.

    The batch dim grows, so this can only be a leaf/output of the segmented plan
    or feed another netplist cut (segment outputs are threaded as host arrays);
    feeding it into a fused e5rt region changes the batch the region expects,
    which is fine since each region is compiled from its own input shapes."""
    if len(x.shape) != 4:
        raise ValueError(f"space_to_batch expects 4D [N,C,H,W], got {x.shape}")
    N, C, H, W = x.shape
    if H % bh or W % bw:
        raise ValueError(f"space_to_batch: H={H},W={W} not divisible by (bh={bh}, bw={bw})")
    return Tensor((N * bh * bw, C, H // bh, W // bw), "space_to_batch", [x],
                  {"bh": int(bh), "bw": int(bw)})


def batch_to_space(x: Tensor, bh: int, bw: int) -> Tensor:
    """Move batch blocks back into space on the ANE's native `BatchToSpace` layer
    (inverse of `space_to_batch`): `[N*bh*bw, C, H, W] -> [N, C, H*bh, W*bw]`.
    Graph cut.

    ARCH-GATED: the validator requires the input batch divisible by `bh*bw`
    (string: "Input batch n is not divisible by factor x * factor y"); a
    non-divisible batch fails compilation, so it is rejected here."""
    if len(x.shape) != 4:
        raise ValueError(f"batch_to_space expects 4D [N,C,H,W], got {x.shape}")
    B, C, H, W = x.shape
    if B % (bh * bw):
        raise ValueError(f"batch_to_space: input batch {B} not divisible by bh*bw={bh * bw} "
                         f"(arch-gated on this ANE)")
    return Tensor((B // (bh * bw), C, H * bh, W * bw), "batch_to_space", [x],
                  {"bh": int(bh), "bw": int(bw)})


def flatten(x: Tensor) -> Tensor:
    """Flatten on the ANE's native `Flatten` layer (NCHW): collapse to a 1-D
    vector of `prod(shape)` elements. The bridge takes a [C,H,W] input, so this
    requires a 3D graph tensor. Graph cut."""
    if len(x.shape) != 3:
        raise ValueError(f"flatten expects 3D [C,H,W] (the native bridge layout); got {x.shape}")
    return Tensor((int(np.prod(x.shape)),), "flatten", [x])


def input_view(x: Tensor, offset: int, size: int) -> Tensor:
    """Contiguous view `x[offset:offset+size]` along Width on the ANE's native
    `InputView` layer. `x` is flattened to 1-D (length W); returns `[size]`.
    Graph cut."""
    W = int(np.prod(x.shape))
    if offset < 0 or size <= 0 or offset + size > W:
        raise ValueError(f"input_view: window [{offset}:{offset + size}] out of range for W={W}")
    return Tensor((size,), "input_view", [x], {"offset": int(offset), "size": int(size)})


def dynamic_slice(x: Tensor, start: int, size: int = 2) -> Tensor:
    """Runtime-parametric slice `x[start:start+size]` on the ANE's native
    `DynamicSlice` layer (start bound through a netplist constant). Graph cut.

    ARCH-NOTE: the only verified/accepted netplist variant of this layer on this
    host fixes Width=4 and SliceSize=2, so this op requires a length-4 input and
    `size==2`. The static-start API is general in spirit; only the hardware variant
    is verified."""
    W = int(np.prod(x.shape))
    if W != 4 or size != 2:
        raise ValueError("dynamic_slice: the verified ANE variant requires a length-4 "
                         f"input and size==2; got W={W}, size={size}")
    if start < 0 or start + size > W:
        raise ValueError(f"dynamic_slice: window [{start}:{start + size}] out of range for W={W}")
    return Tensor((size,), "dynamic_slice", [x], {"start": int(start), "size": int(size)})


def scaled_elementwise(x: Tensor, z: Tensor, op: str = "Add", scale: float = 1.0) -> Tensor:
    """`scale * (x OP z)` on the ANE's native `ScaledElementWise` layer (a fused
    binary-op + scalar-scale). `op` in {Add, Mult, Min, Max}; inputs are flattened
    to equal-length 1-D Width vectors. Graph cut.

    Two arch quirks of the native layer are guarded here (found by tests/gen_random):
    `Sub` is rejected by ANECCompile, and `Mult` ignores `scale` on-silicon - reject
    those configs rather than emit a wrong/uncompilable program."""
    ops = ("Add", "Mult", "Min", "Max")
    if op not in ops:
        raise ValueError(f"scaled_elementwise: op must be one of {ops}; got {op!r} "
                         f"('Sub' is rejected by the ANE ScaledElementWise layer)")
    if op == "Mult" and float(scale) != 1.0:
        raise ValueError("scaled_elementwise: the native layer ignores `scale` for op='Mult' "
                         "(would silently give x*z, not scale*(x*z)); use scale=1.0 or a separate mul")
    if int(np.prod(x.shape)) != int(np.prod(z.shape)):
        raise ValueError(f"scaled_elementwise: x and z must have equal size; got {x.shape}, {z.shape}")
    return Tensor((int(np.prod(x.shape)),), "scaled_elementwise", [x, z],
                  {"op": op, "scale": float(scale)})


def topk(x: Tensor, k: int, largest: bool = True) -> Tensor:
    """Top-`k` values along the last axis of a 2D input [C, W], keyed per row.
    Runs as a native-ANE TopK sub-program (netplist bridge, like af.sdpa) - a cut.

    `k` in {3, 4} is ARCH-GATED on this hardware (ANECCompile fails) and rejected
    here; the rest of `k` in [1, W] is supported."""
    if len(x.shape) != 2:
        raise ValueError(f"topk: only 2D [C,W] inputs are supported; got {x.shape}")
    C, W = x.shape
    if not (1 <= k <= W):
        raise ValueError(f"topk: k={k} out of range [1, {W}]")
    if k in (3, 4):
        raise ValueError(f"topk: k={k} is arch-gated on this ANE (ANECCompile fails for k in {{3,4}})")
    return Tensor((C, k), "topk", [x], {"k": int(k), "largest": bool(largest)})


def sort(x: Tensor, descending: bool = False, return_indices: bool = False) -> Tensor:
    """Sort each row of a 2D input [C, W] along the last axis (Width).
    Runs as a native-ANE Sort sub-program (netplist bridge, like af.sdpa) - a cut.

    With `return_indices=True` the argsort indices are returned instead of the
    sorted values (fp16-encoded, exact for index < 2048). Output shape is [C, W].

    Like the native TopK, the hardware Sort keys the order on one channel lane and
    permutes all channels by it; for a numpy-like per-row independent sort the
    bridge dispatches each row as its own 1-channel tile."""
    if len(x.shape) != 2:
        raise ValueError(f"sort: only 2D [C,W] inputs are supported; got {x.shape}")
    return Tensor(x.shape, "sort", [x],
                  {"descending": bool(descending), "return_indices": bool(return_indices)})


def cross_product(a: Tensor, b: Tensor) -> Tensor:
    """3-vector cross product `cross(a, b)` on the ANE's native CrossProduct
    layer - a path Apple's MIL frontend rejects. Both inputs are length-3
    (shape (3,) or any shape with 3 elements); returns shape (3,). Graph cut."""
    if int(np.prod(a.shape)) != 3 or int(np.prod(b.shape)) != 3:
        raise ValueError(f"cross_product: both inputs must have 3 elements; got {a.shape}, {b.shape}")
    return Tensor((3,), "cross_product", [a, b])


def cross_correlation(x: Tensor, template: Tensor) -> Tensor:
    """Valid (no-flip) cross-correlation of a single-channel map `x` [H, W] with
    a `template` [Th, Tw] on the ANE's native CrossCorrelation layer:
    `y[i,j] = sum_{u,v} x[i+u, j+v] * template[u,v]` over [(H-Th+1), (W-Tw+1)].
    Graph cut. (True correlation - the template is not flipped.)"""
    if len(x.shape) != 2 or len(template.shape) != 2:
        raise ValueError(f"cross_correlation: x and template must be 2D; got {x.shape}, {template.shape}")
    H, W = x.shape
    Th, Tw = template.shape
    if Th > H or Tw > W:
        raise ValueError(f"cross_correlation: template {template.shape} larger than map {x.shape}")
    return Tensor((H - Th + 1, W - Tw + 1), "cross_correlation", [x, template])


def cost_volume(aux: Tensor, ref: Tensor, disparity_range: int = 1) -> Tensor:
    """L1 stereo/optical-flow matching cost on the ANE's native CostVolume layer.
    `aux` is a length-Wa row, `ref` a length-Wr row with `Wr >= Wa + R`;
    returns `(R+1, Wa)` where `cost[d,x] = |aux[x] - ref[x+d]|`. Graph cut."""
    Wa, Wr = int(np.prod(aux.shape)), int(np.prod(ref.shape))
    R = int(disparity_range)
    if R < 0:
        raise ValueError(f"cost_volume: disparity_range must be >= 0; got {R}")
    if Wr < Wa + R:
        raise ValueError(f"cost_volume: ref width {Wr} must be >= aux width {Wa} + disparity_range {R}")
    return Tensor((R + 1, Wa), "cost_volume", [aux, ref], {"disparity_range": R})


def fps(points: Tensor, k: int) -> Tensor:
    """Furthest-point sampling: greedily pick `k` maximally-far-apart points
    (seeded at index 0) on the ANE's native FurthestPointSampling layer.
    `points` is [N, 3]; returns the [k, 3] selected centroids. Graph cut.

    NOTE: the DistanceMetric param is L2-only on this arch (the bridge always
    uses Euclidean distance regardless of the param), so this is L2 FPS."""
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"fps: points must be [N, 3]; got {points.shape}")
    N = points.shape[0]
    if not (1 <= k <= N):
        raise ValueError(f"fps: k={k} out of range [1, {N}]")
    if k > 1024 or N > 8192:
        raise ValueError(f"fps: arch limits are k<=1024, N<=8192; got k={k}, N={N}")
    return Tensor((k, 3), "fps", [points], {"k": int(k)})


def radius_search(points: Tensor, centroids: Tensor, radius: float) -> Tensor:
    """L2 ball-query membership on the ANE's native RadiusSearch layer: for each
    (point, centroid) pair, 1 iff the point is within `radius` of the centroid.
    `points` is [N, 3], `centroids` is [Nc, 3]; returns an [N, Nc] 0/1
    membership matrix (fp16-encoded). Graph cut."""
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"radius_search: points must be [N, 3]; got {points.shape}")
    if len(centroids.shape) != 2 or centroids.shape[1] != 3:
        raise ValueError(f"radius_search: centroids must be [Nc, 3]; got {centroids.shape}")
    N, Nc = points.shape[0], centroids.shape[0]
    return Tensor((N, Nc), "radius_search", [points, centroids], {"radius": float(radius)})


def minmax_norm(x: Tensor, dimension: str = "Width", eps: float = 1e-4) -> Tensor:
    """Min-max normalize `y = (x - min) / (max - min + eps)` over `dimension`
    on the ANE's native MinMaxNormalization layer. `x` is [1, C, H, W]; reduces
    over "Width" or "Height" ("Channel" is arch-gated and rejected). Graph cut."""
    if len(x.shape) != 4 or x.shape[0] != 1:
        raise ValueError(f"minmax_norm: expects [1,C,H,W]; got {x.shape}")
    if dimension not in ("Width", "Height"):
        raise ValueError(f"minmax_norm: dimension must be 'Width' or 'Height' "
                         f"('Channel' is arch-gated on this ANE); got {dimension!r}")
    return Tensor(x.shape, "minmax_norm", [x], {"dimension": dimension, "eps": float(eps)})


def lrn(x: Tensor, alpha: float = 1.0, beta: float = 0.75, k: float = 1.0) -> Tensor:
    """Cross-channel local response normalization (classic AlexNet LRN) on the ANE's
    native LocalResponseNormalization layer (Channel mode). `x` is [1, C, H, W];
    graph cut.

    Per-channel, per-pixel:
        `y[c] = x[c] / (k + alpha * sum_{j in window(c)} x[j]^2) ** beta`

    The window is a LOCAL channel window of size N = C (the bridge fixes the layer's
    `KernelChannel` to the channel count), asymmetric-centered on c and CLIPPED at
    the channel boundaries:
        `window(c) = [max(0, c-(N-1)//2) : min(C, c + N//2 + 1)]`.
    So only the center channel sees all C channels; edge channels see a partial sum.
    This is NOT a full-channel sum (the old docstring's `sum_j x[j]^2` over all j
    was wrong - see the corrected reference in tests/test_numerical.py and the RE in
    the reverse-engineering corpus).

    `alpha`/`beta`/`k` are the standard LRN coefficients in their natural units;
    `alpha` is the TRUE effective alpha. The bridge encodes `alpha` as an fp16
    bit-pattern and pre-multiplies by KernelChannel to cancel the layer's internal
    divide-by-KernelChannel; callers do not see that.

    ARCH-GATED: the layer compiles only for C <= 15 (KernelChannel = C; C >= 16 fails
    ANECCompile on this hardware), so larger channel counts are rejected here."""
    if len(x.shape) != 4 or x.shape[0] != 1:
        raise ValueError(f"lrn: expects [1,C,H,W]; got {x.shape}")
    C = x.shape[1]
    if C > 15:
        raise ValueError(f"lrn: C={C} is arch-gated on this ANE (LocalResponseNormalization "
                         f"with KernelChannel=C fails ANECCompile for C>=16); got C={C}")
    return Tensor(x.shape, "lrn", [x], {"alpha": float(alpha), "beta": float(beta), "k": float(k)})


def _heads(t: Tensor, n: int, dh: int) -> Tensor:           # [S, D] -> [H, S, dh]
    return t.reshape(t.shape[0], n, dh).transpose([1, 0, 2])


def mha(x: Tensor, Wq, bq, Wk, bk, Wv, bv, Wo, bo, n_heads: int) -> Tensor:
    """Multi-head self-attention on `x` [S, D]. Weights [out,in]; biases [D] or None.
    Builds split-heads -> per-head SDPA -> concat -> output-proj from graph ops."""
    S, D = x.shape
    if D % n_heads:
        raise ValueError(f"mha: D={D} not divisible by n_heads={n_heads}")
    dh = D // n_heads
    q, k, v = x.linear(Wq, bq), x.linear(Wk, bk), x.linear(Wv, bv)
    qh, kh, vh = _heads(q, n_heads, dh), _heads(k, n_heads, dh), _heads(v, n_heads, dh)
    kt = kh.transpose([0, 2, 1])
    scale = 1.0 / dh ** 0.5
    # Query-tiling: compute [tile, S] score tiles per head instead of the full [H, S, S]
    # score matrix. Exact (each tile attends to all keys) and ~3x faster on the ANE at
    # large S, which pipelines the smaller tiles better (same fission as af.sdpa). The
    # count is the S-based heuristic unless a tuned value is cached (af.tune_attention).
    from . import _optimize as _opt
    n_tiles = _opt.attention_tiles(S, n_heads, dh)
    if n_tiles == 1:
        o = ((qh @ kt) * scale).softmax(-1) @ vh                # [H, S, dh]
    else:
        tile = -(-S // n_tiles)                                 # ceil -> near-even chunks
        parts = []
        for start in range(0, S, tile):
            n = min(tile, S - start)
            qt = qh.slice_by_size([0, start, 0], [n_heads, n, dh])
            parts.append(((qt @ kt) * scale).softmax(-1) @ vh)  # [H, n, dh]
        o = concat(parts, axis=1)                               # [H, S, dh]
    o = o.transpose([1, 0, 2]).reshape(S, D)
    return o.linear(Wo, bo)


def cross_attention(x: Tensor, context: Tensor, Wq, Wk, Wv, Wo, n_heads: int,
                    bq=None, bk=None, bv=None, bo=None) -> Tensor:
    """Cross-attention: queries from `x` [S, D], keys/values from `context`
    [T, Dctx]. Wq:[D,D]; Wk,Wv:[D,Dctx]; Wo:[D,D]. (SD UNet text conditioning.)"""
    S, D = x.shape
    T = context.shape[0]
    dh = D // n_heads
    qh = _heads(x.linear(Wq, bq), n_heads, dh)                              # [H,S,dh]
    kh = _heads(context.linear(Wk, bk), n_heads, dh)                        # [H,T,dh]
    vh = _heads(context.linear(Wv, bv), n_heads, dh)
    kt = kh.transpose([0, 2, 1])                                            # [H,dh,T]
    scale = 1.0 / dh ** 0.5
    # Query-tiling: when query and context are both long the [H, S, T] score matrix hits
    # the ANE's materialization wall (~2.4x slower past it). Tiling the query into
    # [tile, T] score blocks avoids it, exact, the same fission as af.sdpa/af.mha. Gated
    # on score area so small-T cross-attention (SD text conditioning, T=77) stays
    # single-shot, where it is byte-identical and already fastest.
    from . import _optimize as _opt
    n_tiles = _opt.attention_tiles(S, n_heads, dh, T=T) if (S >= 768 and T >= 512) else 1
    if n_tiles == 1:
        o = ((qh @ kt) * scale).softmax(-1) @ vh                # [H, S, dh]
    else:
        tile = -(-S // n_tiles)                                 # ceil -> near-even chunks
        parts = []
        for start in range(0, S, tile):
            n = min(tile, S - start)
            qt = qh.slice_by_size([0, start, 0], [n_heads, n, dh])
            parts.append(((qt @ kt) * scale).softmax(-1) @ vh)  # [H, n, dh]
        o = concat(parts, axis=1)                               # [H, S, dh]
    o = o.transpose([1, 0, 2]).reshape(S, D)
    return o.linear(Wo, bo)


# The native fused-attention layer is numerically reliable only up to this sequence
# length. Measured 2026-06-02 (the reverse-engineering corpus): native vs an fp32
# reference is ~3e-3 at S<=2048 but ~1.0 (garbage) at S=4096, while the decomposed
# matmul/softmax/matmul stays ~5e-3 (the wide matmul accumulator holds). So above
# this S, af.sdpa emits the accurate decomposition instead of the native node.
SDPA_NATIVE_MAX_SEQ = 2048
# The native fused-attention layer returns garbage once BOTH the query and key sequence axes
# are large: reliable only while min(q_seq, k_seq) < this bound (measured 2026-06-09: a hard
# 512x512 score-tile cliff - cos collapses to ~0.67 at 512x512, while the KV-cache decode
# shape min=1 stays correct to large k_seq). Outside it we decompose (non-causal) or refuse
# (causal - the decomposition can't build the additive mask here).
SDPA_NATIVE_MIN_BOTH = 512


def sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float | None = None,
         is_causal: bool = False, attn_mask: "Tensor | None" = None) -> Tensor:
    """Scaled-dot-product attention. Uses the ANE's *native* fused-attention hardware
    layer (ANECSDPALayerDesc) - a path Apple's user-space MIL compiler never emits
    (it always decomposes SDPA) - for sequence lengths where it is numerically
    reliable (S <= SDPA_NATIVE_MAX_SEQ); above that it emits the accurate fused
    decomposition instead (the native layer returns garbage at large S). q/k/v:
    [1, heads, seq, d_head], fp16. Returns the same shape. `scale` defaults to
    1/sqrt(d_head).

    Where the native layer is used this is a graph-cut boundary: the surrounding graph
    runs as e5rt program(s) and this node runs as a separate native-SDPA ANE
    sub-program (see _compile.compile). `is_causal=True` is NATIVE: the causal additive
    mask rides the SDPA layer's optional 5th bottom (kept on the native bridge route -
    the route optimizer does not decompose it, since the decomposition is unmasked).
    Validated on M1: cos 1.0 vs softmax(QK^T*scale + causal)*V, single + multi-head.
    Requires S <= SDPA_NATIVE_MAX_SEQ (above that the op decomposes, which has no mask)."""
    # K and V share shape (the cached sequence); Q's SEQUENCE may differ from K/V's - the
    # KV-cache DECODE shape (q seq_q attends to cached k/v of length seq_kv). Q,K share H+D.
    if not (len(q.shape) == 4 == len(k.shape) == len(v.shape)):
        raise ValueError(f"af.sdpa expects 4D q,k,v of [1,H,S,D]; got {q.shape}, {k.shape}, {v.shape}")
    if k.shape != v.shape:
        raise ValueError(f"af.sdpa: k,v must share shape (the cached sequence); got {k.shape}, {v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
        raise ValueError(f"af.sdpa: q,k must share H (heads) and D (embedding); got {q.shape}, {k.shape}")
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError(f"af.sdpa: batch must be 1 (native layer); got {q.shape[0]}/{k.shape[0]}")
    if is_causal and q.shape[2] != k.shape[2]:
        raise ValueError("af.sdpa: is_causal requires equal q/k seq (prefill); for KV-cache "
                         "decode (seq_q < seq_kv) the new tokens attend to all cached k/v - "
                         "pass is_causal=False (or a runtime attn_mask).")
    if attn_mask is not None:
        if is_causal:
            raise ValueError("af.sdpa: pass either is_causal or an explicit attn_mask, not both")
        # attn_mask is a RUNTIME additive bias broadcastable with the [.,.,Sq,Skv] scores
        # (rides the native SDPA layer's 5th bottom). The native layer applies ONE additive-mask
        # plane shared across all heads, over the FULL query axis: shape must be [1,1,Sq,Skv].
        # A per-head mask ([1,H,Sq,Skv], H>1) is silently mis-applied (one plane used for all
        # heads -> wrong), and a query-broadcast mask ([1,1,1,Skv] with q_seq>1) underflows the
        # bridge -- reject both rather than return garbage. (For KV-cache decode q_seq==1, so
        # [1,1,1,Skv] is a full-query plane and is accepted.)
        if (len(attn_mask.shape) != 4 or attn_mask.shape[0] != 1 or attn_mask.shape[1] != 1
                or attn_mask.shape[2] != q.shape[2] or attn_mask.shape[3] != k.shape[2]):
            raise ValueError(
                f"af.sdpa: attn_mask must be a single shared plane [1,1,Sq,Skv]="
                f"{[1, 1, q.shape[2], k.shape[2]]} (one mask for all heads, full query axis); "
                f"got {list(attn_mask.shape)}. Per-head masks (H>1) and query-broadcast "
                f"(Sq-axis=1 while q_seq>1) are not supported by the native layer.")
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5
    seq = max(q.shape[2], k.shape[2])               # attention spans the K/V (cached) length
    both = min(q.shape[2], k.shape[2])              # the native layer breaks when BOTH axes are large
    # Native fused-attention is reliable only inside this envelope; outside it returns garbage.
    native_ok = both < SDPA_NATIVE_MIN_BOTH and seq <= SDPA_NATIVE_MAX_SEQ
    if not native_ok:
        if is_causal:
            # The accurate decomposition would need a causal additive mask, but there's no
            # host-constant add on the graph here, and the native layer is unreliable at this
            # size -- refuse rather than return a wrong answer. Chunk the query into tiles whose
            # min(q,k) seq stays < SDPA_NATIVE_MIN_BOTH.
            raise NotImplementedError(
                f"af.sdpa: causal attention at min(q,k)seq={both} (>= {SDPA_NATIVE_MIN_BOTH}) "
                f"or seq={seq} (> {SDPA_NATIVE_MAX_SEQ}) is outside the reliable native regime "
                f"and the causal decomposition is not wired; chunk the query so each tile's "
                f"min(seq) < {SDPA_NATIVE_MIN_BOTH}.")
        # non-causal (optionally with a runtime Tensor mask): the accurate fused decomposition
        # (handles the decode shape too - Q[.,.,Sq,.] @ K^T -> [.,.,Sq,Skv].softmax @ V).
        #
        # Query-tiling: for a large query axis, compute the attention in [tile, Skv] score
        # tiles instead of materializing the full [Sq, Skv] score matrix. This is exact (each
        # tile attends to all keys) and is markedly faster on the ANE, which pipelines the
        # smaller score tiles far better than one big matmul+softmax (~3x at Sq=1500, H=16).
        # Tiles target ~512 query rows; a small query (e.g. KV-cache decode) takes one shot.
        kt = k.transpose([0, 1, 3, 2])
        from . import _optimize as _opt
        n_tiles = _opt.attention_tiles(q.shape[2], q.shape[1], q.shape[3], T=k.shape[2])
        if n_tiles == 1:
            scores = q @ kt * float(scale)
            if attn_mask is not None:
                scores = scores + attn_mask
            return scores.softmax(-1) @ v
        Sq = q.shape[2]
        tile = -(-Sq // n_tiles)                            # ceil -> near-even chunks
        out_tiles = []
        for start in range(0, Sq, tile):
            n = min(tile, Sq - start)
            qt = q.slice_by_size([0, 0, start, 0], [q.shape[0], q.shape[1], n, q.shape[3]])
            st = qt @ kt * float(scale)
            if attn_mask is not None:
                st = st + attn_mask.slice_by_size(
                    [0, 0, start, 0], [attn_mask.shape[0], attn_mask.shape[1], n, attn_mask.shape[3]])
            out_tiles.append(st.softmax(-1) @ v)
        return concat(out_tiles, axis=2)
    if attn_mask is not None:                       # runtime mask rides the 5th bottom (stays native)
        return Tensor(q.shape, "sdpa", [q, k, v, attn_mask], {"scale": float(scale), "masked": True})
    return Tensor(q.shape, "sdpa", [q, k, v], {"scale": float(scale), "causal": bool(is_causal)})


def geglu(x: Tensor, W, b) -> Tensor:
    """GEGLU FFN gate: split the [2*Dff, D] projection into value/gate halves
    (weight-split at build, no slice op), out = value * gelu(gate)."""
    W = np.asarray(W); Dff = W.shape[0] // 2
    bv = bg = None
    if b is not None:
        b = np.asarray(b); bv, bg = b[:Dff], b[Dff:]
    return x.linear(W[:Dff], bv) * x.linear(W[Dff:], bg).gelu()
