"""aneforge - a clean graph->compile->run frontend for the Apple Neural Engine.

Build a small tensor graph, `compile` it into ONE fused e5rt program, and run it
on the ANE. Fusing is the point: the ANE penalises many tiny dispatches, so a whole
subgraph becomes a single program. Weights pack automatically into one BLOBFILE -
fp16, or per-channel int8 *streamed* (dequantised during the tile DMA) when
`int8=True`.

    import aneforge as af

    x = af.input((1, 3, 32, 32))
    h = af.conv(x, W1, pad=1).relu()
    h = af.conv(h, W2, pad=1).relu()
    y = h.mean((2, 3)).reshape(1, C) @ Wfc
    net = af.compile(y, int8=True)      # one fused ANE program
    net = af.compile(y, compress="int4")   # 4-bit LUT weights, accuracy-gated
    out = net(image)                    # run on the ANE

Op surface:
  - linear algebra: conv, conv_transpose; matmul/linear via `@`; bmm
  - dynamic_conv: conv with a RUNTIME-tensor weight (hypernetworks / per-sample kernels;
    native ANE dynamic kernel, batch-1 only)
  - activations: relu/silu/gelu/sigmoid/tanh/exp/log/sqrt/rsqrt/abs/square/
    sin/cos/erf/softplus/relu6/elu/leaky_relu/clip
  - arithmetic: add/sub/mul/div(`/`)/maximum/minimum/pow
  - reductions/norms: mean/sum/amax/amin, softmax, l2_norm, rms_norm/layer_norm/
    group_norm/batch_norm
  - spatial/shape: max_pool/avg_pool, upsample, concat, reshape/transpose,
    pixel_shuffle/pixel_unshuffle
  - nn helpers: mha, cross_attention, geglu

Two op routes. Most ops are FUSED e5rt-MIL: they lower to MIL and fuse into ONE
program (no graph cut). A second family are NETPLIST-BRIDGE ops - native Path-A
hardware layers Apple's MIL frontend never emits (sdpa, argmax/topk/sort,
cross_product/cross_correlation/cost_volume, fps/radius_search, minmax_norm/lrn,
the space/channel/batch rearranges, flatten/input_view/dynamic_slice/
scaled_elementwise). Each bridge op CUTS the graph: surrounding regions run as
e5rt programs, the bridge node runs as a separate native sub-program (sub-ms via
the A2 persistent worker), and `compile` returns a SegmentedModel.

Image input: `af.image_input(shape, scale=1/255, bias=0.0)` declares a uint8 input
port and dequantises it on the engine (`cast -> scale -> bias`), so raw camera /
decoded-video bytes feed the model directly (host skips the float-convert/repack);
`scale`/`bias` are scalar or per-channel (length-C, broadcast over NCHW).

Pretrained loaders: `af.load(".../all-MiniLM-L6-v2")` (sentence encoder),
`af.load_resnet18()` (ImageNet classifier).

Design rules: compute is fp16 only (fp32/int32/bf16
rejected); reductions/matmuls use a WIDE (fp32-class) accumulator fed by radix-4
fp16-rounded input tiles - representable sums are near-exact (a sum/dot of 16384 ones is
bit-exact, where naive fp16 would stall at ~2048), and a +1 survives next to a 16000
partial that an fp16 running sum would swallow. The fp16 limit is at the products and the
I/O cast, not the running sum, so cancellation-heavy reductions still lose precision;
`int8=True` streams weights at half the bytes. `compress=`
chooses weight encoding: None (fp16, default), 'int8' (per-channel), 'int4'
(LUT palettization, per-tensor, with an accuracy-gated fallback to int8/fp16 set by
`compress_atol`), 'sparse' (unstructured bitmask, emitted when the weight is >=50%
zeros, else fp16), or 'auto' (per-weight: sparse if sparse, else int4 if accurate,
else int8, else fp16). `int8=True` is the alias for `compress='int8'`. Wraps the
unentitled Espresso `e5rt` runtime only - no CoreML, no entitlement.

aneforge also has a tiny reverse-mode autograd (`autograd.py`): `af.parameter` /
`af.backward` / `af.mse` / `af.SGD` / `af.Trainer` train a small model with the
forward and backward passes compiled and run on the ANE. It also does
classification: `af.softmax_cross_entropy` (analytic fp16-stable on-ANE gradient) +
`af.Adam` train a 784->128->10 MLP on MNIST to ~97% test accuracy.
`Trainer(..., device_optimizer=True)` additionally runs the OPTIMIZER STEP on the
ANE (SGD/Adam update as graph ops), so all training tensor-math is on the engine;
the host only computes the scalar lr_t and shuttles state/grads (the host<->device
state round-trip remains). See examples/train_mnist_mlp.py.

Layout: graph.py (Tensor + ops), _compile.py (per-op emit registry + compile),
_blob.py (weight packing), autograd.py (on-ANE autograd), models.py (pretrained loaders).
"""
# Tolerate a duplicate OpenMP runtime in one process: numpy/MKL and the ANE
# runtime dylib each bring their own libomp, and without this the second to load
# aborts the process. Set before any import that pulls in numpy so OpenMP sees it
# at initialization; setdefault keeps it user-overridable.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ._version import __version__            # written by hatch-vcs at build time
except ImportError:                              # raw source checkout, not yet built
    from importlib.metadata import version, PackageNotFoundError
    try: __version__ = version("aneforge")
    except PackageNotFoundError: __version__ = "0+unknown"

from ._op_catalog import (OP_CATALOG, op_info, device_status, is_native, ops_on,
                          min_native_family, walled_everywhere, categories)
from .graph import (Tensor, affine, batch_norm, batch_to_space, channel_to_space, concat,
                    conv, conv_transpose, crop, dynamic_conv, cross_attention, cross_correlation,
                    cross_product, cost_volume, depth_to_space, dynamic_slice, einsum_native,
                    flatten, fps, gather, geglu, image_input, input, input_view, instance_norm,
                    local_response_norm, lrn, maximum, minimum, mha, minmax_norm,
                    pixel_shuffle, pixel_unshuffle, radius_search, resize_bilinear,
                    resize_nearest_neighbor, scaled_elementwise, sdpa, select, space_to_batch,
                    space_to_channel, space_to_depth, split, sort, stack, topk,
                    upsample_bilinear, where)
from ._compile import (Model, SegmentedModel, compile, PrecisionWarning,
                       CrossChipFP16Warning, DispatchFloorWarning)
from ._paired import Paired, paired
from ._optimize import tune, tune_precision
from ._cost import estimate, estimate_provenance, precision_risk, project_peak
from ._circuit import CompileBackoffError, reset as reset_compile_breaker
from ._rewrite import reduce_sum_to_matmul, paired_subtract
from .autograd import (Adam, adam_step, backward, backward_from, conv2d, conv_param,
                       mse, parameter, SGD, softmax_cross_entropy, Trainer, UnrolledTrainer)
from .streaming import CheckpointedStack
from .models import Encoder, Vision, load, load_resnet18, conv_block, cifar_cnn, group_norm_train

__all__ = [
    "Tensor", "Model", "SegmentedModel", "PrecisionWarning", "CrossChipFP16Warning",
    "DispatchFloorWarning",
    "input", "image_input", "conv", "conv_transpose", "dynamic_conv", "concat",
    "batch_norm", "maximum", "minimum", "mha", "cross_attention", "geglu", "sdpa",
    "pixel_shuffle", "pixel_unshuffle", "topk", "sort", "cross_product",
    "cross_correlation", "cost_volume", "fps", "radius_search", "minmax_norm", "lrn",
    "space_to_channel", "channel_to_space", "space_to_batch", "batch_to_space",
    "flatten", "input_view", "dynamic_slice", "scaled_elementwise",
    "stack", "split", "select", "where", "OP_CATALOG", "op_info", "device_status", "is_native", "ops_on", "min_native_family", "walled_everywhere", "categories", "gather", "instance_norm", "local_response_norm", "einsum_native",
    "space_to_depth", "depth_to_space", "crop", "resize_nearest_neighbor",
    "resize_bilinear", "upsample_bilinear", "affine",
    "compile", "tune", "tune_precision", "estimate", "estimate_provenance",
    "precision_risk", "project_peak",
    "CompileBackoffError", "reset_compile_breaker",
    "reduce_sum_to_matmul", "paired_subtract",
    "load", "load_resnet18", "Encoder", "Vision", "conv_block", "cifar_cnn", "group_norm_train",
    "Paired", "paired",
    "parameter", "backward", "backward_from", "mse", "SGD", "Adam",
    "softmax_cross_entropy", "Trainer", "UnrolledTrainer", "adam_step",
    "conv_param", "conv2d", "CheckpointedStack",
    "fft", "linalg", "special", "einsum", "dsp",
]

# Applied-math submodules (import for discoverability: af.fft / af.linalg / af.special /
# af.dsp). Each is self-contained over the public ops.
from . import fft as fft
from . import linalg as linalg
from . import special as special
from . import einsum as einsum
from . import dsp as dsp

# `af.einsum(...)` is the general decomposer, directly callable. The package attribute
# shadows the submodule of the same name; `import aneforge.einsum` and
# `from aneforge.einsum import ...` still resolve to the module via sys.modules.
from .einsum import einsum  # noqa: F811
