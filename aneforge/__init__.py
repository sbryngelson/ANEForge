"""aneforge - a graph->compile->run frontend for the Apple Neural Engine.

Build a tensor graph, `compile` it into one fused e5rt program (or a SegmentedModel
when a netplist-bridge op cuts the graph), and run it on the ANE. fp16 compute with a
wide reduction accumulator; weights pack into one BLOBFILE (fp16 / int8 / int4 /
sparse). Wraps the unentitled Espresso `e5rt` runtime only - no CoreML, no entitlement.

See docs/developer/overview.md for the op surface, fused-vs-bridge routes, numerics,
weight compression, image input, and on-ANE autograd.
"""
# Tolerate a duplicate OpenMP runtime (numpy/MKL and the ANE dylib each bring their
# own libomp; without this the second to load aborts). Must precede any numpy import.
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
from ._optimize import tune, tune_precision, tune_attention, attention_tiles
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
    "compile", "tune", "tune_precision", "tune_attention", "attention_tiles", "estimate", "estimate_provenance",
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
