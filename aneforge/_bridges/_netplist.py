#!/usr/bin/env python3
"""Shared netplist author for the Path-A bridges: generate net.plist + weights.* programs (no CoreML / MLCompute) and dispatch them."""

from __future__ import annotations

import argparse
import json
import plistlib
import struct
import subprocess
from pathlib import Path

_INVOKERS_DIR = Path(__file__).resolve().parents[1] / "_invokers"     # aneforge/_invokers/


def bin_dir() -> Path:
  """Per-machine compiled-invoker binary cache, outside the package tree (read-only site-packages safe)."""
  return Path.home() / ".cache" / "aneforge" / "bin"


def ensure_invoker(name: str) -> Path:
  """Compile `_invokers/<name>.mm` -> `bin_dir()/<name>` on demand (rebuilt when source is newer); `sdpa_invoker` is the generic invoker shared by geometry/structural/rearrange ops."""
  src = _INVOKERS_DIR / f"{name}.mm"
  binp = bin_dir() / name
  if binp.exists() and src.exists() and binp.stat().st_mtime >= src.stat().st_mtime: return binp
  if not src.exists(): raise FileNotFoundError(f"invoker source missing: {src}")
  binp.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["xcrun", "clang++", "-O2", "-fobjc-arc", "-std=gnu++17",
       "-framework", "Foundation", "-framework", "IOSurface", str(src), "-o", str(binp)]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0: raise RuntimeError(f"failed to build invoker {name}:\n{proc.stderr}")
  return binp


def invoke_netplist(invoker, net_plist, *, weights=(), inputs=(), outputs=(), repeats: int | None = 1, warmup=None, extra=()):
  """Shared Path-A dispatch: build the invoker command and run it. `inputs`/`outputs` are (name, path) pairs; `repeats`/`warmup` omitted when None (rank_invoker rejects them). Raises on non-zero rc or non-ok status; returns the status dict."""
  cmd = [str(invoker), "--net-plist", str(net_plist)]
  for w in weights: cmd += ["--weights", str(w)]
  for name, path in inputs: cmd += ["--input", f"{name}={path}"]
  for name, path in outputs: cmd += ["--output", f"{name}={path}"]
  if repeats is not None: cmd += ["--repeats", str(repeats)]
  if warmup is not None: cmd += ["--warmup", str(warmup)]
  cmd += list(extra)
  proc = subprocess.run(cmd, capture_output=True, text=True)
  name = Path(invoker).name
  if proc.returncode != 0:
    raise RuntimeError(
      f"{name} failed (rc={proc.returncode}):\n"
      f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
  # Prefer JSON lines (layer_invoker may emit non-JSON noise before the status line).
  json_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
  if not json_lines: raise RuntimeError(f"no JSON from {name}:\n{proc.stdout}\n{proc.stderr}")
  info = json.loads(json_lines[-1])
  if info.get("status") != "ok": raise RuntimeError(f"{name} non-ok status: {info}")
  return info


UNARY_NEURONS = {
  "relu": "ReLU",
  "sigmoid": "SigmoidHighPrecision",
  "gelu": "GELU",
  "tanh": "Tanh",
  "sqrt": "Sqrt",
  "rsqrt": "Rsqrt",
  "exp2": "Exp2",
  "log2": "Log2",
  "inv": "Inv",
  "sin": "Sin",
  "cos": "Cos",
  "ceil": "Ceil",
  "floor": "Floor",
  "round": "RoundNearest",
}

UNARY_ELEMENTWISE = {
  "abs": "Abs",
}

ELEMENTWISE = {
  "add": "Add",
  "mul": "Mult",
  "sub": "Sub",
  "min": "Min",
  "max": "Max",
  "greater": "GreaterThan",
  "less": "LessThan",
  "equal": "Equal",
  "not_equal": "NotEqual",
}

RUNTIME_CONST_TEMPLATE_OPS = [
  "add_const",
  "broadcast_const_add",
  "fill_like_as_broadcast_add",
  "gather_const_indices",
  "gather_along_axis_as_gather",
]

NATIVE_NETPLIST_OPS = {
  "cost_volume",
  "cross_product",
  "cross_correlation",
  "cross_correlation_add_relu",
  "cross_correlation_relu",
  "dynamic_goc_bias",
  "dynamic_goc_scale_bias",
  "goc_scalar_scale_bias",
  "fps_radius_search",
  "furthest_point_sampling",
  "kernel_rasterizer",
  "kernel_rasterizer_add_relu",
  "kernel_rasterizer_relu",
  "matrix_decomposition",
  "matrix_decomposition_add",
  "matrix_decomposition_relu",
  "matrix_decomposition_matmul",
  "dynamic_slice_broadcast_const_f16",
  "dynamic_slice_const_f16",
  "dynamic_slice_const_u16",
  "dynamic_slice_const_u16_w4",
  "minmax_norm",
  "radius_search",
}

OP_ALIASES = {
  "farthest_point_sampling": "furthest_point_sampling",
  "filllike_as_broadcast_add": "fill_like_as_broadcast_add",
  "filllike_lowered_to_broadcast_const_add": "fill_like_as_broadcast_add",
  "fps": "furthest_point_sampling",
  "gather_along_axis_lowered_to_gather": "gather_along_axis_as_gather",
  "conv1x1_gelu": "conv_gelu",
  "rs": "radius_search",
  "slice_by_index": "slice",
  "upsample_nearest_neighbor": "upsample_nearest",
}


def canonical_op(op: str) -> str:
  return OP_ALIASES.get(op, op)


def input_entry(
  symbol: str,
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  entry_name: str | None = None,
  batch_size: int = 1,
) -> dict:
  return {
    "BatchSize": batch_size,
    "InputChannels": channels,
    "InputDepth": 1,
    "InputHeight": height,
    "InputInterleave": 1,
    "InputName": symbol,
    "InputType": "Float16",
    "InputWidth": width,
    "Name": entry_name or symbol,
    "OperationName": "op0",
  }


def output_entry(symbol: str = "y", unit: str = "op") -> dict:
  return {
    "Name": unit,
    "OperationName": "op0",
    "OutputInterleave": 1,
    "OutputName": symbol,
    "OutputType": "Float16",
  }


def fp16_bits(value: float) -> int:
  return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def _canonical_dimension(dimension: str) -> str:
  normalized = dimension.strip().lower()
  names = {
    "w": "Width",
    "width": "Width",
    "h": "Height",
    "height": "Height",
    "c": "Channel",
    "channel": "Channel",
    "channels": "Channel",
    "d": "Depth",
    "depth": "Depth",
  }
  if normalized not in names: raise ValueError(f"unsupported dimension {dimension!r}")
  return names[normalized]


def build_unary(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": UNARY_NEURONS[op]},
    "Type": "Neuron",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_elementwise(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  unit = {
    "Bottom": ["x", "z"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": ELEMENTWISE[op]},
    "Type": "ElementWise",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name),
      input_entry("z", width=width, height=height, channels=channels, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_unary_elementwise(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": UNARY_ELEMENTWISE[op]},
    "Type": "ElementWise",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_parametric_neuron(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  params_by_op = {
    "elu": {"EluAlpha": 15360, "Type": "ELU"},
    "leaky_relu": {"ReluOffset": 0, "ReluSlope": 8479, "Type": "LeakyReLU"},
    "relu6": {"ReluMax": 17920, "ReluMin": 0, "Type": "ClampedReLU"},
    "clamped_relu": {"ReluMax": fp16_bits(6.0), "ReluMin": fp16_bits(0.125), "Type": "ClampedReLU"},
    "prelu": {"ReluOffset": 0, "ReluSlope": fp16_bits(0.125), "Type": "LeakyReLU"},
  }
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": params_by_op[op],
    "Type": "Neuron",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_silu(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_silu-1"
  units = {
    "silu-1_0": {
      "Bottom": ["x"],
      "InputType": ["Float16"],
      "Name": "silu-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "SigmoidHighPrecision"},
      "Type": "Neuron",
    },
    "silu-1_1": {
      "Bottom": ["x", "silu-1_0"],
      "InputType": ["Float16", "Float16"],
      "Name": "silu-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
  }
  return build_plist(
    network_name,
    "silu-1_1",
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name="silu-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="silu-1_1"),
    ],
    [output_entry("y", "silu-1_1")],
    units,
  )


def build_softsign(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_softsign-1"
  units = {
    "softsign-1_0": {
      "Bottom": ["x"],
      "InputType": ["Float16"],
      "Name": "softsign-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Abs"},
      "Type": "ElementWise",
    },
    "softsign-1_1": {
      "Bottom": ["c"],
      "InputType": ["Float16"],
      "Name": "softsign-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BroadcastInfo": [{"Dimension": "Width", "Size": width}]},
      "Type": "Broadcast",
    },
    "softsign-1_2": {
      "Bottom": ["softsign-1_0", "softsign-1_1"],
      "InputType": ["Float16", "Float16"],
      "Name": "softsign-1_2",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Add"},
      "Type": "ElementWise",
    },
    "softsign-1_3": {
      "Bottom": ["softsign-1_2"],
      "InputType": ["Float16"],
      "Name": "softsign-1_3",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Inv"},
      "Type": "Neuron",
    },
    "softsign-1_4": {
      "Bottom": ["x", "softsign-1_3"],
      "InputType": ["Float16", "Float16"],
      "Name": "softsign-1_4",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
  }
  plist = build_plist(
    network_name,
    "softsign-1_4",
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name="softsign-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="softsign-1_4"),
    ],
    [output_entry("y", "softsign-1_4")],
    units,
  )
  return attach_constant(plist, network_name, constant_entry("c", "Float16", width=1))


def build_thresholded_relu(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_thresholded_relu-1"
  units = {
    "thresholded_relu-1_0": {
      "Bottom": ["c"],
      "InputType": ["Float16"],
      "Name": "thresholded_relu-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BroadcastInfo": [{"Dimension": "Width", "Size": width}]},
      "Type": "Broadcast",
    },
    "thresholded_relu-1_1": {
      "Bottom": ["x", "thresholded_relu-1_0"],
      "InputType": ["Float16", "Float16"],
      "Name": "thresholded_relu-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "GreaterThan"},
      "Type": "ElementWise",
    },
    "thresholded_relu-1_2": {
      "Bottom": ["x", "thresholded_relu-1_1"],
      "InputType": ["Float16", "Float16"],
      "Name": "thresholded_relu-1_2",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
  }
  plist = build_plist(
    network_name,
    "thresholded_relu-1_2",
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name="thresholded_relu-1_1"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="thresholded_relu-1_2"),
    ],
    [output_entry("y", "thresholded_relu-1_2")],
    units,
  )
  return attach_constant(plist, network_name, constant_entry("c", "Float16", width=1))


def build_linear_activation(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "linear_activation-1"
  network_name = "network_linear_activation-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"BiasScalar": fp16_bits(0.2), "ScaleScalar": fp16_bits(1.25)},
    "Type": "GOC",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_hard_sigmoid(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_hard_sigmoid-1"
  units = {
    "hard_sigmoid-1": {
      "Bottom": ["x"],
      "InputType": ["Float16"],
      "Name": "hard_sigmoid-1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BiasScalar": 16640, "ScaleScalar": 12902},
      "Type": "GOC",
    },
    "hard_sigmoid-1_1": {
      "Bottom": ["hard_sigmoid-1"],
      "InputType": ["Float16"],
      "Name": "hard_sigmoid-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"ReluMax": 15360, "ReluMin": 0, "Type": "ClampedReLU"},
      "Type": "Neuron",
    },
  }
  return build_plist(
    network_name,
    "hard_sigmoid-1_1",
    [input_entry("x", width=width, height=height, channels=channels, entry_name="hard_sigmoid-1")],
    [output_entry("y", "hard_sigmoid-1_1")],
    units,
  )


def build_hard_swish(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_hard_swish-1"
  units = {
    "hard_swish-1_0": {
      "Bottom": ["x"],
      "InputType": ["Float16"],
      "Name": "hard_swish-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BiasScalar": 16896},
      "Type": "GOC",
    },
    "hard_swish-1_1": {
      "Bottom": ["hard_swish-1_0"],
      "InputType": ["Float16"],
      "Name": "hard_swish-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"ReluMax": 17920, "ReluMin": 0, "Type": "ClampedReLU"},
      "Type": "Neuron",
    },
    "hard_swish-1_2": {
      "Bottom": ["hard_swish-1_1"],
      "InputType": ["Float16"],
      "Name": "hard_swish-1_2",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"ScaleScalar": 12629},
      "Type": "GOC",
    },
    "hard_swish-1_3": {
      "Bottom": ["x", "hard_swish-1_2"],
      "InputType": ["Float16"],
      "Name": "hard_swish-1_3",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
  }
  return build_plist(
    network_name,
    "hard_swish-1_3",
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name="hard_swish-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="hard_swish-1_3"),
    ],
    [output_entry("y", "hard_swish-1_3")],
    units,
  )


def build_div(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_div-1"
  units = {
    "div-1_0": {
      "Bottom": ["z"],
      "InputType": ["Float16"],
      "Name": "div-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Inv"},
      "Type": "Neuron",
    },
    "div-1_1": {
      "Bottom": ["x", "div-1_0"],
      "InputType": ["Float16", "Float16"],
      "Name": "div-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
  }
  return build_plist(
    network_name,
    "div-1_1",
    [
      input_entry("z", width=width, height=height, channels=channels, entry_name="div-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="div-1_1"),
    ],
    [output_entry("y", "div-1_1")],
    units,
  )


def build_floor_div(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_floor_div-1"
  units = {
    "floor_div-1_0": {
      "Bottom": ["z"],
      "InputType": ["Float16"],
      "Name": "floor_div-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Inv"},
      "Type": "Neuron",
    },
    "floor_div-1_1": {
      "Bottom": ["x", "floor_div-1_0"],
      "InputType": ["Float16", "Float16"],
      "Name": "floor_div-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Mult"},
      "Type": "ElementWise",
    },
    "floor_div-1_2": {
      "Bottom": ["floor_div-1_1"],
      "InputType": ["Float16"],
      "Name": "floor_div-1_2",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Floor"},
      "Type": "Neuron",
    },
  }
  return build_plist(
    network_name,
    "floor_div-1_2",
    [
      input_entry("z", width=width, height=height, channels=channels, entry_name="floor_div-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name="floor_div-1_1"),
    ],
    [output_entry("y", "floor_div-1_2")],
    units,
  )


def build_comparison_or(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  if op not in {"less_equal", "greater_equal"}: raise ValueError(op)
  primary_type = "LessThan" if op == "less_equal" else "GreaterThan"
  network_name = f"network_{op}-1"
  units = {
    f"{op}-1_0": {
      "Bottom": ["x", "z"],
      "InputType": ["Float16", "Float16"],
      "Name": f"{op}-1_0",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": primary_type},
      "Type": "ElementWise",
    },
    f"{op}-1_1": {
      "Bottom": ["x", "z"],
      "InputType": ["Float16", "Float16"],
      "Name": f"{op}-1_1",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Equal"},
      "Type": "ElementWise",
    },
    f"{op}-1_2": {
      "Bottom": [f"{op}-1_0", f"{op}-1_1"],
      "InputType": ["Float16", "Float16"],
      "Name": f"{op}-1_2",
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Max"},
      "Type": "ElementWise",
    },
  }
  return build_plist(
    network_name,
    f"{op}-1_2",
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name=f"{op}-1_0"),
      input_entry("z", width=width, height=height, channels=channels, entry_name=f"{op}-1_0"),
      input_entry("x", width=width, height=height, channels=channels, entry_name=f"{op}-1_1"),
      input_entry("z", width=width, height=height, channels=channels, entry_name=f"{op}-1_1"),
    ],
    [output_entry("y", f"{op}-1_2")],
    units,
  )


def build_softmax(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "softmax-1"
  network_name = "network_softmax-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Dimension": "Width"},
    "Type": "Softmax",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_pooling(op: str) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  pool_type = {"maxpool2x2": "Max", "avgpool2x2": "Avg"}[op]
  params = {
    "KernelDepth": 1,
    "KernelHeight": 2,
    "KernelWidth": 2,
    "PadBack": 0,
    "PadBot": 0,
    "PadFront": 0,
    "PadLeft": 0,
    "PadRight": 0,
    "PadTop": 0,
    "PaddingMode": "Negative" if op == "maxpool2x2" else "Zero",
    "Step": [2, 2, 1],
    "Type": pool_type,
  }
  if op == "avgpool2x2": params["AverageCountExcludePadding"] = True
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": 1,
    "OutputType": "Float16",
    "Params": params,
    "Type": "Pooling",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=4, height=4, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_reduction(op: str, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = f"{op}-1"
  network_name = f"network_{op}-1"
  reduce_type = {
    "reduce_sum": "Sum",
    "reduce_mean": "Avg",
    "reduce_max": "Max",
    "reduce_min": "Min",
  }[op]
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Dimension": ["Width"], "Type": reduce_type},
    "Type": "Reduction",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_minmax_norm(
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  dimension: str = "Width",
  epsilon: float = 0.001,
) -> dict:
  unit_name = "minmax_norm-1"
  network_name = "network_minmax_norm-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {
      "Dimension": _canonical_dimension(dimension),
      "Epsilon": fp16_bits(epsilon),
    },
    "Type": "MinMaxNormalization",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_kernel_rasterizer(
  width: int = 4,
  height: int = 4,
  channels: int = 1,
  kernel_width: int = 3,
  kernel_height: int = 3,
  rasterizer_mode: str | int = "UnfoldedChannels",
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  if kernel_width < 1 or kernel_height < 1: raise ValueError("kernel_rasterizer kernel dimensions must be positive")
  if num_groups < 1 or output_channels < 1:
    raise ValueError("kernel_rasterizer num_groups and output_channels must be positive")
  unit_name = "kernel_rasterizer-1"
  network_name = "network_kernel_rasterizer-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": output_channels,
    "OutputType": "Float16",
    "Params": {
      "KernelHeight": kernel_height,
      "KernelWidth": kernel_width,
      "NumGroups": num_groups,
      "RasterizerMode": rasterizer_mode,
    },
    "Type": "KernelRasterizer",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_cross_correlation(
  width: int = 4,
  height: int = 4,
  channels: int = 1,
  template_width: int = 3,
  template_height: int = 3,
  pad_top: int = 0,
  pad_bottom: int = 0,
  pad_left: int = 0,
  pad_right: int = 0,
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  if template_width < 1 or template_height < 1:
    raise ValueError("cross_correlation template dimensions must be positive")
  if num_groups < 1 or output_channels < 1:
    raise ValueError("cross_correlation num_groups and output_channels must be positive")
  unit_name = "cross_correlation-1"
  network_name = "network_cross_correlation-1"
  template_input_width = template_width * template_height * channels
  unit = {
    "Bottom": ["x", "template"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": output_channels,
    "OutputType": "Float16",
    "Params": {
      "NumGroups": num_groups,
      "PadBot": pad_bottom,
      "PadLeft": pad_left,
      "PadRight": pad_right,
      "PadTop": pad_top,
      "TemplateHeight": template_height,
      "TemplateWidth": template_width,
    },
    "Type": "CrossCorrelation",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name),
      input_entry("template", width=template_input_width, height=1, channels=1, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_cost_volume(
  ref_width: int = 4,
  aux_width: int = 3,
  height: int = 3,
  channels: int = 1,
  disparity_direction: int = 0,
  disparity_range: int = 1,
  output_channels: int | None = None,
  scale_factor_x: float | None = None,
  scale_factor_y: float | None = None,
) -> dict:
  if ref_width < 1 or aux_width < 1 or height < 1 or channels < 1:
    raise ValueError("cost_volume dimensions must be positive")
  if disparity_direction != 0: raise ValueError("cost_volume currently supports disparity_direction=0")
  if disparity_range < 0: raise ValueError("cost_volume disparity_range must be nonnegative")
  if ref_width < aux_width + disparity_range:
    raise ValueError("cost_volume requires ref_width >= aux_width + disparity_range")
  out_channels = disparity_range + 1 if output_channels is None else int(output_channels)
  if out_channels < 1: raise ValueError("cost_volume output_channels must be positive")
  params: dict[str, object] = {
    "DisparityDirection": int(disparity_direction),
    "DisparityRange": int(disparity_range),
  }
  if scale_factor_x is not None: params["ScaleFactorX"] = float(scale_factor_x)
  if scale_factor_y is not None: params["ScaleFactorY"] = float(scale_factor_y)
  unit_name = "cost_volume-1"
  network_name = "network_cost_volume-1"
  unit = {
    "Bottom": ["aux", "ref"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": out_channels,
    "OutputType": "Float16",
    "Params": params,
    "Type": "CostVolume",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("aux", width=aux_width, height=height, channels=channels, entry_name=unit_name),
      input_entry("ref", width=ref_width, height=height, channels=channels, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_cross_product(batch_size: int = 1, output_channels: int = 3, params_style: str = "empty") -> dict:
  if batch_size < 1: raise ValueError("cross_product batch_size must be positive")
  if output_channels < 1: raise ValueError("cross_product output_channels must be positive")
  unit_name = "cross_product-1"
  network_name = "network_cross_product-1"
  unit = {
    "Bottom": ["x", "z"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": int(output_channels),
    "OutputType": "Float16",
    "Type": "CrossProduct",
  }
  if params_style == "empty": unit["Params"] = {}
  elif params_style == "type_cross_product": unit["Params"] = {"Type": "CrossProduct"}
  elif params_style != "absent":
    raise ValueError("cross_product params_style must be empty, absent, or type_cross_product")
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("x", width=1, height=1, channels=3, entry_name=unit_name, batch_size=batch_size),
      input_entry("z", width=1, height=1, channels=3, entry_name=unit_name, batch_size=batch_size),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_dynamic_goc(mode: str = "bias", channels: int = 4, params_style: str = "minimal") -> dict:
  if channels not in {1, 2, 3, 4, 8}: raise ValueError("dynamic_goc channels must be one of 1, 2, 3, 4, or 8")
  if mode not in {"bias", "scale_bias"}: raise ValueError("dynamic_goc mode must be bias or scale_bias")
  unit_name = f"dynamic_goc_{mode}-1"
  network_name = f"network_dynamic_goc_{mode}-1"
  bottoms = ["x", "bias"] if mode == "bias" else ["x", "scale", "bias"]
  unit = {
    "Bottom": bottoms,
    "InputType": ["Float16"] * len(bottoms),
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Type": "DynamicGOC",
  }
  if mode == "bias":
    if params_style == "minimal": unit["Params"] = {"BiasOnly": True}
    elif params_style == "ternary": unit["Params"] = {"BiasOnly": True, "Type": "TERNARY_DYNAMIC_GOC"}
    elif params_style != "absent": raise ValueError("dynamic_goc_bias params_style must be minimal, ternary, or absent")
  else:
    if params_style == "minimal": unit["Params"] = {}
    elif params_style == "ternary": unit["Params"] = {"Type": "TERNARY_DYNAMIC_GOC"}
    elif params_style != "absent":
      raise ValueError("dynamic_goc_scale_bias params_style must be minimal, ternary, or absent")
  inputs = [
    input_entry("x", width=1, height=1, channels=channels, entry_name=unit_name),
  ]
  if mode == "scale_bias":
    inputs.append(input_entry("scale", width=1, height=1, channels=channels, entry_name=unit_name))
  inputs.append(input_entry("bias", width=1, height=1, channels=channels, entry_name=unit_name))
  return build_plist(
    network_name,
    unit_name,
    inputs,
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_goc_scalar_scale_bias(
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  scale: float = 2.0,
  bias: float = 1.0,
) -> dict:
  unit_name = "goc_scalar_scale_bias-1"
  network_name = "network_goc_scalar_scale_bias-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"ScaleScalar": fp16_bits(scale), "BiasScalar": fp16_bits(bias)},
    "Type": "GOC",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_furthest_point_sampling(
  width: int = 4,
  centroid_count: int = 2,
  distance_metric: str | None = "L1",
  output_channels: int = 3,
) -> dict:
  if width < 2: raise ValueError("furthest_point_sampling requires at least two input points")
  if centroid_count < 2: raise ValueError("furthest_point_sampling requires centroid_count >= 2 on this runtime path")
  if centroid_count > width: raise ValueError("furthest_point_sampling requires centroid_count <= width")
  if output_channels < 1: raise ValueError("furthest_point_sampling output_channels must be positive")
  params: dict[str, object] = {"CentroidCount": int(centroid_count)}
  if distance_metric is not None:
    metric = str(distance_metric).upper()
    if metric not in {"L1", "L2"}: raise ValueError("furthest_point_sampling distance_metric must be L1, L2, or None")
    params["DistanceMetric"] = metric
  unit_name = "furthest_point_sampling-1"
  network_name = "network_furthest_point_sampling-1"
  unit = {
    "Bottom": ["points"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": int(output_channels),
    "OutputType": "Float16",
    "Params": params,
    "Type": "FurthestPointSampling",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("points", width=width, height=1, channels=3, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_radius_search(
  point_width: int = 4,
  centroid_width: int = 2,
  radius: float = 1.0,
  output_channels: int = 1,
) -> dict:
  if point_width < 1 or centroid_width < 1: raise ValueError("radius_search dimensions must be positive")
  if radius <= 0.0: raise ValueError("radius_search radius must be positive")
  if output_channels < 1: raise ValueError("radius_search output_channels must be positive")
  unit_name = "radius_search-1"
  network_name = "network_radius_search-1"
  unit = {
    "Bottom": ["centroids", "points"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": int(output_channels),
    "OutputType": "Float16",
    "Params": {"Radius": float(radius)},
    "Type": "RadiusSearch",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("centroids", width=centroid_width, height=1, channels=3, entry_name=unit_name),
      input_entry("points", width=point_width, height=1, channels=3, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_fps_radius_search(
  width: int = 4,
  centroid_count: int = 2,
  distance_metric: str | None = "L1",
  radius: float = 1.0,
  output_channels: int = 1,
) -> dict:
  if width < 2: raise ValueError("fps_radius_search requires at least two input points")
  if centroid_count < 2: raise ValueError("fps_radius_search requires centroid_count >= 2 on this runtime path")
  if centroid_count > width: raise ValueError("fps_radius_search requires centroid_count <= width")
  if radius <= 0.0: raise ValueError("fps_radius_search radius must be positive")
  if output_channels < 1: raise ValueError("fps_radius_search output_channels must be positive")
  fps_name = "fps-1"
  rs_name = "radius_search-1"
  network_name = "network_fps_radius_search-1"
  fps_params: dict[str, object] = {"CentroidCount": int(centroid_count)}
  if distance_metric is not None:
    metric = str(distance_metric).upper()
    if metric not in {"L1", "L2"}: raise ValueError("fps_radius_search distance_metric must be L1, L2, or None")
    fps_params["DistanceMetric"] = metric
  units = {
    fps_name: {
      "Bottom": ["points"],
      "InputType": ["Float16"],
      "Name": fps_name,
      "OutputChannels": 3,
      "OutputType": "Float16",
      "Params": fps_params,
      "Type": "FurthestPointSampling",
    },
    rs_name: {
      "Bottom": [fps_name, "points"],
      "InputType": ["Float16", "Float16"],
      "Name": rs_name,
      "OutputChannels": int(output_channels),
      "OutputType": "Float16",
      "Params": {"Radius": float(radius)},
      "Type": "RadiusSearch",
    },
  }
  return build_plist(
    network_name,
    rs_name,
    [
      input_entry("points", width=width, height=1, channels=3, entry_name=fps_name),
      input_entry("points", width=width, height=1, channels=3, entry_name=rs_name),
    ],
    [output_entry("y", rs_name)],
    units,
  )


def _relu_unit(name: str, bottom: str, channels: int = 1) -> dict:
  return {
    "Bottom": [bottom],
    "InputType": ["Float16"],
    "Name": name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": "ReLU"},
    "Type": "Neuron",
  }


def _add_unit(name: str, left: str, right: str, channels: int = 1) -> dict:
  return {
    "Bottom": [left, right],
    "InputType": ["Float16", "Float16"],
    "Name": name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": "Add"},
    "Type": "ElementWise",
  }


def build_kernel_rasterizer_relu(
  width: int = 5,
  height: int = 4,
  channels: int = 1,
  kernel_width: int = 3,
  kernel_height: int = 3,
  rasterizer_mode: str | int = "UnfoldedChannels",
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  plist = build_kernel_rasterizer(
    width,
    height,
    channels,
    kernel_width,
    kernel_height,
    rasterizer_mode,
    num_groups,
    output_channels,
  )
  network_name = "network_kernel_rasterizer-1"
  plist["Networks"] = ["network_kernel_rasterizer_relu-1"]
  plist["ProcedureList"][0]["Name"] = "procedure_kernel_rasterizer_relu-1"
  plist["ProcedureList"][0]["OperationList"][0]["NetworkName"] = "network_kernel_rasterizer_relu-1"
  network = plist.pop(network_name)
  network["relu-1"] = _relu_unit("relu-1", "kernel_rasterizer-1", output_channels)
  network["Units"] = ["kernel_rasterizer-1", "relu-1"]
  network["y"]["Bottom"] = "relu-1"
  plist["network_kernel_rasterizer_relu-1"] = network
  plist["ProcedureList"][0]["OutputList"] = [output_entry("y", "relu-1")]
  return plist


def build_kernel_rasterizer_add_relu(
  width: int = 5,
  height: int = 4,
  channels: int = 1,
  kernel_width: int = 3,
  kernel_height: int = 3,
  rasterizer_mode: str | int = "UnfoldedChannels",
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  plist = build_kernel_rasterizer(
    width,
    height,
    channels,
    kernel_width,
    kernel_height,
    rasterizer_mode,
    num_groups,
    output_channels,
  )
  network_name = "network_kernel_rasterizer-1"
  plist["Networks"] = ["network_kernel_rasterizer_add_relu-1"]
  plist["ProcedureList"][0]["Name"] = "procedure_kernel_rasterizer_add_relu-1"
  plist["ProcedureList"][0]["OperationList"][0]["NetworkName"] = "network_kernel_rasterizer_add_relu-1"
  network = plist.pop(network_name)
  network["add-1"] = _add_unit("add-1", "kernel_rasterizer-1", "residual", output_channels)
  network["relu-1"] = _relu_unit("relu-1", "add-1", output_channels)
  network["Units"] = ["kernel_rasterizer-1", "add-1", "relu-1"]
  network["y"]["Bottom"] = "relu-1"
  plist["network_kernel_rasterizer_add_relu-1"] = network
  plist["ProcedureList"][0]["InputList"].append(
    input_entry("residual", width=width * height * channels, height=1, channels=1, entry_name="add-1")
  )
  plist["ProcedureList"][0]["OutputList"] = [output_entry("y", "relu-1")]
  return plist


def _cross_correlation_output_shape(
  width: int,
  height: int,
  template_width: int,
  template_height: int,
  pad_top: int,
  pad_bottom: int,
  pad_left: int,
  pad_right: int,
) -> tuple[int, int]:
  out_width = width + pad_left + pad_right - template_width + 1
  out_height = height + pad_top + pad_bottom - template_height + 1
  if out_width < 1 or out_height < 1: raise ValueError("cross_correlation output shape would be empty")
  return out_width, out_height


def build_cross_correlation_relu(
  width: int = 4,
  height: int = 4,
  channels: int = 1,
  template_width: int = 3,
  template_height: int = 3,
  pad_top: int = 0,
  pad_bottom: int = 0,
  pad_left: int = 0,
  pad_right: int = 0,
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  plist = build_cross_correlation(
    width,
    height,
    channels,
    template_width,
    template_height,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
    num_groups,
    output_channels,
  )
  network_name = "network_cross_correlation-1"
  plist["Networks"] = ["network_cross_correlation_relu-1"]
  plist["ProcedureList"][0]["Name"] = "procedure_cross_correlation_relu-1"
  plist["ProcedureList"][0]["OperationList"][0]["NetworkName"] = "network_cross_correlation_relu-1"
  network = plist.pop(network_name)
  network["relu-1"] = _relu_unit("relu-1", "cross_correlation-1", output_channels)
  network["Units"] = ["cross_correlation-1", "relu-1"]
  network["y"]["Bottom"] = "relu-1"
  plist["network_cross_correlation_relu-1"] = network
  plist["ProcedureList"][0]["OutputList"] = [output_entry("y", "relu-1")]
  return plist


def build_cross_correlation_add_relu(
  width: int = 4,
  height: int = 4,
  channels: int = 1,
  template_width: int = 3,
  template_height: int = 3,
  pad_top: int = 0,
  pad_bottom: int = 0,
  pad_left: int = 0,
  pad_right: int = 0,
  num_groups: int = 1,
  output_channels: int = 1,
) -> dict:
  out_width, out_height = _cross_correlation_output_shape(
    width,
    height,
    template_width,
    template_height,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
  )
  plist = build_cross_correlation(
    width,
    height,
    channels,
    template_width,
    template_height,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right,
    num_groups,
    output_channels,
  )
  network_name = "network_cross_correlation-1"
  plist["Networks"] = ["network_cross_correlation_add_relu-1"]
  plist["ProcedureList"][0]["Name"] = "procedure_cross_correlation_add_relu-1"
  plist["ProcedureList"][0]["OperationList"][0]["NetworkName"] = "network_cross_correlation_add_relu-1"
  network = plist.pop(network_name)
  network["add-1"] = _add_unit("add-1", "cross_correlation-1", "residual", output_channels)
  network["relu-1"] = _relu_unit("relu-1", "add-1", output_channels)
  network["Units"] = ["cross_correlation-1", "add-1", "relu-1"]
  network["y"]["Bottom"] = "relu-1"
  plist["network_cross_correlation_add_relu-1"] = network
  plist["ProcedureList"][0]["InputList"].append(
    input_entry("residual", width=out_width, height=out_height, channels=output_channels, entry_name="add-1")
  )
  plist["ProcedureList"][0]["OutputList"] = [output_entry("y", "relu-1")]
  return plist


def _canonical_matrix_decomposition_type(decomposition_type: str) -> str:
  normalized = decomposition_type.strip().lower().replace("-", "_")
  aliases = {
    "nonqr": "NonQRGivens",
    "nonqr_givens": "NonQRGivens",
    "non_qr_givens": "NonQRGivens",
    "nonqrgivens": "NonQRGivens",
    "givens": "NonQRGivens",
    "cross": "CrossProductOrthoNormalization",
    "cross_product": "CrossProductOrthoNormalization",
    "cross_product_ortho_normalization": "CrossProductOrthoNormalization",
    "crossproductorthonormalization": "CrossProductOrthoNormalization",
  }
  if decomposition_type in {"NonQRGivens", "CrossProductOrthoNormalization"}: return decomposition_type
  if normalized not in aliases: raise ValueError(f"unsupported matrix decomposition type {decomposition_type!r}")
  return aliases[normalized]


def build_matrix_decomposition(
  decomposition_type: str = "NonQRGivens",
  batch_size: int = 1,
  epsilon: float = 0.001,
  rotation_axis: list[int] | None = None,
) -> dict:
  if batch_size < 1: raise ValueError("matrix_decomposition batch_size must be positive")
  unit_type = _canonical_matrix_decomposition_type(decomposition_type)
  input_channels = 9 if unit_type == "NonQRGivens" else 6
  output_channels = 4 if unit_type == "NonQRGivens" else 3
  params: dict[str, object] = {"Type": unit_type}
  if unit_type == "NonQRGivens":
    params["Epsilon"] = fp16_bits(epsilon)
    axes = rotation_axis if rotation_axis is not None else [7] * batch_size
    if len(axes) != batch_size: raise ValueError("matrix_decomposition rotation_axis length must equal batch_size")
    params["RotationAxis"] = [int(axis) for axis in axes]
  elif rotation_axis is not None:
    if len(rotation_axis) != batch_size:
      raise ValueError("matrix_decomposition rotation_axis length must equal batch_size")
    params["RotationAxis"] = [int(axis) for axis in rotation_axis]
  unit_name = "matrix_decomposition-1"
  network_name = "network_matrix_decomposition-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": output_channels,
    "OutputType": "Float16",
    "Params": params,
    "Type": "MatrixDecomposition",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry(
        "x",
        width=1,
        height=1,
        channels=input_channels,
        entry_name=unit_name,
        batch_size=batch_size,
      )
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_matrix_decomposition_relu(
  decomposition_type: str = "NonQRGivens",
  batch_size: int = 1,
  epsilon: float = 0.001,
  rotation_axis: list[int] | None = None,
) -> dict:
  unit_type = _canonical_matrix_decomposition_type(decomposition_type)
  if unit_type != "NonQRGivens":
    raise ValueError("matrix_decomposition_relu is currently recovered only for NonQRGivens")
  if batch_size < 1: raise ValueError("matrix_decomposition_relu batch_size must be positive")
  axes = rotation_axis if rotation_axis is not None else [7] * batch_size
  if len(axes) != batch_size: raise ValueError("matrix_decomposition_relu rotation_axis length must equal batch_size")
  md_name = "matrix_decomposition_relu-1_md"
  relu_name = "matrix_decomposition_relu-1_relu"
  network_name = "network_matrix_decomposition_relu-1"
  md_unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": md_name,
    "OutputChannels": 4,
    "OutputType": "Float16",
    "Params": {
      "Type": unit_type,
      "Epsilon": fp16_bits(epsilon),
      "RotationAxis": [int(axis) for axis in axes],
    },
    "Type": "MatrixDecomposition",
  }
  relu_unit = {
    "Bottom": [md_name],
    "InputType": ["Float16"],
    "Name": relu_name,
    "OutputChannels": 4,
    "OutputType": "Float16",
    "Params": {"Type": "ReLU"},
    "Type": "Neuron",
  }
  return build_plist(
    network_name,
    relu_name,
    [input_entry("x", width=1, height=1, channels=9, batch_size=batch_size, entry_name=md_name)],
    [output_entry("y", relu_name)],
    {md_name: md_unit, relu_name: relu_unit},
  )


def build_matrix_decomposition_add(
  decomposition_type: str = "NonQRGivens",
  batch_size: int = 1,
  epsilon: float = 0.001,
  rotation_axis: list[int] | None = None,
) -> dict:
  unit_type = _canonical_matrix_decomposition_type(decomposition_type)
  if unit_type != "NonQRGivens":
    raise ValueError("matrix_decomposition_add is currently recovered only for NonQRGivens")
  if batch_size < 1: raise ValueError("matrix_decomposition_add batch_size must be positive")
  axes = rotation_axis if rotation_axis is not None else [7] * batch_size
  if len(axes) != batch_size: raise ValueError("matrix_decomposition_add rotation_axis length must equal batch_size")
  md_name = "matrix_decomposition_add-1_md"
  add_name = "matrix_decomposition_add-1_add"
  network_name = "network_matrix_decomposition_add-1"
  md_unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": md_name,
    "OutputChannels": 4,
    "OutputType": "Float16",
    "Params": {
      "Type": unit_type,
      "Epsilon": fp16_bits(epsilon),
      "RotationAxis": [int(axis) for axis in axes],
    },
    "Type": "MatrixDecomposition",
  }
  add_unit = {
    "Bottom": [md_name, "residual"],
    "InputType": ["Float16", "Float16"],
    "Name": add_name,
    "OutputChannels": 4,
    "OutputType": "Float16",
    "Params": {"Type": "Add"},
    "Type": "ElementWise",
  }
  return build_plist(
    network_name,
    add_name,
    [
      input_entry("x", width=1, height=1, channels=9, batch_size=batch_size, entry_name=md_name),
      input_entry("residual", width=4, height=1, channels=4, batch_size=batch_size, entry_name=add_name),
    ],
    [output_entry("y", add_name)],
    {md_name: md_unit, add_name: add_unit},
  )


def build_matrix_decomposition_matmul(
  decomposition_type: str = "NonQRGivens",
  batch_size: int = 1,
  epsilon: float = 0.001,
  rotation_axis: list[int] | None = None,
  matmul_order: str = "md_v",
) -> dict:
  if batch_size < 1: raise ValueError("matrix_decomposition_matmul batch_size must be positive")
  unit_type = _canonical_matrix_decomposition_type(decomposition_type)
  input_channels = 9 if unit_type == "NonQRGivens" else 6
  output_channels = 4 if unit_type == "NonQRGivens" else 3
  order = matmul_order.strip().lower()
  if order not in {"md_v", "v_md"}:
    raise ValueError("matrix_decomposition_matmul matmul_order must be 'md_v' or 'v_md'")

  md_name = "matrix_decomposition_matmul-1_md"
  mm_name = "matrix_decomposition_matmul-1_mm"
  network_name = "network_matrix_decomposition_matmul-1"
  params: dict[str, object] = {"Type": unit_type}
  if unit_type == "NonQRGivens":
    axes = rotation_axis if rotation_axis is not None else [7] * batch_size
    if len(axes) != batch_size:
      raise ValueError("matrix_decomposition_matmul rotation_axis length must equal batch_size")
    params["Epsilon"] = fp16_bits(epsilon)
    params["RotationAxis"] = [int(axis) for axis in axes]
  elif rotation_axis is not None:
    if len(rotation_axis) != batch_size:
      raise ValueError("matrix_decomposition_matmul rotation_axis length must equal batch_size")
    params["RotationAxis"] = [int(axis) for axis in rotation_axis]
  md_unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": md_name,
    "OutputChannels": output_channels,
    "OutputType": "Float16",
    "Params": params,
    "Type": "MatrixDecomposition",
  }
  mm_unit = {
    "Bottom": [md_name, "v"] if order == "md_v" else ["v", md_name],
    "InputType": ["Float16", "Float16"],
    "Name": mm_name,
    "OutputChannels": output_channels,
    "OutputType": "Float16",
    "Type": "MatrixMultiplication",
  }
  return build_plist(
    network_name,
    mm_name,
    [
      input_entry("x", width=1, height=1, channels=input_channels, batch_size=batch_size, entry_name=md_name),
      input_entry("v", width=output_channels, height=1, channels=output_channels, batch_size=batch_size, entry_name=mm_name),
    ],
    [output_entry("y", mm_name)],
    {md_name: md_unit, mm_name: mm_unit},
  )


def build_slice(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "slice-1_0"
  network_name = "network_slice-1"
  if width < 3: raise ValueError("slice requires width >= 3 for the fixed offset=1,size=2 route")
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputType": "Float16",
    "Params": {"Dimension": "Width", "Offset": 1, "Size": 2},
    "Type": "InputView",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_upsample_nearest() -> dict:
  unit_name = "upsample_nearest-1_0"
  network_name = "network_upsample_nearest-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "NumGroups": 1,
    "OutputChannels": 1,
    "OutputType": "Float16",
    "Params": {
      "KernelGroupReuse": True,
      "KernelHeight": 2,
      "KernelMode": "Unity",
      "KernelType": "Float16",
      "KernelWidth": 2,
      "PadBot": 0,
      "PadLeft": 1,
      "PadRight": 0,
      "PadTop": 1,
      "PaddingMode": "Zero",
      "Step": [2, 2],
      "Type": "ChannelWiseDeConv",
    },
    "Type": "Conv",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=2, height=2, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_concat(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "concat-1"
  network_name = "network_concat-1"
  unit = {
    "Bottom": ["x", "z"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": 1,
    "OutputType": "Float16",
    "Params": {"Dimension": "Width"},
    "Type": "Concat",
  }
  return build_plist(
    network_name,
    unit_name,
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name),
      input_entry("z", width=width, height=height, channels=channels, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_reshape() -> dict:
  unit_name = "reshape-1"
  network_name = "network_reshape-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputType": "Float16",
    "Params": {
      "ReshapedBatch": 1,
      "ReshapedChannel": 1,
      "ReshapedHeight": 2,
      "ReshapedWidth": 2,
    },
    "Type": "Reshape",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=4, height=1, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_reshape_alias_3d(width: int = 16, height: int = 16, channels: int = 4) -> dict:
  unit_name = "reshape_alias_3d-1"
  network_name = "network_reshape_alias_3d-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputType": "Float16",
    "Params": {
      "ReshapedBatch": 1,
      "ReshapedChannel": channels,
      "ReshapedHeight": 1,
      "ReshapedWidth": width * height,
    },
    "Type": "Reshape",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def input_view_unit(name: str, bottom: str, dimension: str, offset: int, size: int) -> dict:
  return {
    "Bottom": [bottom],
    "InputType": ["Float16"],
    "Name": name,
    "OutputType": "Float16",
    "Params": {"Dimension": dimension, "Offset": offset, "Size": size},
    "Type": "InputView",
  }


def concat_unit(name: str, bottom: list[str], dimension: str, channels: int) -> dict:
  return {
    "Bottom": bottom,
    "InputType": ["Float16"] * len(bottom),
    "Name": name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Dimension": dimension},
    "Type": "Concat",
  }


def constant_entry(
  name: str,
  constant_type: str,
  width: int,
  height: int = 1,
  channels: int = 1,
  batch_size: int = 1,
  depth: int = 1,
  index: int = 1,
  byte_offset: int = 0,
) -> dict:
  return {
    "ConstantBatchSize": batch_size,
    "ConstantByteOffset": byte_offset,
    "ConstantChannels": channels,
    "ConstantDepth": depth,
    "ConstantHeight": height,
    "ConstantIndex": index,
    "ConstantInterleave": 1,
    "ConstantName": name,
    "ConstantType": constant_type,
    "ConstantWidth": width,
  }


def attach_constant(plist: dict, network_name: str, constant: dict) -> dict:
  network = plist[network_name]
  name = constant["ConstantName"]
  constants = list(network.get("Constants", []))
  if name not in constants: constants.append(name)
  network["Constants"] = constants
  network[name] = constant
  network["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_add_const(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "add_const-1"
  network_name = "network_add_const-1"
  unit = {
    "Bottom": ["x", "c"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"Type": "Add"},
    "Type": "ElementWise",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  return attach_constant(
    plist,
    network_name,
    constant_entry("c", "Float16", width=width, height=height, channels=channels),
  )


def build_broadcast_const_add(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_broadcast_const_add-1"
  broadcast = "broadcast_const-1"
  add = "add-1"
  units = {
    broadcast: {
      "Bottom": ["c"],
      "InputType": ["Float16"],
      "Name": broadcast,
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BroadcastInfo": [{"Dimension": "Width", "Size": width}]},
      "Type": "Broadcast",
    },
    add: {
      "Bottom": ["x", broadcast],
      "InputType": ["Float16", "Float16"],
      "Name": add,
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Add"},
      "Type": "ElementWise",
    },
  }
  plist = build_plist(
    network_name,
    add,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=add)],
    [output_entry("y", add)],
    units,
  )
  return attach_constant(plist, network_name, constant_entry("c", "Float16", width=1))


def build_fill_like_as_broadcast_add(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = "network_filllike_lowered_broadcast-1"
  broadcast = "filllike_lowered_broadcast-1"
  add = "filllike_lowered_add-1"
  units = {
    broadcast: {
      "Bottom": ["c"],
      "InputType": ["Float16"],
      "Name": broadcast,
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"BroadcastInfo": [{"Dimension": "Width", "Size": width}]},
      "Type": "Broadcast",
    },
    add: {
      "Bottom": ["x", broadcast],
      "InputType": ["Float16", "Float16"],
      "Name": add,
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "Add"},
      "Type": "ElementWise",
    },
  }
  plist = build_plist(
    network_name,
    add,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=add)],
    [output_entry("y", add)],
    units,
  )
  return attach_constant(plist, network_name, constant_entry("c", "Float16", width=1))


def build_gather_const_indices(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "gather_const-1"
  network_name = "network_gather_const_indices-1"
  unit = {
    "Bottom": ["x", "idx"],
    "InputType": ["Float16", "UInt16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"GatherNDAxes": ["Width"]},
    "Type": "Gather",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  return attach_constant(plist, network_name, constant_entry("idx", "UInt16", width=width))


def build_gather_along_axis_as_gather(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  unit_name = "gather_along_axis_lowered-1"
  network_name = "network_gather_along_axis_lowered-1"
  unit = {
    "Bottom": ["x", "idx"],
    "InputType": ["Float16", "UInt16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": {"GatherNDAxes": ["Width"]},
    "Type": "Gather",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  return attach_constant(plist, network_name, constant_entry("idx", "UInt16", width=width))


def build_slice_update_concat_equiv(
  width: int = 8,
  height: int = 8,
  channels: int = 1,
  patch_width: int = 4,
  patch_height: int = 4,
) -> dict:
  if patch_width < 1 or patch_width >= width: raise ValueError("patch_width must be between 1 and width - 1")
  if patch_height < 1 or patch_height >= height: raise ValueError("patch_height must be between 1 and height - 1")

  network_name = "network_slice_update_concat_equiv-1"
  top = "slice_update_top"
  top_right = "slice_update_top_right"
  top_concat = "slice_update_top_concat"
  bottom = "slice_update_bottom"
  final = "slice_update_concat_equiv-1"
  units = {
    top: input_view_unit(top, "x", "Height", 0, patch_height),
    top_right: input_view_unit(top_right, top, "Width", patch_width, width - patch_width),
    top_concat: concat_unit(top_concat, ["u", top_right], "Width", channels),
    bottom: input_view_unit(bottom, "x", "Height", patch_height, height - patch_height),
    final: concat_unit(final, [top_concat, bottom], "Height", channels),
  }
  return build_plist(
    network_name,
    final,
    [
      input_entry("x", width=width, height=height, channels=channels, entry_name=top),
      input_entry("x", width=width, height=height, channels=channels, entry_name=bottom),
      input_entry("u", width=patch_width, height=patch_height, channels=channels, entry_name=top_concat),
    ],
    [output_entry("y", final)],
    units,
  )


def build_dynamic_slice_const_f16(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  if width != 4 or height != 1 or channels != 1:
    raise ValueError("dynamic_slice_const_f16 currently requires width=4, height=1, channels=1")
  unit_name = "dynamic_slice_const_f16-1"
  network_name = "network_dynamic_slice_const_f16-1"
  unit = {
    "Bottom": ["x", "idx"],
    "InputType": ["Float16", "Float16"],
    "Name": unit_name,
    "OutputType": "Float16",
    "Params": {
      "BackgroundValue": 0.0,
      "CoordinateInfo": [{"Coordinate": "Width", "CoordinateMode": "NonNormalized"}],
      "CoordinateMode": "NonNormalized",
      "DynamicSliceAxisOrder": ["Width"],
      "DynamicSliceInfo": [{"Coordinate": "Width", "SliceSize": 2}],
      "PaddingInfo": [{"Coordinate": "Width", "PaddingMode": "Background"}],
    },
    "Type": "DynamicSlice",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=width, height=height, channels=channels, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  return attach_constant(
    plist,
    network_name,
    constant_entry("idx", "Float16", width=1, height=1, channels=1, index=1),
  )


def _build_dynamic_slice_constant_index(*, op_name: str, constant_type: str, constant_width: int) -> dict:
  unit_name = f"{op_name}-1"
  network_name = f"network_{op_name}-1"
  unit = {
    "Bottom": ["x", "idx"],
    "InputType": ["Float16", constant_type],
    "Name": unit_name,
    "OutputType": "Float16",
    "Params": {
      "BackgroundValue": 0.0,
      "CoordinateInfo": [{"Coordinate": "Width", "CoordinateMode": "NonNormalized"}],
      "CoordinateMode": "NonNormalized",
      "DynamicSliceAxisOrder": ["Width"],
      "DynamicSliceInfo": [{"Coordinate": "Width", "SliceSize": 2}],
      "PaddingInfo": [{"Coordinate": "Width", "PaddingMode": "Background"}],
    },
    "Type": "DynamicSlice",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=4, height=1, channels=1, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  return attach_constant(
    plist,
    network_name,
    constant_entry("idx", constant_type, width=constant_width, height=1, channels=1, index=1),
  )


def build_dynamic_slice_const_u16(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  if width != 4 or height != 1 or channels != 1:
    raise ValueError("dynamic_slice_const_u16 currently requires width=4, height=1, channels=1")
  return _build_dynamic_slice_constant_index(op_name="dynamic_slice_const_u16", constant_type="UInt16", constant_width=1)


def build_dynamic_slice_const_u16_w4(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  if width != 4 or height != 1 or channels != 1:
    raise ValueError("dynamic_slice_const_u16_w4 currently requires width=4, height=1, channels=1")
  return _build_dynamic_slice_constant_index(op_name="dynamic_slice_const_u16_w4", constant_type="UInt16", constant_width=4)


def build_dynamic_slice_broadcast_const_f16(width: int = 4, height: int = 1, channels: int = 1) -> dict:
  if width != 4 or height != 1 or channels != 1:
    raise ValueError("dynamic_slice_broadcast_const_f16 currently requires width=4, height=1, channels=1")
  network_name = "network_dynamic_slice_broadcast_const_f16-1"
  broadcast_name = "idx_broadcast-1"
  unit_name = "dynamic_slice_broadcast_const_f16-1"
  units = {
    broadcast_name: {
      "Bottom": ["idx_scalar"],
      "InputType": ["Float16"],
      "Name": broadcast_name,
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"BroadcastInfo": [{"Dimension": "Width", "Size": 4}]},
      "Type": "Broadcast",
    },
    unit_name: {
      "Bottom": ["x", broadcast_name],
      "InputType": ["Float16", "Float16"],
      "Name": unit_name,
      "OutputType": "Float16",
      "Params": {
        "BackgroundValue": 0.0,
        "CoordinateInfo": [{"Coordinate": "Width", "CoordinateMode": "NonNormalized"}],
        "CoordinateMode": "NonNormalized",
        "DynamicSliceAxisOrder": ["Width"],
        "DynamicSliceInfo": [{"Coordinate": "Width", "SliceSize": 2}],
        "PaddingInfo": [{"Coordinate": "Width", "PaddingMode": "Background"}],
      },
      "Type": "DynamicSlice",
    },
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=4, height=1, channels=1, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    units,
  )
  return attach_constant(
    plist,
    network_name,
    constant_entry("idx_scalar", "Float16", width=1, height=1, channels=1, index=1),
  )


def build_transpose() -> dict:
  unit_name = "transpose-1"
  network_name = "network_transpose-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "OutputChannels": 1,
    "OutputType": "Float16",
    "Params": {
      "TransposeDimensions": [
        {
          "TransposeDestinationDimension": "Height",
          "TransposeSourceDimension": "Width",
        },
        {
          "TransposeDestinationDimension": "Width",
          "TransposeSourceDimension": "Height",
        },
      ]
    },
    "Type": "Transpose",
  }
  return build_plist(
    network_name,
    unit_name,
    [input_entry("x", width=2, height=2, entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )


def build_matmul() -> dict:
  network_name = "network_matmul-1"
  units = {
    "matmul-1_0": {
      "Bottom": ["a"],
      "InputType": ["Float16"],
      "Name": "matmul-1_0",
      "OutputChannels": 4,
      "OutputType": "Float16",
      "Params": {
        "TransposeDimensions": [
          {
            "TransposeDestinationDimension": "Channel",
            "TransposeSourceDimension": "Height",
          },
          {
            "TransposeDestinationDimension": "Height",
            "TransposeSourceDimension": "Channel",
          },
        ]
      },
      "Type": "Transpose",
    },
    "matmul-1_1": {
      "Bottom": ["x"],
      "InputType": ["Float16"],
      "Name": "matmul-1_1",
      "OutputChannels": 4,
      "OutputType": "Float16",
      "Params": {
        "TransposeDimensions": [
          {
            "TransposeDestinationDimension": "Channel",
            "TransposeSourceDimension": "Height",
          },
          {
            "TransposeDestinationDimension": "Height",
            "TransposeSourceDimension": "Channel",
          },
        ]
      },
      "Type": "Transpose",
    },
    "matmul-1_2": {
      "Bottom": ["matmul-1_0", "matmul-1_1"],
      "InputType": ["Float16", "Float16"],
      "Name": "matmul-1_2",
      "OutputChannels": 4,
      "OutputType": "Float16",
      "Type": "MatrixMultiplication",
    },
    "matmul-1_3": {
      "Bottom": ["matmul-1_2"],
      "InputType": ["Float16"],
      "Name": "matmul-1_3",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {
        "TransposeDimensions": [
          {
            "TransposeDestinationDimension": "Channel",
            "TransposeSourceDimension": "Height",
          },
          {
            "TransposeDestinationDimension": "Height",
            "TransposeSourceDimension": "Channel",
          },
        ]
      },
      "Type": "Transpose",
    },
  }
  return build_plist(
    network_name,
    "matmul-1_3",
    [
      input_entry("a", width=4, height=4, entry_name="matmul-1_0"),
      input_entry("x", width=1, height=4, entry_name="matmul-1_1"),
    ],
    [output_entry("y", "matmul-1_3")],
    units,
  )


def build_conv1x1() -> dict:
  unit_name = "conv1x1-1"
  network_name = "network_conv1x1-1"
  unit = {
    "Bottom": ["x"],
    "InputType": ["Float16"],
    "Name": unit_name,
    "NumGroups": 1,
    "OutputChannels": 1,
    "OutputType": "Float16",
    "Params": {
      "KernelDepth": 1,
      "KernelGroupReuse": False,
      "KernelHeight": 1,
      "KernelIndex": 1,
      "KernelMode": "Dense",
      "KernelMutable": False,
      "KernelOffset": 0,
      "KernelType": "Float16",
      "KernelWidth": 1,
      "PadBack": 0,
      "PadBot": 0,
      "PadFront": 0,
      "PadLeft": 0,
      "PadRight": 0,
      "PadTop": 0,
      "PaddingMode": "Zero",
      "Step": [1, 1, 1],
      "Type": "Conv",
    },
    "Type": "Conv",
  }
  plist = build_plist(
    network_name,
    unit_name,
    [input_entry("x", entry_name=unit_name)],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )
  plist[network_name]["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_add_relu() -> dict:
  network_name = "network_add_relu-1"
  units = {
    "add-1": {
      "Bottom": ["x", "z"],
      "InputType": ["Float16", "Float16"],
      "Name": "add-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "Add"},
      "Type": "ElementWise",
    },
    "relu-1": {
      "Bottom": ["add-1"],
      "InputType": ["Float16"],
      "Name": "relu-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "ReLU"},
      "Type": "Neuron",
    },
  }
  return build_plist(
    network_name,
    "relu-1",
    [input_entry("x", entry_name="add-1"), input_entry("z", entry_name="add-1")],
    [output_entry("y", "relu-1")],
    units,
  )


def build_conv_relu() -> dict:
  network_name = "network_conv_relu-1"
  conv = build_conv1x1()["network_conv1x1-1"]["conv1x1-1"]
  conv["Bottom"] = ["x"]
  units = {
    "conv1x1-1": conv,
    "relu-1": {
      "Bottom": ["conv1x1-1"],
      "InputType": ["Float16"],
      "Name": "relu-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "ReLU"},
      "Type": "Neuron",
    },
  }
  plist = build_plist(
    network_name,
    "relu-1",
    [input_entry("x", entry_name="conv1x1-1")],
    [output_entry("y", "relu-1")],
    units,
  )
  plist[network_name]["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_conv_gelu() -> dict:
  network_name = "network_conv_gelu-1"
  conv = build_conv1x1()["network_conv1x1-1"]["conv1x1-1"]
  conv["Bottom"] = ["x"]
  units = {
    "conv1x1-1": conv,
    "gelu-1": {
      "Bottom": ["conv1x1-1"],
      "InputType": ["Float16"],
      "Name": "gelu-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "GELU"},
      "Type": "Neuron",
    },
  }
  plist = build_plist(
    network_name,
    "gelu-1",
    [input_entry("x", entry_name="conv1x1-1")],
    [output_entry("y", "gelu-1")],
    units,
  )
  plist[network_name]["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_conv_add_relu() -> dict:
  network_name = "network_conv_add_relu-1"
  conv = build_conv1x1()["network_conv1x1-1"]["conv1x1-1"]
  conv["Bottom"] = ["x"]
  units = {
    "conv1x1-1": conv,
    "add-1": {
      "Bottom": ["conv1x1-1", "z"],
      "InputType": ["Float16", "Float16"],
      "Name": "add-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "Add"},
      "Type": "ElementWise",
    },
    "relu-1": {
      "Bottom": ["add-1"],
      "InputType": ["Float16"],
      "Name": "relu-1",
      "OutputChannels": 1,
      "OutputType": "Float16",
      "Params": {"Type": "ReLU"},
      "Type": "Neuron",
    },
  }
  plist = build_plist(
    network_name,
    "relu-1",
    [input_entry("x", entry_name="conv1x1-1"), input_entry("z", entry_name="add-1")],
    [output_entry("y", "relu-1")],
    units,
  )
  plist[network_name]["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_relu_chain(depth: int, width: int = 4, height: int = 1, channels: int = 1) -> dict:
  network_name = f"network_relu_chain_d{depth}"
  units = {}
  bottom = "x"
  for i in range(depth):
    name = f"relu-{i + 1}"
    units[name] = {
      "Bottom": [bottom],
      "InputType": ["Float16"],
      "Name": name,
      "OutputChannels": channels,
      "OutputType": "Float16",
      "Params": {"Type": "ReLU"},
      "Type": "Neuron",
    }
    bottom = name
  return build_plist(
    network_name,
    bottom,
    [input_entry("x", width=width, height=height, channels=channels, entry_name="relu-1")],
    [output_entry("y", bottom)],
    units,
  )


def build_conv_chain(depth: int) -> dict:
  network_name = f"network_conv_chain_d{depth}"
  units = {}
  bottom = "x"
  for i in range(depth):
    name = f"conv1x1-{i + 1}"
    conv = build_conv1x1()["network_conv1x1-1"]["conv1x1-1"]
    conv["Name"] = name
    conv["Bottom"] = [bottom]
    units[name] = conv
    bottom = name
  plist = build_plist(
    network_name,
    bottom,
    [input_entry("x", entry_name="conv1x1-1")],
    [output_entry("y", bottom)],
    units,
  )
  plist[network_name]["Weights"] = ["weights.0", "weights.1"]
  return plist


def build_sdpa(
  channels: int = 64,
  sequence: int = 16,
  dim: int = 16,
  scale: float | None = None,
  constant_flag_spelling: str = "Constants_array",
  subtract_max: bool = True,
) -> dict:
  """Single-op SDPA netplist mirroring the fused-hardware route. `constant_flag_spelling` selects how the Scale tensor is flagged constant ("Constants_array" is the only spelling that loads here; "all" = probe). `subtract_max` -> Params.SubtractMax (default True; off breaks softmax). `scale` defaults to 1/sqrt(dim)."""
  if scale is None: scale = 1.0 / (float(dim) ** 0.5)
  unit_name = "sdpa-1"
  network_name = "network_sdpa-1"

  # Q, K, V are tensor inputs; Scale is the 4th Bottom (weight-backed constant).
  params: dict[str, object] = {"SubtractMax": bool(subtract_max)}
  unit: dict[str, object] = {
    "Bottom": ["query", "key", "value", "scale"],
    "InputType": ["Float16", "Float16", "Float16", "Float16"],
    "Name": unit_name,
    "OutputChannels": channels,
    "OutputType": "Float16",
    "Params": params,  # SubtractMax is the only Params key the SDPA parser reads
    "Type": "SDPA",
  }

  use_constants_array = constant_flag_spelling in {"Constants_array", "all"}
  if constant_flag_spelling in {"IsConstant_unit", "all"}: unit["IsConstant"] = [False, False, False, True]
  if constant_flag_spelling in {"is_constant_unit", "all"}: unit["is_constant"] = [False, False, False, True]
  if constant_flag_spelling in {"ConstantTensor_unit", "all"}: unit["ConstantTensor"] = [False, False, False, True]
  if constant_flag_spelling in {"ScaleMutable_false", "all"}:
    unit["ScaleMutable"] = False
    params["ScaleMutable"] = False

  plist = build_plist(
    network_name,
    unit_name,
    [
      input_entry("query", width=dim, height=sequence, channels=channels, entry_name=unit_name),
      input_entry("key",   width=dim, height=sequence, channels=channels, entry_name=unit_name),
      input_entry("value", width=dim, height=sequence, channels=channels, entry_name=unit_name),
    ],
    [output_entry("y", unit_name)],
    {unit_name: unit},
  )

  if use_constants_array:
    # Scale as a 1-element fp16 weight-backed constant (canonical Apple route).
    plist = attach_constant(
      plist,
      network_name,
      constant_entry("scale", "Float16", width=1, height=1, channels=1),
    )
  else:
    # No Constants array: add Scale as a plain input port.
    plist["ProcedureList"][0]["InputList"].append(
      input_entry("scale", width=1, height=1, channels=1, entry_name=unit_name)
    )

  return plist


def build_plist(
  network_name: str,
  output_bottom: str,
  inputs: list[dict],
  outputs: list[dict],
  units_by_name: dict[str, dict],
) -> dict:
  network: dict[str, object] = dict(units_by_name)
  network["Units"] = list(units_by_name)
  network["Weights"] = ["weights.0"]
  network["y"] = {
    "Bottom": output_bottom,
    "OutputInterleave": 1,
    "OutputName": "y",
    "OutputType": "Float16",
  }
  return {
    "Version": "1.0.10",
    "Networks": [network_name],
    "ProcedureList": [
      {
        "Name": network_name.replace("network_", "procedure_"),
        "InputList": inputs,
        "OperationList": [{"NetworkName": network_name, "OperationName": "op0"}],
        "OutputList": outputs,
      }
    ],
    network_name: network,
  }


def build(
  op: str,
  depth: int = 1,
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  dimension: str = "Width",
  epsilon: float = 0.001,
  kernel_width: int = 3,
  kernel_height: int = 3,
  rasterizer_mode: str | int = "UnfoldedChannels",
  num_groups: int = 1,
  template_width: int = 3,
  template_height: int = 3,
  pad_top: int = 0,
  pad_bottom: int = 0,
  pad_left: int = 0,
  pad_right: int = 0,
  output_channels: int = 1,
  batch_size: int = 1,
  matrix_decomposition_type: str = "NonQRGivens",
  rotation_axis: list[int] | None = None,
  matmul_order: str = "md_v",
  disparity_direction: int = 0,
  disparity_range: int = 1,
  scale_factor_x: float | None = None,
  scale_factor_y: float | None = None,
  centroid_count: int = 2,
  distance_metric: str | None = "L1",
  radius: float = 1.0,
  params_style: str = "minimal",
  goc_scale: float = 2.0,
  goc_bias: float = 1.0,
  sdpa_channels: int = 64,
  sdpa_sequence: int = 16,
  sdpa_dim: int = 16,
  sdpa_scale: float | None = None,
  sdpa_constant_flag_spelling: str = "Constants_array",
  sdpa_subtract_max: bool = True,
) -> dict:
  op = canonical_op(op)
  if op in {"sdpa", "scaled_dot_product_attention"}:
    return build_sdpa(
      channels=sdpa_channels,
      sequence=sdpa_sequence,
      dim=sdpa_dim,
      scale=sdpa_scale,
      constant_flag_spelling=sdpa_constant_flag_spelling,
      subtract_max=sdpa_subtract_max,
    )
  if op in UNARY_NEURONS: return build_unary(op, width, height, channels)
  if op in UNARY_ELEMENTWISE: return build_unary_elementwise(op, width, height, channels)
  if op in ELEMENTWISE: return build_elementwise(op, width, height, channels)
  if op in ("elu", "leaky_relu", "relu6", "clamped_relu", "prelu"):
    return build_parametric_neuron(op, width, height, channels)
  if op == "silu": return build_silu(width, height, channels)
  if op == "softsign": return build_softsign(width, height, channels)
  if op == "thresholded_relu": return build_thresholded_relu(width, height, channels)
  if op == "linear_activation": return build_linear_activation(width, height, channels)
  if op == "hard_sigmoid": return build_hard_sigmoid(width, height, channels)
  if op == "hard_swish": return build_hard_swish(width, height, channels)
  if op == "div": return build_div(width, height, channels)
  if op == "floor_div": return build_floor_div(width, height, channels)
  if op in {"less_equal", "greater_equal"}: return build_comparison_or(op, width, height, channels)
  if op == "softmax": return build_softmax(width, height, channels)
  if op in ("maxpool2x2", "avgpool2x2"): return build_pooling(op)
  if op in ("reduce_sum", "reduce_mean", "reduce_max", "reduce_min"):
    return build_reduction(op, width, height, channels)
  if op == "minmax_norm": return build_minmax_norm(width, height, channels, dimension, epsilon)
  if op == "kernel_rasterizer":
    return build_kernel_rasterizer(
      width,
      height,
      channels,
      kernel_width,
      kernel_height,
      rasterizer_mode,
      num_groups,
      output_channels,
    )
  if op == "kernel_rasterizer_relu":
    return build_kernel_rasterizer_relu(
      width,
      height,
      channels,
      kernel_width,
      kernel_height,
      rasterizer_mode,
      num_groups,
      output_channels,
    )
  if op == "kernel_rasterizer_add_relu":
    return build_kernel_rasterizer_add_relu(
      width,
      height,
      channels,
      kernel_width,
      kernel_height,
      rasterizer_mode,
      num_groups,
      output_channels,
    )
  if op == "cross_correlation":
    return build_cross_correlation(
      width,
      height,
      channels,
      template_width,
      template_height,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right,
      num_groups,
      output_channels,
    )
  if op == "cross_correlation_relu":
    return build_cross_correlation_relu(
      width,
      height,
      channels,
      template_width,
      template_height,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right,
      num_groups,
      output_channels,
    )
  if op == "cross_correlation_add_relu":
    return build_cross_correlation_add_relu(
      width,
      height,
      channels,
      template_width,
      template_height,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right,
      num_groups,
      output_channels,
    )
  if op == "cost_volume":
    return build_cost_volume(
      ref_width=width,
      aux_width=template_width,
      height=height,
      channels=channels,
      disparity_direction=disparity_direction,
      disparity_range=disparity_range,
      output_channels=output_channels,
      scale_factor_x=scale_factor_x,
      scale_factor_y=scale_factor_y,
    )
  if op == "cross_product":
    return build_cross_product(
      batch_size=batch_size,
      output_channels=output_channels,
    )
  if op == "dynamic_goc_bias":
    return build_dynamic_goc(
      mode="bias",
      channels=channels,
      params_style=params_style,
    )
  if op == "dynamic_goc_scale_bias":
    return build_dynamic_goc(
      mode="scale_bias",
      channels=channels,
      params_style=params_style,
    )
  if op == "goc_scalar_scale_bias":
    return build_goc_scalar_scale_bias(
      width=width,
      height=height,
      channels=channels,
      scale=goc_scale,
      bias=goc_bias,
    )
  if op == "furthest_point_sampling":
    return build_furthest_point_sampling(
      width=width,
      centroid_count=centroid_count,
      distance_metric=distance_metric,
      output_channels=output_channels,
    )
  if op == "radius_search":
    return build_radius_search(
      point_width=width,
      centroid_width=template_width,
      radius=radius,
      output_channels=output_channels,
    )
  if op == "fps_radius_search":
    return build_fps_radius_search(
      width=width,
      centroid_count=centroid_count,
      distance_metric=distance_metric,
      radius=radius,
      output_channels=output_channels,
    )
  if op == "matrix_decomposition":
    return build_matrix_decomposition(
      decomposition_type=matrix_decomposition_type,
      batch_size=batch_size,
      epsilon=epsilon,
      rotation_axis=rotation_axis,
    )
  if op == "matrix_decomposition_relu":
    return build_matrix_decomposition_relu(
      decomposition_type=matrix_decomposition_type,
      batch_size=batch_size,
      epsilon=epsilon,
      rotation_axis=rotation_axis,
    )
  if op == "matrix_decomposition_add":
    return build_matrix_decomposition_add(
      decomposition_type=matrix_decomposition_type,
      batch_size=batch_size,
      epsilon=epsilon,
      rotation_axis=rotation_axis,
    )
  if op == "matrix_decomposition_matmul":
    return build_matrix_decomposition_matmul(
      decomposition_type=matrix_decomposition_type,
      batch_size=batch_size,
      epsilon=epsilon,
      rotation_axis=rotation_axis,
      matmul_order=matmul_order,
    )
  if op == "slice": return build_slice(width, height, channels)
  if op == "upsample_nearest": return build_upsample_nearest()
  if op == "concat": return build_concat(width, height, channels)
  if op == "reshape": return build_reshape()
  if op == "reshape_alias_3d": return build_reshape_alias_3d(width, height, channels)
  if op == "slice_update_concat_equiv": return build_slice_update_concat_equiv(width, height, channels)
  if op == "dynamic_slice_const_f16": return build_dynamic_slice_const_f16(width, height, channels)
  if op == "dynamic_slice_const_u16": return build_dynamic_slice_const_u16(width, height, channels)
  if op == "dynamic_slice_const_u16_w4": return build_dynamic_slice_const_u16_w4(width, height, channels)
  if op == "dynamic_slice_broadcast_const_f16": return build_dynamic_slice_broadcast_const_f16(width, height, channels)
  if op == "add_const": return build_add_const(width, height, channels)
  if op == "broadcast_const_add": return build_broadcast_const_add(width, height, channels)
  if op == "fill_like_as_broadcast_add": return build_fill_like_as_broadcast_add(width, height, channels)
  if op == "gather_const_indices": return build_gather_const_indices(width, height, channels)
  if op == "gather_along_axis_as_gather": return build_gather_along_axis_as_gather(width, height, channels)
  if op == "transpose": return build_transpose()
  if op == "matmul": return build_matmul()
  if op == "conv1x1": return build_conv1x1()
  if op == "add_relu": return build_add_relu()
  if op == "conv_relu": return build_conv_relu()
  if op == "conv_gelu": return build_conv_gelu()
  if op == "conv_add_relu": return build_conv_add_relu()
  if op == "relu_chain": return build_relu_chain(depth, width, height, channels)
  if op == "conv_chain": return build_conv_chain(depth)
  raise ValueError(f"unsupported op: {op}")


def _float16_payload(value: float, count: int) -> bytes:
  if count < 1: raise ValueError("constant payload count must be positive")
  return struct.pack("<" + "e" * count, *([value] * count))


def _uint16_payload(values: list[int]) -> bytes:
  if not values: raise ValueError("index payload must not be empty")
  for value in values:
    if value < 0 or value > 0xFFFF: raise ValueError(f"UInt16 index out of range: {value}")
  return struct.pack("<" + "H" * len(values), *values)


def parse_indices(text: str | None, width: int) -> list[int]:
  if not text: return list(range(width))
  values = [int(item.strip()) for item in text.split(",") if item.strip()]
  if len(values) != width: raise ValueError(f"--indices must contain exactly {width} entries")
  return values


def parse_rotation_axis(text: str | None) -> list[int] | None:
  if not text: return None
  return [int(item.strip()) for item in text.split(",") if item.strip()]


def weights_for_op(
  op: str,
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  const_value: float = 1.0,
  indices: list[int] | None = None,
  sdpa_scale: float | None = None,
  sdpa_dim: int = 16,
) -> list[bytes]:
  op = canonical_op(op)
  if op in {"sdpa", "scaled_dot_product_attention"}:
    scale = sdpa_scale if sdpa_scale is not None else (1.0 / float(sdpa_dim) ** 0.5)
    return [b"", _float16_payload(scale, 1)]
  if op == "add_const": return [b"", _float16_payload(const_value, width * height * channels)]
  if op in {"broadcast_const_add", "fill_like_as_broadcast_add"}: return [b"", _float16_payload(const_value, 1)]
  if op == "softsign": return [b"", _float16_payload(1.0, 1)]
  if op == "thresholded_relu": return [b"", _float16_payload(0.25, 1)]
  if op in {"gather_const_indices", "gather_along_axis_as_gather"}:
    return [b"", _uint16_payload(indices or list(range(width)))]
  if op == "dynamic_slice_const_f16": return [b"", _float16_payload(1.0, 1)]
  if op == "dynamic_slice_const_u16": return [b"", _uint16_payload([1])]
  if op == "dynamic_slice_const_u16_w4": return [b"", _uint16_payload([1, 1, 1, 1])]
  if op == "dynamic_slice_broadcast_const_f16": return [b"", _float16_payload(1.0, 1)]
  if "conv" in op: return [b"", struct.pack("<e", 2.0)]
  return [b""]


def write_model(
  op: str,
  outdir: Path,
  depth: int = 1,
  width: int = 4,
  height: int = 1,
  channels: int = 1,
  const_value: float = 1.0,
  indices: list[int] | None = None,
  dimension: str = "Width",
  epsilon: float = 0.001,
  kernel_width: int = 3,
  kernel_height: int = 3,
  rasterizer_mode: str | int = "UnfoldedChannels",
  num_groups: int = 1,
  template_width: int = 3,
  template_height: int = 3,
  pad_top: int = 0,
  pad_bottom: int = 0,
  pad_left: int = 0,
  pad_right: int = 0,
  output_channels: int = 1,
  batch_size: int = 1,
  matrix_decomposition_type: str = "NonQRGivens",
  rotation_axis: list[int] | None = None,
  matmul_order: str = "md_v",
  disparity_direction: int = 0,
  disparity_range: int = 1,
  scale_factor_x: float | None = None,
  scale_factor_y: float | None = None,
  centroid_count: int = 2,
  distance_metric: str | None = "L1",
  radius: float = 1.0,
  params_style: str = "minimal",
  goc_scale: float = 2.0,
  goc_bias: float = 1.0,
  sdpa_channels: int = 64,
  sdpa_sequence: int = 16,
  sdpa_dim: int = 16,
  sdpa_scale: float | None = None,
  sdpa_constant_flag_spelling: str = "Constants_array",
  sdpa_subtract_max: bool = True,
) -> None:
  outdir.mkdir(parents=True, exist_ok=True)
  for stale_weight in outdir.glob("weights.*"): stale_weight.unlink()
  with (outdir / "net.plist").open("wb") as f:
    plistlib.dump(
      build(
        op,
        depth,
        width,
        height,
        channels,
        dimension,
        epsilon,
        kernel_width,
        kernel_height,
        rasterizer_mode,
        num_groups,
        template_width,
        template_height,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        output_channels,
        batch_size,
        matrix_decomposition_type,
        rotation_axis,
        matmul_order,
        disparity_direction,
        disparity_range,
        scale_factor_x,
        scale_factor_y,
        centroid_count,
        distance_metric,
        radius,
        params_style,
        goc_scale,
        goc_bias,
        sdpa_channels=sdpa_channels,
        sdpa_sequence=sdpa_sequence,
        sdpa_dim=sdpa_dim,
        sdpa_scale=sdpa_scale,
        sdpa_constant_flag_spelling=sdpa_constant_flag_spelling,
        sdpa_subtract_max=sdpa_subtract_max,
      ),
      f,
      sort_keys=True,
    )
  for index, payload in enumerate(
    weights_for_op(
      op, width, height, channels, const_value, indices,
      sdpa_scale=sdpa_scale, sdpa_dim=sdpa_dim,
    )
  ):
    (outdir / f"weights.{index}").write_bytes(payload)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "op",
    choices=sorted(
      [
        *UNARY_NEURONS,
        *UNARY_ELEMENTWISE,
        *ELEMENTWISE,
        "elu",
        "leaky_relu",
        "relu6",
        "clamped_relu",
        "prelu",
        "silu",
        "softsign",
        "thresholded_relu",
        "linear_activation",
        "hard_sigmoid",
        "hard_swish",
        "div",
        "floor_div",
        "less_equal",
        "greater_equal",
        "softmax",
        "maxpool2x2",
        "avgpool2x2",
        "reduce_sum",
        "reduce_mean",
        "reduce_max",
        "reduce_min",
        *NATIVE_NETPLIST_OPS,
        "slice",
        "upsample_nearest",
        "concat",
        "reshape",
        "reshape_alias_3d",
        "slice_update_concat_equiv",
        *RUNTIME_CONST_TEMPLATE_OPS,
        "transpose",
        "matmul",
        "conv1x1",
        "add_relu",
        "conv_relu",
        "conv_gelu",
        "conv1x1_gelu",
        "conv_add_relu",
        "relu_chain",
        "conv_chain",
        "sdpa",
        "scaled_dot_product_attention",
      ]
    ),
  )
  parser.add_argument("outdir", type=Path)
  parser.add_argument("--depth", type=int, default=4)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--height", type=int, default=1)
  parser.add_argument("--channels", type=int, default=1)
  parser.add_argument("--const-value", type=float, default=1.0)
  parser.add_argument("--dimension", default="Width")
  parser.add_argument("--epsilon", type=float, default=0.001)
  parser.add_argument("--kernel-width", type=int, default=3)
  parser.add_argument("--kernel-height", type=int, default=3)
  parser.add_argument("--rasterizer-mode", default="UnfoldedChannels")
  parser.add_argument("--num-groups", type=int, default=1)
  parser.add_argument("--template-width", type=int, default=3)
  parser.add_argument("--template-height", type=int, default=3)
  parser.add_argument("--pad-top", type=int, default=0)
  parser.add_argument("--pad-bottom", type=int, default=0)
  parser.add_argument("--pad-left", type=int, default=0)
  parser.add_argument("--pad-right", type=int, default=0)
  parser.add_argument("--output-channels", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--matrix-decomposition-type", default="NonQRGivens")
  parser.add_argument("--rotation-axis", help="comma-separated MatrixDecomposition rotation-axis masks")
  parser.add_argument("--matmul-order", default="md_v", choices=["md_v", "v_md"])
  parser.add_argument("--disparity-direction", type=int, default=0)
  parser.add_argument("--disparity-range", type=int, default=1)
  parser.add_argument("--scale-factor-x", type=float)
  parser.add_argument("--scale-factor-y", type=float)
  parser.add_argument("--centroid-count", type=int, default=2)
  parser.add_argument("--distance-metric", default="L1", choices=["L1", "L2", "none"])
  parser.add_argument("--radius", type=float, default=1.0)
  parser.add_argument("--goc-scale", type=float, default=2.0)
  parser.add_argument("--goc-bias", type=float, default=1.0)
  parser.add_argument("--indices", help="comma-separated UInt16 index constants")
  parser.add_argument("--sdpa-channels", type=int, default=64,
            help="SDPA heads count (C dimension)")
  parser.add_argument("--sdpa-sequence", type=int, default=16,
            help="SDPA sequence length (H dimension)")
  parser.add_argument("--sdpa-dim", type=int, default=16,
            help="SDPA head dim (W dimension)")
  parser.add_argument("--sdpa-scale", type=float, default=None,
            help="SDPA scale constant (defaults to 1/sqrt(d_head))")
  parser.add_argument(
    "--sdpa-constant-flag",
    default="Constants_array",
    choices=[
      "Constants_array",
      "IsConstant_unit",
      "is_constant_unit",
      "ConstantTensor_unit",
      "ScaleMutable_false",
      "all",
    ],
    help="Which spelling of the constant-Scale flag to emit",
  )
  parser.add_argument(
    "--sdpa-subtract-max",
    dest="sdpa_subtract_max",
    action="store_true",
    default=True,
    help="Set Params.SubtractMax=True on the SDPA unit (default; required "
       "for numerically correct softmax)",
  )
  parser.add_argument(
    "--no-sdpa-subtract-max",
    dest="sdpa_subtract_max",
    action="store_false",
    help="Disable Params.SubtractMax (i.e. omit max-stabilization - the "
       "previous default before the netplist parser was reverse-engineered)",
  )
  args = parser.parse_args()
  if args.depth < 1: parser.error("--depth must be positive")
  if args.width < 1 or args.height < 1 or args.channels < 1 or args.batch_size < 1:
    parser.error("--width, --height, --channels, and --batch-size must be positive")
  rasterizer_mode: str | int = args.rasterizer_mode
  if isinstance(rasterizer_mode, str) and rasterizer_mode.isdigit(): rasterizer_mode = int(rasterizer_mode)

  write_model(
    args.op,
    args.outdir,
    args.depth,
    args.width,
    args.height,
    args.channels,
    args.const_value,
    parse_indices(args.indices, args.width),
    args.dimension,
    args.epsilon,
    args.kernel_width,
    args.kernel_height,
    rasterizer_mode,
    args.num_groups,
    args.template_width,
    args.template_height,
    args.pad_top,
    args.pad_bottom,
    args.pad_left,
    args.pad_right,
    args.output_channels,
    args.batch_size,
    args.matrix_decomposition_type,
    parse_rotation_axis(args.rotation_axis),
    args.matmul_order,
    args.disparity_direction,
    args.disparity_range,
    args.scale_factor_x,
    args.scale_factor_y,
    args.centroid_count,
    None if args.distance_metric == "none" else args.distance_metric,
    args.radius,
    "minimal",
    args.goc_scale,
    args.goc_bias,
    sdpa_channels=args.sdpa_channels,
    sdpa_sequence=args.sdpa_sequence,
    sdpa_dim=args.sdpa_dim,
    sdpa_scale=args.sdpa_scale,
    sdpa_constant_flag_spelling=args.sdpa_constant_flag,
    sdpa_subtract_max=args.sdpa_subtract_max,
  )
  print(args.outdir)
  return 0


if __name__ == "__main__": raise SystemExit(main())
