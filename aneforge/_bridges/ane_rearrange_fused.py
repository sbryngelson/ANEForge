"""Native ANE spatial-rearrange layers via hand-authored ANECIR netplists.

Six fp16 spatial-rearrange layers running natively on the ANE:

    PixelShuffle    [N, C*fx*fy, H, W]  -> [N, C, H*fy, W*fx]   (depth->space)
    PixelUnshuffle  [N, C, H*fy, W*fx]  -> [N, C*fx*fy, H, W]   (space->depth)
    ChannelToSpace  depth->space, SAME shapes as PixelShuffle
    SpaceToChannel  space->depth, SAME shapes as PixelUnshuffle
    SpaceToBatch    [N, C, H, W]        -> [N*fx*fy, C, H/fy, W/fx]
    BatchToSpace    [N*fx*fy, C, H, W]  -> [N, C, H*fy, W*fx]   (inverse of S2B)

PixelShuffle / PixelUnshuffle use the PyTorch (channel-major) convention;
ChannelToSpace / SpaceToChannel use the TensorFlow space_to_depth /
depth_to_space (block-major) convention.  They coincide only when C==1.
"""

from __future__ import annotations

import plistlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ._netplist import ensure_invoker, invoke_netplist

# The native netplist Type token for each public layer name.
_LAYER_TYPES = {
    "PixelShuffle",
    "PixelUnshuffle",
    "SpaceToChannel",
    "ChannelToSpace",
    "SpaceToBatch",
    "BatchToSpace",
}


def _in_entry(sym: str, w: int, h: int, c: int, unit: str, b: int = 1) -> dict:
    return {
        "BatchSize": b,
        "InputChannels": c,
        "InputDepth": 1,
        "InputHeight": h,
        "InputInterleave": 1,
        "InputName": sym,
        "InputType": "Float16",
        "InputWidth": w,
        "Name": unit,
        "OperationName": "op0",
    }


def _out_entry(sym: str, unit: str) -> dict:
    return {
        "Name": unit,
        "OperationName": "op0",
        "OutputInterleave": 1,
        "OutputName": sym,
        "OutputType": "Float16",
    }


def build_netplist(
    layer: str,
    *,
    in_channels: int,
    in_height: int,
    in_width: int,
    out_channels: int,
    factor_x: int,
    factor_y: int,
    factor_z: int = 1,
    in_batch: int = 1,
) -> dict:
    """Author a one-op spatial-rearrange netplist (plist dict)."""
    if layer not in _LAYER_TYPES:
        raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(_LAYER_TYPES)}")
    unit_name = f"{layer.lower()}-1"
    net_name = f"network_{unit_name}"
    unit = {
        "Bottom": ["x"],
        "InputType": ["Float16"],
        "Name": unit_name,
        "OutputChannels": out_channels,
        "OutputType": "Float16",
        "Params": {"FactorX": factor_x, "FactorY": factor_y, "FactorZ": factor_z},
        "Type": layer,
    }
    network = {
        unit_name: unit,
        "Units": [unit_name],
        "Weights": ["weights.0"],
        "y": {"Bottom": unit_name, "OutputInterleave": 1, "OutputName": "y", "OutputType": "Float16"},
    }
    return {
        "Version": "1.0.10",
        "Networks": [net_name],
        "ProcedureList": [
            {
                "Name": f"procedure_{unit_name}",
                "InputList": [_in_entry("x", in_width, in_height, in_channels, unit_name, in_batch)],
                "OperationList": [{"NetworkName": net_name, "OperationName": "op0"}],
                "OutputList": [_out_entry("y", unit_name)],
            }
        ],
        net_name: network,
    }


def _run_netplist(plist: dict, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    invoker = ensure_invoker("sdpa_invoker")
    with tempfile.TemporaryDirectory(prefix="ane_rearrange_") as d:
        wd = Path(d)
        with (wd / "net.plist").open("wb") as f:
            plistlib.dump(plist, f, sort_keys=True)
        (wd / "weights.0").write_bytes(b"")
        (wd / "in_x.f16").write_bytes(np.ascontiguousarray(x.astype(np.float16)).tobytes())
        info = invoke_netplist(
            invoker, wd / "net.plist",
            weights=[wd / "weights.0"],
            inputs=[("x", wd / "in_x.f16")],
            outputs=[("y", wd / "out_y.f16")],
            repeats=1, warmup=0,
        )
        out = np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16)
        return out, info


# --------------------------------------------------------------------------
# Public ops.  Inputs/outputs are fp16 numpy arrays in [N, C, H, W] order.

def pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    """Depth-to-space upscale: [N, C*r*r, H, W] -> [N, C, H*r, W*r]."""
    N, C2, H, W = x.shape
    assert C2 % (r * r) == 0, "in_channels must be divisible by r*r"
    C = C2 // (r * r)
    plist = build_netplist(
        "PixelShuffle", in_channels=C2, in_height=H, in_width=W,
        out_channels=C, factor_x=r, factor_y=r,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N, C, H * r, W * r)


def pixel_unshuffle(x: np.ndarray, r: int) -> np.ndarray:
    """Space-to-depth: [N, C, H*r, W*r] -> [N, C*r*r, H, W]."""
    N, C, H, W = x.shape
    assert H % r == 0 and W % r == 0, "H and W must be divisible by r"
    plist = build_netplist(
        "PixelUnshuffle", in_channels=C, in_height=H, in_width=W,
        out_channels=C * r * r, factor_x=r, factor_y=r,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N, C * r * r, H // r, W // r)


def channel_to_space(x: np.ndarray, r: int) -> np.ndarray:
    """TensorFlow depth_to_space convention (block-major channels)."""
    N, C2, H, W = x.shape
    C = C2 // (r * r)
    plist = build_netplist(
        "ChannelToSpace", in_channels=C2, in_height=H, in_width=W,
        out_channels=C, factor_x=r, factor_y=r,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N, C, H * r, W * r)


def space_to_channel(x: np.ndarray, r: int) -> np.ndarray:
    """TensorFlow space_to_depth convention (block-major channels)."""
    N, C, H, W = x.shape
    plist = build_netplist(
        "SpaceToChannel", in_channels=C, in_height=H, in_width=W,
        out_channels=C * r * r, factor_x=r, factor_y=r,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N, C * r * r, H // r, W // r)


def space_to_batch(x: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """[N, C, H, W] -> [N*bh*bw, C, H/bh, W/bw] (blocks moved to batch).

    Output batch slice `bh_i*bw + bw_i` == `x[..., bh_i::bh, bw_i::bw]`.
    """
    N, C, H, W = x.shape
    assert H % bh == 0 and W % bw == 0
    plist = build_netplist(
        "SpaceToBatch", in_channels=C, in_height=H, in_width=W,
        out_channels=C, factor_x=bw, factor_y=bh, in_batch=N,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N * bh * bw, C, H // bh, W // bw)


def batch_to_space(x: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """[N*bh*bw, C, H, W] -> [N, C, H*bh, W*bw] (inverse of space_to_batch).

    Requires input batch divisible by `bh*bw` (validator constraint).
    """
    B, C, H, W = x.shape
    assert B % (bh * bw) == 0, "input batch must be divisible by bh*bw"
    N = B // (bh * bw)
    plist = build_netplist(
        "BatchToSpace", in_channels=C, in_height=H, in_width=W,
        out_channels=C, factor_x=bw, factor_y=bh, in_batch=B,
    )
    out, _ = _run_netplist(plist, x)
    return out.reshape(N, C, H * bh, W * bw)


# --------------------------------------------------------------------------
# numpy references (exact integer rearranges).

def ref_pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    N, C2, H, W = x.shape
    C = C2 // (r * r)
    return x.reshape(N, C, r, r, H, W).transpose(0, 1, 4, 2, 5, 3).reshape(N, C, H * r, W * r)


def ref_pixel_unshuffle(x: np.ndarray, r: int) -> np.ndarray:
    N, C, H, W = x.shape
    return x.reshape(N, C, H // r, r, W // r, r).transpose(0, 1, 3, 5, 2, 4).reshape(
        N, C * r * r, H // r, W // r
    )


def ref_depth_to_space_tf(x: np.ndarray, r: int) -> np.ndarray:
    """TensorFlow depth_to_space (block-major) - reference for ChannelToSpace."""
    N, C2, H, W = x.shape
    C = C2 // (r * r)
    out = np.zeros((N, C, H * r, W * r), dtype=x.dtype)
    for c in range(C):
        for fy in range(r):
            for fx in range(r):
                out[:, c, fy::r, fx::r] = x[:, (fy * r + fx) * C + c]
    return out


def ref_space_to_depth_tf(x: np.ndarray, r: int) -> np.ndarray:
    """TensorFlow space_to_depth (block-major) - reference for SpaceToChannel."""
    N, C, H, W = x.shape
    out = np.zeros((N, C * r * r, H // r, W // r), dtype=x.dtype)
    for c in range(C):
        for fy in range(r):
            for fx in range(r):
                out[:, (fy * r + fx) * C + c] = x[:, c, fy::r, fx::r]
    return out


def ref_space_to_batch(x: np.ndarray, bh: int, bw: int) -> np.ndarray:
    N, C, H, W = x.shape
    out = np.zeros((N * bh * bw, C, H // bh, W // bw), dtype=x.dtype)
    for n in range(N):
        for i in range(bh):
            for j in range(bw):
                out[(n * bh + i) * bw + j] = x[n, :, i::bh, j::bw]
    return out


def ref_batch_to_space(x: np.ndarray, bh: int, bw: int) -> np.ndarray:
    B, C, H, W = x.shape
    N = B // (bh * bw)
    out = np.zeros((N, C, H * bh, W * bw), dtype=x.dtype)
    for n in range(N):
        for i in range(bh):
            for j in range(bw):
                out[n, :, i::bh, j::bw] = x[(n * bh + i) * bw + j]
    return out


if __name__ == "__main__":
    results = []

    # Integer-valued fp16 inputs (exactly representable) so any mismatch is a true
    # permutation error, not fp16 rounding noise.  C>1 exercises the channel-ordering
    # convention.
    def iota(shape):
        return np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)

    x = iota((1, 8, 3, 5))  # C=2 after r=2
    y = pixel_shuffle(x, 2)
    e = float(np.abs(y.astype(np.float32) - ref_pixel_shuffle(x, 2).astype(np.float32)).max())
    results.append(("PixelShuffle", y.shape, e))

    x = iota((1, 3, 6, 8))
    y = pixel_unshuffle(x, 2)
    e = float(np.abs(y.astype(np.float32) - ref_pixel_unshuffle(x, 2).astype(np.float32)).max())
    results.append(("PixelUnshuffle", y.shape, e))

    x = iota((1, 8, 3, 5))  # C=2 after r=2
    y = channel_to_space(x, 2)
    e = float(np.abs(y.astype(np.float32) - ref_depth_to_space_tf(x, 2).astype(np.float32)).max())
    results.append(("ChannelToSpace", y.shape, e))

    x = iota((1, 3, 6, 8))
    y = space_to_channel(x, 2)
    e = float(np.abs(y.astype(np.float32) - ref_space_to_depth_tf(x, 2).astype(np.float32)).max())
    results.append(("SpaceToChannel", y.shape, e))

    x = iota((1, 2, 4, 6))
    y = space_to_batch(x, 2, 2)
    e = float(np.abs(y.astype(np.float32) - ref_space_to_batch(x, 2, 2).astype(np.float32)).max())
    results.append(("SpaceToBatch", y.shape, e))

    x = iota((4, 2, 2, 3))
    y = batch_to_space(x, 2, 2)
    e = float(np.abs(y.astype(np.float32) - ref_batch_to_space(x, 2, 2).astype(np.float32)).max())
    results.append(("BatchToSpace", y.shape, e))

    print(f"{'layer':<16}{'out shape':<18}{'max-abs-err':>12}")
    for name, shape, err in results:
        print(f"{name:<16}{str(tuple(shape)):<18}{err:>12.6g}")
    assert all(e == 0.0 for _, _, e in results), "numerics mismatch!"
    print("\nALL SIX CRACKED - exact bit-parity with numpy reference.")
