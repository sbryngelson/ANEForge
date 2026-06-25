#!/usr/bin/env python3
"""Shared device-comparison helpers (timing/precision + device runners + numpy conv/torch/HF reference helpers) imported by the other bench/ scripts - ANE vs GPU (MLX) vs CPU. Not run on its own."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

# optional backends; degrades gracefully
HAVE_ANE = HAVE_MLX = HAVE_TORCH = HAVE_TV = HAVE_HF = False
ANE_ERR = MLX_ERR = ""
try:
    import aneforge as af  # noqa: F401
    HAVE_ANE = True
except Exception as e:  # pragma: no cover
    ANE_ERR = f"{type(e).__name__}: {e}"
try:
    import mlx.core as mx
    HAVE_MLX = True
except Exception as e:  # pragma: no cover
    MLX_ERR = f"{type(e).__name__}: {e}"
try:
    import torchvision  # noqa: F401
    HAVE_TV = True
except Exception:
    pass
try:
    import transformers  # noqa: F401
    HAVE_HF = True
except Exception:
    pass

HAVE_SUDO = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


# timing + precision helpers

def min_latency(fn, reps=30, warmup=8) -> float:
    """MIN end-to-end wall time (s) over reps, after warmup; fn must block until device work completes."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def relerr(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    rn = np.linalg.norm(ref)
    return float(np.linalg.norm(got - ref) / rn) if rn > 0 else float(np.linalg.norm(got - ref))


# device runners (each returns (latency_s, output_array) or None if unavailable)

def cpu_run(build_out, reps=30):
    """build_out() -> np.ndarray. Times the whole call (numpy/Accelerate)."""
    out_holder = {}
    def fn():
        out_holder["o"] = build_out()
    lat = min_latency(fn, reps=reps)
    return lat, out_holder["o"]


def mlx_run(build_mx, reps=30):
    """build_mx() -> mlx array (lazy). We force eval each rep."""
    def fn():
        o = build_mx()
        mx.eval(o)
    lat = min_latency(fn, reps=reps)
    o = build_mx(); mx.eval(o)
    return lat, np.array(o, copy=False)


def min_latency_with_out(fn, reps=30, warmup=8):
    holder = {}
    def wrap():
        holder["o"] = fn()
    lat = min_latency(wrap, reps=reps, warmup=warmup)
    return lat, holder["o"]


def _np_conv_stack(x, Ws, pad, relu=False):
    """Naive fp32 NCHW conv stack reference (stride 1, same pad)."""
    h = x.astype(np.float32)
    for w in Ws:
        h = _np_conv2d(h, w, pad)
        if relu:
            h = np.maximum(h, 0.0)
    return h


def _np_conv2d(x, w, pad):
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # im2col
    cols = np.empty((N, Cin, kH, kW, H, W), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            cols[:, :, i, j] = xp[:, :, i:i + H, j:j + W]
    cols = cols.reshape(N, Cin * kH * kW, H * W)
    wf = w.reshape(Cout, Cin * kH * kW)
    out = np.einsum("oc,ncp->nop", wf, cols).reshape(N, Cout, H, W)
    return out


def _torch_fwd(model, img, dev):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(img).to(dev)
        o = model(t)
        if dev == "mps":
            torch.mps.synchronize()
        return o.detach().to("cpu").numpy()[0]


def _hf_embed(hf, ids):
    import torch
    with torch.no_grad():
        v = hf(**ids).last_hidden_state[0].mean(0).numpy()
    return v / np.linalg.norm(v)

