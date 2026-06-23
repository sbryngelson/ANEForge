"""Run an unmodified tinygrad model on the Apple Neural Engine, via `aneforge.tinygrad`.

The model below is plain tinygrad - `nn.Conv2d` / `nn.Linear`, no ANEForge-specific code.
`aneforge.tinygrad.trace` compiles its whole forward into ONE e5rt program and runs it on
the engine; the result matches tinygrad to cosine 1.0000, and as one fused dispatch it beats
tinygrad's JIT'd METAL by a wide margin. tinygrad's larger models (ResNet-18/34/50, a ViT
encoder) trace the same way and run 8-16x faster / lower energy than the GPU.

    pip install "aneforge>=0.1.3" tinygrad
    python3 examples/tinygrad_bridge.py
"""
from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor, Device, TinyJit, nn

from aneforge.tinygrad import trace


class Net:
    """An ordinary tinygrad CNN. Nothing here knows about the Neural Engine."""

    def __init__(self):
        self.c1, self.c2 = nn.Conv2d(1, 8, 3), nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def __call__(self, x):
        x = self.c1(x).relu().max_pool2d(2)
        x = self.c2(x).relu().max_pool2d(2)
        return self.fc(x.flatten(1))


def _median_ms(fn, n=30):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[n // 2]


def main():
    np.random.seed(0); Tensor.manual_seed(0)
    net, shape = Net(), (8, 1, 28, 28)
    x = Tensor(np.random.randn(*shape).astype(np.float32))
    xn = x.numpy().astype(np.float32)

    ref = net(x).numpy()                              # plain tinygrad (the default device)
    run = trace(net, shape)                           # unmodified model -> one ANE program
    out = run(x).numpy()
    cos = float((ref.ravel() @ out.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-30))

    jit = TinyJit(lambda t: net(t).realize())         # fair GPU baseline: JIT'd METAL
    for _ in range(5):
        jit(Tensor(xn))
    ane_ms = _median_ms(lambda: np.asarray(run.program(xn)))
    gpu_ms = _median_ms(lambda: (jit(Tensor(xn)), Device[Device.DEFAULT].synchronize()))

    print("  model         : plain tinygrad CNN, unmodified")
    print(f"  ran on the ANE: {run.on_ane}")
    print(f"  agreement     : cosine {cos:.5f} vs tinygrad")
    print(f"  ANE (fused)   : {ane_ms:.2f} ms")
    print(f"  GPU ({Device.DEFAULT}, JIT) : {gpu_ms:.2f} ms   ({gpu_ms / ane_ms:.1f}x)")
    ok = run.on_ane and cos > 0.999
    print(f"\n  {'PASS' if ok else 'FAIL'}: a tinygrad model ran on the Apple Neural Engine")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
