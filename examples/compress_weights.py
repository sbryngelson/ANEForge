"""aneforge compression demo: int4-LUT / sparse / int8 weight streaming (accuracy x size, then streaming speedup). Run: python3 examples/compress_weights.py"""
import sys, time, statistics, tempfile
from pathlib import Path

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

MODES = [None, "int8", "int4", "sparse", "auto"]


def _weights_bytes(out, **kw):
    """Pack the graph and return the size of the single packed weights.bin."""
    d = tempfile.mkdtemp(prefix="cdemo_")
    af.compile(out, build_dir=d, **kw).release()
    return Path(d, "weights.bin").stat().st_size


def part_a_resnet18():
    _common.head("PART A - accuracy x size : ImageNet ResNet-18, whole net as one ANE program")
    try:
        import torch, torchvision
    except Exception as e:  # noqa: BLE001
        print(f"(skipped: torch/torchvision not available: {type(e).__name__})")
        return

    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(img)).numpy()[0].astype(np.float64)
    ref_top1 = int(ref.argmax())

    # last row loosens the int4 accuracy gate to force more int4 adoption.
    rows = [(m, 0.05, m if m else "None") for m in MODES]
    rows.append(("int4", 0.5, "int4@0.5"))

    print(f"\n{'compress':>9} | {'logit cosine':>12} | {'top-1 match':>11} | "
          f"{'weights.bin':>12} | {'vs fp16':>8}")
    fp16_bytes = None
    for mode, atol, label in rows:
        d = tempfile.mkdtemp(prefix="cdemo_rn_")
        clf = af.load_resnet18(compress=mode, compress_atol=atol, build_dir=d)
        ane = np.asarray(clf(img)[0], np.float64)   # ANE returns fp16; widen for the metric
        nbytes = Path(d, "weights.bin").stat().st_size
        clf.release()
        if mode is None:
            fp16_bytes = nbytes
        cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref)))
        match = "yes" if int(ane.argmax()) == ref_top1 else "NO"
        ratio = f"{fp16_bytes / nbytes:.2f}x" if fp16_bytes else "-"
        print(f"{label:>9} | {cos:>12.4f} | {match:>11} | {nbytes/1e6:>9.2f} MB | {ratio:>8}")
    print("\n  reading: the int4/blockwise accuracy gate (compress_atol) keeps ResNet near-")
    print("  lossless (cosine 1.0) by FALLING BACK per-weight to int8/fp16, so the packed size")
    print("  barely moves and top-1 holds. An already-compact CNN is not where compression pays;")
    print("  raising compress_atol forces more int4 adoption (more size, some cosine cost). The")
    print("  real lever is a big bandwidth-bound weight -> Part B.")


def _median_ms(net, x, warmup=20, iters=100):
    for _ in range(warmup):
        net(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        net(x)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def part_b_streaming_speedup():
    _common.head("PART B - streaming SPEEDUP : one big bandwidth-bound matmul (4096x4096, B=1)")
    rng = np.random.default_rng(1)
    K = N = 4096
    x = rng.standard_normal((1, K)).astype(np.float16)

    Wd = rng.standard_normal((K, N)).astype(np.float16)   # dense; int4 needs a generous gate
    Ws = Wd.copy(); Ws[np.abs(Ws) < 1.0] = 0.0            # ~68% zeros -> sparse encoding emitted

    print(f"\n{'compress':>9} | {'weight':>7} | {'latency (ms)':>12} | {'speedup':>8} | "
          f"{'weights.bin':>12}")
    rows = [
        ("None",   Wd, {}),
        ("int8",   Wd, {"compress": "int8"}),
        ("int4",   Wd, {"compress": "int4", "compress_atol": 1.0}),
        ("sparse", Ws, {"compress": "sparse"}),
    ]
    fp16_ms = None
    for label, W, kw in rows:
        out = af.input((1, K)) @ W
        net = af.compile(out, **kw)
        ms = _median_ms(net, x)
        net.release()
        nbytes = _weights_bytes(af.input((1, K)) @ W, **kw)
        if label == "None":
            fp16_ms = ms
        speed = f"{fp16_ms / ms:.2f}x" if fp16_ms else "-"
        wtag = "dense" if W is Wd else "68%-0"
        print(f"{label:>9} | {wtag:>7} | {ms:>12.3f} | {speed:>8} | {nbytes/1e6:>9.2f} MB")
    print("\n  reading: compressed weights move fewer bytes per tile, so on a big weight-bound")
    print("  matmul int4/sparse beat fp16 on latency AND size. Speedup is vs the fp16 baseline")
    print("  (first row); an fp16 weight ignores zeros, so that baseline holds for the sparse row")
    print("  too. int8 is ~neutral here (the int8->fp16 dequant offsets the halved bytes).")


def main():
    part_a_resnet18()
    part_b_streaming_speedup()


if __name__ == "__main__":
    sys.exit(main())
