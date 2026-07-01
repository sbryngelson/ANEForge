"""ResNet-50 workload for the MLPerf-lite harness: an ONNX ResNet-50 compiled to one ANE program, wrapped as a
SUT, over an ImageNet-val QSL (real preprocessed images for accuracy) or a synthetic QSL (random inputs for a
performance-only run; a conv net's latency is data-independent, so synthetic inputs give valid timing).

The MLPerf reference ResNet-50 is the torchvision model at 224x224; `demo_onnx` exports it (pretrained by
default so accuracy mode is meaningful). Pass your own MLPerf-provided ResNet-50 ONNX to `build_sut` to match
the official model exactly.
"""
from __future__ import annotations
import os

import numpy as np

import loadgen_lite as lg

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def demo_onnx(path, pretrained=True):
    """Export torchvision ResNet-50 to `path`. Pretrained (IMAGENET1K_V2) by default so accuracy mode means
    something; random weights are fine for a performance-only run."""
    import torch, torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=weights).eval()
    torch.onnx.export(m, (torch.randn(1, 3, 224, 224),), path, opset_version=13,
                      do_constant_folding=True, input_names=["x"], output_names=["y"], dynamo=False)
    return path


class ResNet50SUT(lg.SUT):
    """Wrap a compiled ANE ResNet-50 as a SUT: one fp16 forward per sample, prediction = argmax of the logits."""
    name = "resnet50-ane-fp16"

    def __init__(self, net, name=None):
        self.net = net
        if name:
            self.name = name

    def issue(self, qsl, indices):
        out = []
        for i in indices:
            x = np.asarray(qsl.get(i), np.float16)
            logits = np.asarray(self.net(x)).astype(np.float32).ravel()
            out.append(int(logits.argmax()))
        return out


def build_sut(model_path=None, compress=None, cache_dir=None):
    """Compile a ResNet-50 to a single ANE program and return the SUT. With no `model_path`, export the
    torchvision reference to a cache dir first. `compress` ("int8"/"int4"/...) quantizes the ANE weights."""
    import aneforge as af
    if model_path is None:
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), "Models", "mlperf")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "resnet50.onnx")
        if not os.path.exists(model_path):
            demo_onnx(model_path)
    net = af.load_onnx(model_path, compress=compress)     # compress=None -> fp16 (the compile default)
    tag = "resnet50-ane-" + (compress or "fp16")
    return ResNet50SUT(net, name=tag)


def synthetic_qsl(count=256, seed=0):
    """A QSL of random [1, 3, 224, 224] fp16 inputs, for a performance-only run (no dataset needed)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((count, 1, 3, 224, 224)).astype(np.float16)
    return lg.QSL(count, lambda i: data[i], name="synthetic")


def _preprocess(img):
    """Standard ResNet eval transform: resize shortest side to 256, center-crop 224, scale to [0,1], normalize
    by the ImageNet mean/std. Returns a [1, 3, 224, 224] fp16 array."""
    from PIL import Image
    im = img.convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    bilinear = getattr(Image, "Resampling", Image).BILINEAR   # Pillow >= 9.1 moved it under Resampling
    im = im.resize((round(w * s), round(h * s)), bilinear)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    a = (a - _MEAN) / _STD
    return a[None].astype(np.float16)


def imagenet_qsl(val_dir, count=None):
    """A QSL over an ImageNet val directory laid out as `val/<wnid>/<image>.JPEG` (the standard torchvision
    ImageFolder layout). Returns (qsl, labels) where labels[i] is the ground-truth class index for sample i,
    sorted-wnid order (the torchvision/ImageNet convention). Requires Pillow."""
    from PIL import Image
    wnids = sorted(d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
    cls_of = {w: i for i, w in enumerate(wnids)}
    files, labels = [], []
    for w in wnids:
        for fn in sorted(os.listdir(os.path.join(val_dir, w))):
            files.append(os.path.join(val_dir, w, fn)); labels.append(cls_of[w])
    if count is not None:
        files, labels = files[:count], labels[:count]
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = _preprocess(Image.open(files[i]))
        return cache[i]

    return lg.QSL(len(files), get, name="imagenet-val"), labels


def accuracy(sut, qsl, labels, count=None):
    """Top-1 accuracy of the SUT over a labeled QSL (MLPerf accuracy mode). Returns (top1, n)."""
    n = qsl.count if count is None else min(count, qsl.count)
    preds = sut.issue(qsl, list(range(n)))
    correct = sum(int(p == labels[i]) for i, p in enumerate(preds))
    return correct / n, n
