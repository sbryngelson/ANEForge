"""Transfer-learning from an imported ONNX model, entirely on the ANE: import a CNN as a frozen feature
extractor (af.onnx_to_features), extract MNIST features on the Neural Engine, then train a fresh linear
head on those features on the ANE (forward + backward + Adam). Run: python3 examples/onnx_finetune.py"""
import os
import sys
import tempfile
from pathlib import Path

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def _export_backbone(path):
    """A small conv backbone -> ONNX. Random weights here; in practice you import a *pretrained* model."""
    import torch, torch.nn as nn
    import torch.nn.functional as fn

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 14 * 14, 10)                  # the classifier we discard

        def forward(self, x):
            x = fn.max_pool2d(self.c1(x).relu(), 2); x = self.c2(x).relu().flatten(1); return self.fc(x)

    torch.onnx.export(Net().eval(), (torch.randn(1, 1, 28, 28),), path, opset_version=13, input_names=["x"], dynamo=False)
    return path


def main():
    d = np.load(Path(__file__).resolve().parent / "data" / "mnist_subset.npz")
    Xtr = (d["Xtr"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28); ytr = d["ytr"].astype(np.int64)
    Xte = (d["Xte"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28); yte = d["yte"].astype(np.int64)

    onnx_path = _export_backbone(os.path.join(tempfile.mkdtemp(), "backbone.onnx"))
    _, features = af.onnx_to_features(onnx_path)           # penultimate features (input to the discarded classifier)
    extractor = af.compile(features)                       # frozen backbone -> features, on the ANE
    D = features.shape[-1]

    def feats(X):                                          # run the frozen extractor on the ANE
        return np.stack([np.asarray(extractor(X[i:i + 1].astype(np.float16))).ravel() for i in range(len(X))]).astype(np.float32)
    Ftr, Fte = feats(Xtr), feats(Xte)
    print(f"imported ONNX backbone -> {D}-d features (extracted on the ANE: {Ftr.shape})")

    rng = np.random.default_rng(0); onehot = np.eye(10, dtype=np.float32)[ytr]
    B, K = 100, 10
    P = [af.parameter((rng.standard_normal((D, 10)) * np.sqrt(1 / D)).astype(np.float32)), af.parameter(np.zeros((1, 10), np.float32))]
    xs = [af.input((B, D)) for _ in range(K)]; ts = [af.input((B, 10)) for _ in range(K)]
    tr = af.UnrolledTrainer(P, lambda W, x: x @ W[0] + W[1], "ce", xs, ts, (Ftr, onehot), lr=0.1, loss_scale=1024.0)

    acc = lambda: float((tr.predict(Fte).argmax(1) == yte).mean())
    print(f"\n{'steps':>6} | {'test acc':>9}")
    print(f"{0:>6} | {acc():>9.4f}  (init)")
    for e in range(1, 31):
        tr.step()                                          # K on-engine training steps per dispatch
        if e % 5 == 0:
            print(f"{e*K:>6} | {acc():>9.4f}")
    print(f"\ntrained a head on imported-ONNX features to {acc():.3f} test accuracy - entirely on the ANE")
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
