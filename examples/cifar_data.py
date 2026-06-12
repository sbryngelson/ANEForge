"""CIFAR-10 loading for the on-ANE training demo. Uses torchvision to download the
dataset once into examples/data/cifar10 (the [models] extra). Returns normalized
fp32 NCHW arrays + integer labels, plus a deterministic mini-batch iterator. No
training logic here."""
from pathlib import Path
import numpy as np

# CIFAR-10 channel mean/std (standard normalization).
_MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)


def load_cifar10(root: str | None = None):
    """Return (Xtr, ytr, Xte, yte): Xtr/Xte are [N,3,32,32] fp32 normalized; y are
    int64 labels in [0,9]. Downloads via torchvision on first call."""
    import torchvision  # part of the [models] extra
    root = str(root or (Path(__file__).resolve().parent / "data" / "cifar10"))
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)

    def pack(ds):
        X = ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0   # NHWC uint8 -> NCHW fp32
        X = (X - _MEAN) / _STD
        y = np.asarray(ds.targets, np.int64)
        return X, y

    Xtr, ytr = pack(tr); Xte, yte = pack(te)
    return Xtr, ytr, Xte, yte


def onehot(y, classes: int = 10):
    return np.eye(classes, dtype=np.float32)[y]


def batches(X, y, batch: int, seed: int = 0):
    """Yield (Xb, yb) mini-batches of exactly `batch` rows, reshuffled each epoch
    forever (drops the final short batch so the compiled graph's N is fixed)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    while True:
        idx = rng.permutation(n)
        for s in range(0, n - batch + 1, batch):
            j = idx[s:s + batch]
            yield X[j], y[j]
