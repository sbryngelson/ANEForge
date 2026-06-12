"""One-time: fetch MNIST and save a small subset for the on-ANE classifier
demo/test. Run once: python examples/make_mnist_subset.py  (needs network + sklearn).

The subset is 1000 train / 1000 test (equal sizes): a single forward-logits program
at the N=1000 batch shape then serves BOTH the train loss and the test accuracy with
no different-shape recompile (see the classifier plan, Task 5/7 scope).

Provenance: sklearn `fetch_openml("mnist_784", version=1)`. If OpenML is unavailable,
falls back to the `ossci-datasets` MNIST IDX mirror
(https://ossci-datasets.s3.amazonaws.com/mnist/), decoding the IDX files directly and
taking the same seed-0 permutation subset.
"""
import gzip
import struct
from pathlib import Path
from urllib.request import urlopen

import numpy as np

OSSCI = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _load_idx(url: str) -> np.ndarray:
    with urlopen(url) as r:                       # noqa: S310 (documented MNIST mirror)
        raw = gzip.decompress(r.read())
    magic, = struct.unpack(">I", raw[:4])
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8)
    return data.reshape(dims)


def _fetch():
    """Return (X uint8 [70000,784] 0..255, y int64 [70000]); openml first, IDX mirror fallback."""
    try:
        from sklearn.datasets import fetch_openml
        mn = fetch_openml("mnist_784", version=1, as_frame=False)
        return mn.data.astype(np.uint8), mn.target.astype(np.int64), "sklearn fetch_openml('mnist_784', v1)"
    except Exception as e:                         # noqa: BLE001
        print("fetch_openml failed (%s); falling back to ossci-datasets IDX mirror" % e)
        Xtr = _load_idx(OSSCI + "train-images-idx3-ubyte.gz").reshape(-1, 784)
        ytr = _load_idx(OSSCI + "train-labels-idx1-ubyte.gz")
        Xte = _load_idx(OSSCI + "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
        yte = _load_idx(OSSCI + "t10k-labels-idx1-ubyte.gz")
        X = np.concatenate([Xtr, Xte]).astype(np.uint8)
        y = np.concatenate([ytr, yte]).astype(np.int64)
        return X, y, "ossci-datasets MNIST IDX mirror"


def main():
    X, y, source = _fetch()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, te = idx[:1000], idx[1000:2000]            # 1000 train / 1000 test (equal sizes)
    out = Path(__file__).parent / "data" / "mnist_subset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, Xtr=X[tr], ytr=y[tr].astype(np.uint8),
                        Xte=X[te], yte=y[te].astype(np.uint8))
    print("wrote", out, X[tr].shape, X[te].shape, "| source:", source)


if __name__ == "__main__":
    main()
