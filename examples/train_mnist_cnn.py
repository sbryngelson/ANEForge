"""Train a CNN on MNIST fully on the ANE with a trainable conv, K steps unrolled into one program. Run: python3 examples/train_mnist_cnn.py"""
import sys, time
from pathlib import Path
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

B, K, DISPATCHES = 100, 5, 60      # K conv steps unrolled per dispatch


def main():
    d = np.load(Path(__file__).resolve().parent / "data" / "mnist_subset.npz")
    Xtr = (d["Xtr"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    ytr = d["ytr"].astype(np.int64)
    Xte = (d["Xte"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    yte = d["yte"].astype(np.int64)

    Cout, k, pk, Cls = 8, 3, 2, 10
    Hc = 28 - k + 1                    # 26 (pad-0 conv)
    Hp = Hc // pk                      # 13
    flat = Cout * Hp * Hp
    onehot = np.eye(Cls, dtype=np.float32)[ytr]

    rng = np.random.default_rng(0)
    convW = af.conv_param((rng.standard_normal((Cout, 1, k, k)) * np.sqrt(2 / (k * k))).astype(np.float32))
    Wfc = af.parameter((rng.standard_normal((flat, Cls)) * np.sqrt(2 / flat)).astype(np.float32))
    bfc = af.parameter(np.zeros((1, Cls), np.float32))
    P = [convW, Wfc, bfc]

    def forward(W, x):
        cW, Wf, bf = W
        h = af.conv2d(x, cW).relu().avg_pool(pk)      # [B, Cout, 13, 13]
        return (h.reshape(x.shape[0], flat) @ Wf) + bf

    xs = [af.input((B, 1, 28, 28)) for _ in range(K)]
    ts = [af.input((B, Cls)) for _ in range(K)]
    tr = af.UnrolledTrainer(P, forward, "ce", xs, ts, (Xtr, onehot),
                            lr=0.01, loss_scale=1024.0)

    acc = lambda: float((tr.predict(Xte).argmax(1) == yte).mean())
    print(f"MNIST CNN (conv->relu->avg_pool->fc), {K} steps unrolled into ONE ANE program "
          f"({len(tr._net.input_tensors)} inputs, {len(tr._net.output_tensors)} outputs)")
    print(f"\n{'dispatch':>9} | {'steps':>6} | {'test acc':>9}")
    print(f"{0:>9} | {0:>6} | {acc():>9.4f}  (init)")
    t0 = time.perf_counter()
    for it in range(1, DISPATCHES + 1):
        tr.step()
        if it % 10 == 0 or it == DISPATCHES:
            print(f"{it:>9} | {it*K:>6} | {acc():>9.4f}")
    print(f"\nfinal test accuracy {acc():.4f}  ({DISPATCHES*K} steps, {DISPATCHES} dispatches, "
          f"{time.perf_counter()-t0:.1f}s) - trained entirely on the ANE")
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
