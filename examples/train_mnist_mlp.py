"""Train an MNIST MLP fully on the ANE (forward + backward + Adam), K steps unrolled into one fused program. Run: python3 examples/train_mnist_mlp.py"""
import sys, time
from pathlib import Path
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

B, K, EPOCHS = 100, 10, 40        # K steps/dispatch; B*K = 1000 = one epoch per dispatch


def main():
    d = np.load(Path(__file__).resolve().parent / "data" / "mnist_subset.npz")
    Xtr = d["Xtr"].astype(np.float32) / 255.0
    ytr = d["ytr"].astype(np.int64)
    Xte = d["Xte"].astype(np.float32) / 255.0
    yte = d["yte"].astype(np.int64)
    DIN, HID, C = 784, 128, 10
    onehot = np.eye(C, dtype=np.float32)[ytr]

    rng = np.random.default_rng(0)
    P = [af.parameter((rng.standard_normal((DIN, HID)) * np.sqrt(2 / DIN)).astype(np.float32)),
         af.parameter(np.zeros((1, HID), np.float32)),
         af.parameter((rng.standard_normal((HID, C)) * np.sqrt(2 / HID)).astype(np.float32)),
         af.parameter(np.zeros((1, C), np.float32))]

    def forward(W, x):
        W1, b1, W2, b2 = W
        return (x @ W1 + b1).relu() @ W2 + b2        # logits

    xs = [af.input((B, DIN)) for _ in range(K)]
    ts = [af.input((B, C)) for _ in range(K)]
    tr = af.UnrolledTrainer(P, forward, "ce", xs, ts, (Xtr, onehot),
                            lr=0.01, loss_scale=1024.0)

    acc = lambda: float((tr.predict(Xte).argmax(1) == yte).mean())
    print(f"MNIST MLP, {K} steps unrolled into ONE ANE program "
          f"({len(tr._net.input_tensors)} inputs, {len(tr._net.output_tensors)} outputs)")
    print(f"\n{'epoch':>6} | {'steps':>6} | {'test acc':>9}")
    print(f"{0:>6} | {0:>6} | {acc():>9.4f}  (init)")
    t0 = time.perf_counter()
    for e in range(1, EPOCHS + 1):
        tr.step()                                    # ONE dispatch = K on-engine steps
        if e % 5 == 0 or e == EPOCHS:
            print(f"{e:>6} | {e*K:>6} | {acc():>9.4f}")
    print(f"\nfinal test accuracy {acc():.4f}  ({EPOCHS*K} steps, {EPOCHS} dispatches, "
          f"{time.perf_counter()-t0:.1f}s) - trained entirely on the ANE")
    tr.release()


if __name__ == "__main__":
    sys.exit(main())
