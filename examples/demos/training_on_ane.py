"""DEMO: training ON the ANE - forward, backward, and the optimizer all on-device.

Exercises:
  - af.parameter (trainable tensors) + af.mse loss
  - af.Trainer(device_optimizer=True): forward + backward + the Adam update run on the ANE
  - a real fit: recover a known linear map from data, watching the loss fall

Run:  python3 examples/demos/training_on_ane.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    Bsz, Din = 32, 4
    true_w = (rng.standard_normal((Din, 1)) * 1.0).astype(np.float32)

    def batch():
        X = (rng.standard_normal((Bsz, Din))).astype(np.float32)
        return X, (X @ true_w).astype(np.float32)

    w = af.parameter(np.zeros((Din, 1), np.float32))     # learned from scratch
    x = af.input((Bsz, Din)); target = af.input((Bsz, 1))
    obj = af.mse(x @ w, target)

    X0, y0 = batch()
    tr = af.Trainer(obj, [w], lr=0.05, optimizer="adam", device_optimizer=True,
                    data_inputs={x: X0, target: y0})

    print("on-ANE training (forward + backward + Adam all on the engine):")
    print(f"{'step':>5} | {'mse loss':>10}")
    print("-" * 20)
    for step in range(1, 401):
        Xb, yb = batch()
        tr.data[x] = Xb; tr.data[target] = yb
        tr.step()
        if step % 80 == 0 or step == 1:
            print(f"{step:>5} | {tr.loss():>10.5f}")
    learned = np.asarray(w.attrs["value"]).reshape(-1)
    err = float(np.linalg.norm(learned - true_w.reshape(-1)) / np.linalg.norm(true_w))
    tr.release()
    print(f"\nrecovered weights vs truth: relerr = {err:.3f}")
    print("Loss falls + weights converge - the whole training step ran on the ANE, the host")
    print("only fed mini-batches. Same machinery trains real CNNs (device_optimizer / resident state).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
