"""Train a real CNN on CIFAR-10 ENTIRELY on the Apple Neural Engine, and compare to
a PyTorch model with the same layer topology (same conv/GroupNorm/pool/fc shapes
and Adam). The comparison is close but not exact: the ANE convs have no bias and
use He-normal init, while torch's nn.Conv2d adds a bias and uses its default
Kaiming-uniform init - so expect the curves to track, not coincide.

  conv(pad=1)->GroupNorm->ReLU->maxpool  x2,  then conv->GroupNorm->ReLU,
  global-avg-pool, fc.  forward + backward + Adam all run on the ANE
  (device_optimizer=True); the host only feeds mini-batches and the scalar lr_t.
  Includes a cosine LR schedule, periodic eval, and a checkpoint of the trained
  params.

  python3 examples/train_cifar_cnn.py
"""
import math, sys, time
from pathlib import Path
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar_data import load_cifar10, onehot, batches

BATCH, STEPS, EVAL_EVERY, LR, LOSS_SCALE, SEED = 64, 8000, 1000, 3e-3, 128.0, 0
CKPT = Path(__file__).resolve().parent / "data" / "cifar_cnn.npz"


def save_checkpoint(params, path=CKPT):
    """Save the trained param master values to a numpy .npz (one array per param)."""
    np.savez(path, *[p.attrs["value"] for p in params])


def load_checkpoint(params, path=CKPT):
    """Load param master values from a checkpoint into the given param list in order."""
    d = np.load(path)
    for p, key in zip(params, d.files):
        p.attrs["value"] = d[key].astype(np.float32)


def cosine_lr(step, total, base):
    """Cosine-decayed learning rate from `base` to ~0 over `total` steps."""
    return base * 0.5 * (1.0 + math.cos(math.pi * step / total))


def main():
    Xtr, ytr, Xte, yte = load_cifar10()
    x, logits, params = af.cifar_cnn(BATCH, seed=SEED)
    target = af.input((BATCH, 10))
    obj = af.softmax_cross_entropy(logits, target)
    b0 = next(batches(Xtr, ytr, BATCH, seed=SEED))
    tr = af.Trainer(obj, params, lr=LR, loss_scale=LOSS_SCALE, optimizer="adam",
                    device_optimizer=True,
                    data_inputs={x: b0[0].astype(np.float32), target: onehot(b0[1])})

    gen = batches(Xtr, ytr, BATCH, seed=SEED)
    print(f"CIFAR-10 CNN trained on the ANE - batch {BATCH}, {STEPS} steps")
    print(f"\n{'step':>6} | {'train loss':>10} | {'test acc':>9}")
    print("-" * 34)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        Xb, yb = next(gen)
        tr.data[x] = Xb.astype(np.float32)
        tr.data[target] = onehot(yb)
        tr.lr = cosine_lr(step - 1, STEPS, LR)   # device-Adam step reads self.lr
        tr.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            acc = tr.accuracy(Xte, yte)
            print(f"{step:>6} | {tr.loss():>10.4f} | {acc:>9.4f}")
    dt = time.perf_counter() - t0
    ane_acc = tr.accuracy(Xte, yte)
    save_checkpoint(params)
    tr.release()
    print(f"\nANE final test accuracy {ane_acc:.4f}  ({STEPS} steps, {dt:.1f}s)")
    print(f"checkpoint saved to {CKPT}")

    # --- torch reference: same layer topology + Adam (note: torch convs add a bias
    # and use default Kaiming-uniform init; the ANE model has no conv bias + He-normal) ---
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
    except ImportError:
        print("(torch not installed - skipping reference comparison)")
        return 0
    torch.manual_seed(SEED)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 32, 3, padding=1); self.n1 = nn.GroupNorm(8, 32)
            self.c2 = nn.Conv2d(32, 64, 3, padding=1); self.n2 = nn.GroupNorm(8, 64)
            self.c3 = nn.Conv2d(64, 128, 3, padding=1); self.n3 = nn.GroupNorm(8, 128)
            self.fc = nn.Linear(128, 10)

        def forward(self, z):
            z = F.max_pool2d(F.relu(self.n1(self.c1(z))), 2)
            z = F.max_pool2d(F.relu(self.n2(self.c2(z))), 2)
            z = F.relu(self.n3(self.c3(z)))
            return self.fc(z.mean((2, 3)))

    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen_t = batches(Xtr, ytr, BATCH, seed=SEED)
    for step in range(STEPS):
        Xb, yb = next(gen_t)
        opt.zero_grad()
        out = net(torch.tensor(Xb))
        F.cross_entropy(out, torch.tensor(yb)).backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(Xte)).argmax(1).numpy()
    ref_acc = float((pred == yte).mean())
    print(f"torch reference test accuracy {ref_acc:.4f}")
    print(f"\nANE within {abs(ane_acc - ref_acc):.4f} of torch "
          f"({'OK' if ane_acc >= 0.70 and abs(ane_acc - ref_acc) <= 0.03 else 'CHECK'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
