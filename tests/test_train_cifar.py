import numpy as np
import pytest
import aneforge as af
from aneforge import autograd as agrad


def _cos(a, b):
  a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _eval(t):
  """Compile a graph whose inputs carry attrs['value'], eval on the ANE, copy out, and release."""
  net = af.compile(t)
  feed = [s.attrs["value"].astype(np.float16) for s in net._input_tensors]
  out = np.asarray(net(*feed)).copy()
  net.release()
  return out


def test_conv2d_pad_forward_shape():
  """conv2d(pad=1) on a 3x3 kernel preserves spatial dims (a 'same' conv)."""
  rng = np.random.default_rng(0)
  x = af.input((2, 3, 8, 8)); x.attrs["value"] = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
  w = af.conv_param((rng.standard_normal((5, 3, 3, 3)) * 0.1).astype(np.float32))
  y = af.conv2d(x, w, pad=1)
  assert y.shape == (2, 5, 8, 8)


def test_conv2d_pad_grad_matches_torch():
  """Input- and weight-gradients of conv2d(pad=1) match torch F.conv2d(padding=1)."""
  torch = pytest.importorskip("torch")
  import torch.nn.functional as F
  rng = np.random.default_rng(1)
  N, Cin, Hh, Ww, Cout, k = 4, 3, 8, 8, 5, 3
  x_np = rng.standard_normal((N, Cin, Hh, Ww)).astype(np.float32)
  w_np = (rng.standard_normal((Cout, Cin, k, k)) * 0.1).astype(np.float32)

  x = af.input(x_np.shape); x.attrs["value"] = x_np
  w = af.conv_param(w_np)
  y = af.conv2d(x, w, pad=1)
  # mean over batch only: a full 4-dim mean shrinks grads to ~1e-4 (fp16 underflow)
  loss = (y * y).mean((0,))
  grads = agrad.backward(loss, [x, w], loss_scale=1.0)
  gx_ane = _eval(grads[x])
  gw_ane = _eval(grads[w])

  xt = torch.tensor(x_np, requires_grad=True)
  wt = torch.tensor(w_np, requires_grad=True)
  yt = F.conv2d(xt, wt, padding=1)
  (yt * yt).mean(0).sum().backward()
  # w grad: conv_param stores the flat [Cin*kH*kW, Cout] patch matrix; remap torch's grad.
  gw_ref = wt.grad.numpy().reshape(Cout, Cin * k * k).T
  assert _cos(gx_ane.reshape(x_np.shape), xt.grad.numpy()) > 0.99
  assert _cos(gw_ane.reshape(gw_ref.shape), gw_ref) > 0.99


def test_group_norm_train_grad_matches_torch():
  """group_norm_train input/gamma/beta grads match torch F.group_norm at N>1."""
  torch = pytest.importorskip("torch")
  import torch.nn.functional as F
  from aneforge.models import group_norm_train
  rng = np.random.default_rng(2)
  N, C, Hh, Ww, G = 4, 6, 5, 5, 3
  x_np = rng.standard_normal((N, C, Hh, Ww)).astype(np.float32)
  g_np = (rng.standard_normal((1, C, 1, 1)) * 0.5 + 1.0).astype(np.float32)
  b_np = (rng.standard_normal((1, C, 1, 1)) * 0.1).astype(np.float32)

  x = af.input(x_np.shape); x.attrs["value"] = x_np
  gp = af.parameter(g_np); bp = af.parameter(b_np)
  y = group_norm_train(x, gp, bp, G)
  loss = (y * y).mean(tuple(range(4)))
  grads = agrad.backward(loss, [x, gp, bp], loss_scale=1.0)
  gx = _eval(grads[x]); gg = _eval(grads[gp]); gb = _eval(grads[bp])

  xt = torch.tensor(x_np, requires_grad=True)
  gt = torch.tensor(g_np.reshape(C), requires_grad=True)
  bt = torch.tensor(b_np.reshape(C), requires_grad=True)
  yt = F.group_norm(xt, G, gt, bt, eps=1e-5)
  (yt * yt).mean().backward()
  assert _cos(gx.reshape(x_np.shape), xt.grad.numpy()) > 0.99
  assert _cos(gg.reshape(C), gt.grad.numpy()) > 0.99
  assert _cos(gb.reshape(C), bt.grad.numpy()) > 0.99


def test_cifar_cnn_forward_runs_on_ane():
  """The full CNN graph compiles and produces [B, 10] logits on the ANE."""
  from aneforge.models import cifar_cnn
  B = 8
  x, logits, params = cifar_cnn(B, seed=0)
  assert logits.shape == (B, 10)
  rng = np.random.default_rng(3)
  net = af.compile(logits)
  feed = []
  for t in net._input_tensors:
    if t is x:
      feed.append(rng.standard_normal((B, 3, 32, 32)).astype(np.float16))
    else:
      feed.append(t.attrs["value"].astype(np.float16))
  out = np.asarray(net(*feed))
  assert out.shape == (B, 10) and np.isfinite(out).all()
  net.release()


def test_cifar_cnn_short_run_learns():
  """A short on-ANE training run reduces loss and clears a low accuracy floor."""
  import sys as _sys
  from pathlib import Path as _Path
  _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "examples"))
  cifar_data = pytest.importorskip("cifar_data")  # needs torchvision (CIFAR download)
  B = 64
  Xtr, ytr, Xte, yte = cifar_data.load_cifar10()
  # Use a fixed subset for speed and determinism.
  Xtr, ytr = Xtr[:5000], ytr[:5000]
  x, logits, params = af.cifar_cnn(B, seed=0)
  target = af.input((B, 10))
  obj = af.softmax_cross_entropy(logits, target)
  gen = cifar_data.batches(Xtr, ytr, B, seed=0)
  Xb0, yb0 = next(gen)
  tr = af.Trainer(obj, params, lr=3e-3, loss_scale=128.0, optimizer="adam",
                  device_optimizer=True,
                  data_inputs={x: Xb0.astype(np.float32), target: cifar_data.onehot(yb0)})
  tr.data[x] = Xb0.astype(np.float32); tr.data[target] = cifar_data.onehot(yb0)
  tr.step()
  loss0 = tr.loss()
  for _ in range(300):
    Xb, yb = next(gen)
    tr.data[x] = Xb.astype(np.float32); tr.data[target] = cifar_data.onehot(yb)
    tr.step()
  loss1 = tr.loss()
  acc = tr.accuracy(Xte[:2000], yte[:2000])
  tr.release()
  assert np.isfinite(loss1), f"loss went non-finite ({loss1}); lower loss_scale"
  assert loss1 < loss0, f"loss did not decrease: {loss0:.3f} -> {loss1:.3f}"
  assert acc > 0.35, f"accuracy floor not cleared: {acc:.3f}"
