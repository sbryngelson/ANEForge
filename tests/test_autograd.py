from pathlib import Path

import numpy as np
import pytest
import aneforge as af
from aneforge import autograd as agrad


def test_parameter_is_trainable_input():
  p = agrad.parameter(np.ones((2, 3), np.float32))
  assert p.op == "input"
  assert p.attrs.get("trainable") is True
  assert p.attrs["value"].shape == (2, 3)
  assert p.attrs["value"].dtype == np.float32


def test_vjp_registry_and_topo():
  x = af.input((1, 4))
  y = x * 2.0
  order = agrad._topo(y)
  assert order[0] is x and order[-1] is y
  assert isinstance(agrad.VJP, dict)


def _ane_grad(loss, params, loss_scale=1.0):
  """Compile the backward graph (concatenated param grads), eval on ANE, return
    a list of numpy grads (one per param) in fp32."""
  grads = agrad.backward(loss, params, loss_scale=loss_scale)
  flat = [grads[p].reshape(1, int(np.prod(p.shape))) for p in params]
  out = af.concat(flat, axis=1) if len(flat) > 1 else flat[0]
  net = af.compile(out)
  feed = []
  # inputs in creation order: data inputs need values; here params are the only inputs
  # (the harness builds graphs whose only inputs are the params)
  import aneforge._compile as _c
  order = [t for t in _c._topo(out) if t.op == "input"]
  order.sort(key=lambda t: t.attrs.get("idx", 0))
  for t in order:
    feed.append(t.attrs["value"].astype(np.float16))
  res = net(*feed).reshape(-1) / loss_scale
  net.release()
  # split by param sizes
  sizes = [int(np.prod(p.shape)) for p in params]
  outg, k = [], 0
  for p, s in zip(params, sizes):
    outg.append(res[k:k + s].reshape(p.shape)); k += s
  return outg


def test_muls_grad():
  pv = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
  p = agrad.parameter(pv)
  loss = (p * 3.0).sum((0, 1))            # d loss/d p = 3
  (gane,) = _ane_grad(loss, [p])
  assert np.allclose(gane, 3.0, atol=2e-2)


def _check(build, params, ref_grads, loss_scale=1.0, atol=3e-2):
  """build() -> scalar loss using params; ref_grads: list of numpy fp32 grads."""
  loss = build()
  ane = _ane_grad(loss, params, loss_scale)
  for ga, gr in zip(ane, ref_grads):
    cos = float(ga.ravel() @ gr.ravel() / (np.linalg.norm(ga) * np.linalg.norm(gr) + 1e-9))
    assert cos > 0.99 and np.allclose(ga, gr, atol=atol), (cos, ga, gr)


def test_add_sub_mul_square():
  rng = np.random.default_rng(1)
  a = agrad.parameter(rng.standard_normal((1, 6)).astype(np.float32))
  b = agrad.parameter(rng.standard_normal((1, 6)).astype(np.float32))
  av, bv = a.attrs["value"], b.attrs["value"]
  # loss = sum( (a+b)*a - (a-b) + a^2 ) ; grads by hand
  _check(lambda: (((a + b) * a) - (a - b) + a.square()).sum((0, 1)), [a, b],
         [ (av + bv) + av - 1.0 + 2 * av,    # d/da
           av + 1.0 ])                       # d/db


def test_reduce_mean_grad():
  rng = np.random.default_rng(2)
  p = agrad.parameter(rng.standard_normal((1, 10)).astype(np.float32))
  # loss = mean(p) ; d/dp = 1/N
  _check(lambda: p.mean((0, 1)), [p], [np.full((1, 10), 1.0 / 10, np.float32)])


def test_bmm_linear_grad():
  rng = np.random.default_rng(3)
  x = agrad.parameter(rng.standard_normal((4, 5)).astype(np.float32))   # treat x as a param to grad-check both
  W = agrad.parameter(rng.standard_normal((5, 3)).astype(np.float32))
  xv, Wv = x.attrs["value"], W.attrs["value"]
  # loss = sum(x @ W); g = ones[4,3]; dW = x^T @ ones ; dx = ones @ W^T
  ones = np.ones((4, 3), np.float32)
  _check(lambda: (x @ W).sum((0, 1)), [x, W],
         [ones @ Wv.T, xv.T @ ones])


def test_activation_grads():
  rng = np.random.default_rng(4)
  xv = rng.standard_normal((1, 12)).astype(np.float32)
  from math import sqrt, pi
  try:
    from scipy.special import erf
  except ImportError:
    import math
    erf = np.vectorize(math.erf)
  def dgelu(x):
    return 0.5 * (1 + erf(x / sqrt(2))) + x * (1 / sqrt(2 * pi)) * np.exp(-0.5 * x * x)
  for opname, fn, dfn in [
    ("gelu", lambda p: p.gelu(), lambda x: dgelu(x)),
    ("tanh", lambda p: p.tanh(), lambda x: 1 - np.tanh(x) ** 2),
    ("sigmoid", lambda p: p.sigmoid(), lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x)))),
  ]:
    p = agrad.parameter(xv.copy())
    _check(lambda: fn(p).sum((0, 1)), [p], [dfn(xv).astype(np.float32)], atol=4e-2)


def test_softmax_grad():
  rng = np.random.default_rng(20)
  xv = rng.standard_normal((4, 6)).astype(np.float32)
  p = agrad.parameter(xv)
  # loss = sum(w * softmax(p)) for fixed random w -> grad has a clean numpy form
  wv = rng.standard_normal((4, 6)).astype(np.float32)
  w = agrad.parameter(wv)
  loss = (w * p.softmax(-1)).sum((0, 1))
  grads = _ane_grad(loss, [p])
  # numpy reference: y=softmax(xv); g_into_softmax = wv; dx = y*(g - sum(g*y,axis,keepdims))
  y = np.exp(xv - xv.max(1, keepdims=True)); y /= y.sum(1, keepdims=True)
  dx = y * (wv - (wv * y).sum(1, keepdims=True))
  ga = grads[0]
  cos = float(ga.ravel() @ dx.ravel() / (np.linalg.norm(ga) * np.linalg.norm(dx) + 1e-9))
  assert cos > 0.99 and np.allclose(ga, dx, atol=3e-2)


def _eval_grad_tensor(gt, params):
  """Compile a single grad Tensor (flattened), eval on ANE feeding params, return flat numpy."""
  out = gt.reshape(1, int(np.prod(gt.shape)))
  net = af.compile(out)
  import aneforge._compile as _c
  order = [t for t in _c._topo(out) if t.op == "input"]
  order.sort(key=lambda t: t.attrs.get("idx", 0))
  feed = [t.attrs["value"].astype(np.float16) for t in order]
  res = np.asarray(net(*feed)).reshape(-1)
  net.release()
  return res


# grad-checks for the conv/attention coverage vjps (transpose, reshape, concat, #
# slice_by_size, relu, conv-grad-input, avg_pool, trainable conv2d, mha). Each  #
# compares the on-ANE analytic gradient to a numpy/finite-difference reference  #
# (torch is unavailable on this interpreter; numpy FD is the agreed reference). #
# Require cos > 0.99 and relerr in the fp16 band.                               #

def _ane_grad_shaped(loss, params, loss_scale=1.0):
  """Eval each param's analytic gradient on the ANE in its natural shape."""
  grads = agrad.backward(loss, params, loss_scale=loss_scale)
  return [_eval_grad_tensor(grads[p], [p]).reshape(p.shape) / loss_scale for p in params]


def _cos(a, b):
  return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_transpose_grad():
  rng = np.random.default_rng(50)
  xv = rng.standard_normal((2, 3, 4)).astype(np.float32)
  wv = rng.standard_normal((4, 3, 2)).astype(np.float32)   # weight on the transposed shape
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.transpose([2, 1, 0]) * w).sum((0, 1, 2))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = wv.transpose(2, 1, 0)                              # d/dx sum(x.T * w) = w transposed back
  assert _cos(gx, ref) > 0.99 and np.allclose(gx, ref, atol=3e-2), (_cos(gx, ref),)


def test_reshape_grad():
  rng = np.random.default_rng(51)
  xv = rng.standard_normal((2, 6)).astype(np.float32)
  wv = rng.standard_normal((3, 4)).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.reshape(3, 4) * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = wv.reshape(2, 6)
  assert _cos(gx, ref) > 0.99 and np.allclose(gx, ref, atol=3e-2)


def test_concat_and_slice_grad():
  # concat vjp slices g per source; slice_by_size vjp scatters g back. Compose
  # them: y = concat([x[:, :3], x[:, 3:]], axis=1) should give d/dx = w (identity).
  rng = np.random.default_rng(52)
  xv = rng.standard_normal((4, 6)).astype(np.float32)
  wv = rng.standard_normal((4, 6)).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  left = x.slice_by_size([0, 0], [4, 3]); right = x.slice_by_size([0, 3], [4, 3])
  y = af.concat([left, right], axis=1)
  loss = (y * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  assert _cos(gx, wv) > 0.99 and np.allclose(gx, wv, atol=3e-2)


def test_relu_grad():
  rng = np.random.default_rng(53)
  xv = rng.standard_normal((4, 5)).astype(np.float32)
  wv = rng.standard_normal((4, 5)).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.relu() * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = (xv > 0).astype(np.float32) * wv                  # relu'(x) = 1[x>0]
  assert _cos(gx, ref) > 0.99 and np.allclose(gx, ref, atol=3e-2)


def test_silu_grad():
  rng = np.random.default_rng(60)
  xv = rng.standard_normal((4, 5)).astype(np.float32)
  wv = rng.standard_normal((4, 5)).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.silu() * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  s = 1.0 / (1.0 + np.exp(-xv))
  ref = (s * (1.0 + xv * (1.0 - s))) * wv                  # silu'(x) * w
  assert _cos(gx, ref) > 0.99 and np.allclose(gx, ref, atol=3e-2), (_cos(gx, ref),)


def _fd_grad(fwd, xv, wv, eps=1e-2):
  G = np.zeros_like(xv)
  it = np.nditer(xv, flags=["multi_index"])
  for _ in it:
    i = it.multi_index
    xp = xv.copy(); xp[i] += eps; xm = xv.copy(); xm[i] -= eps
    G[i] = ((fwd(xp) * wv).sum() - (fwd(xm) * wv).sum()) / (2 * eps)
  return G


def test_rms_norm_grad():
  rng = np.random.default_rng(61); D = 16
  xv = rng.standard_normal((4, D)).astype(np.float32)
  wv = rng.standard_normal((4, D)).astype(np.float32)
  gam = rng.standard_normal(D).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.rms_norm(gam) * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = _fd_grad(lambda z: z / np.sqrt((z**2).mean(-1, keepdims=True) + 1e-5) * gam, xv, wv)
  assert _cos(gx, ref) > 0.99, (_cos(gx, ref),)


def test_layer_norm_grad():
  rng = np.random.default_rng(62); D = 16
  xv = rng.standard_normal((4, D)).astype(np.float32)
  wv = rng.standard_normal((4, D)).astype(np.float32)
  gam = rng.standard_normal(D).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.layer_norm(gam, np.zeros(D, np.float32)) * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  def ln(z):
    mu = z.mean(-1, keepdims=True); v = ((z - mu)**2).mean(-1, keepdims=True)
    return (z - mu) / np.sqrt(v + 1e-5) * gam
  ref = _fd_grad(ln, xv, wv)
  assert _cos(gx, ref) > 0.99, (_cos(gx, ref),)


def test_channel_layer_norm_grad():
  rng = np.random.default_rng(63); C = 16
  xv = rng.standard_normal((1, C, 1, 4)).astype(np.float32)
  wv = rng.standard_normal((1, C, 1, 4)).astype(np.float32)
  gam = rng.standard_normal(C).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.channel_layer_norm(gam, np.zeros(C, np.float32)) * w).sum((0, 1, 2, 3))
  (gx,) = _ane_grad_shaped(loss, [x])
  def cln(z):
    mu = z.mean(1, keepdims=True); v = ((z - mu)**2).mean(1, keepdims=True)
    return (z - mu) / np.sqrt(v + 1e-5) * gam.reshape(1, C, 1, 1)
  ref = _fd_grad(cln, xv, wv)
  assert _cos(gx, ref) > 0.99, (_cos(gx, ref),)


def test_conv_grad_wrt_input():
  # native conv node: grad wrt input = conv_transpose(g, W). stride=1, pad in {0,1}.
  rng = np.random.default_rng(54)
  N, Cin, H, W, Cout = 2, 3, 7, 7, 4
  for pad in (0, 1):
    xv = rng.standard_normal((N, Cin, H, W)).astype(np.float32)
    Wv = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    x = agrad.parameter(xv)
    y = af.conv(x, Wv, pad=pad)
    wout = rng.standard_normal(y.shape).astype(np.float32)
    loss = (y * agrad.parameter(wout)).sum((0, 1, 2, 3))
    (gx,) = _ane_grad_shaped(loss, [x])
    # numpy reference: d/dx sum(conv(x,W)*wout) = conv_transpose(wout, W)
    ref = _np_conv_transpose(wout, Wv, pad=pad)
    assert _cos(gx, ref) > 0.99 and gx.shape == xv.shape, (_cos(gx, ref), gx.shape)


def _np_conv2d(x, W, pad=0, stride=1):
  N, Cin, H, Wd = x.shape; Cout, _, kH, kW = W.shape
  xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
  Ho = (xp.shape[2] - kH) // stride + 1; Wo = (xp.shape[3] - kW) // stride + 1
  y = np.zeros((N, Cout, Ho, Wo), np.float32)
  for i in range(Ho):
    for j in range(Wo):
      patch = xp[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
      y[:, :, i, j] = np.einsum('nchw,ochw->no', patch, W)
  return y


def _np_conv_transpose(g, W, pad=0):
  # gradient of conv wrt input: scatter-add W-weighted g back into the padded input
  N, Cout, Ho, Wo = g.shape; _, Cin, kH, kW = W.shape
  H = Ho + kH - 1; Wd = Wo + kW - 1
  dxp = np.zeros((N, Cin, H, Wd), np.float32)
  for i in range(Ho):
    for j in range(Wo):
      dxp[:, :, i:i+kH, j:j+kW] += np.einsum('no,ochw->nchw', g[:, :, i, j], W)
  return dxp[:, :, pad:H-pad, pad:Wd-pad] if pad else dxp


def test_avg_pool_grad():
  rng = np.random.default_rng(55)
  N, C, H, W, k = 2, 3, 8, 8, 2
  xv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  x = agrad.parameter(xv); y = x.avg_pool(k)
  wout = rng.standard_normal(y.shape).astype(np.float32)
  loss = (y * agrad.parameter(wout)).sum((0, 1, 2, 3))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = np.kron(wout, np.ones((1, 1, k, k), np.float32)) / (k * k)  # uniform spread
  assert _cos(gx, ref) > 0.99 and np.allclose(gx, ref, atol=3e-2)


def test_max_pool_grad():
  # max_pool backward routes g to the argmax of each window, built without
  # gather/scatter (upsample the pooled max back over each k*k block, mask the
  # cells equal to it via greater+select). Non-overlapping windows (stride==k,
  # pad==0). On random continuous inputs ties are rare; matches a numpy/torch
  # max-pool backward within the fp16 band (cos > 0.99).
  rng = np.random.default_rng(58)
  N, C, H, W, k = 2, 3, 8, 8, 2
  xv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  x = agrad.parameter(xv); y = x.max_pool(k)
  wout = rng.standard_normal(y.shape).astype(np.float32)
  loss = (y * agrad.parameter(wout)).sum((0, 1, 2, 3))
  (gx,) = _ane_grad_shaped(loss, [x])
  # numpy reference: route each window's cotangent to its argmax cell
  Ho, Wo = H // k, W // k
  ref = np.zeros_like(xv)
  for n in range(N):
    for c in range(C):
      for i in range(Ho):
        for j in range(Wo):
          win = xv[n, c, i*k:i*k+k, j*k:j*k+k]
          a = np.unravel_index(np.argmax(win), win.shape)
          ref[n, c, i*k+a[0], j*k+a[1]] += wout[n, c, i, j]
  assert _cos(gx, ref) > 0.99 and gx.shape == xv.shape, (_cos(gx, ref), gx.shape)


def test_conv2d_trainable_grads():
  # the trainable conv (weight is a real parameter): BOTH input and weight grads
  # are produced on the ANE and match a numpy conv reference (stride=1, pad=0).
  rng = np.random.default_rng(56)
  N, Cin, H, W, Cout, kH, kW = 2, 2, 6, 6, 3, 3, 3
  xv = rng.standard_normal((N, Cin, H, W)).astype(np.float32)
  Wv = rng.standard_normal((Cout, Cin, kH, kW)).astype(np.float32)
  x = agrad.parameter(xv); Wp = agrad.conv_param(Wv)
  y = agrad.conv2d(x, Wp)
  loss = (y.square()).sum((0, 1, 2, 3)) * 0.5
  gx, gW = _ane_grad_shaped(loss, [x, Wp], loss_scale=256.0)
  # numpy reference (loss = 0.5*sum(y^2), y=conv(x,W))
  yv = _np_conv2d(xv, Wv)
  dx_ref = _np_conv_transpose(yv, Wv)                      # d/dx
  # d/dW[o,c,u,v] = sum_{n,i,j} x[n,c,i+u,j+v]*y[n,o,i,j]
  Ho, Wo = yv.shape[2], yv.shape[3]
  dW_ref = np.zeros_like(Wv)
  for u in range(kH):
    for v in range(kW):
      dW_ref[:, :, u, v] = np.einsum('ncij,noij->oc', xv[:, :, u:u+Ho, v:v+Wo], yv)
  dW_ref = dW_ref.reshape(Cout, Cin*kH*kW).T               # match conv_param flat layout
  assert _cos(gx, dx_ref) > 0.99, _cos(gx, dx_ref)
  assert _cos(gW, dW_ref) > 0.99, _cos(gW, dW_ref)


def test_mha_is_differentiable():
  # a full multi-head attention block is differentiable end to end on the ANE:
  # the input gradient matches a numpy finite-difference reference within fp16.
  rng = np.random.default_rng(57)
  S, D, Hh = 4, 8, 2
  xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
  Wq = (rng.standard_normal((D, D)) * 0.2).astype(np.float32)
  Wk = (rng.standard_normal((D, D)) * 0.2).astype(np.float32)
  Wv = (rng.standard_normal((D, D)) * 0.2).astype(np.float32)
  Wo = (rng.standard_normal((D, D)) * 0.2).astype(np.float32)
  wout = rng.standard_normal((S, D)).astype(np.float32)

  def fwd(xin):
    xp = agrad.parameter(xin)
    o = af.mha(xp, Wq, None, Wk, None, Wv, None, Wo, None, Hh)
    loss = (o * agrad.parameter(wout)).sum((0, 1))
    net = af.compile(loss)
    import aneforge._compile as _c
    order = sorted([t for t in _c._topo(loss) if t.op == "input"],
                   key=lambda t: t.attrs.get("idx", 0))
    v = float(np.asarray(net(*[t.attrs["value"].astype(np.float16) for t in order])).reshape(-1)[0])
    net.release(); return v

  x = agrad.parameter(xv)
  o = af.mha(x, Wq, None, Wk, None, Wv, None, Wo, None, Hh)
  loss = (o * agrad.parameter(wout)).sum((0, 1))
  (g,) = _ane_grad_shaped(loss, [x])
  eps = 1e-2
  fd = np.zeros_like(xv)
  for i in range(S):
    for j in range(D):
      xp1 = xv.copy(); xp1[i, j] += eps
      xm = xv.copy(); xm[i, j] -= eps
      fd[i, j] = (fwd(xp1) - fwd(xm)) / (2 * eps)
  assert _cos(g, fd) > 0.97, _cos(g, fd)    # FD on fp16 forward -> 0.97 band


def test_backward_from_matches_backward():
  rng = np.random.default_rng(21)
  p = agrad.parameter(rng.standard_normal((1, 5)).astype(np.float32))
  loss = (p * 3.0).sum((0, 1))
  g_full = agrad.backward(loss, [p])[p]
  # backward_from with an explicit ones-seed at `loss` must reproduce backward
  g_from = agrad.backward_from(agrad._const_like(loss, 1.0), loss, [p])[p]
  va = _eval_grad_tensor(g_full, [p])
  vb = _eval_grad_tensor(g_from, [p])
  assert np.allclose(va, vb, atol=1e-3), (va, vb)


def test_softmax_ce_gradient():
  rng = np.random.default_rng(22)
  N, K = 5, 10
  logits_v = rng.standard_normal((N, K)).astype(np.float32)
  onehot = np.zeros((N, K), np.float32)
  onehot[np.arange(N), rng.integers(0, K, N)] = 1.0
  logits = agrad.parameter(logits_v)
  target = agrad.parameter(onehot)                 # treat as a param leaf to grad-check
  ce = agrad.softmax_cross_entropy(logits, target)
  # the seed dL/dlogits = (softmax(logits) - onehot)/N ; build it and eval on ANE
  seed = (logits.softmax(-1) - target) * (1.0 / N)
  # direct: compile the seed itself and compare to numpy (softmax-onehot)/N
  net = af.compile(seed.reshape(1, N * K))
  import aneforge._compile as _c
  order = [t for t in _c._topo(seed) if t.op == "input"]
  order.sort(key=lambda t: t.attrs.get("idx", 0))
  vals = [t.attrs["value"].astype(np.float16) for t in order]
  out = np.asarray(net(*vals)).reshape(N, K); net.release()
  y = np.exp(logits_v - logits_v.max(1, keepdims=True)); y /= y.sum(1, keepdims=True)
  ref = (y - onehot) / N
  assert np.allclose(out, ref, atol=3e-2), (out, ref)
  assert ce.n == N and ce.logits is logits and ce.target is target


def test_adam_minimizes_quadratic():
  c = np.array([[1.0, -2.0, 3.0]], np.float32)
  w = agrad.parameter(np.zeros((1, 3), np.float32))
  opt = agrad.Adam([w], lr=0.1)
  for _ in range(500):
    grad = 2.0 * (w.attrs["value"] - c)     # d/dw sum((w-c)^2)
    opt.step([grad])
  assert np.allclose(w.attrs["value"], c, atol=1e-2)


def test_cnn_trains_on_subset():
  # conv -> relu -> avg_pool -> flatten -> linear -> softmax-CE, forward+backward
  # on the ANE. The conv weight is a trainable parameter (built from primitives).
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  # Cap the full-batch size: aneforge's trainable-conv im2col tensors grow with the
  # batch, so the COMPILE cost scales with B - B=1000 hangs the M1/h13 compiler for
  # minutes (it compiles fine on M5). 64 keeps the compile fast and is host-independent
  # (identical on M1 and M5); we trade the headline accuracy for a quick, portable check.
  N = min(Xtr.shape[0], 64)
  Xtr = Xtr[:N]; ytr = ytr[:N]
  Xtr_img = Xtr.reshape(N, 1, 28, 28); Xte_img = Xte.reshape(-1, 1, 28, 28)
  K, Cout, k, pk = 10, 8, 3, 2
  Hp = (28 - k + 1) // pk; flat = Cout * Hp * Hp; B = N
  rng = np.random.default_rng(0)
  x = af.input((B, 1, 28, 28)); target = af.input((B, K))
  convW = agrad.conv_param((rng.standard_normal((Cout, 1, k, k)) * np.sqrt(2 / (k * k))).astype(np.float32))
  Wfc = agrad.parameter((rng.standard_normal((flat, K)) * np.sqrt(2 / flat)).astype(np.float32))
  bfc = agrad.parameter(np.zeros((1, K), np.float32))
  logits = (agrad.conv2d(x, convW).relu().avg_pool(pk).reshape(B, flat) @ Wfc) + bfc
  onehot = np.eye(K, dtype=np.float32)[ytr]
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [convW, Wfc, bfc],
                     lr=0.01, loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: Xtr_img, target: onehot})
  l0 = tr.loss(); acc0 = tr.accuracy(Xte_img, yte)
  for _ in range(300):
    tr.step()
  l1 = tr.loss(); acc1 = tr.accuracy(Xte_img, yte)
  tr.release()
  # At the small batch the model can't reach the full-data ~0.95; assert it clearly
  # LEARNS (loss collapses, test accuracy jumps well above its starting point).
  assert l1 < 0.4 * l0 and acc1 > acc0 + 0.3, (l0, l1, acc0, acc1)


def test_cnn_maxpool_trains_on_subset():
  # the avg_pool CNN with the pool swapped to max_pool: conv -> relu -> max_pool
  # -> flatten -> linear -> softmax-CE, forward+backward on the ANE. Exercises the
  # max_pool vjp (argmax routing via upsample + greater + select) in a full train.
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  # Same 64-sample batch cap as the avg_pool test above: keeps the compile fast and
  # host-independent, trading headline accuracy for a quick portable check.
  N = min(Xtr.shape[0], 64)
  Xtr = Xtr[:N]; ytr = ytr[:N]
  Xtr_img = Xtr.reshape(N, 1, 28, 28); Xte_img = Xte.reshape(-1, 1, 28, 28)
  K, Cout, k, pk = 10, 8, 3, 2
  Hp = (28 - k + 1) // pk; flat = Cout * Hp * Hp; B = N
  rng = np.random.default_rng(0)
  x = af.input((B, 1, 28, 28)); target = af.input((B, K))
  convW = agrad.conv_param((rng.standard_normal((Cout, 1, k, k)) * np.sqrt(2 / (k * k))).astype(np.float32))
  Wfc = agrad.parameter((rng.standard_normal((flat, K)) * np.sqrt(2 / flat)).astype(np.float32))
  bfc = agrad.parameter(np.zeros((1, K), np.float32))
  logits = (agrad.conv2d(x, convW).relu().max_pool(pk).reshape(B, flat) @ Wfc) + bfc
  onehot = np.eye(K, dtype=np.float32)[ytr]
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [convW, Wfc, bfc],
                     lr=0.01, loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: Xtr_img, target: onehot})
  l0 = tr.loss(); acc0 = tr.accuracy(Xte_img, yte)
  for _ in range(300):
    tr.step()
  l1 = tr.loss(); acc1 = tr.accuracy(Xte_img, yte)
  tr.release()
  # At the small batch the model can't reach the full-data ~0.95; assert it clearly
  # LEARNS (loss collapses, test accuracy jumps well above its starting point).
  assert l1 < 0.4 * l0 and acc1 > acc0 + 0.3, (l0, l1, acc0, acc1)


def test_transformer_block_trains():
  # mha + residual + MLP + residual, forward+backward on the ANE; attention is
  # differentiable end to end. Loss drops well below the initial value on a toy task.
  rng = np.random.default_rng(0)
  S, D, n_heads = 8, 16, 4; dh = D // n_heads
  Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
  Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)
  x = af.input((S, D)); y = af.input((S, D))
  P = lambda shp, s=0.2: agrad.parameter((rng.standard_normal(shp) * s).astype(np.float32))
  Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
  W1 = P((D, 4 * D)); b1 = agrad.parameter(np.zeros((1, 4 * D), np.float32))
  W2 = P((4 * D, D)); b2 = agrad.parameter(np.zeros((1, D), np.float32))
  params = [Wq, Wk, Wv, Wo, W1, b1, W2, b2]
  heads = lambda t: t.reshape(S, n_heads, dh).transpose([1, 0, 2])
  q, k, v = heads(x @ Wq), heads(x @ Wk), heads(x @ Wv)
  scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
  attn = (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
  h = x + attn
  out = h + ((((h @ W1) + b1).gelu() @ W2) + b2)
  tr = agrad.Trainer(agrad.mse(out, y), params, lr=0.01, loss_scale=1024.0,
                     optimizer="adam", data_inputs={x: Xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(300):
    tr.step()
  l1 = tr.loss()
  tr.release()
  assert l1 < 0.25 * l0, (l0, l1)


def test_layer_norm_trainable_affine_forward_matches_baked():
  # passing parameter Tensors for gamma/beta composes normalize + learnable affine;
  # the forward must equal the native baked-affine layer_norm with the same values.
  rng = np.random.default_rng(70); M, D = 8, 16
  xv = rng.standard_normal((M, D)).astype(np.float32)
  gv = (rng.standard_normal(D) * 0.5 + 1.0).astype(np.float32)
  bv = (rng.standard_normal(D) * 0.3).astype(np.float32)
  baked = af.compile(af.input((M, D)).layer_norm(gv, bv))(xv)
  g = agrad.parameter(gv.reshape(1, D)); b = agrad.parameter(bv.reshape(1, D))
  x = af.input((M, D))
  comp = af.compile(x.layer_norm(g, b))
  feed_for = {id(x): xv, id(g): gv.reshape(1, D), id(b): bv.reshape(1, D)}
  got = comp(*[feed_for[id(t)].astype(np.float16) for t in comp._input_tensors])
  comp.release()
  assert _cos(got, baked) > 0.999, _cos(got, baked)


def test_layer_norm_affine_params_train():
  # the affine gamma/beta are trainable: recover a known target affine from unit init.
  rng = np.random.default_rng(71); M, D = 8, 16
  xv = rng.standard_normal((M, D)).astype(np.float32)
  gv = (rng.standard_normal(D) * 0.5 + 1.0).astype(np.float32)
  bv = (rng.standard_normal(D) * 0.3).astype(np.float32)
  Tgt = af.compile(af.input((M, D)).layer_norm(gv, bv))(xv).astype(np.float32)
  x = af.input((M, D)); y = af.input((M, D))
  g = agrad.parameter(np.ones((1, D), np.float32)); b = agrad.parameter(np.zeros((1, D), np.float32))
  out = x.layer_norm(g, b)
  tr = agrad.Trainer(agrad.mse(out, y), [g, b], lr=0.05, loss_scale=1024.0,
                     optimizer="adam", data_inputs={x: xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(500):
    tr.step()
  l1 = tr.loss()
  g1 = g.attrs["value"].ravel(); b1 = b.attrs["value"].ravel()
  tr.release()
  assert l1 < 0.05 * l0, (l0, l1)
  assert _cos(g1, gv) > 0.99 and _cos(b1, bv) > 0.99      # affine recovered


def test_rms_norm_affine_param_trains():
  rng = np.random.default_rng(72); M, D = 8, 16
  xv = rng.standard_normal((M, D)).astype(np.float32)
  gv = (rng.standard_normal(D) * 0.5 + 1.0).astype(np.float32)
  Tgt = af.compile(af.input((M, D)).rms_norm(gv))(xv).astype(np.float32)
  x = af.input((M, D)); y = af.input((M, D))
  g = agrad.parameter(np.ones((1, D), np.float32))
  tr = agrad.Trainer(agrad.mse(x.rms_norm(g), y), [g], lr=0.05, loss_scale=1024.0,
                     optimizer="adam", data_inputs={x: xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(500):
    tr.step()
  l1 = tr.loss()
  g1 = g.attrs["value"].ravel()
  tr.release()
  assert l1 < 0.05 * l0, (l0, l1)
  assert _cos(g1, gv) > 0.99


def test_charlm_trains_and_predicts():
  # A small multi-layer causal char-LM trains end to end on the engine: one-hot
  # embedding (matmul, not gather), causal-masked attention, RMSNorm + SwiGLU, and a
  # next-token cross-entropy objective. Loss falls and next-char accuracy reaches ~1.0.
  text = "ane trains a tiny language model. " * 2
  chars = sorted(set(text)); V = len(chars)
  stoi = {c: i for i, c in enumerate(chars)}
  ids = np.array([stoi[c] for c in text], np.int64)
  S, D, Hh, dh, NL, FF = 24, 32, 2, 16, 2, 64

  def onehot(idx):
    o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
    return o
  Xv, Tv = onehot(ids[:S]), onehot(ids[1:S + 1])
  cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)
  rng = np.random.default_rng(0)
  P = lambda sh, s=0.08: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
  W_emb, W_pos, W_out = P((V, D)), P((S, D)), P((D, V)); fin = agrad.parameter(np.ones((1, D), np.float32))
  blocks = [dict(Wq=P((D, D)), Wk=P((D, D)), Wv=P((D, D)), Wo=P((D, D)), Wg=P((D, FF)), Wu=P((D, FF)),
                 Wd=P((FF, D)), rn1=agrad.parameter(np.ones((1, D), np.float32)),
                 rn2=agrad.parameter(np.ones((1, D), np.float32))) for _ in range(NL)]
  params = [W_emb, W_pos, W_out, fin] + [p for b in blocks for p in b.values()]
  heads = lambda t: t.reshape(S, Hh, dh).transpose([1, 0, 2])
  x = af.input((S, V)); y = af.input((S, V)); mask = af.input((S, S)); mask.attrs["value"] = cmask
  h = x @ W_emb + W_pos
  for b in blocks:
    xn = h.rms_norm(b["rn1"])
    q, k, v = heads(xn @ b["Wq"]), heads(xn @ b["Wk"]), heads(xn @ b["Wv"])
    sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
    h = h + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ b["Wo"]
    hn = h.rms_norm(b["rn2"])
    h = h + ((hn @ b["Wg"]).silu() * (hn @ b["Wu"])) @ b["Wd"]
  logits = h.rms_norm(fin) @ W_out
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, y), params, lr=0.004,
                     loss_scale=1024.0, optimizer="adam", data_inputs={x: Xv, y: Tv})
  l0 = tr.loss()
  for _ in range(200):
    tr.step()
  l1 = tr.loss()
  pred = np.asarray(tr._fwd(*tr._feed(tr._fwd))).argmax(-1)
  acc = float((pred == ids[1:S + 1]).mean())
  tr.release()
  assert l1 < 0.1 * l0, (l0, l1)
  assert acc >= 0.95, acc


def test_charlm_generalizes_on_corpus():
  # Trained on random windows of a structured corpus's TRAIN split, the char-LM
  # predicts held-out VAL windows it never trained on, well above the unigram
  # baseline - generalization (transferable structure), not memorization.
  from collections import Counter
  rng = np.random.default_rng(0)
  animals, verbs, advs = ["cat", "dog", "bird"], ["runs", "jumps", "sleeps"], ["fast", "slow"]
  corpus = "".join(f"the {rng.choice(animals)} {rng.choice(verbs)} {rng.choice(advs)}. "
                   for _ in range(120))
  chars = sorted(set(corpus)); V = len(chars)
  stoi = {c: i for i, c in enumerate(chars)}
  ids = np.array([stoi[c] for c in corpus], np.int64)
  n = len(ids); ntr = int(n * 0.85)
  S, D, Hh, dh, NL, FF = 48, 48, 4, 12, 3, 96

  def onehot(idx):
    o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
    return o
  cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)
  P = lambda sh, s=0.08: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
  W_emb, W_pos, W_out = P((V, D)), P((S, D)), P((D, V)); fin = agrad.parameter(np.ones((1, D), np.float32))
  blocks = [dict(Wq=P((D, D)), Wk=P((D, D)), Wv=P((D, D)), Wo=P((D, D)), Wg=P((D, FF)), Wu=P((D, FF)),
                 Wd=P((FF, D)), rn1=agrad.parameter(np.ones((1, D), np.float32)),
                 rn2=agrad.parameter(np.ones((1, D), np.float32))) for _ in range(NL)]
  params = [W_emb, W_pos, W_out, fin] + [p for b in blocks for p in b.values()]
  heads = lambda t: t.reshape(S, Hh, dh).transpose([1, 0, 2])
  x = af.input((S, V)); y = af.input((S, V)); mask = af.input((S, S)); mask.attrs["value"] = cmask
  h = x @ W_emb + W_pos
  for b in blocks:
    xn = h.rms_norm(b["rn1"])
    q, k, v = heads(xn @ b["Wq"]), heads(xn @ b["Wk"]), heads(xn @ b["Wv"])
    sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
    h = h + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ b["Wo"]
    hn = h.rms_norm(b["rn2"])
    h = h + ((hn @ b["Wg"]).silu() * (hn @ b["Wu"])) @ b["Wd"]
  logits = h.rms_norm(fin) @ W_out
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, y), params, lr=0.003,
                     loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: onehot(ids[:S]), y: onehot(ids[1:S + 1])})

  def val_acc():
    c = t = 0
    for i in range(ntr, n - S - 1, S):
      tr.data[x] = onehot(ids[i:i + S])
      c += int((np.asarray(tr._fwd(*tr._feed(tr._fwd))).argmax(-1) == ids[i + 1:i + S + 1]).sum()); t += S
    return c / t

  for _ in range(400):
    i = int(rng.integers(0, ntr - S - 1))
    tr.data[x] = onehot(ids[i:i + S]); tr.data[y] = onehot(ids[i + 1:i + S + 1])
    tr.step()
  acc = val_acc()
  mfc = Counter(ids[1:ntr].tolist()).most_common(1)[0][0]
  baseline = float((ids[ntr + 1:n] == mfc).mean())
  tr.release()
  assert acc > 2 * baseline and acc > 0.45, (acc, baseline)   # generalizes, not memorizes


def test_prenorm_transformer_block_trains():
  # Pre-norm transformer block (layer_norm before attention and before the MLP),
  # forward+backward on the ANE. The gradient flows THROUGH layer_norm to the
  # trainable projections/MLP; norm affine (gamma/beta) is fixed at unit init.
  # This is the end-to-end proof that layer_norm no longer blocks backprop.
  rng = np.random.default_rng(0)
  S, D, n_heads = 8, 16, 4; dh = D // n_heads
  Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
  Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)
  g1 = np.ones(D, np.float32); b0 = np.zeros(D, np.float32)
  x = af.input((S, D)); y = af.input((S, D))
  P = lambda shp, s=0.2: agrad.parameter((rng.standard_normal(shp) * s).astype(np.float32))
  Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
  W1 = P((D, 4 * D)); b1 = agrad.parameter(np.zeros((1, 4 * D), np.float32))
  W2 = P((4 * D, D)); b2 = agrad.parameter(np.zeros((1, D), np.float32))
  params = [Wq, Wk, Wv, Wo, W1, b1, W2, b2]
  heads = lambda t: t.reshape(S, n_heads, dh).transpose([1, 0, 2])
  xn = x.layer_norm(g1, b0)
  q, k, v = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
  scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
  h = x + (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
  hn = h.layer_norm(g1, b0)
  out = h + ((((hn @ W1) + b1).gelu() @ W2) + b2)
  tr = agrad.Trainer(agrad.mse(out, y), params, lr=0.01, loss_scale=1024.0,
                     optimizer="adam", data_inputs={x: Xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(300):
    tr.step()
  l1 = tr.loss()
  tr.release()
  assert l1 < 0.25 * l0, (l0, l1)


def test_llama_block_trains():
  # LLaMA-style block: rms_norm (pre-norm) + attention + rms_norm + SwiGLU
  # (silu gate). Exercises rms_norm and silu gradients end to end; trainable
  # projections + the three SwiGLU weights. Norm affine (gamma) fixed at unit.
  rng = np.random.default_rng(1)
  S, D, n_heads = 8, 16, 4; dh = D // n_heads; FF = 4 * D
  Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
  Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)
  g1 = np.ones(D, np.float32)
  x = af.input((S, D)); y = af.input((S, D))
  P = lambda shp, s=0.2: agrad.parameter((rng.standard_normal(shp) * s).astype(np.float32))
  Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
  Wg, Wu, Wd = P((D, FF)), P((D, FF)), P((FF, D))      # SwiGLU gate / up / down
  params = [Wq, Wk, Wv, Wo, Wg, Wu, Wd]
  heads = lambda t: t.reshape(S, n_heads, dh).transpose([1, 0, 2])
  xn = x.rms_norm(g1)
  q, k, v = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
  scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
  h = x + (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
  hn = h.rms_norm(g1)
  out = h + ((hn @ Wg).silu() * (hn @ Wu)) @ Wd
  tr = agrad.Trainer(agrad.mse(out, y), params, lr=0.01, loss_scale=1024.0,
                     optimizer="adam", data_inputs={x: Xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(300):
    tr.step()
  l1 = tr.loss()
  tr.release()
  assert l1 < 0.25 * l0, (l0, l1)


def test_linear_trains():
  # lr=0.2 / loss_scale=1024.0 converge (loss drops well below 25% of initial).
  rng = np.random.default_rng(5)
  Xv = rng.standard_normal((8, 4)).astype(np.float32)
  Wtrue = rng.standard_normal((4, 3)).astype(np.float32)
  Yv = (Xv @ Wtrue).astype(np.float32)
  x = af.input((8, 4)); y = af.input((8, 3))
  W = agrad.parameter((rng.standard_normal((4, 3)) * 0.1).astype(np.float32))
  loss = agrad.mse(x @ W, y)
  tr = agrad.Trainer(loss, [W], lr=0.2, loss_scale=1024.0,
                     data_inputs={x: Xv, y: Yv})
  l0 = tr.loss()
  for _ in range(200):
    tr.step()
  l1 = tr.loss()
  assert l1 < 0.25 * l0, (l0, l1)


def test_classifier_trains_synthetic():
  rng = np.random.default_rng(23)
  N, D, K = 64, 8, 3
  W = rng.standard_normal((D, K)).astype(np.float32)
  X = rng.standard_normal((N, D)).astype(np.float32)
  labels = (X @ W).argmax(1)
  onehot = np.eye(K, dtype=np.float32)[labels]
  x = af.input((N, D)); target = af.input((N, K))
  P1 = agrad.parameter((rng.standard_normal((D, 32)) * 0.2).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, 32), np.float32))
  P2 = agrad.parameter((rng.standard_normal((32, K)) * 0.2).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.05, loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: X, target: onehot})
  acc0 = tr.accuracy(X, labels)
  for _ in range(300):
    tr.step()
  acc1 = tr.accuracy(X, labels)
  assert acc1 > 0.9 and acc1 > acc0, (acc0, acc1)


def test_mnist_mlp_trains():
  # 784->128->10 GELU MLP, full-batch (N=1000), softmax-CE + Adam, forward+backward
  # on the ANE. Train==test==1000 so the one forward-logits program serves both the
  # train loss and the test accuracy (no different-shape recompile; Task 5/7 scope).
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  N, Dn, K, H = Xtr.shape[0], 784, 10, 128
  onehot = np.eye(K, dtype=np.float32)[ytr]
  rng = np.random.default_rng(0)
  x = af.input((N, Dn)); target = af.input((N, K))
  P1 = agrad.parameter((rng.standard_normal((Dn, H)) * (1/np.sqrt(Dn))).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  P2 = agrad.parameter((rng.standard_normal((H, K)) * (1/np.sqrt(H))).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.01, loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: Xtr, target: onehot})
  for _ in range(400):
    tr.step()
  acc = tr.accuracy(Xte, yte)        # Xte is 1000 rows == train batch shape
  assert acc > 0.85, acc


def test_accuracy_chunking_returns_one_pred_per_row():
  # a trained-enough tiny classifier; check accuracy() handles non-B-multiple lengths
  rng = np.random.default_rng(30)
  B, D, K = 4, 5, 3
  Wt = rng.standard_normal((D, K)).astype(np.float32)
  x = af.input((B, D)); target = af.input((B, K))
  P = agrad.parameter(Wt.copy())                      # logits = x @ P (identity-ish)
  logits = x @ P
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P], lr=0.0,
                     loss_scale=1.0, optimizer="sgd", data_inputs={x: np.zeros((B, D), np.float32),
                                                                   target: np.zeros((B, K), np.float32)})
  Xe = rng.standard_normal((7, D)).astype(np.float32)   # 7 rows, B=4 -> chunks [4,3]
  ye = (Xe @ Wt).argmax(1)
  acc = tr.accuracy(Xe, ye)
  assert 0.0 <= acc <= 1.0
  # since P==Wt and logits=x@P, argmax matches the reference exactly -> acc==1.0
  assert acc == 1.0


def test_minibatch_trains_subset():
  from pathlib import Path
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  B, Dn, K, H = 200, 784, 10, 128
  onehot = np.eye(K, dtype=np.float32)[ytr]
  rng = np.random.default_rng(0)
  x = af.input((B, Dn)); target = af.input((B, K))
  P1 = agrad.parameter((rng.standard_normal((Dn, H)) * (1/np.sqrt(Dn))).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  P2 = agrad.parameter((rng.standard_normal((H, K)) * (1/np.sqrt(H))).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.01, loss_scale=1024.0, optimizer="adam",
                     data_inputs={x: np.zeros((B, Dn), np.float32), target: np.zeros((B, K), np.float32)})
  tr.set_dataset(x, Xtr, target, onehot, seed=0)
  acc0 = tr.accuracy(Xte, yte)
  for _ in range(400):
    tr.step()
  acc1 = tr.accuracy(Xte, yte)
  assert acc1 > 0.85 and acc1 > acc0, (acc0, acc1)


def test_on_ane_adam_update_matches_numpy():
  rng = np.random.default_rng(40)
  shp = (8, 6)
  wv, mv, vv, gv = [rng.standard_normal(shp).astype(np.float32) for _ in range(4)]
  vv = np.abs(vv)                                   # v >= 0
  b1, b2, eps, lr_t = 0.9, 0.999, 1e-8, 0.01
  w = agrad.parameter(wv); m = agrad.parameter(mv); v = agrad.parameter(vv); g = agrad.parameter(gv)
  lr = af.input((1, 1))
  w2, m2, v2 = agrad._adam_update(w, m, v, g, lr, b1, b2, eps)
  import aneforge._compile as _c

  def run(t):
    net = _c.compile(t)
    feed = []
    for ti in net._input_tensors:
      feed.append((ti.attrs["value"] if ti.attrs.get("trainable")
                   else np.full((1, 1), lr_t, np.float32)).astype(np.float16))
    out = np.asarray(net(*feed)); net.release(); return out
  W2, M2, V2 = run(w2), run(m2), run(v2)
  m_ref = b1 * mv + (1 - b1) * gv; v_ref = b2 * vv + (1 - b2) * gv * gv
  w_ref = wv - lr_t * m_ref / (np.sqrt(v_ref) + eps)
  assert np.allclose(M2.reshape(shp), m_ref, atol=3e-2) and np.allclose(V2.reshape(shp), v_ref, atol=3e-2)
  assert np.allclose(W2.reshape(shp), w_ref, atol=3e-2)


def test_on_ane_stack3_update_returns_three_arrays():
  # The STACK multi-output path: a [3,n] update program returns the 3 correct
  # arrays on the ANE after a host row-wise split (no _compile.py change).
  rng = np.random.default_rng(41)
  shp = (8, 6)
  wv, mv, vv, gv = [rng.standard_normal(shp).astype(np.float32) for _ in range(4)]
  vv = np.abs(vv)
  b1, b2, eps, lr_t = 0.9, 0.999, 1e-8, 0.01
  w = agrad.parameter(wv); m = agrad.parameter(mv); v = agrad.parameter(vv); g = agrad.parameter(gv)
  lr = af.input((1, 1))
  w2, m2, v2 = agrad._adam_update(w, m, v, g, lr, b1, b2, eps)
  stacked = agrad._stack3(w2, m2, v2)
  import aneforge._compile as _c
  net = _c.compile(stacked)
  feed = [(ti.attrs["value"] if ti.attrs.get("trainable")
           else np.full((1, 1), lr_t, np.float32)).astype(np.float16) for ti in net._input_tensors]
  out = np.asarray(net(*feed)); net.release()
  W2, M2, V2 = agrad._split3(out, shp)
  m_ref = b1 * mv + (1 - b1) * gv; v_ref = b2 * vv + (1 - b2) * gv * gv
  w_ref = wv - lr_t * m_ref / (np.sqrt(v_ref) + eps)
  assert np.allclose(W2, w_ref, atol=3e-2)
  assert np.allclose(M2, m_ref, atol=3e-2)
  assert np.allclose(V2, v_ref, atol=3e-2)


def test_on_ane_sgd_trains_subset():
  from pathlib import Path
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  B, Dn, K, H = 200, 784, 10, 128
  oh = np.eye(K, dtype=np.float32)[ytr]; rng = np.random.default_rng(0)
  x = af.input((B, Dn)); target = af.input((B, K))
  P1 = agrad.parameter((rng.standard_normal((Dn, H)) * (1 / np.sqrt(Dn))).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  P2 = agrad.parameter((rng.standard_normal((H, K)) * (1 / np.sqrt(H))).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.2, loss_scale=1024.0, optimizer="sgd", device_optimizer=True,
                     data_inputs={x: np.zeros((B, Dn), np.float32), target: np.zeros((B, K), np.float32)})
  tr.set_dataset(x, Xtr, target, oh, seed=0)
  for _ in range(400):
    tr.step()
  assert tr.accuracy(Xte, yte) > 0.80


def test_on_ane_adam_trains_subset():
  from pathlib import Path
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  B, Dn, K, H = 200, 784, 10, 128
  oh = np.eye(K, dtype=np.float32)[ytr]; rng = np.random.default_rng(0)
  x = af.input((B, Dn)); target = af.input((B, K))
  P1 = agrad.parameter((rng.standard_normal((Dn, H)) * (1 / np.sqrt(Dn))).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  P2 = agrad.parameter((rng.standard_normal((H, K)) * (1 / np.sqrt(H))).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.01, loss_scale=1024.0, optimizer="adam", device_optimizer=True,
                     data_inputs={x: np.zeros((B, Dn), np.float32), target: np.zeros((B, K), np.float32)})
  tr.set_dataset(x, Xtr, target, oh, seed=0)
  for _ in range(400):
    tr.step()
  assert tr.accuracy(Xte, yte) > 0.85


def test_mlp_trains():
  # lr=0.1 / loss_scale=1024.0 converge (loss drops below 50% of initial).
  rng = np.random.default_rng(6)
  N, D, H = 16, 4, 16
  Xv = rng.standard_normal((N, D)).astype(np.float32)
  Yv = np.tanh(Xv @ rng.standard_normal((D, 1)).astype(np.float32)).astype(np.float32)  # smooth target
  x = af.input((N, D)); y = af.input((N, 1))
  W1 = agrad.parameter((rng.standard_normal((D, H)) * 0.2).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  W2 = agrad.parameter((rng.standard_normal((H, 1)) * 0.2).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, 1), np.float32))
  h = ((x @ W1) + b1).gelu()
  pred = (h @ W2) + b2
  loss = agrad.mse(pred, y)
  tr = agrad.Trainer(loss, [W1, b1, W2, b2], lr=0.1, loss_scale=1024.0,
                     data_inputs={x: Xv, y: Yv})
  l0 = tr.loss()
  for _ in range(400):
    tr.step()
  assert tr.loss() < 0.5 * l0   # learns a meaningful fit


def test_compile_multi_two_outputs():
  # compile_multi lowers a graph with N output Tensors into ONE program with N
  # named output ports; __call__ returns them by name, matching a numpy reference.
  from aneforge import _compile as _c
  rng = np.random.default_rng(3)
  N, D = 8, 5
  Xv = rng.standard_normal((N, D)).astype(np.float32)
  Wv = rng.standard_normal((D, 1)).astype(np.float32)
  x = af.input((N, D))
  a = x @ Wv                      # baked-weight matmul -> (N,1)
  b = a.gelu()
  mm = _c.compile_multi([a, b])
  out = mm(Xv)
  ref_a = Xv @ Wv
  name_a, name_b = mm.output_tensors[0]._name, mm.output_tensors[1]._name
  assert np.allclose(out[name_a], ref_a, atol=1e-2)
  # b = gelu(a): monotone, finite, and matches a numpy gelu within fp16 noise
  g = 0.5 * ref_a * (1 + np.vectorize(__import__("math").erf)(ref_a / np.sqrt(2)))
  assert np.allclose(out[name_b], g, atol=2e-2)
  mm.release()


def test_resident_sgd_matches_host_reference():
  # A fused fwd+bwd+SGD step holds both weights RESIDENT on-device across steps
  # (host feeds only x,y,lr); the resident weights match a host SGD reference.
  from aneforge import _compile as _c
  rng = np.random.default_rng(1)
  D, H, N, STEPS, LR = 6, 4, 12, 40, 0.02
  W1_0 = (0.2 * rng.standard_normal((D, H))).astype(np.float32)
  W2_0 = (0.2 * rng.standard_normal((H, 1))).astype(np.float32)
  Wt1 = rng.standard_normal((D, H)).astype(np.float32)
  Wt2 = rng.standard_normal((H, 1)).astype(np.float32)
  Xv = rng.standard_normal((N, D)).astype(np.float32)
  Yv = (Xv @ Wt1 @ Wt2).astype(np.float32)

  x = af.input((N, D)); y = af.input((N, 1)); lr = af.input((1, 1))
  W1 = agrad.parameter(W1_0); W2 = agrad.parameter(W2_0)
  pred = (x @ W1) @ W2
  grads = agrad.backward(agrad.mse(pred, y), [W1, W2], loss_scale=1.0)
  W1n = agrad._sgd_update(W1, grads[W1], lr)
  W2n = agrad._sgd_update(W2, grads[W2], lr)
  mm = _c.compile_multi([W1n, W2n]); prog = mm.prog
  inn = {t: n for t, n in mm.input_ports}; out = {t: n for t, n in mm.output_ports}
  prog.share_buffer(0, out[W1n], 0, inn[W1]); prog.share_buffer(0, out[W2n], 0, inn[W2])
  prog.set_input(inn[W1], W1_0); prog.set_input(inn[W2], W2_0)
  lr_arr = np.array([[LR]], np.float32)
  for _ in range(STEPS):
    prog.set_input(inn[x], Xv); prog.set_input(inn[y], Yv); prog.set_input(inn[lr], lr_arr)
    prog.execute()                                  # NO state write/read
  W1r = prog.read_output(out[W1n]).astype(np.float32)
  W2r = prog.read_output(out[W2n]).astype(np.float32)
  mm.release()

  W1h, W2h = W1_0.copy(), W2_0.copy()
  for _ in range(STEPS):
    h = Xv @ W1h; diff = (h @ W2h) - Yv
    gp = (2.0 / N) * diff
    W1h = W1h - LR * (Xv.T @ (gp @ W2h.T)); W2h = W2h - LR * (h.T @ gp)

  def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
  assert cos(W1r, W1h) > 0.99 and cos(W2r, W2h) > 0.99


def test_resident_adam_trains_subset_and_state_stays_resident():
  from pathlib import Path
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  B, Dn, K, H = 100, 784, 10, 64
  oh = np.eye(K, dtype=np.float32)[ytr]; rng = np.random.default_rng(0)
  x = af.input((B, Dn)); target = af.input((B, K))
  P1 = agrad.parameter((rng.standard_normal((Dn, H)) * (1 / np.sqrt(Dn))).astype(np.float32))
  b1 = agrad.parameter(np.zeros((1, H), np.float32))
  P2 = agrad.parameter((rng.standard_normal((H, K)) * (1 / np.sqrt(H))).astype(np.float32))
  b2 = agrad.parameter(np.zeros((1, K), np.float32))
  logits = ((x @ P1) + b1).gelu() @ P2 + b2
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [P1, b1, P2, b2],
                     lr=0.01, loss_scale=1024.0, optimizer="adam", resident_state=True,
                     data_inputs={x: np.zeros((B, Dn), np.float32), target: np.zeros((B, K), np.float32)})
  # the per-step feed touches ONLY data + lr -- no param/moment port is fed each
  # step (state stays resident on-device). Verify structurally.
  state_ids = set()
  for e in tr._res_state:
    state_ids.add(id(e["p"]))
    if "m_in" in e: state_ids.update({id(e["m_in"]), id(e["v_in"])})
  assert all(id(t) not in state_ids for t in tr._res_data_inputs)
  assert tr._res_lr not in tr._res_data_inputs
  tr.set_dataset(x, Xtr, target, oh, seed=0)
  for _ in range(8 * (len(Xtr) // B)):
    tr.step()
  assert tr.accuracy(Xte, yte) > 0.85
  tr.release()


def test_resident_cnn_trains_subset():
  # the resident-state path (previously MLP-validated) also compiles + trains for
  # a CNN: the whole step (primitive-built trainable-conv forward + backward +
  # per-param Adam) is ONE fused multi-output program with state aliased
  # on-device. Host feeds only the minibatch + lr each step.
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  # The resident path compiles ONE program at the fixed batch B below, so its compile
  # cost is governed by B (modest, ~B=100), not the dataset size N. Unlike the
  # full-batch CNN tests, N must stay >= B here (minibatches of B are sampled from N),
  # so it is not capped - capping it below B would yield zero training steps.
  N = Xtr.shape[0]
  Xtr_img = Xtr.reshape(N, 1, 28, 28); Xte_img = Xte.reshape(-1, 1, 28, 28)
  K, Cout, k, pk = 10, 8, 3, 2
  Hp = (28 - k + 1) // pk; flat = Cout * Hp * Hp; B = 100
  rng = np.random.default_rng(0)
  x = af.input((B, 1, 28, 28)); target = af.input((B, K))
  convW = agrad.conv_param((rng.standard_normal((Cout, 1, k, k)) * np.sqrt(2 / (k * k))).astype(np.float32))
  Wfc = agrad.parameter((rng.standard_normal((flat, K)) * np.sqrt(2 / flat)).astype(np.float32))
  bfc = agrad.parameter(np.zeros((1, K), np.float32))
  logits = (agrad.conv2d(x, convW).relu().avg_pool(pk).reshape(B, flat) @ Wfc) + bfc
  onehot = np.eye(K, dtype=np.float32)[ytr]
  tr = agrad.Trainer(agrad.softmax_cross_entropy(logits, target), [convW, Wfc, bfc],
                     lr=0.01, loss_scale=1024.0, optimizer="adam", resident_state=True,
                     data_inputs={x: np.zeros((B, 1, 28, 28), np.float32),
                                  target: np.zeros((B, K), np.float32)})
  # state ports are never fed per step (resident on-device)
  state_ids = set()
  for e in tr._res_state:
    state_ids.add(id(e["p"]))
    if "m_in" in e: state_ids.update({id(e["m_in"]), id(e["v_in"])})
  assert all(id(t) not in state_ids for t in tr._res_data_inputs)
  tr.set_dataset(x, Xtr_img, target, onehot, seed=0)
  acc0 = tr.accuracy(Xte_img, yte)
  for _ in range(8 * (N // B)):
    tr.step()
  acc1 = tr.accuracy(Xte_img, yte)
  tr.release()
  assert acc1 > 0.85 and acc1 > acc0, (acc0, acc1)


def test_resident_transformer_block_trains():
  # the resident-state path also compiles + trains for a transformer block:
  # mha + residual + MLP + residual, the whole step (forward + backward + eight-
  # param Adam) is ONE fused multi-output program with state aliased on-device.
  rng = np.random.default_rng(0)
  S, D, n_heads = 8, 16, 4; dh = D // n_heads
  Xv = (rng.standard_normal((S, D)) * 0.5).astype(np.float32)
  Tgt = np.tanh(Xv @ (rng.standard_normal((D, D)) * 0.3).astype(np.float32)).astype(np.float32)
  x = af.input((S, D)); y = af.input((S, D))
  P = lambda shp, s=0.2: agrad.parameter((rng.standard_normal(shp) * s).astype(np.float32))
  Wq, Wk, Wv, Wo = P((D, D)), P((D, D)), P((D, D)), P((D, D))
  W1 = P((D, 4 * D)); b1 = agrad.parameter(np.zeros((1, 4 * D), np.float32))
  W2 = P((4 * D, D)); b2 = agrad.parameter(np.zeros((1, D), np.float32))
  params = [Wq, Wk, Wv, Wo, W1, b1, W2, b2]
  heads = lambda t: t.reshape(S, n_heads, dh).transpose([1, 0, 2])
  q, k, v = heads(x @ Wq), heads(x @ Wk), heads(x @ Wv)
  scores = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)).softmax(-1)
  attn = (scores @ v).transpose([1, 0, 2]).reshape(S, D) @ Wo
  h = x + attn
  out = h + ((((h @ W1) + b1).gelu() @ W2) + b2)
  tr = agrad.Trainer(agrad.mse(out, y), params, lr=0.01, loss_scale=1024.0,
                     optimizer="adam", resident_state=True, data_inputs={x: Xv, y: Tgt})
  l0 = tr.loss()
  for _ in range(300):
    tr.step()
  l1 = tr.loss()
  tr.release()
  assert l1 < 0.25 * l0, (l0, l1)


def test_unrolled_trainer_resident_trains_and_state_stays_resident():
  # K Adam steps unrolled into ONE program, optimizer state RESIDENT on-device.
  d = np.load(Path(__file__).resolve().parent.parent / "examples" / "data" / "mnist_subset.npz")
  Xtr = d["Xtr"].astype(np.float32) / 255.0; ytr = d["ytr"].astype(np.int64)
  Xte = d["Xte"].astype(np.float32) / 255.0; yte = d["yte"].astype(np.int64)
  B, K, DIN, H, C = 100, 10, 784, 64, 10
  oh = np.eye(C, dtype=np.float32)[ytr]; rng = np.random.default_rng(0)
  P = [agrad.parameter((rng.standard_normal((DIN, H)) / np.sqrt(DIN)).astype(np.float32)),
       agrad.parameter(np.zeros((1, H), np.float32)),
       agrad.parameter((rng.standard_normal((H, C)) / np.sqrt(H)).astype(np.float32)),
       agrad.parameter(np.zeros((1, C), np.float32))]
  fwd = lambda W, x: ((x @ W[0]) + W[1]).relu() @ W[2] + W[3]
  xs = [af.input((B, DIN)) for _ in range(K)]; ts = [af.input((B, C)) for _ in range(K)]
  tr = af.UnrolledTrainer(P, fwd, "ce", xs, ts, (Xtr, oh), lr=0.01, loss_scale=1024.0,
                          resident=True)
  # structural: the per-dispatch feed touches ONLY data + lr ports; the param/m/v
  # state ports are never fed (they stay resident on-device, aliased out->in).
  assert tr.resident
  data_names = {n for n, _, _ in tr._res_data} | set(tr._res_lr_names)
  state_names = {tr._res_inm[id(t)] for t in (P + tr._m_in + tr._v_in)}
  assert data_names.isdisjoint(state_names)
  for _ in range(40):
    tr.step()
  assert float((tr.predict(Xte).argmax(1) == yte).mean()) > 0.85
  tr.release()


def test_unrolled_trainer_resident_matches_nonresident():
  # resident and host-shuttle paths must produce the SAME trained weights (the only
  # difference is WHERE state lives, not the math).
  rng = np.random.default_rng(1)
  B, K, DIN, H, C, N = 16, 4, 12, 10, 3, 256
  X = (rng.standard_normal((N, DIN)) * 0.5).astype(np.float32)
  y = (np.tanh(X @ rng.standard_normal((DIN, C))).argmax(1)).astype(np.int64)
  oh = np.eye(C, dtype=np.float32)[y]
  def build(resident):
    r = np.random.default_rng(0)
    P = [agrad.parameter((r.standard_normal((DIN, H)) / np.sqrt(DIN)).astype(np.float32)),
         agrad.parameter(np.zeros((1, H), np.float32)),
         agrad.parameter((r.standard_normal((H, C)) / np.sqrt(H)).astype(np.float32)),
         agrad.parameter(np.zeros((1, C), np.float32))]
    fwd = lambda W, x: ((x @ W[0]) + W[1]).relu() @ W[2] + W[3]
    xs = [af.input((B, DIN)) for _ in range(K)]; ts = [af.input((B, C)) for _ in range(K)]
    tr = af.UnrolledTrainer(P, fwd, "ce", xs, ts, (X, oh), lr=0.02, loss_scale=1024.0,
                            seed=0, resident=resident)
    for _ in range(6):
      tr.step()
    w = tr.predict(X)
    tr.release()
    return w
  wr, wn = build(True), build(False)
  assert np.allclose(wr, wn, atol=1e-2), f"resident vs host-shuttle diverged: {np.abs(wr-wn).max()}"


def _fd_unary(fwd, xv, wv, eps=1e-2):
  G = np.zeros_like(xv); it = np.nditer(xv, flags=["multi_index"])
  for _ in it:
    i = it.multi_index; xp = xv.copy(); xp[i] += eps; xm = xv.copy(); xm[i] -= eps
    G[i] = ((fwd(xp) * wv).sum() - (fwd(xm) * wv).sum()) / (2 * eps)
  return G


@pytest.mark.parametrize("name,build,fwd,pos", [
  ("exp", lambda x: x.exp(), np.exp, False),
  ("sqrt", lambda x: x.sqrt(), np.sqrt, True),
  ("clip", lambda x: x.clip(-2.0, 2.0), lambda z: np.clip(z, -2, 2), False),
  ("rsqrt", lambda x: x.rsqrt(), lambda z: 1/np.sqrt(z), True),
  ("inverse", lambda x: x.inverse(), lambda z: 1/z, True),
  ("log", lambda x: x.log(), np.log, True),
  ("abs", lambda x: x.abs(), np.abs, False),
  ("cos", lambda x: x.cos(), np.cos, False),
  ("leaky_relu", lambda x: x.leaky_relu(0.1), lambda z: np.where(z > 0, z, 0.1*z), False),
  ("elu", lambda x: x.elu(1.0), lambda z: np.where(z > 0, z, np.exp(z)-1), False),
  ("relu6", lambda x: x.relu6(), lambda z: np.clip(z, 0, 6), False),
  ("scaled_tanh", lambda x: x.scaled_tanh(1.5, 0.7), lambda z: 1.5*np.tanh(0.7*z), False),
  ("erf", lambda x: x.erf(), lambda z: np.vectorize(__import__("math").erf)(z), False),
  ("sigmoid_hard", lambda x: x.sigmoid_hard(0.2, 0.5), lambda z: np.clip(0.2*z+0.5, 0, 1), False),
  ("l2_norm", lambda x: x.l2_norm(), lambda z: z/np.sqrt((z**2).sum(-1, keepdims=True)+1e-12), False),
])
def test_unary_vjps(name, build, fwd, pos):
  rng = np.random.default_rng(hash(name) % 1000)
  sh = (4, 8)
  xv = (np.abs(rng.standard_normal(sh)) + 0.5 if pos else rng.standard_normal(sh)).astype(np.float32)
  wv = rng.standard_normal(sh).astype(np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (build(x) * w).sum((0, 1))
  (gx,) = _ane_grad_shaped(loss, [x])
  ref = _fd_unary(fwd, xv, wv)
  assert _cos(gx, ref) > 0.97, (name, _cos(gx, ref))


def test_group_norm_grad():
  rng = np.random.default_rng(63); N, C, H, W, G = 1, 8, 4, 4, 2
  xv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  wv = rng.standard_normal((N, C, H, W)).astype(np.float32)
  gam = rng.standard_normal(C).astype(np.float32); bet = np.zeros(C, np.float32)
  x = agrad.parameter(xv); w = agrad.parameter(wv)
  loss = (x.group_norm(gam, bet, G) * w).sum((0, 1, 2, 3))
  (gx,) = _ane_grad_shaped(loss, [x])
  def gn(z):
    zg = z.reshape(N, G, C // G * H * W); mu = zg.mean(2, keepdims=True)
    v = ((zg - mu)**2).mean(2, keepdims=True)
    return ((zg - mu) / np.sqrt(v + 1e-5)).reshape(N, C, H, W) * gam.reshape(1, C, 1, 1)
  G2 = np.zeros_like(xv); it = np.nditer(xv, flags=["multi_index"])
  for _ in it:
    i = it.multi_index; xp = xv.copy(); xp[i] += 1e-2; xm = xv.copy(); xm[i] -= 1e-2
    G2[i] = ((gn(xp) * wv).sum() - (gn(xm) * wv).sum()) / 2e-2
  assert _cos(gx, G2) > 0.97, (_cos(gx, G2),)


def test_sgd_skips_nonfinite_grad_step():
  w = agrad.parameter(np.ones((1, 3), np.float32))
  opt = agrad.SGD([w], lr=0.1)
  before = w.attrs["value"].copy()
  with pytest.warns(UserWarning, match="non-finite"):
    opt.step([np.array([[1.0, np.inf, 0.0]], np.float32)])
  assert np.array_equal(w.attrs["value"], before)
  g = np.ones((1, 3), np.float32)
  opt.step([g])                                   # a later finite step applies normally
  assert np.allclose(w.attrs["value"], before - 0.1 * g)


def test_adam_skips_nonfinite_grad_step_atomically():
  w = agrad.parameter(np.zeros((1, 3), np.float32))
  opt = agrad.Adam([w], lr=0.1)
  before = w.attrs["value"].copy()
  with pytest.warns(UserWarning, match="non-finite"):
    opt.step([np.array([[np.nan, 1.0, 1.0]], np.float32)])
  assert np.array_equal(w.attrs["value"], before)
  assert opt.t == 0                               # skipped atomically: no t advance,
  assert not opt.m[0].any() and not opt.v[0].any()  # moments untouched
  opt.step([np.full((1, 3), 2.0, np.float32)])
  assert opt.t == 1
  assert np.isfinite(w.attrs["value"]).all()
  assert not np.array_equal(w.attrs["value"], before)


def test_nonfinite_skip_warns_first_then_throttles():
  import warnings
  w = agrad.parameter(np.zeros((1, 2), np.float32))
  opt = agrad.SGD([w], lr=0.1)
  bad = [np.array([[np.inf, 0.0]], np.float32)]
  with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    for _ in range(5):
      opt.step(bad)
  assert sum("non-finite" in str(r.message) for r in rec) == 1
  opt.step([np.zeros((1, 2), np.float32)])        # a finite step resets the streak
  with pytest.warns(UserWarning, match="non-finite"):
    opt.step(bad)
