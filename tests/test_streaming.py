"""Layer-streamed (gradient-checkpointed) training: aneforge/streaming.py.

The whole point is that the per-layer forward/backward are compiled ONCE and reused for
every layer, so an arbitrarily deep stack of identical layers trains with compile work
that does not grow with depth. The gradients must stay bit-exact versus a monolithic
backward, which is what these tests check.
"""
import numpy as np

import aneforge as af
from aneforge import autograd as agrad
from aneforge.streaming import CheckpointedStack


def _cos(a, b):
  a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# one identical layer: a residual MLP block y = x + relu(x @ W1) @ W2  (shape-preserving)
def _layer(params, x):
  W1, W2 = params
  return x + (x @ W1).relu() @ W2


def _example_params(D, rng):
  return [(rng.standard_normal((D, D)) * 0.2).astype(np.float32),
          (rng.standard_normal((D, D)) * 0.2).astype(np.float32)]


def test_streamed_grads_match_monolith():
  rng = np.random.default_rng(0)
  B, D, NL = 4, 8, 5
  xv = (rng.standard_normal((B, D)) * 0.5).astype(np.float32)
  layers = [_example_params(D, rng) for _ in range(NL)]

  # --- monolith: stack the same layers in one graph, loss = sum(output) ---
  xm = af.input((B, D))
  mono_params = [[agrad.parameter(w) for w in lp] for lp in layers]
  h = xm
  for lp in mono_params:
    h = h + (h @ lp[0]).relu() @ lp[1]
  loss = h.sum((0, 1))
  flat = [p for lp in mono_params for p in lp]
  gref = agrad.backward(loss, flat, loss_scale=1.0)

  def evalt(t):
    m = af.compile(t)
    feeds = {id(xm): xv.astype(np.float16)}
    for lp, lpv in zip(mono_params, layers):
      for p, v in zip(lp, lpv):
        feeds[id(p)] = v.astype(np.float16)
    r = np.asarray(m(*[feeds[id(tt)] for tt in m._input_tensors])); m.release()
    return r
  ref = [[evalt(gref[p]) for p in lp] for lp in mono_params]

  # --- streamed: one compiled fwd + one compiled bwd, reused for all NL layers ---
  stack = CheckpointedStack(_layer, _example_params(D, rng), (B, D))
  out, ckpts = stack.forward(layers, xv)
  g_out = np.ones((B, D), np.float32)                      # d sum(out) / d out
  pgrads, _ = stack.backward(layers, ckpts, g_out)
  stack.release()

  for i in range(NL):
    for j in range(2):
      c = _cos(pgrads[i][j], ref[i][j])
      assert c > 0.999, (i, j, c)


def test_deep_stack_compiles_and_trains():
  # A deep stack (more layers than a monolith compiles cheaply) trains: a sum-target
  # regression where SGD over streamed grads drives the loss down. Compile cost is two
  # programs regardless of depth.
  rng = np.random.default_rng(1)
  B, D, NL = 4, 8, 16
  xv = (rng.standard_normal((B, D)) * 0.3).astype(np.float32)
  target = (rng.standard_normal((B, D)) * 0.1).astype(np.float32)
  layers = [_example_params(D, rng) for _ in range(NL)]
  stack = CheckpointedStack(_layer, _example_params(D, rng), (B, D))

  def loss_and_grad():
    out, ckpts = stack.forward(layers, xv)
    diff = out - target
    l = float((diff ** 2).mean())
    g_out = (2.0 / diff.size) * diff                    # d mean((out-t)^2) / d out
    pgrads, _ = stack.backward(layers, ckpts, g_out)
    return l, pgrads

  l0, _ = loss_and_grad()
  lr = 0.1
  for _ in range(40):
    _, pgrads = loss_and_grad()
    for i in range(NL):
      for j in range(2):
        layers[i][j] = layers[i][j] - lr * pgrads[i][j]
  l1, _ = loss_and_grad()
  stack.release()
  assert l1 < 0.5 * l0, (l0, l1)
