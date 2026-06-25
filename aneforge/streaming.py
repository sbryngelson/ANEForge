"""Layer-streamed (gradient-checkpointed) training for deep stacks of identical layers.

Compile one layer's forward and backward ONCE and reuse them for every layer, so compile
work is independent of depth (a monolithic compile grows superlinearly). Backward uses
recompute-in-backward checkpointing; result is bit-identical to a monolithic `backward`.
See docs/developer/compile-pipeline.md.
"""
from __future__ import annotations

import numpy as np

from . import autograd as _ag
from . import graph as _g
from ._compile import Model as _Model, compile as _compile, compile_multi as _compile_multi

_F16 = np.float16


class CheckpointedStack:
  """A depth-independent compile for a stack of identical layers.

    `layer_fn(params, x)` builds one layer (returns an output Tensor of the same shape as
    `x`). `example_params` gives one layer's parameter shapes; `io_shape` is the activation
    shape flowing between layers.
    """

  def __init__(self, layer_fn, example_params, io_shape):
    self.io_shape = tuple(io_shape)
    self._nparam = len(example_params)

    # per-layer forward: y = layer_fn(params, x)
    self._x = _g.input(self.io_shape)
    self._p = [_ag.parameter(np.asarray(p, np.float32)) for p in example_params]
    y = layer_fn(self._p, self._x)
    if tuple(y.shape) != self.io_shape:
      raise ValueError(f"layer_fn output shape {y.shape} != io_shape {self.io_shape}")
    self._fwd = _compile(y)

    # per-layer backward: given the upstream gradient at the output, return the
    # param gradients and the input gradient (recompute-in-backward checkpointing).
    self._xb = _g.input(self.io_shape)
    self._pb = [_ag.parameter(np.asarray(p, np.float32)) for p in example_params]
    self._gout = _g.input(self.io_shape)
    yb = layer_fn(self._pb, self._xb)
    grads = _ag.backward_from(self._gout, yb, [*self._pb, self._xb])
    self._g_param = [grads[p] for p in self._pb]
    self._g_in = grads[self._xb]
    self._bwd = _compile_multi([*self._g_param, self._g_in])
    self._bwd_in = {id(t): n for t, n in self._bwd.input_ports}
    self._bwd_out = dict(self._bwd.output_ports)
    # baked-constant input ports (e.g. a causal mask) carried into the backward graph
    _fed = {id(self._xb), id(self._gout), *(id(p) for p in self._pb)}
    self._bwd_consts = [(t, n) for t, n in self._bwd.input_ports if id(t) not in _fed]

  def forward(self, layers_params, x0):
    """Run the stack. `layers_params[i]` is layer i's list of parameter arrays. Returns
        `(output, checkpoints)`; `checkpoints[i]` is layer i's input activation (for `backward`)."""
    x = np.asarray(x0, np.float32)
    checkpoints = []
    for lp in layers_params:
      checkpoints.append(x)
      feed = {id(self._x): x.astype(_F16),
          **{id(t): np.asarray(v, _F16) for t, v in zip(self._p, lp)}}
      # any other input port is a baked constant (e.g. a causal mask): feed its value
      assert isinstance(self._fwd, _Model)  # layer_fn never contains sdpa nodes
      vals = [feed[id(t)] if id(t) in feed else np.asarray(t.attrs["value"], _F16)
          for t in self._fwd._input_tensors]
      x = np.asarray(self._fwd(*vals), np.float32)
    return x, checkpoints

  def backward(self, layers_params, checkpoints, g_out):
    """Backprop the stack from `g_out` (gradient at the output). Returns
        `(param_grads, g_in)`: `param_grads[i]` is layer i's param gradients; `g_in` is the
        gradient at the stack input."""
    g = np.asarray(g_out, np.float32)
    param_grads: list[list[np.ndarray] | None] = [None] * len(layers_params)
    for i in range(len(layers_params) - 1, -1, -1):
      self._bwd.prog.set_input(self._bwd_in[id(self._gout)], g.astype(_F16))
      self._bwd.prog.set_input(self._bwd_in[id(self._xb)], checkpoints[i].astype(_F16))
      for t, v in zip(self._pb, layers_params[i]):
        self._bwd.prog.set_input(self._bwd_in[id(t)], np.asarray(v, _F16))
      for t, n in self._bwd_consts:                   # baked constants (e.g. mask)
        self._bwd.prog.set_input(n, np.asarray(t.attrs["value"], _F16))
      self._bwd.prog.execute()
      param_grads[i] = [np.asarray(self._bwd.prog.read_output(self._bwd_out[gp]), np.float32)
                for gp in self._g_param]
      g = np.asarray(self._bwd.prog.read_output(self._bwd_out[self._g_in]), np.float32)
    return param_grads, g

  def release(self):
    self._fwd.release()
    self._bwd.release()
