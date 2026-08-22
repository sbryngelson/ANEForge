# Training on the ANE

ANEForge is not inference-only. `aneforge/autograd.py` is a small reverse-mode
autograd in which both the forward and the backward pass compile to, and run on,
the Apple Neural Engine through the e5rt path. A model's weights, gradients, and
the optimizer update all live as ANE graph ops; the host computes only a scalar
learning rate, samples minibatches, and reads results at checkpoints.

Training runs on the same e5rt dispatch backend as the rest of ANEForge - no
CoreML, no entitlement.

The import alias throughout is `import aneforge as af`.

## The model: tiny reverse-mode autograd

### Trainable weights are graph inputs, not constants

A native ANE program bakes its weights into the compiled blob. That is fine for
inference but useless for training, because every weight update would force a
recompile. ANEForge sidesteps this with `af.parameter`:

```python
W = af.parameter(np.random.randn(784, 128).astype(np.float32))
```

`af.parameter(init)` creates a graph input tagged trainable. It carries an
fp32 master value in `attrs["value"]`, is used in the graph like any other
input, and is fed its current value on each evaluation. Updating a weight is
then just writing a new value into the master array and feeding it on the
next step - no recompile per step. This mutable-weight mechanism underlies the
whole training stack.

### The VJP registry

`backward` walks the forward graph in reverse, and for each op it looks up a
vector-Jacobian-product rule in a registry:

```python
@af.autograd.vjp("mul")
def _vjp_mul(t, g):
    a, b = t.srcs
    return [_unbroadcast(g * b, a.shape), _unbroadcast(g * a, b.shape)]
```

Each rule takes the node and its incoming cotangent `g` and returns one gradient
per source. The gradients are ordinary ANEForge tensor ops, so the backward pass
is a graph that compiles and runs on the engine like the forward pass.

### `backward` builds an on-ANE backward graph

```python
grads = af.backward(loss, params, loss_scale=1024.0)   # {param: grad Tensor}
```

`af.backward(loss, params, loss_scale)` returns, for each parameter, a gradient
Tensor expressed entirely in ANE ops. The seed `dL/dloss = loss_scale` is folded
into an additive constant rather than emitted as a multiply, which sidesteps the
`mul(reduce_output, 0.0)` compile wall described below. A companion
`af.backward_from(grad_root, root, params)` seeds from an explicit gradient at an
intermediate tensor (for example the logits) instead of from a scalar loss.

Both functions accept a `stop` stop-gradient frontier (default `params`).
For ordinary leaf weights this is a no-op, but it is what makes multi-step
unrolling possible: when one step's updated-weight tensors thread into the next
step's forward, each step's gradient must treat the current weights as leaves
(plain SGD/Adam) rather than differentiate through the previous update.

## The Trainer

`af.Trainer` compiles a forward program once and one backward program per
parameter, then runs the training loop. It accepts either a scalar `loss` Tensor
(regression) or a `CEHandle` from `af.softmax_cross_entropy(logits, target)`
(classification). For classification the cross-entropy gradient at the logits is
the analytic, fp16-stable form `(softmax(logits) - target) / N` - no `log`
appears in the backward path, which would otherwise overflow fp16.

```python
logits = forward(x)                                   # an ANEForge graph
obj = af.softmax_cross_entropy(logits, target)
tr = af.Trainer(obj, params, lr=0.005, loss_scale=1024.0, optimizer="adam")
tr.set_dataset(x, X_full, target, Y_onehot)
for _ in range(steps):
    tr.step()
print(tr.accuracy(X_test, y_test))
```

By default the host runs an fp32 SGD or Adam update over the parameters' master
values (`af.SGD` / `af.Adam`), which is the byte-for-byte baseline path.

### `device_optimizer=True`: the update on the engine

With `device_optimizer=True` the optimizer arithmetic compiles to ANE graph
ops too. Alongside the per-parameter backward programs, a per-parameter
update program computes the new state on the engine, so no training tensor math
runs on the host. The host only computes the scalar `lr_t`, shuttles state and
gradients in and out, samples the minibatch, and prints.

One subtlety is load-bearing. The learning rate `lr_t` fed to the on-engine
Adam update must not divide by `loss_scale`. Adam's step is the ratio
`m / sqrt(v)`; with a scaled gradient `g' = S*g` the moments scale as `m' = S*m`
and `v' = S^2*v`, so the ratio is scale-invariant and the loss-scale cancels.
Dividing `lr_t` by `S` (as a naive SGD-style unscale would) double-unscales and
collapses the step. SGD is linear, so its `lr_t = lr / loss_scale` is correct;
Adam's is not. The on-engine Adam moments `m` and `v` are held as fp16 arrays,
fed each step and read back.

### `resident_state=True`: optimizer state stays on the engine

`resident_state=True` assembles the whole step - forward, backward, and the
per-parameter update - as one fused multi-output program and keeps the optimizer
state resident on-device across steps. Each updated-state output is aliased back
onto its own input port with `Program.share_buffer` (via `compile_multi` /
`MultiModel`), the shared buffers seeded once. The host then feeds only
the minibatch and `lr_t` each step, and reads the weights off the device only at
checkpoints; nothing is shuttled in and out.

Full MNIST (a 784-256-10 GELU MLP, Adam) trains to 97.79% test accuracy with
all twelve state tensors (four parameters times weight/`m`/`v`) resident across
roughly 2,340 steps, in about 1.0 s - faster than the host round-trip path
(~4 s) because there is no per-step shuttle, and with no compile wall.
(The resident-state mechanism is verified bit-for-bit against the host-shuttle path in
`tests/test_autograd.py`.)

## Loss scaling and fp16

Gradients are computed in fp16. A loss scale (commonly 1024) lifts small
gradients out of fp16 underflow before the backward pass and is divided back out
before the optimizer step (or, for Adam, cancels in the ratio as above). The
optimizer state is fine in fp16: the Adam moments stay O(1-20) and do not
overflow, so fp16 optimizer state suffices and paired-fp16 is not needed
for the models trained here.

One chip-specific caveat. On the A13-class ANE (M1), the trainable
conv's width-offset im2col slices route through a fixed-point crop-DMA that
saturates any value past 4094 (= 65504/16) to infinity, so an extreme
`loss_scale` times a large backward activation could corrupt a conv
weight-gradient. ANEForge warns in that case and never caps
`loss_scale`. The auto-cap was refuted end-to-end on M1: a real normalized CNN
trains identically at loss_scale 128, 1024, and 65536 - only a synthetic
`0.5*sum(y^2)` repro with random inputs ever reaches the threshold. The warning
fires solely on A13 targets that train a conv weight with `kW > 1`; M5 / A16 has
no such route.

## Measured results

### MNIST MLP

A 784-256-10 GELU MLP trains on full MNIST to 97.57% test accuracy with the
forward, backward, and Adam update all on the ANE - about 0.5 point under the
host-fp32-Adam baseline of 98.05% on the same model. With resident state the
same model reaches 97.79% in ~1.0 s (above).

### Cross-chip parity

The identical seeded MNIST CNN (`examples/train_mnist_cnn.py`, fixed
`np.random.default_rng(0)`, deterministic data order, default `loss_scale=1024`)
trains to 0.9080 on M1 (h13/A13), 0.9080 on M2 Pro (h14g/A14, deterministic
x3), and 0.9070 on M5 (h17/A16-class) over 300 steps. Every run is internally
deterministic across repeats. A14 lands exactly on M1's number, so the one-sample gap
(1 in 1000, 0.1%) appears only at the A16 generation - the end-to-end signature of the
ANE's <=1-ULP-per-op cross-chip fp16 difference accumulating over 300 steps, with the
drift boundary at A15/A16 rather than per-chip noise: real, but far too small to
matter. Training is chip-portable across all three measured generations. See
[cross-chip.md](cross-chip.md).

### Versus the GPU and CPU

For a small MLP, training on the ANE ties the fastest Apple GPU path
and both beat the CPU. On the same 784-256-10 GELU MLP (Adam, B=128, 5 epochs =
2,340 steps on full MNIST), all reaching ~97.5% accuracy:

| device / framework | train time | vs fastest |
|---|---|---|
| GPU Metal (MLX, fp32) | 0.87 s | 1.00x |
| ANE (ANEForge fp16, resident) | 0.93 s | 1.08x |
| CPU (PyTorch fp32) | 1.40 s | 1.62x |
| CPU (MLX fp32) | 1.77 s | 2.05x |
| GPU Metal (PyTorch MPS, fp32) | 1.87 s | 2.16x |

This is the dispatch-bound, tiny-model regime (~0.4 ms/step, ~200K parameters),
so the ANE matching the GPU is about per-step overhead, not peak FLOPS; for
larger compute-bound models the GPU's raw throughput is expected to pull ahead.
ANE training is a capability and efficiency story, not a peak-speed claim.

## Host-free multi-step dispatch

A fixed number of training steps has no data-dependent control flow, so the whole
forward -> backward -> optimizer-update recurrence unrolls into a single graph,
the same trick the iterative solvers in `aneforge.linalg` use. `af.UnrolledTrainer`
builds it: each `step()` runs K steps in one dispatch on the engine, with the
host feeding K minibatches plus the per-step learning rates (an array move, no
per-step host round-trip inside the block). With `resident=True` (the default)
the optimizer state is `share_buffer`-aliased across dispatches, so the host
feeds only the K minibatches and per-step lr, reading weights at checkpoints.

```python
xs = [af.input((B, DIN)) for _ in range(K)]
ts = [af.input((B, C)) for _ in range(K)]
tr = af.UnrolledTrainer(P, forward, "ce", xs, ts, (Xtr, onehot),
                        lr=0.01, loss_scale=1024.0)
for _ in range(epochs):
    tr.step()                 # ONE dispatch = K on-engine steps
acc = (tr.predict(Xte).argmax(1) == yte).mean()
```

One `execute_multi` drives K on-engine steps with no Path B and no
entitlement. This is performance-neutral: on the resident-buffer path the per-step
host dispatch was never the bottleneck, so collapsing the per-step `execute()`
calls into one buys nothing measurable. The bounded multi-step path is reachable
without an entitlement, and by this measurement the entitlement-gated
fully-autonomous loop would not run these workloads any faster.

For a full-batch task the data and learning rate can also be seeded once, after
which the training loop is literally `prog.execute()` and the host feeds nothing
per dispatch; `examples/train_transformer.py` trains an attention block this way.

## Worked examples

The committed demos train a 1000-sample MNIST subset (`examples/data/mnist_subset.npz`)
end to end on the engine:

- `examples/train_mnist_mlp.py` - a GELU MLP, K steps unrolled per dispatch.
- `examples/train_mnist_cnn.py` - conv -> relu -> avg_pool -> fc, with a
  trainable conv. The native ANE conv needs a baked weight, so
  `af.conv_param` / `af.conv2d` build the conv from primitives (static im2col plus
  a batched matmul) for a weight gradient that runs on the engine; the
  `conv_shape` is carried across each in-graph optimizer update automatically.
- `examples/train_transformer.py` - a multi-head self-attention block trained
  fully host-free with `af.adam_step` and resident state.
- `examples/train_transformer_prenorm.py` - a pre-norm transformer block
  (`layer_norm` before attention and before the MLP). The gradient flows through
  LayerNorm to the trainable projections and MLP; the block converges.
- `examples/train_llama_block.py` - a LLaMA-style block (`rms_norm`
  pre-normalization + a SwiGLU feed-forward built on `silu`). Exercises the
  RMSNorm and SiLU gradients end to end. The char-LM examples apply causal
  attention as an additive mask before a decomposed softmax; for a forward decoder
  block, causal attention is also available on the native fused-attention layer via
  `af.sdpa(q, k, v, is_causal=True)` (see `examples/llama_block_causal.py`).
- `examples/train_charlm.py` - training at scale: a 4-layer causal LLaMA-style
  character language model (token + positional embeddings, RMSNorm + SwiGLU blocks,
  output projection, next-token cross-entropy), forward + backward + optimizer all
  on the engine. Cross-entropy falls to near zero, next-char accuracy reaches 100%,
  and it reconstructs its training text from a seed. Token embedding is a one-hot
  matmul (`onehot @ W_emb`) rather than a `gather` (which has no VJP), so the
  embedding gradient is an ordinary matmul gradient; causal masking is an additive
  upper-triangular bias before softmax. The fused forward + backward +
  optimizer program converges through eight layers; compile time, not correctness,
  is the practical ceiling on depth.
- `examples/train_charlm_corpus.py` - the data-scale companion: the same model trains
  on random windows from a structured corpus's training split and is evaluated on a
  disjoint validation split it never trains on. Held-out next-char accuracy reaches
  about 65% against a roughly 20% unigram baseline, so the model learned transferable
  structure rather than memorizing a single sequence. Each window is re-based to
  absolute positions 0..S-1 so the learned positional embeddings apply to every
  sampled window.
- `examples/train_charlm_deep.py` - a 16-layer char-LM trained with a
  layer-streamed compile (`aneforge.streaming.CheckpointedStack`). When the layers
  are identical, the per-layer forward and per-layer backward each depend only on one
  layer's shape, so each is compiled once and reused for every layer; the six programs
  (embed, layer, head, each with a forward and a backward) compile in about half a
  second regardless of depth, where the monolithic eight-layer compile alone took
  about 162 seconds. Cross-entropy falls from about 3.0 to 0.03, removing compile
  size as the ceiling on training depth.

## Depth without a compile-size wall

A monolithic compile fuses a model's whole forward, backward, and optimizer step into
one program, so compile time grows superlinearly with depth and caps how deep a model
can train. `aneforge.streaming.CheckpointedStack` removes that ceiling for a stack of
identical layers. The per-layer forward and backward are each compiled
once and reused for every layer, fed that layer's parameters and its checkpointed
input activation, so compile work is independent of layer count. The backward is
standard gradient checkpointing: each layer's input is stored, the layer's forward is
recomputed inside the reused backward program, and that program returns the parameter
gradients plus the gradient with respect to the input (the upstream gradient
for the layer below). The streamed gradients are bit-exact against a monolithic
backward. The caller drives the embedding and output stages (each compiled once) and a
host-side optimizer over the streamed gradients, with all forward and backward compute
on the engine.

The two normalization layers in those blocks show that a real
norm-bearing architecture trains on the engine, affine included. The norm methods
are polymorphic in their affine arguments: a numpy `gamma` / `beta` bakes a fixed
affine (the native lowering, gradient flows through the input only), while a
parameter `Tensor` for `gamma` / `beta` composes the unit-affine normalize op
with an explicit learnable scale and shift, so the affine trains alongside the
rest of the model (its gradient flows through the broadcast-aware `mul` / `add`
VJPs). Pass `gamma` / `beta` shaped to broadcast over the normalized axis
(`[1, D]` for `layer_norm` / `rms_norm`, `[1, C, 1, 1]` for `group_norm`). Models
using `prelu` are likewise trainable through the `Trainer`.

## Honest limits

### Gradient vocabulary

The registered VJPs are all numerically correct - every one matches its
closed-form derivative (verified to cos ~1.0 against finite differences on M1).
The differentiable vocabulary covers the structural and linear-algebra ops, the
common activations, and the three normalization layers:

- structure / linalg: `matmul`, `bmm`, `conv`, `avg_pool`, `max_pool`,
  `reduce_sum`, `reduce_mean`, `transpose`, `reshape`, `flatten2d`,
  `slice_by_size`, `concat`, `add`, `sub`, `mul`, `muls`, `adds`, `square`
- activations: `relu`, `relu6`, `leaky_relu`, `elu`, `gelu`, `silu`,
  `sigmoid`, `sigmoid_hard`, `tanh`, `scaled_tanh`, `softmax`, `clip`
- math: `exp`, `log`, `sqrt`, `rsqrt`, `inverse`, `erf`, `cos`, `abs`
- normalization: `layer_norm`, `rms_norm`, `group_norm`, `l2_norm`

This trains transformer and LLaMA-style blocks, diffusion-UNet and GroupNorm
CNNs, plain CNNs, and MLPs end to end on the engine. The normalization VJPs use
the exact closed-form input gradient; because ANEForge has no constant-tensor op,
the backward graph re-injects `gamma` as a fed value-input.

A forward op with no registered VJP compiles and runs its forward
pass, then fails at `backward` with `no vjp for op '<name>'`. The norms and `silu` are
covered (they appear in nearly every modern architecture). The remaining gaps are
low-value and do not block mainstream training: the reduction-routing ops `amax` / `amin`
/ `cumsum`, and a few parametric activations (`atan`, `softplus`, clamped variants).

Two pre-existing test caveats are unrelated to gradient coverage and predate this
work: `test_conv2d_trainable_grads` fails with NaN (the conv weight-gradient
slice-saturation that is root-caused separately - it fails identically with the
op removed), and `test_cnn_trains_on_subset` can hang late in a full suite run
under many rapid compiles, which the compile backoff in `aneforge/_circuit.py`
paces as a defensive backstop.

### The `mul(reduce_output, 0.0)` compile wall

Multiplying a `reduce_sum` / `reduce_mean` result by zero trips an ANECCompile
fusion wall (`reduce * nonzero` compiles, and a bare mul-by-zero compiles; only
the combination fails). The autograd builds constant tensors with
`_const_like(t, c) = (t - t).adds(c)` - an exact-zero subtract rather than a
multiply by zero - which sidesteps the wall for any finite tensor, and the
backward seed folds the loss scale into an additive constant for the same reason.
Handled internally; it matters only if you author backward graphs by hand.

### Throughput

End-to-end training throughput is dispatch- and round-trip-bound like the rest of
the ANE surface at this model scale. On-engine training is a capability (train
on-device, low power, chip-portable), not a peak-speed claim against the GPU on
large compute-bound models.

## See also

- [`aneforge-api.md`](aneforge-api.md) - the full frontend reference.
- [`capabilities.md`](capabilities.md) - the hardware op surface and dtype limits.
- [`getting-started.md`](getting-started.md) - building the e5rt dispatch dylib.
