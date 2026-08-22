# On-ANE autograd

ANEForge implements a small reverse-mode autograd over its own graph IR in which forward, backward, and the optimizer update all compile to ANE programs. The host never runs tensor math during a training step - it only samples minibatches, computes a scalar learning rate, shuttles state, and prints. This page documents how that works and the ANE/numerics reasons behind the non-obvious choices.

## Core model

A training run treats trainable weights as ordinary graph inputs:

- A trainable leaf (`af.parameter`, `af.conv_param`) is a graph input tagged `trainable`, holding an fp32 master value in `attrs['value']`. It is fed its current value each eval and updated by the optimizer afterwards.
- Because parameters are fed inputs rather than baked constants, weight updates need no recompile: the host updates the master and feeds it back on the next eval.

`backward(loss, params, stop=...)` returns `{param: grad_Tensor}`, the reverse-mode gradients of scalar `loss` with respect to each parameter. The seed `dL/dloss = loss_scale` is folded into the seed's additive constant, which avoids a `muls` on the reduced loss output.

### The stop-gradient frontier

`stop` is the detach frontier: gradient reaches these tensors but does not propagate past them. It defaults to `params`, which is a no-op when the params are true graph leaves (the usual case).

It becomes load-bearing for unrolled training, where one step's updated-weight tensors are threaded into the next step's forward. There each step's gradient must treat the current weights as leaves (plain SGD/Adam), not differentiate through the previous update - otherwise the unroll becomes a second-order computation.

## The VJP registry

Each forward op already runs natively on the ANE; adding a vector-Jacobian product (VJP) makes it trainable. VJPs are built from existing ANE ops, reusing the forward output `t` where it is itself the derivative term (`exp`/`inverse`/`rsqrt`), else recomputing from `t.srcs[0]`. The registry was grown by a gradient audit that surfaced which forward ops lacked a backward rule.

| VJP group | What it unlocks / notes |
|---|---|
| Unary math / activations | Closed-form derivatives from existing ops; surfaced by the gradient audit. |
| `transpose` / `reshape` | Pure index re-arrangements; gradient is the exact inverse re-arrangement of `g` (no numerics, fp16-exact). Together with linear/bmm/softmax these make a whole `af.mha` transformer block differentiable end to end. |
| `rms_norm` / `layer_norm` / `channel_layer_norm` / `group_norm` | Unlock transformer / LLM / CNN training. All reduce over the **last** dim; closed-form `grad_x` is exact. |
| `matmul` (baked weight) | Frozen baked weight stores `W^T` as `attrs['wt']` shaped `[N,K]`; only the activation gets a grad: `gx = g @ wt`. |
| `bmm` (broadcast batch) | `ga = g @ b^T`, `gb = a^T @ g`. When an operand broadcasts over the batch dim (e.g. `patches[N,L,K] @ W[1,K,Cout]`), `_unbroadcast` sums the products back to each operand's own shape; equal-batch bmm is unchanged. |
| `relu` | `dx = select(x > 0, g, 0)` via the exposed `greater` + `select`; the zero rhs is an exact-zero tensor `x - x`. |
| `slice_by_size` | `dx` scatters `g` into a zeros-like-`x` at the same offset, built on-ANE by concatenating zero-slices (no walled scatter op). Overlapping im2col slices accumulate through the reverse grad summation. Verified cos 1.0 vs a numpy scatter. |
| `conv` (input grad only) | See below. |
| `avg_pool` / `max_pool` | See below. |

### gamma as a fed value-input

`gamma` in the norm VJPs is a per-channel constant, not a graph Tensor. ANEForge has no const-tensor op, so `_gamma_input` re-injects it as a fed value-input shaped `[1,...,1,D]` (the trainer feeds any input carrying `attrs['value']`).

### The `reduce * 0` fusion wall

`_const_like(t, c)` builds a constant-valued tensor shaped like `t` from existing ops as `(t - t).adds(c)` rather than the obvious `(t * 0.0).adds(c)`.

!!! warning "ANECCompile fusion wall"
    `mul(reduce_output, 0.0)` trips an ANECCompile fusion wall: a `reduce_sum`/`reduce_mean` result multiplied by zero fails to compile. Only that specific combination fails - `reduce * nonzero` compiles, and mul-by-zero without a preceding reduce compiles. `t - t` is an exact-zero `sub` (not a mul-by-zero) and sidesteps the wall for any finite `t`.

For the same reason `_vjp_reduce_mean` uses the tensor-tensor form `g * _const_like(x, 1/n)` (which broadcasts `g` to `x` and scales by `1/n` in one op) as the uniform, wall-proof pattern, even though `1/n` is nonzero and a plain `muls` would also compile.

## Convolution

The native ANE conv requires a const (baked) weight - a runtime tensor-weight conv is rejected by Espresso ("Not implemented ... not supported on any backend"). This drives two separate paths.

### conv VJP (input gradient only)

For a native `conv` node the weight is not a graph input, so autograd never asks for its gradient; only the input gradient is defined, the standard transposed-conv backward:

```
grad_input = conv_transpose(grad_out, W)   # forward stride/pad/dilation
```

The conv weight `[Cout,Cin,kH,kW]` is passed as-is to `conv_transpose` (whose `[Cin_ct,Cout_ct,kH,kW]` layout matches because the transposed conv's input channels are the forward conv's output channels). Verified cos 1.0 vs torch for stride=1 (pad 0/1). stride>1 needs an `output_padding` the op lacks and is rejected.

### Trainable conv (input + weight gradients)

To train a conv weight, ANEForge builds the conv from primitives so the weight is a real graph parameter:

- im2col via static `slice_by_size` + `concat` (not the walled `KernelRasterizer`) into patches,
- then a broadcast batched-matmul by the weight parameter.

Every op has a VJP, so both input and weight gradients are produced automatically and run on the ANE. Verified cos 1.0 vs torch for forward and both gradients (stride=1, pad=0). Downsampling uses `avg_pool`, not stride.

- `conv2d(x, weight, stride, pad)` - `x` is `[N,Cin,H,W]`; returns `[N,Cout,Hout,Wout]`. `stride` must be 1 (strided slicing is unavailable). `pad>=0` zero-pads in-graph via a zero-border `concat` before the im2col, so a 'same' conv stays inside one fused program and the padding differentiates through the existing concat VJP. With `pad=0` the behaviour is byte-for-byte the earlier implementation.
- `conv_param(weight_init)` - `weight_init` is `[Cout,Cin,kH,kW]` (PyTorch layout), stored internally as the flat patch matrix `[Cin*kH*kW, Cout]` that `conv2d` consumes. Patch (row) order is `ci*(kH*kW) + (u*kW + v)`, matching the im2col.

!!! note "Compile scales with batch N"
    The im2col materialises `[N, Cin*kH*kW, Hout*Wout]` tensors, so compile (tiling/partition) time grows with N. On M1/A13 a very large full batch (e.g. N~1000 over 28x28) can take minutes or hang the compiler (M5 compiles it fine). Train in mini-batches (e.g. N <= 128 fed per step), not one full-batch graph.

### A13 width-slice saturation

The patch index is placed on axis 2, not the last (width) axis. The concat's backward is a slice along the patch axis, and the A13 x16 crop-DMA saturation (any `|value| > 4094` -> +/-inf) fires only on a nonzero begin-offset of the last (width) axis. With patches on a non-last axis, the large loss-scaled input-gradient never transits the saturating slice, so multi-layer conv training is numerically correct on M1 at any `loss_scale`. (The forward x-slice still carries the small input; the transpose moving K to the last axis has a pure-permute backward with no slice.)

This is a warn, never auto-cap condition:

- `_A13_CONV_WGRAD_LOSS_SCALE_MAX` documents the ceiling: on A13 the weight-grad runs loss-scaled backward activations through width-offset slices (`conv2d`'s `for u,v` with `v>0`), so `loss_scale x |backward activation|` must stay under 4094 (= fp16max/16). A synthetic repro (`0.5*sum(y^2)` loss + random inputs) breaks at `loss_scale >= 512`, but a real normalized CNN trains identically at loss_scale 128/1024/65536 on M1 - real nets never reach those magnitudes. M5/A16 have a clean route.
- `_has_conv_wgrad` is true iff a param is a trainable conv weight with `kW>1` (a `kW=1` conv has no width-offset slice).
- `_guard_a13_conv_loss_scale` warns (returns `loss_scale` unchanged) when the target is A13-class, a conv weight is trained, and `loss_scale` could push activations past 4094. Target resolves via the same `detect_family()` / `ANEFORGE_TARGET` path as compile.

## Cross-entropy: analytic fp16-stable gradient

`CEHandle` is a softmax-cross-entropy objective carrying the logits and a one-hot target (a graph input). The gradient at the logits is the analytic fused form:

```
dL/dlogits = (softmax(logits) - target) / N
```

This is fp16-stable because it contains no `log`. With the seed:

```
CEHandle.seed = (softmax(logits) - target) * (loss_scale / N)
```

The loss value and accuracy are computed host-side in fp32 by the Trainer.

## On-ANE optimizer update

The optimizer update arithmetic is built as graph ops so it runs on the ANE. The learning rate `lr_t` is a fed `[1,1]` input (broadcast tensor-mul over the param), not a baked `muls`, so per-step bias correction / loss-scale folding varies correctly each step. Verified on-device (cos 1.0 vs numpy).

- `_adam_update` - Adam as ANE ops; `b1`/`b2`/`eps` baked, `lr_t` fed (bias correction folded into `lr_t` host-side). Returns `(w', m', v')`. `lr_t * m2` is a tensor-mul broadcasting the fed `[1,1]`; the baked scalars (`m*b1`, `g*(1-b1)`, `.adds(eps)`) are fine because none is a mul-by-zero on a reduce output.
- `adam_step` - one Adam update over lists `params`/`m`/`v`, returning new tensor lists; used to unroll K steps (thread the returned lists into the next step's forward). Propagates a `conv_param`'s `conv_shape` onto the updated tensor so it still works as a `conv2d` weight downstream.

### Adam loss-scale cancels in the ratio

`_device_adam_step` computes the host scalar `lr_t = lr * sqrt(1-b2^t)/(1-b1^t)` (bias correction folded in) and does not divide out the loss-scale. Adam's update is the ratio `m/sqrt(v)`; with scaled grad `g' = scale*g` the moments scale as `m' = scale*m`, `v' = scale^2*v`, so `m'/sqrt(v') = m/sqrt(v)` - the loss-scale cancels. Dividing `lr_t` by `scale` (a naive SGD-style unscale) would double-unscale and collapse the step. (`eps` is the only scale-sensitive term and is negligible vs `scale*sqrt(v)`.)

### The 3-output Adam stack and the wide-row wall

`_stack3` packs Adam's three same-shape outputs into one output by concatenating along axis 0 in their natural row width: `concat([w,m,v], axis=0)` -> `[3*rows, cols]`. This makes the 3-output update compile as a single-output program (no `_compile.py` change); the host splits it back row-block-wise via `_split3`.

!!! warning "ANECCompile wide-row wall"
    The natural-width axis-0 stack is load-bearing. Reshaping each output to a wide `[1, prod(shape)]` row and concatenating those trips the wide-row wall once `prod(shape)` is large (a 784x128 = 100352-wide row segfaults; verified). Keeping the column width and stacking rows stays inside the verified 2-D envelope. Params are flattened to 2-D first so `cols` is well defined.

## Loss-scaling overflow guard

`_check_finite_grads` returns true iff every gradient is finite (one `np.isfinite` reduction per array). Otherwise it bumps the optimizer's consecutive-skip counter, warns (first skip, then every 100th), and the caller skips the entire step - the standard loss-scaling overflow idiom. A skipped step leaves the fp32 masters (and Adam's `t`/moments) untouched, so a later finite step is unaffected. The warning advises lowering `loss_scale` on inf/nan weight-grads (on the ANE this means a loss-scaled fp16 gradient overflowed).

## Trainer

`Trainer` compiles a forward program once plus one backward program per parameter (each emitting that param's gradient in its natural 2-D shape). `step` evals the backward programs on the ANE and applies the optimizer; params update host-side and are fed back next eval with no recompile.

!!! note "Why one program per parameter"
    Concatenating all grads into a single wide row trips the ANECCompile wide-row wall when a large weight grad (e.g. 784x128) is reshaped to a 100k-wide row and concatenated with a differently-sized row (the reshape-to-wide-row-into-concat is the trigger, not the matmul). Per-param programs stay inside the verified 2-D matmul envelope, and the optimizer consumes the grads in param shape anyway.

It accepts either:

- a scalar `loss` Tensor (regression) - forward outputs the loss scalar; backward seeds from a ones-seed at the loss; or
- a `CEHandle` (classification) - forward outputs the logits; backward seeds from the analytic on-ANE gradient `(softmax(logits) - target) * (loss_scale / N)`. Host-side `loss()` (fp32 cross-entropy) and `accuracy(X, y_labels)` (argmax) read the logits program.

`optimizer="sgd"|"adam"` selects the optimizer.

### device_optimizer and resident state

`device_optimizer=True` runs the optimizer step on the ANE: alongside the per-param backward programs, a per-param update program computes the new state with ANE graph ops. The host then only computes the scalar `lr_t`, shuttles state/grads in/out, samples minibatch indices, and prints. Adam's `m`/`v` are held host-side as fp16 arrays, fed each step and read back. `device_optimizer=False` (the default) keeps the host fp32 optimizer path byte-for-byte unchanged (regression guard + baseline).

- `_build_device_optimizer` compiles a per-param update program. SGD: `(w_p, g_p, lr_t) -> w_p'` (single output). Adam: `(w_p, m_p, v_p, g_p, lr_t) -> stack(w_p', m_p', v_p')` via `_stack3`, split host-side. `g_p`/`m_p`/`v_p`/`lr_t` are plain (non-trainable) leaves fed each step; `w_p` is the param leaf itself.
- `_build_resident` goes further: the whole training step is one fused multi-output program with optimizer state resident on-device. Each param `p` (and, for Adam, `m_p`/`v_p`) is a graph input whose updated value is a program output aliased back onto that input port via `share_buffer` - so state lives on the engine across steps and the host feeds only the minibatch `(x, target)` + scalar `lr_t`, reading state back at checkpoints. Within one execute, the stream's FIFO ordering has the forward read the pre-step param and the update overwrite it last (the next step reads the advanced value).

!!! note "Precision-check skip"
    ANEForge's own training kernels (forward loss, per-param backward, optimizer update) contain structural subtracts - the loss `pred - target`, gradient axpys, the `w - lr*g` update - that trip the generic `cancel_sub` precision heuristic. These are vouched, accuracy-tested kernels (the MNIST baselines + the corpus), not user-data modeling choices, so they skip the user-facing precision check.

## UnrolledTrainer

`UnrolledTrainer` trains with K Adam steps unrolled into one fused ANE program, so the entire forward -> backward -> optimizer-update recurrence runs on the engine with no per-step host loop. Each `step()` runs K steps in a single dispatch: the host feeds K minibatches plus K per-step learning rates and shuttles `(params, m, v)` in and out between K-step blocks (an array move, not tensor math; no per-step host<->device round-trip inside the block). It is the bounded-K, fully-on-engine analogue of `Trainer` (whose `step()` dispatches once per step), enabled by the stop-gradient frontier above.

Key arguments:

- `params` - trainable leaves (`af.parameter` / `af.conv_param`).
- `forward(P, x) -> output` - builds the model from the current-step weight tensors `P` (aligned with `params`) and data input `x`; used per unrolled step and for `predict`.
- `kind` - `"ce"` (logits + softmax-cross-entropy) or `"mse"`.
- `x_inputs` / `t_inputs` - K data / target input placeholders, one per step.
- `dataset` - `(X, Y)` numpy arrays (`X` shaped per sample, `Y` one-hot `[N,C]` for `"ce"`).
- `resident` (default True) - optimizer state stays resident on-device across dispatches: each updated-state output is aliased (`share_buffer`) onto its own input port, seeded once. The host feeds only the K minibatches + per-step lr and reads weights at checkpoints; nothing is shuttled between dispatches. This is the end-to-end fully-on-engine path (no per-step loop and no state move).

!!! note "Separate eval program for predict"
    Checkpoint `predict` uses a separate single-batch forward program with its own weight-input leaves (fed the masters), not the trainable params: compile mutates each Tensor's internal name, so sharing the param objects between the two programs would clobber the training program's ports.

## Measured results

Training fully on the Neural Engine reaches roughly:

- ~97% on MNIST, and
- 71% on CIFAR-10,

with the device-optimizer / resident-state paths matching the host fp32 baseline (the ~98% MNIST regression guard).
