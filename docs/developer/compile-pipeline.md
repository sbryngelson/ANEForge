# The compile pipeline

This page traces how `compile(out)` lowers an ANEForge graph into **one fused e5rt program** that runs on the Apple Neural Engine: topological ordering, the per-op MIL emit registry, weight packing into a single BLOBFILE, the dispatch-floor signal, and the fused-vs-segmented split.

## From graph to one fused program

`compile(out)` topologically orders the graph rooted at `out`, emits a single MIL function — one op per graph node — packs every weight into one BLOBFILE, and hands the result to e5rt on the ANE. Each op's MIL is produced by a small handler registered with `@op(...)`. That `_EMIT` registry does double duty: it is also the exact set of ops ANEForge can reach through the fused MIL door on-device.

When the graph contains `af.sdpa` (or any other netplist-bridge op), the result is instead a **segmented plan** — e5rt program segments split around each native sub-program. See [Fused vs. segmented](#fused-vs-segmented) below.

!!! note "One program, one dispatch"
    The whole point of the fused path is that a graph of many nodes pays a single ANE dispatch. The cost model and the dispatch-floor warning both lean on this — see [The optimizer and the cost model](optimizer-and-cost.md).

### Topological ordering

`_topo` is an **iterative** post-order DFS over an explicit stack, deliberately not recursive: deep unrolled graphs would otherwise blow Python's recursion limit. Sources are appended before their consumers, which yields a valid topological order.

### Per-family gating before lowering

`_retarget_for` gates the graph for a target ANE family *before* lowering. It raises a clear compile-time error on the H13+ hard floor, on an unreachable op, or on a tensor exceeding the family's limits; where a below-floor op has an in-graph decomposition (e.g. `sin`/`cos` → `aneforge.special`), it substitutes it and returns the rewritten graph. On the host family with an all-native graph this is a no-op fast path that returns `out` unchanged.

ANEForge requires an H13+ (A13+) ANE — MIL is only supported for H13+ architectures, and a target family below that floor is rejected up front.

## The `compile()` knobs: `opt` and `compress`

`opt` selects the graph optimizer (default `'routes'`). The full semantics live in [the optimizer page](optimizer-and-cost.md#opt-levels); in brief:

| `opt` | Behavior |
|-------|----------|
| `'routes'` (default) | Lossless, cost-model-driven route selection per route-bearing bridge node. Never changes numerics; a cut-free graph compiles to exactly the `opt=0` program. |
| `0` | No optimization — the historical, byte-identical path. Use for byte-identity tests. |
| `1` | Cost-model pick over the full variant set (route swaps + the lossy whole-graph int8 variant), no on-device measurement. |
| `2` / `'max'` | Autotune — measure legal proven-safe variants on the ANE, validate each against the `opt=0` baseline, return the fastest correct one (cached). |

`compress` is the unified weight-encoding knob:

- `None` — fp16 (default, byte-identical at `opt=0`)
- `'int8'` — per-channel affine
- `'int4'` — LUT, accuracy-gated
- `'sparse'` — bitmask, when the weight is ≥50% zeros
- `'blockwise'` — per-inner-block int8 (`constexpr_blockwise_shift_scale`, `block_size` columns per scale), accuracy-gated → int8 → fp16
- `'auto'` — per-weight, family-aware (below)

`int8=True` is the back-compat alias for `compress='int8'`. `compress_atol` is the int4/blockwise fallback budget (relative L2); `block_size` is the inner-dim block width for `'blockwise'`.

!!! warning "compress stays on the byte-identical path"
    Compressed weights ride the `opt=0` lowering. Passing `compress` together with an explicit `opt>=1` is **rejected**, and the default route pass is skipped for compressed compiles — they take the no-op path.

### `compress='auto'` is family-aware

`'auto'` picks, per weight, the most aggressive encoding that stays correct: sparse if the weight is sparse, else int4 if accurate, else int8, else fp16. But it only considers encodings that **stream natively** on the target family (host-detected when `target=None`):

- On **h13/M1** the native-streaming set is int4-LUT + sparse, so auto streams those (sparse for ≥50%-zero weights) and skips int8/blockwise — they fold to dense fp16, costing accuracy for zero bandwidth win, so fp16 dominates them. A rejected int4 falls to fp16, not int8.
- On **A14+** all four encodings stream, so all are auto candidates.
- `family=None` keeps every branch enabled (historical behavior).

Explicit single-mode knobs are **never** filtered — that's the user's call.

## Weight encoding precedence

`_Emitter.weight(name)` declares a constant weight and chooses its encoding by this precedence:

```
sparse  →  int4-LUT  →  per-channel int8  →  fp16
```

- **sparse** when `compress='sparse'`, allowed, and the weight is genuinely sparse (≥50% zeros, not all-zero).
- **int4-LUT** when `compress='int4'`, allowed, inner dim even, and within the accuracy budget. Gated by per-tensor relative-L2 reconstruction error vs `compress_atol`; it falls back rather than emitting a too-lossy weight. The int4 fallback chain is **int4 → int8 → fp16**: on a gate miss or odd inner dim it falls to int8 if `allow_int8` (denser than fp16, and the user opted into compression), else fp16.
- **per-channel int8** on the `compress='int8'`/`int8` override.
- **fp16** otherwise. `compress=None`/`opt=0` stays byte-identical (the fp16 branch).

Branch order *is* the `'auto'` precedence. Single-mode knobs enable exactly their one compressed branch; `'auto'` enables the branches that stream natively and takes the most-aggressive that stays correct (sparse lossless, int4 accuracy-gated, int8 the dense fallback). Under `'auto'`, each compressed branch is additionally gated on streaming natively on the target family (`self._auto_streams`).

!!! note "fp16 norm in fp32"
    The int4-LUT / blockwise relative-error checks compute the weight norm in fp32: `W2.dot(W2)` overflows fp16 for large weights.

### int4-LUT uses ND indices (a routability wall)

LUT indices always have the same shape as the weight (`lut.rank == W.ndim + 2`). For 2-D weights this is `[1,1,16,1]` exactly as before (byte-identical). For ND weights (e.g. conv) the lut is `[1]*N + [16,1]`; Espresso accepts any rank as long as `lut.rank == indices.rank + 2`. The reason ND indices are *mandatory*: a reshape of a `constexpr_` output is **not routable** on the ANE backend, so the indices must already carry the weight's rank.

### Blockwise dequant needs a +0 bridge

For `'blockwise'`, `data` has the weight's shape and `scale` is `[OUT, nblocks]`. On-device `constexpr_blockwise_shift_scale` reconstructs `data.reshape(OUT, nblocks, bs) * scale[:,:,None]` (zero offset; verified multi-block on M5 at relerr ~3e-4). The wall: that constexpr output is **not routable** as a matmul/conv weight operand (Espresso "Not implemented") — it only feeds elementwise consumers. So ANEForge **bridges it through a `+0` add** to a dense fp16 result that can back a matmul/conv. Net effect: the weight is dequantized in-program, not streamed as int8 — the disk artifact is ~2× smaller but there is **no bandwidth win** (unlike int4-LUT/sparse, which stream compressed). Per-channel int8 is the exception: it *is* routable as a conv weight (`constexpr_affine_dequantize` feeds the conv weight operand directly, no +0 bridge) and so halves DRAM weight bytes at the MAC port.

## Multi-output ops

`split` emits N equal outputs once (keyed on `which == 0`); later splits of the same source reuse the already-emitted multi-output statement by name (`n0_<i>`). A per-source marker on the emitter guards against re-emit.

## Model surface

A compiled `Model` carries a few internal hooks useful when modifying the pipeline:

- **`Model._input_tensors`** — the ordered input `Tensor` objects (same order as `inputs`). It lets callers such as `autograd.Trainer` map each compiled input back to its source Tensor (trainable parameter vs. provided data). Additive; no effect on inference.
- **Zero-copy hot-loop API** — skip the per-call host↔device memcpy:

    ```python
    v = model.input_view()      # fp16 view; write into it
    model.execute()
    out = model.output_view()   # fp16 view onto the result, no copy
    ```

    Use this for tight inference/training loops; for one-off calls just use `__call__`.

- **`MultiModel`** — a compiled fused program with N *named* outputs, e.g. a resident training step `(x, y, lr, w, m, v, ...) -> (w', m', v', ...)`. Unlike `Model` it keeps the ordered input/output Tensors and exposes the raw `Program`, so callers (the `autograd.Trainer` resident path) can alias state outputs onto their inputs via `share_buffer` and drive a resident loop. `__call__` evals once and returns outputs by name. Port names are snapshotted at compile time — `compile()`/`compile_multi()` reassign `t._name` on shared Tensor objects, so a later compile of another graph sharing these tensors must not break this model's feed/read names.

## The dispatch-floor signal

`compile` warns (once) via `DispatchFloorWarning` when a program is **dispatch-floor-bound**: its predicted ANE time is dominated by the fixed per-call dispatch + firmware round-trip (~0.2 ms on M1-class parts), so each call costs about the same however small the work is.

The check (`_dispatch_floor_signal`) is a cheap structural estimate — no device work — and fires only when the predicted cost is dominated by the dispatch floor (it stays silent on already compute/bytes-bound programs, where amortizing buys little). The mechanics, and why this matters:

!!! warning "Dispatch is single-in-flight"
    The ANE per-call latency is a fixed dispatch + firmware round-trip. Batching and op-chaining amortize it; **concurrency does not** — threads do not amortize a single-in-flight dispatch. A caller looping per-sample silently pays the full fixed cost every iteration. To amortize: increase batch N (per-sample cost falls ~linearly toward the compute rate) or chain more ops into one program (`share_buffer` keeps state resident). Silence with `warnings.filterwarnings('ignore', category=aneforge.DispatchFloorWarning)`.

A companion `PrecisionWarning` fires when a graph has fp16-cancellation-risk nodes whose results may be inaccurate; the structural model behind it is documented under [the precision risk model](optimizer-and-cost.md#the-precision-risk-model).

## Fused vs. segmented

Most graphs lower to one fused program. A graph that contains a **netplist-bridge op** — a node that must run as a native-ANE Path-A sub-program, like `af.sdpa`, `argmax`, `topk`, `sort` — instead compiles to a **segmented plan**: fused e5rt program segments interleaved with native sub-program calls.

The `NETPLIST_OPS` table maps each bridge op name to a runner that, given materialized fp16 inputs plus the node attrs, returns the fp16 output. Adding an op there makes it a legal cut point. The fused MIL emitters are always preferred; bridges exist only for ops the ANE backend doesn't expose through MIL. The bridge closures live inside the package (lazily importing from `aneforge._bridges.<module>`), so ANEForge stays self-contained with no runtime dependency on the reverse-engineering corpus.

### How tensors thread between segments

A `SegmentedModel` is a compiled plan of e5rt segments interleaved with native-ANE sub-program calls. Tensors thread between segments as **host fp16 arrays** — correctness-first.

The throughput follow-up is **persistent Path-A workers** (the "A2" path): one worker per netplist stage that has a worker route, built lazily on first call (keyed by stage), reused for the model's lifetime, and released in `release()`. Set `ANEFORGE_NETPLIST_WORKER=0` to force the A1 (subprocess-per-call) path.

Runner resolution prefers a persistent worker (load-once-eval-many) and falls back to the subprocess bridge when:

- **No worker route exists** for the op — the subprocess bridge *is* the normal path, not a degradation, so it stays silent.
- **Causal/masked SDPA** — the per-call additive-mask injection lives on the subprocess bridge (`_run_sdpa`); the persistent worker pre-builds a mask-less netplist, so masked SDPA routes to the bridge (correctness over the worker).
- **Genuine worker failure** — fall back to the verified subprocess path (correctness-preserving, slower per call) and remember it so the worker isn't retried every call. Signaled once per op.

## Cross-family compile validation

`cross_compile_check` answers: does the graph rooted at `out` compile for another ANE family, checked from *this* host? `target` is a compiler arch string (`'h13'`) or a family int. It returns True iff the e5rt compiler produces a library for that `TargetArchitecture` — **compile-level validation only**; it cannot execute on a different family from this host. This is the keystone of cross-chip CI: validate that the op corpus compiles for every chip family on one box. Numeric correctness still needs the real silicon. The full gating story (including why a static pre-gate is required) is on the [capabilities & targets page](capabilities-and-targets.md#cross-compile-validation).
