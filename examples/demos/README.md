# aneforge demos

Small, self-contained programs that **run on real ANE silicon** and demonstrate, one topic
at a time, what the Apple Neural Engine does and how to drive it well. Each script's docstring
lists exactly what it exercises; run any directly:

```sh
python3 examples/demos/what_the_ane_is.py
```

They are grouped to mirror the ANE guide's flow (the machine -> reaching it -> performance ->
workloads -> practice), but each stands alone.

### The machine
| demo | exercises |
|------|-----------|
| `what_the_ane_is.py` | build a graph, compile to one ANE program, dispatch; matmul vs numpy (cos ~1) |
| `capability_surface.py` | `OP_CATALOG` / `is_native` / `min_native_family` / `ops_on` / `walled_everywhere` |
| `execution_model_floor.py` | the fixed ~0.2 ms per-call firmware round-trip + `DispatchFloorWarning` |
| `single_in_flight.py` | one request in-flight per die; threads don't amortize; the multi-die lever |
| `numerics_fp16.py` | fp16 compute/accumulator; cancellation pitfall + `PrecisionWarning` |

### Reaching the ANE
| demo | exercises |
|------|-----------|
| `dispatch_no_coreml.py` | MIL -> on-device ANECompiler -> e5rt execute; compile-once/eval-many; no CoreML |
| `mil_dialect.py` | the CoreML MIL aneforge emits (BLOBFILE weights), via `compile(build_dir=...)` |
| `entitlement_boundary.py` | unentitled dispatch: compile via the XPC daemon, submit direct over IOKit |
| `weights_compression.py` | int8/compression is per-chip gated (M1 folds; M4+ streams); half the weight bytes |

### Performance
| demo | exercises |
|------|-----------|
| `roofline_compute.py` | measured fp16 compute peak (~4.8 TFLOP/s on M1) |
| `roofline_bandwidth.py` | measured streaming BW (~51 GB/s = the compiler's own constant) |
| `power_efficiency.py` | sustained throughput + energy/op at the ~1.48 W ANE rail |
| `ane_vs_gpu_cpu.py` | the same matmul on ANE vs numpy CPU vs torch-MPS GPU |
| `cross_chip_cost_model.py` | `af.estimate` (no-device prediction) + `af.project_peak` cross-chip |

### Optimization levers
| demo | exercises |
|------|-----------|
| `batching_amortization.py` | batch N collapses per-sample cost ~127x (per-call ~flat) |
| `chaining_depth.py` | a deep model = one fused program; per-op cost collapses |
| `zero_copy_io.py` | `input_view`/`output_view` skip the host<->device memcpy (~30% on large I/O) |
| `resident_state.py` | resident on-device state via `share_buffer` (no host re-feed) |
| `optimization_autotune.py` | `af.estimate` pruner + `af.tune` measured variant selection |

### Workloads
| demo | exercises |
|------|-----------|
| `vision_conv_encoder.py` | a conv->relu->poolx2->GAP->fc encoder fused into one program |
| `llm_attention_kvcache.py` | native SDPA hardware layer (causal + KV-cache decode shape) |
| `training_on_ane.py` | forward + backward + Adam all on the ANE (`af.Trainer`) |
| `numerical_scientific.py` | a DFT as ANE matmuls; magnitude spectrum vs numpy FFT |

### Practice & limits
| demo | exercises |
|------|-----------|
| `hidden_layers.py` | hidden hardware layers via the bridge route (native SDPA) + family gating |
| `pitfalls_limits.py` | the compile-time warnings + the hard limits to design around |

### The one-paragraph model these teach
An ANE call is a fixed firmware round-trip (~0.2 ms on M1), dispatched one-at-a-time per die,
around microseconds of compute. You make a *workload* fast by doing more work per submit
(`batching_amortization`, `chaining_depth`, `resident_state`), using all the dies
(`single_in_flight`), removing host copies (`zero_copy_io`), staying compute-bound
(`roofline_compute` vs `roofline_bandwidth`), and quantizing (`weights_compression`) - while
the compiler fuses your graph (`chaining_depth`) into one signature-gated program from your
MIL (`mil_dialect`), whose cost you can predict without the device
(`cross_chip_cost_model` / `optimization_autotune`).
