# aneforge/_invokers

The ObjC sources the native-layer **bridge ops** compile and exec at runtime. Each
authors a netplist (`Type=<Layer>`) program and invokes it on the Apple Neural Engine
through Apple's private frameworks, returning the result to the Python `_bridges/`
module that called it. Kept in the package so the bridges are self-contained - no
dependency on the reverse-engineering corpus.

Each is a generic invoker shared across an op family:

| invoker | ops |
| --- | --- |
| `layer_invoker.mm`       | `lrn`, `minmax_norm`, `scaled_elementwise` |
| `rank_invoker.mm`        | `topk`, `argmax`, `sort` |
| `sdpa_invoker.mm`        | `sdpa`, rearrange / space-batch |
| `persistent_worker.mm`   | the A2 persistent worker (sub-millisecond native dispatch) |

They link only system frameworks (Foundation / IOSurface), so they compile on any
Apple Silicon Mac; the bridge builds each once into a per-machine binary cache.

(Reverse-engineering origin in the private ane-research corpus, where these were the
ane_*_numerics_probe.mm / ane_*_invoke_probe.mm discovery tools.)
