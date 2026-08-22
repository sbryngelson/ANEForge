# ANE op catalog: every native MIL op x device (M1-M5)

> Generated from `aneforge/_op_catalog.py` in the ANEForge repository (`python docs/gen_op_catalog.py`); do not hand-edit. Query the same data at runtime via `af.op_info`, `af.is_native(op, chip)`, `af.ops_on(chip)`, `af.min_native_family(op)`, `af.walled_everywhere()`.

187 native MIL ops. Device ladder: m1=A13, m2=A14, m3=A15, m4_m5=A16/A17. Cells: Y native, ~ bridge/decompose, N walled. aneforge's higher-level ops (rms_norm/group_norm/mha/sdpa/fft/linalg/...) are composites that lower to these.

## Activations (incl. LUT)
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `ceil` | Y | Y | Y | Y | ElementWise | F2 LUT (probed native M1) |
| `clamped_relu` | Y | Y | Y | Y | ClampedRelu | LUT |
| `clip` | Y | Y | Y | Y | ElementWise | user-facing `clamp`; LUT |
| `elu` | Y | Y | Y | Y | Elu | LUT (effectively F2 -> A13+) |
| `erf` | Y | Y | Y | Y | SimpleActivation | F2 LUT |
| `exp` | Y | Y | Y | Y | ElementWise | LUT |
| `exp2` | Y | Y | Y | Y | ElementWise | F2 LUT |
| `floor` | Y | Y | Y | Y | ElementWise | F2 LUT (probed native M1) |
| `gelu` | Y | Y | Y | Y | Gelu | LUT (M1 probe: ~0.08 rel err vs exact - LUT approximation, still native) |
| `leaky_relu` | Y | Y | Y | Y | LeakyRelu | LUT |
| `log` | Y | Y | Y | Y | ElementWise | LUT (ln2 immediate) |
| `prelu` | Y | Y | Y | Y | PRelu | per-channel alpha (LUT); native at rank >=3 (M1-confirmed) |
| `relu` | Y | Y | Y | Y | SimpleActivation | F0 SimpleActivation |
| `relu6` | Y | Y | Y | Y |  | LUT |
| `round` | Y | Y | Y | Y | ElementWise | F2 LUT round-nearest (probed native M1) |
| `scaled_tanh` | Y | Y | Y | Y | ScaledTanh | LUT |
| `sigmoid` | Y | Y | Y | Y | SimpleActivation | F0/LUT (incl. hard variant) |
| `sigmoid_hard` | Y | Y | Y | Y | SigmoidHard | LUT |
| `sign` | Y | Y | Y | Y | ElementWise | F2 LUT (probed native M1) |
| `silu` | Y | Y | Y | Y | SimpleActivation | a.k.a. swish; LUT |
| `softmax` | Y | Y | Y | Y | Softmax | F2 LUT (log2e immediate) |
| `softplus` | Y | Y | Y | Y | Softplus | LUT (+ parametric) |
| `softplus_parametric` | Y | Y | Y | Y | Softplus | LUT |
| `softsign` | Y | Y | Y | Y | Softsign | LUT |
| `tanh` | Y | Y | Y | Y | ElementWise | LUT |
| `threshold` | Y | Y | Y | Y | ElementWise | LUT |
| `thresholded_relu` | Y | Y | Y | Y | ThresholdedRelu | LUT |

## Comparison / logical
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `equal` | Y | Y | Y | Y | ElementWise | F0 compare -> bool (probed native M1) |
| `greater` | Y | Y | Y | Y | ElementWise | F0 compare |
| `greater_equal` | Y | Y | Y | Y | ElementWise | F0 compare (probed native M1) |
| `less` | Y | Y | Y | Y | ElementWise | F0 compare (probed native M1) |
| `less_equal` | Y | Y | Y | Y | ElementWise | F0 compare (probed native M1) |
| `logical_and` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose via `min`/`mul` on host |
| `logical_not` | Y | Y | Y | Y | ElementWise | F0 (probed native M1) |
| `logical_or` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose via `max` on host |
| `logical_xor` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose via `!=` on host |
| `not_equal` | Y | Y | Y | Y | ElementWise | F0 compare (probed native M1) |
| `select` | Y | Y | Y | Y | Select | user-facing `where`; template_text backend |

## Control flow
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `call` | ~ | ~ | ~ | ~ | Call | function call; `mapped_no_current_hwx_case` (inlined) |
| `cond` | ~ | ~ | ~ | ~ |  | `mapped_no_current_hwx_case` + `Unsupported` converter - no standalone ANE codegen; flatten on host |
| `while_loop` | ~ | ~ | ~ | ~ |  | `mapped_no_current_hwx_case` + `Unsupported`/`WhileLoop` - unroll on host |

## Conv / MatMul / Pooling
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `avg_pool` | Y | Y | Y | Y | Pool | F0; window <=29 (M1) / 31 (A14+); 3D window A13+ |
| `conv` | Y | Y | Y | Y | Conv | F0; M1 kernels <=29x29 (13x13 fp16), 3D depth native A13+; M5 <=32x32 |
| `conv_transpose` | Y | Y | Y | Y | Conv | F0 deconv; strided axes use small-kernel caps |
| `einsum` | Y | Y | Y | Y | Einsum | lowers to matmul/transpose chain |
| `l2_pool` | Y | Y | Y | Y | Pool | special LUT pool (1024-entry fp16) |
| `linear` | Y | Y | Y | Y | Linear | folds to conv when RHS <=2 MB SRAM working set |
| `linear_activation` | Y | Y | Y | Y | LinearActivation | fused linear+activation |
| `matmul` | Y | Y | Y | Y | Matmul | NE lane / conv-fold; same tensor caps as conv |
| `max_pool` | Y | Y | Y | Y | Pool | F0 |
| `ne_bypass` | ~ | ~ | ~ | ~ | NEBypass | private NEBypass unit; `mapped_no_current_hwx_case` |
| `ne_conv` | Y | Y | Y | Y | NEConv | private NEConv unit (fill=0x44/mir=0x5d) |
| `ne_matmul` | Y | Y | Y | Y | NEMatMul | private NEMatMul unit |
| `ne_pool` | Y | Y | Y | Y | NEPool | private NEPool unit (probe-pending codegen, treated reachable) |
| `pe_elementwise` | Y | Y | Y | Y | PEElementWise | private PEElementWise unit (fill=0x49/mir=0x59) |
| `pe_goc` | ~ | ~ | ~ | ~ | PEGOC | private PEGOC unit; `mapped_no_current_hwx_case` (compiler-internal) |
| `pe_pool` | Y | Y | Y | Y | PEPool | private PEPool unit |
| `scaled_dot_product_attention` | Y | Y | Y | Y | SDPA | F2; rides matmul+softmax (NOT texture-gated) - native on M1. user-facing `sdpa` |

## Detection / sampling
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `argsort` | N | Y | Y | Y | Sort | **Sort family, A14+**; codegen-rejected on M1 (= `sort` floor) |
| `list_gather` | N | N | N | N | Unsupported | TensorList op - `Unsupported` everywhere |
| `list_length` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `list_read` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `list_scatter` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `list_write` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `make_list` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `non_maximum_suppression` | Y | Y | Y | Y | NonMaximumSuppression | template_text NMS backend |
| `random_bernoulli` | N | N | N | N | Unsupported | `Unsupported` everywhere - host RNG |
| `random_categorical` | N | N | N | N | Unsupported | `Unsupported` everywhere - host RNG |
| `random_normal` | N | N | N | N | Unsupported | `Unsupported` everywhere - host RNG |
| `random_uniform` | ~ | ~ | Y | Y | RandomUniform | **RNG, A15+** (HAL 0x4a9=0 on M1/M2); aneforge uses host RNG below A15 (`dropout`/`random` decomposable) |
| `topk` | N | Y | Y | Y | TopK | **rank/sort bridge, A14+** (`_OP_FLOOR`); bridge validator callable on M1 but **codegen-rejected** (measured) |

## Elementwise arithmetic
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `abs` | Y | Y | Y | Y | ElementWise | PEElementWise (F0) |
| `add` | Y | Y | Y | Y | ElementWise | const + tensor forms; text-immediate fused const |
| `cumsum` | Y | Y | Y | Y | CumSum | runs ON the ANE as a single op (verified M1 2026-06-09: cos 1.0) - NOT host-decomposed. The standard MIL `cumsum` op is unimplemented, so it is reached via the curated e5rt path (see _capabilities). |
| `floor_div` | Y | Y | Y | Y | ElementWise | LUT-assisted (actlut:2) |
| `inverse` | Y | Y | Y | Y | ElementWise | reciprocal LUT |
| `maximum` | Y | Y | Y | Y | ElementWise | const + tensor (LUT) |
| `minimum` | Y | Y | Y | Y | ElementWise | const + tensor (LUT) |
| `mod` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose on host |
| `mul` | Y | Y | Y | Y | ElementWise | const + tensor forms |
| `pow` | Y | Y | Y | Y | ElementWise | `pow_const`; user-facing `x ** y` (probed native M1) |
| `real_div` | Y | Y | Y | Y | ElementWise | general divide; A11/A12 = const-fp16 reciprocal only. user-facing `truediv`/`div` |
| `rsqrt` | Y | Y | Y | Y | ElementWise | F2 LUT |
| `sqrt` | Y | Y | Y | Y | ElementWise | F2 LUT activation (native A13+, decomposed on A11/A12) |
| `square` | Y | Y | Y | Y | ElementWise | F0 PEElementWise |
| `sub` | Y | Y | Y | Y | ElementWise | lowered to add-of-negated-const |

## Image / resize / texture
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `affine` | N | Y | Y | Y | Affine | **texture-engine only (A14+)**; "affine transform is not supported on this architecture" on M1 |
| `crop_resize` | N | Y | Y | Y | CropResize | **texture-engine only (A14+, HAL 0x81d)** - `_OP_FLOOR`; unavailable on M1, no host substitution wired |
| `degamma` | ~ | ~ | ~ | ~ | DeGamma | ISP/image op; `mapped_no_current_hwx_case` |
| `gamma` | ~ | ~ | ~ | ~ | Gamma | ISP/image op; `mapped_no_current_hwx_case` |
| `pixel_buffer_to_tensor` | ~ | ~ | ~ | ~ | PixelBufferToTensor | 4CC image input; `mapped_no_current_hwx_case`. Does not lower on the unentitled direct path (entitlement gate, not chip gate); use `af.image_input`. |
| `resample` | N | Y | Y | Y | Resample | **texture-engine only (A14+)**; warp depth=1, channel in {1,2}. Walled on M1 |
| `resize` | ~ | Y | Y | Y | Resize | F2 but **texture-gated**: M1 = software deconv/transpose fallback (different rounding; some modes hard-abort); native A14+ |
| `resize_bilinear` | ~ | Y | Y | Y | ResizeBilinear | NE lane; sw-fallback on M1 |
| `resize_nearest_neighbor` | ~ | Y | Y | Y | ResizeNearestNeighbor | NE lane; sw-fallback on M1 (1x1-source fast path exists) |
| `tensor_to_pixel_buffer` | ~ | ~ | ~ | ~ | TensorToPixelBuffer | `mapped_no_current_hwx_case` (compiler-internal) |
| `upsample_bilinear` | ~ | Y | Y | Y | UpsampleBilinear | NE lane; sw-fallback on M1 |
| `upsample_nearest_neighbor` | ~ | Y | Y | Y | UpsampleNearestNeighbor | NE lane; sw-fallback on M1 |

## Normalization
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `batch_norm` | Y | Y | Y | Y | BatchNorm | inference fold-to-affine runs everywhere (incl. A11/A12); native stats form is A13+ |
| `instance_norm` | Y | Y | Y | Y | InstanceNorm | F2 |
| `l2_norm` | Y | Y | Y | Y |  | F2 |
| `layer_norm` | Y | Y | Y | Y | LayerNorm | F2 (native A13+) |
| `local_response_norm` | Y | Y | Y | Y | LRNorm | LRN bridge (measured Y on M1) |

## Quantization / dtype
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `cast` | Y | Y | Y | Y | Cast | F0 format primitive. **fp16<->fp32/bool native on M1**; `cast(->int32)` is **walled on M1** (empirically confirmed) - keep dtype fp on h13 |
| `const` | ~ | ~ | ~ | ~ | ConstOps | `mapped_no_current_hwx_case` - folded at compile, not a standalone codegen op |
| `constexpr_affine_dequantize` | ~ | ~ | ~ | ~ | ConstOps | weight-compression const; folded. **int4-LUT streams natively from M1; int8/affine fold to fp16 below A15** (HAL +0x520-0x539). |
| `constexpr_blockwise_shift_scale` | ~ | ~ | Y | Y | ConstOps | blockwise stream gate A15+; folds to fp16 on M1/M2 |
| `constexpr_cast` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `constexpr_lut_to_dense` | Y | Y | Y | Y | ConstOps | palette/LUT stream gate (+0x529) is **A13-on** -> int4-LUT streams natively from M1 (*the one compressed format that wins on M1) |
| `constexpr_lut_to_sparse` | ~ | ~ | ~ | ~ | ConstOps | folded const; sparse stream A15+ |
| `constexpr_sparse_blockwise_shift_scale` | ~ | ~ | Y | Y | ConstOps | sparse+blockwise stream A15+ |
| `constexpr_sparse_to_dense` | ~ | ~ | Y | Y | ConstOps | sparse stream A15+ |
| `dequantize` | Y | Y | Y | Y | Dequantize | F0 |
| `quantize` | Y | Y | Y | Y | Quantize | F0 (not texture-gated) |

## Recurrent
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `gru` | N | N | N | N | Unsupported | `Unsupported` everywhere - unroll to conv/matmul+activation on host |
| `lstm` | N | N | N | N | Unsupported | `Unsupported` everywhere - unroll on host |
| `rnn` | N | N | N | N | Unsupported | `Unsupported` everywhere - unroll on host |

## Reductions
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `reduce_argmax` | Y | Y | Y | Y | ReduceArg | per-axis ArgMax - F0, all chips (bridge `ArgMax` measured Y on M1) |
| `reduce_argmin` | ~ | ~ | Y | Y | ReduceArg | per-axis argmin; **M1/M2 walled on the MIL route** (HAL 0x4f2, A15+), bridge mirrors argmax. user-facing `argmin` |
| `reduce_l1_norm` | Y | Y | Y | Y | Reduce | F2 Reduce |
| `reduce_l2_norm` | Y | Y | Y | Y | Reduce | F2 Reduce |
| `reduce_log_sum` | Y | Y | Y | Y | Reduce | LUT-assisted Reduce (ln2 immediate) |
| `reduce_log_sum_exp` | Y | Y | Y | Y | Reduce | LUT Reduce; aneforge wires its vjp (probed native M1) |
| `reduce_max` | Y | Y | Y | Y | Reduce | F2 |
| `reduce_mean` | Y | Y | Y | Y | Reduce | F2 |
| `reduce_min` | Y | Y | Y | Y | Reduce | F2 |
| `reduce_prod` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose (log-sum-exp / scan) on host |
| `reduce_sum` | Y | Y | Y | Y | Reduce | F2 (native A13+; decomposed on A11/A12). reduced-axis >=192 -> transpose route (>=384 on A15+) |
| `reduce_sum_square` | Y | Y | Y | Y | Reduce | F2; the 0x494 `reduce->square` fusion is M2+ only - M1 emits an extra fp16 round (<=1-round numeric, not a wall) |

## Special / math
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `acos` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `acosh` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `asin` | N | N | N | N | Unsupported | `Unsupported` everywhere - host decomposition |
| `asinh` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `atan` | Y | Y | Y | Y | ElementWise | F2 LUT - **native on M1** (probe: WORKS; the one trig in vocab on h13) |
| `atanh` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `cos` | ~ | ~ | Y | Y | ElementWise | **F4 trig, native A15+ only** (REJECTED on M1/A14); M1/M2 Horner |
| `cosh` | N | N | N | N | Unsupported | `Unsupported` everywhere (REJECTED M1 probe) |
| `cost_volume` | ~ | ~ | ~ | ~ | CostVolume | bridge `CostVolume` (measured Y on M1); `mapped_no_current_hwx_case` |
| `cross_product` | ~ | ~ | ~ | ~ | CrossProduct | bridge `CrossProduct` (measured Y on M1) but `mapped_no_current_hwx_case` in MIL map - reachable via bridge |
| `matrix_decomposition` | ~ | ~ | ~ | ~ | MatrixDecomposition | `mapped_no_current_hwx_case` - no observed codegen |
| `sin` | ~ | ~ | Y | Y | ElementWise | **F4 trig, native A15+ only** (REJECTED on M1/A14 - silicon-measured); M1/M2 use `special.py` Horner |
| `sinh` | N | N | N | N | Unsupported | `Unsupported` everywhere (REJECTED M1 probe) - `(exp(x)-exp(-x))/2` on host |
| `tan` | N | N | N | N | Unsupported | `Unsupported` everywhere (REJECTED M1 probe) - `sin/cos` Horner identity on host |

## Stateful (state / buffers)
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `circular_buffer_to_tensor` | ~ | ~ | ~ | ~ | CircularBufferToTensor | `mapped_no_current_hwx_case`; ring-buffer reader |
| `read_state` | Y | Y | Y | Y | ReadState | F2 stateful; reachable on M1 but needs the e5rt inout-tensor-desc plumbing for KV-cache. |
| `tensor_buffer_to_tensor` | ~ | ~ | ~ | ~ | TensorBufferToTensor | `mapped_no_current_hwx_case`; F2 ring/streaming buffer mover (A13+, reachable inside stateful graph) |
| `tensor_to_circular_buffer` | ~ | ~ | ~ | ~ | TensorToCircularBuffer | `mapped_no_current_hwx_case`; ring-buffer writer |
| `tensor_to_tensor_buffer` | ~ | ~ | ~ | ~ | TensorToTensorBuffer | `mapped_no_current_hwx_case` |
| `write_state` | Y | Y | Y | Y | WriteState | F2 stateful |

## Structural / shape
| op | M1 | M2 | M3 | M4/M5 | kernel | note |
|---|:--:|:--:|:--:|:--:|---|---|
| `band_part` | N | N | N | N | Unsupported | `Unsupported` everywhere (mask via host) |
| `batch_to_space` | Y | Y | Y | Y | BatchToSpace | inverse of above |
| `concat` | Y | Y | Y | Y | Concat | F0 DMA |
| `crop` | Y | Y | Y | Y | Crop | F0 slice/crop (distinct from texture `crop_resize`) |
| `depth_to_space` | Y | Y | Y | Y | DepthToSpace | F2 NE lane; user-facing `pixel_shuffle` |
| `expand_dims` | Y | Y | Y | Y | ExpandDims | F0 |
| `fill` | Y | Y | Y | Y | Fill | const tensor producer |
| `fill_like` | Y | Y | Y | Y | FillLike | const tensor producer |
| `flatten2d` | Y | Y | Y | Y |  | F0 |
| `gather` | Y | Y | Y | Y | Gather | software gather on M1 in narrow envelope (batch=1,depth=1); **hw `gather_hw` path is A14+** (`_OP_FLOOR`) |
| `gather_along_axis` | Y | Y | Y | Y | GatherAlongAxis | template_text; same M1 envelope caveat |
| `gather_nd` | ~ | Y | Y | Y | GatherND | M1 = sw-envelope only (`IsValidForH13`: batch=1,depth=1,idx-ch=3); outside it **rejected**. Native (texture) A14+ |
| `identity` | Y | Y | Y | Y | Cast | aliases Cast/no-op |
| `non_zero` | N | N | N | N | Unsupported | `Unsupported` everywhere (data-dependent shape) |
| `one_hot` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose (eye-gather) on host |
| `pad` | Y | Y | Y | Y | Pad | const pad F0 (NE lane). **symmetric/reflect pad is texture-gated -> ~/sw on M1**, native A14+ |
| `pixel_shuffle` | Y | Y | Y | Y | PixelShuffle | F2 NE lane |
| `pixel_unshuffle` | Y | Y | Y | Y | PixelUnshuffle | template_text |
| `range_1d` | ~ | ~ | ~ | ~ |  | template_text; **M1 raw-MIL probe: walled** (positional-encoding range rejects on h13 codegen) - host-precompute the const |
| `reshape` | Y | Y | Y | Y | Reshape | F0 metadata (A11/A12 = fp16-only/Flatten-or-abort) |
| `reshape_like` | Y | Y | Y | Y | ReshapeLike | F0 |
| `reverse` | Y | Y | Y | Y | Reverse | NE lane (probed native M1; aneforge wires vjp) |
| `reverse_sequence` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose on host |
| `scatter` | N | N | N | N | Unsupported | `Unsupported` everywhere - decompose on host |
| `scatter_along_axis` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `scatter_nd` | N | N | N | N | Unsupported | `Unsupported` everywhere |
| `shape` | N | N | N | N | Unsupported | `Unsupported` everywhere (static-shape graphs only) |
| `slice_by_index` | ~ | ~ | ~ | ~ | SliceByIndex | `mapped_no_current_hwx_case`; static-offset slice folds into descriptor (reachable inside graph) |
| `slice_by_size` | Y | Y | Y | Y | SliceBySize | F0; **pre-A16 width-offset quirk** (Q.4 x16 crop-DMA): CONCATenating multiple nonzero last-axis (width) slices returns WRONG ELEMENTS on A14 (the gather-axis-1 bug; a SINGLE width slice is exact on A14 - linalg column/element extraction is green on M2); on A13 a width slice also saturates |value|>4094 to +/-inf. A16/M5 exact. gather + conv-grad route off the last axis. |
| `slice_update` | Y | Y | Y | Y | SliceUpdate | template_text backend |
| `sliding_windows` | N | N | N | N | NotImplemented | `NotImplemented` on any backend - decompose on host |
| `space_to_batch` | Y | Y | Y | Y | SpaceToBatch | factor in {2,3,4,8}; batch cap 4096 (older)/65536 |
| `space_to_depth` | Y | Y | Y | Y | SpaceToDepth | F2 NE lane; user-facing `pixel_unshuffle` |
| `split` | Y | Y | Y | Y | Split | F0 |
| `squeeze` | Y | Y | Y | Y | Squeeze | F0 |
| `stack` | Y | Y | Y | Y | Stack | F0 |
| `tile` | Y | Y | Y | Y | Tile | F2 (A13+); factors of {2,3,4,8}. Absent on A11/A12 |
| `transpose` | Y | Y | Y | Y | Transpose | F0 but capped by max-transpose-extent (16384 M1-A15 -> 65536 M5; 0 on A11/A12) |
