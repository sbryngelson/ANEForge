# Per-operator ANE quirks

A reference for the hardware constraints baked into ANEForge's op constructors. Each is guarded at build time (a clear error, not a cryptic `ANECCompile` crash), and most were found by reverse-engineering or fuzzing the silicon. The pattern throughout: the ANE has the capability, but only within a narrow envelope.

## Global constraints

| Constraint | Limit | Why |
|---|---|---|
| Tensor rank | rank <= 5 | The ANE dimension model is rank 0..5 ("Rank ... must be between 0 and 5"). A rank-6+ tensor (reshape/stack/expand_dims past 5D) fails ANECCompile; guarded at construction. |
| Compute dtype | fp16 only | fp32/int32/bf16 are not implemented on the backend; rejected. |
| Per-axis size | <= 65536 | Hard per-axis cap; relevant to tiled-axis lowerings (see `group_norm`). |

## Convolution family

| Op | Quirk | Why |
|---|---|---|
| `conv` | kernel width `kW <= 15` (kH unconstrained) | The ANE conv tiles along the kernel **width**; `kW >= 16` is unsupported by ANECCompile (verified: 16x3 compiles, 3x16 does not). |
| `conv_transpose` | same `kW <= 15` width limit | Same kernel-width tiling limit. |
| `dynamic_conv` | same `kW <= 15`; **batch must be 1** | The native dynamic-kernel path (`CreateDynamicKernel`/DynamicGOC) does not support a dynamic-weight conv with batch >= 2. For batched conv use `af.conv` (constant weight) or im2col `conv2d`. |

## Reductions and norms

- `group_norm` tiled-axis bound (`finding_sd15`): the rank-4 tiled lowering reshapes to `[1,G,C/groups,H*W]` and reduces the trailing two axes, so the bound is the largest single axis, `max(C/groups, H*W)`, against the per-axis cap of 65536 - not the flattened `(C/groups)*H*W` product (which overflowed for SD-UNet's 512ch@128 and 640ch@64).
- `channel_layer_norm`: LayerNorm over the channel axis of a channels-first `[N, C, 1, S]` tensor (the ANE-native transformer layout). Same result as `layer_norm` on the `[N*S, C]` view, but with no transpose into/out of `[seq, d]` - which keeps the attention/MLP stack cheap (projections stay 1x1 convs over `[N, C, 1, S]`).
- `l2_norm`: runs as fused e5rt MIL (`reduce_l2_norm` over the axis, then `real_div`), no graph cut. The MIL `l2_norm` op normalizes over all non-batch dims, so the per-axis form `x / sqrt(sum(x2, axis) + eps)` is built explicitly.
- Matmul int8 layout: store as `[N,K]` and consume with `transpose_y=true` (the proven int8 layout).
- Trainable norm affine: `rms_norm`/`layer_norm`/`group_norm` take either arrays (fixed/baked affine) or broadcastable parameter `Tensor`s (trainable). The trainable path normalizes with a unit affine, then scales/shifts by the Tensors so gradients flow via the mul/add VJPs.

## Gather (slice+concat, width-axis bug)

`gather` for static indices lowers to `slice_by_size` + `concat`. But a last-axis (width) gather lowers to a `slice_by_size` with a nonzero width begin-offset, which routes through the A13/A14 x16 fixed-point crop-DMA path and returns the wrong elements there (correct on A16+).

!!! warning "Gather a non-last axis instead"
    For rank>=2, transpose the gathered axis off the last position and transpose back; for rank 1, gather a `[N,1]` view. Both are identity-preserving and correct on every chip family - the same width-axis-slice avoidance the conv im2col backward uses.

## Attention (mha / cross_attention / sdpa)

### Query-tiling (performance, exact)

| Op | Behavior |
|---|---|
| `mha` | Computes `[tile, S]` score tiles per head instead of the full `[H, S, S]` matrix. Exact and ~3x faster at large S (smaller tiles pipeline better). Count is an S-based heuristic unless tuned via `af.tune_attention`. |
| `cross_attention` | When query and context are both long, the `[H, S, T]` score matrix hits the materialization wall (~2.4x slower). Query-tiling into `[tile, T]` blocks avoids it, exact. Gated on score area, so small-T cross-attention (e.g. SD text conditioning, T=77) stays single-shot and byte-identical. |
| `sdpa` (non-causal decomposition) | For a large query axis, compute in `[tile, Skv]` tiles (~512 query rows) instead of the full `[Sq, Skv]` matrix. Exact, ~3x faster at Sq=1500, H=16. A small query (KV-cache decode) takes one shot. |

### Native fused-attention reliability bounds

`sdpa` uses the ANE's native fused-attention layer (`ANECSDPALayerDesc`, a path Apple's MIL compiler never emits - it always decomposes) only where it's numerically reliable; outside, it emits the accurate fused decomposition. Q/K/V are `[1, heads, seq, d_head]`, fp16. Where native is used, it's a graph-cut boundary (see [Native bridges](bridges.md)).

| Constant | Value | Meaning |
|---|---|---|
| `SDPA_NATIVE_MAX_SEQ` | 2048 | Native is reliable only up to this sequence length. Measured 2026-06-02: native vs fp32 ref ~3e-3 at S<=2048 but ~1.0 (garbage) at S=4096, while the decomposed matmul/softmax/matmul stays ~5e-3 (the wide accumulator holds). Above this, `sdpa` emits the decomposition. |
| `SDPA_NATIVE_MIN_BOTH` | 512 | Native returns garbage once **both** query and key seq axes are large: reliable only while `min(q_seq, k_seq) < 512`. Measured 2026-06-09: a hard 512x512 score-tile cliff (cos collapses to ~0.67), while KV-cache decode (min=1) stays correct to large k_seq. Outside, decompose (non-causal) or refuse (causal). |

### Causal masking

`is_causal=True` is native: the causal additive mask rides the SDPA layer's optional 5th bottom and the route optimizer keeps it on the native route (the decomposition is unmasked). Validated on M1: cos 1.0 vs `softmax(QK^T*scale + causal)*V`, single + multi-head. Requires `S <= SDPA_NATIVE_MAX_SEQ`. Causal outside the reliable native regime is refused - the decomposition would need a causal additive mask, but there's no host-constant add on the graph. (Chunk the query into tiles whose `min(q,k)` seq stays < 512.)

### attn_mask restrictions

`attn_mask` is a runtime additive bias riding the native layer's 5th bottom. The native layer applies one additive-mask plane shared across all heads, over the full query axis - shape must be `[1,1,Sq,Skv]`.

!!! warning "Rejected mask shapes"
    A per-head mask (`[1,H,Sq,Skv]`, H>1) is silently mis-applied (one plane used for all heads), and a query-broadcast mask (`[1,1,1,Skv]` with q_seq>1) underflows the bridge - both are rejected rather than returning garbage. (For KV-cache decode, q_seq==1, so `[1,1,1,Skv]` is a full-query plane and is accepted.)

### Decode shapes

K and V share shape (the cached sequence); Q's sequence may differ (KV-cache decode: q seq_q attends to cached k/v of length seq_kv). Q,K share H and D.

## Bridge-op arch gates

These ops cut the graph and run native sub-programs; their hardware variants are narrow.

| Op | Quirk | Why |
|---|---|---|
| `argmax` | 2D inputs `[C, W]` only, last axis (Width) or axis 0 (Channel); indices fp16-encoded (exact for index<2048) | Runs as native `GlobalArgMinMax` sub-program. |
| `topk` | `k` in {3,4} **arch-gated** (rejected); rest of `[1, W]` supported | ANECCompile fails for k in {3,4} on this hardware. Native `TopK` sub-program. |
| `sort` | hardware keys order on **one channel lane** and permutes all channels by it; `return_indices` fp16-encoded (exact for index<2048) | For a numpy-like per-row independent sort the bridge dispatches each row as its own 1-channel tile. Native `Sort` sub-program. |
| `space_to_channel` / `channel_to_space` | batch **N=1 only**; TF (block-major) channel ordering | Native SpaceToChannel/ChannelToSpace layers; same shape law as PixelUnshuffle/PixelShuffle but TF ordering. |
| `space_to_batch` | batch dim **grows** (`[N,C,H,W] -> [N*bh*bw, C, H/bh, W/bw]`) | Can only be a leaf/output of the segmented plan or feed another netplist cut (segment outputs thread as host arrays). Output batch slice `(n*bh+i)*bw+j == x[n, :, i::bh, j::bw]`. |
| `batch_to_space` | input batch must be divisible by `bh*bw` | **Arch-gated**: validator string "Input batch n is not divisible by factor x * factor y"; non-divisible batch fails compilation. |
| `flatten` | needs a 3D (`[C,H,W]`) graph tensor | The native Flatten bridge takes a `[C,H,W]` input. |
| `input_view` | view along Width, x flattened to 1-D length W | Native InputView layer. |
| `dynamic_slice` | only verified variant fixes **Width=4, SliceSize=2** | The static-start API is general in spirit; only this hardware variant is verified/accepted on this host (requires length-4 input, `size==2`). |
| `scaled_elementwise` | `Sub` rejected; `Mult` ignores `scale` | Found by fuzzing: `Sub` is rejected by ANECCompile, `Mult` ignores `scale` on-silicon. Both configs rejected. |
| `lrn` | `C <= 15` only | KernelChannel=C; `C >= 16` fails ANECCompile. Also: `alpha`/`eps` are fp16 bit patterns (see [bridges](bridges.md)). |
| `minmax_norm` | `Dimension="Channel"` arch-gated | Width/Height supported; Channel fails ANECCompile. `eps` is an fp16 bit pattern. |
| `fps` | L2-only metric; `k <= 1024`, `N <= 8192` | The native layer always uses Euclidean distance regardless of `DistanceMetric`. |
| `cross_product` | length-3 inputs only | A path Apple's MIL frontend rejects entirely. |

## Fused-MIL spatial ops (no cut)

For contrast, these spatial ops fuse into the one program with no graph cut: `space_to_depth`, `depth_to_space`, `crop`, `resize_nearest_neighbor`, `resize_bilinear`, `upsample_bilinear`, `affine`, `pixel_shuffle`, `pixel_unshuffle`. (`affine` is a 2-D affine warp via native MIL `affine`/`AffineTransform`: `transform` is `[N,6]` or `[1,6]` broadcast, `[a0,a1,a2, b0,b1,b2]` in normalized `[-1,1]` coords, bilinear sampling with zero padding.)
