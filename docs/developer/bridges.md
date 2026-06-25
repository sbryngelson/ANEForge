# Native bridges and segmentation

Some ANE hardware layers exist on the silicon but Apple's user-space MIL/CoreML compiler never emits them. ANEForge reaches them by hand-authoring native ANECIR netplists and dispatching them directly (Path A) — each such op cuts the graph and runs as a separate native sub-program.

## The segmentation model

Most ops are **fused e5rt-MIL**: they lower to MIL and fuse into one program with no graph cut (see [Architecture overview](overview.md)). A second family are **netplist-bridge** ops — native Path-A hardware layers the MIL frontend never produces. Each bridge op **cuts** the graph:

- surrounding regions compile and run as ordinary `e5rt` programs;
- the bridge node runs as a separate native sub-program (sub-millisecond via the **A2 persistent worker**);
- `compile` returns a `SegmentedModel` that threads segment outputs as host arrays between regions.

Each region is compiled from its own input shapes, so an op that changes the batch (e.g. `space_to_batch`) is fine as a segment boundary even though it would confuse a fused region.

### Path-A dispatch core

`_netplist.py` is the shared netplist author. It generates small ANEF `net.plist` programs **without CoreML or MLCompute**. The generated directory contains `net.plist`, `weights.0`, and `weights.1` (for convolution programs), loadable through `_ANEInMemoryModelDescriptor modelWithNetworkDescription:weights:optionsPlist:`.

- **`bin_dir()`** — a per-machine compiled-invoker binary cache (sibling of the e5rt cache at `~/.cache/aneforge/e5rt`), kept outside the package tree so bridges work when aneforge is installed into a read-only site-packages.
- **`ensure_invoker(name)`** — compiles `aneforge/_invokers/<name>.mm` -> `bin_dir()/<name>` on demand, cached per machine (rebuilt when the source is newer). The invokers link only system frameworks, so this works on any Apple Silicon Mac. Several bridges share one invoker (e.g. `sdpa_invoker` is the generic netplist invoker for the geometry/structural/rearrange ops), so the build is centralized here rather than duplicated per bridge.
- **`invoke_netplist(...)`** — the shared dispatch core: builds the invoker command (`--net-plist`, per-weight `--weights`, per-input/`--output name=path`, `--repeats`, optional `--warmup`), runs it, and raises on a non-zero return code or a non-`ok` status. `inputs`/`outputs` are `(name, path)` pairs; the caller writes input files and reads output files (layouts differ per bridge). Returns the parsed last-line status dict (e.g. timing info).

!!! note "Productized from the RE corpus"
    Each `*_fused` module is a self-contained dispatch closure promoted out of the reverse-engineering corpus. It authors a `Type=<Layer>` netplist and runs it Path-A.

## Native fused-attention (SDPA)

`sdpa_fused` runs SDPA end-to-end through a hand-authored ANECIR netplist with `Type=SDPA`, reaching the native fused-attention hardware layer instead of the HWX-level decomposition Apple's compiler emits. It's a drop-in fused-attention path for `(B, heads, seq, d_head)`-layout Q/K/V at fp16.

### The constant-Scale gate

The 4-input fused-attention layer accepts Q, K, V, and Scale tensor descriptors. The validator (`_ANECValidateSDPALayer`) requires `ANECTensorDesc.byte[0x39] bit 0` set on the Scale tensor (the "Scale is expected to be constant" gate). The netplist spelling of that bit is undocumented; Apple's MIL->ANECIR translator sets it implicitly via the `Constants` array.

The builder supports candidate spellings via `constant_flag_spelling`: `Constants_array` (list Scale in the `Constants` array, backed by weights.0), `IsConstant_unit`/`is_constant_unit`, `ConstantTensor_unit`, `ScaleMutable_false`, and `all` (probe mode). **`Constants_array` is the default and the only spelling observed to compile + load** on this host (others raise `ANECCompile() FAILED`). The Scale value (default `1/sqrt(dim)`) folds into the constant weights blob, and Scale sits as the 4th input (index 3) in the Bottom array, matching Apple's 4-tensor layout.

### SubtractMax (the softmax numerics gate)

`subtract_max` controls `Params.SubtractMax` — the **only** `Params` key the SDPA netplist parser recognizes. It sets `desc[0x00]` (a CFBoolean) controlling whether softmax subtracts the max before `exp`. `ANECSDPALayerDescInitialize` defaults it to `kCFBooleanFalse` (no max-stabilization), which was the source of the original "Y depends on Q,K,V,scale but doesn't equal `softmax(Q@K^T*s)@V`" numerics gap. Apple's MIL->ANECIR translator emits `SubtractMax=True`; numerically correct softmax requires `True`.

### Seq-in-C transpose

The ANE SDPA layer treats the **C tensor dim as the sequence axis** (the validator's "K and V must have same sequence length i.e. C dim"). PyTorch/MIL convention puts heads in C and seq in H, so the bridge pre-transposes to swap them (netplist sees seq-in-C, heads-in-H, the ANE-native layout) and post-transposes Y back to `[B, heads, seq, d_head]`. Internally `channels` is the ANE C dim (sequence after transpose) and `sequence` is the ANE H dim (heads after transpose). ANE output is `(B, S, H, D)` (native C=seq, H=heads, W=d_head).

### Mask and decode

The netplist is edited for two cases:

- **KV-cache DECODE shape** — Q's sequence may differ from K/V's. The validator requires only "Q,K same embedding (W)" and "K,V same seq (C)", so query+output carry `Sq` channels while K/V carry `Skv` (seq_q query tokens attend to seq_kv cached K/V).
- **Optional additive MASK** — bottom shape `[C=Sq, H=1, W=Skv]`, per the validator's "Mask Width axis must match K and V Channel axis or broadcastable".

Validated on M1: decode (`Sq<Skv`) cos ~1.0; causal mask cos 1.0 vs `softmax(QKt*scale+mask)V`.

See [Per-op ANE quirks](op-quirks.md) for the `SDPA_NATIVE_MAX_SEQ`/`SDPA_NATIVE_MIN_BOTH` reliability bounds and the per-head/query-broadcast mask restrictions.

## Rank-family layers

`ane_rank_fused.py` provides four hardware-native fp16 rank layers via netplists: `Sort`, `TopK`, `ArgMinMax`, and `GlobalArgMinMax`. Input `x` is fp16 `[channels, width]` (height=1) for Sort/TopK/GlobalArgMinMax, or `[channels, height, width]` for ArgMinMax.

| Layer | `Params` schema |
|---|---|
| **Sort** | `Direction` (Ascending\|Descending), `SortDimension`, `VectorDimension`, `SortIndices` (key lane), `Indices` (bool: output argsort instead of values) |
| **TopK** | `Type` (Max\|Min), `K` (int), `SortDimension`, `VectorDimension`, `SortIndices`, `Indices` (bool) |
| **ArgMinMax** | `Mode` (SpatialArgMax\|ChannelArgMax\|SpatialArgMin\|ChannelArgMin), `KernelWidth`, `KernelHeight`, `Pad*` |
| **GlobalArgMinMax** | `Type` (Max\|Min), `Dimension` |

- **Sort** sorts the Width axis, permuting *all* channels by the ordering of channel `key_lane` (`SortDimension=Width, VectorDimension=Channel`).
- **TopK** is top-k along Width keyed by channel `key_lane`. **`k` in {3,4} is arch-gated** (ANECCompile fails).
- **ArgMinMax**: `Spatial*` -> one flattened-(H*W) index per channel; `Channel*` -> one channel index per (h,w) position.
- **GlobalArgMinMax**: argmax/argmin index along `Dimension` (Width\|Height\|Channel).

!!! note "fp16-encoded integer indices"
    Integer index outputs come back fp16-encoded in the ordinary Float16 output tensor. For the small index ranges these layers produce (< 2048), fp16 represents every integer exactly, so the indices are exact.

## Spatial-rearrange layers

`ane_rearrange_fused.py` provides six fp16 rearrange layers:

| Layer | Shape law |
|---|---|
| PixelShuffle | `[N, C*fx*fy, H, W] -> [N, C, H*fy, W*fx]` (depth->space) |
| PixelUnshuffle | `[N, C, H*fy, W*fx] -> [N, C*fx*fy, H, W]` (space->depth) |
| ChannelToSpace | depth->space, same shapes as PixelShuffle |
| SpaceToChannel | space->depth, same shapes as PixelUnshuffle |
| SpaceToBatch | `[N, C, H, W] -> [N*fx*fy, C, H/fy, W/fx]` |
| BatchToSpace | `[N*fx*fy, C, H, W] -> [N, C, H*fy, W*fx]` (inverse of S2B) |

PixelShuffle/PixelUnshuffle use the **PyTorch (channel-major)** convention; ChannelToSpace/SpaceToChannel use the **TensorFlow** `space_to_depth`/`depth_to_space` (block-major) convention — they coincide only when `C==1`. For `space_to_batch`, output batch slice `bh_i*bw + bw_i` == `x[..., bh_i::bh, bw_i::bw]`; `batch_to_space` requires the input batch divisible by `bh*bw` (validator constraint). The native rearrange layers support **batch N=1 only**, and `batch_to_space` is arch-gated on the divisibility constraint (see [op-quirks](op-quirks.md)).

## Structural layers

`ane_structural_fused.py` provides three structural kinds (`flatten`/`dropout` are public, `Broadcast` via the model generator):

- **`Flatten`** — NCHW identity reshape to a 1-D vector (`prod(shape)`). The bridge takes a `[C,H,W]` input, so it needs a 3D graph tensor.
- **`Dropout`** — inference-time identity (rate must be 0).
- **`Broadcast`** — replicate a length-1 axis to `Size` along `Dimension`.
- **`input_view`** — contiguous view `x[offset:offset+size]` along Width (x flattened to 1-D length W; returns `[size]`).

## Geometry and point-cloud layers

These are paths Apple's MIL frontend rejects entirely.

- **`cross_product`** — 3-vector cross product (`cross(x, z)`), both inputs length-3, returns `(3,)`; matches `numpy.cross`.
- **`cross_correlation`** — valid (no-flip) cross-correlation of single-channel map `x` `[H,W]` with `template` `[Th,Tw]`: `y[i,j] = sum_{u,v} x[i+u, j+v] * template[u,v]` over `(H-Th+1) x (W-Tw+1)`. True correlation (template not flipped).
- **`cost_volume`** — L1 stereo/optical-flow matching cost: `aux` length-Wa, `ref` length-Wr with `Wr >= Wa + R`, returns `(R+1, Wa)` where `cost[d,x] = |aux[x] - ref[x+d]|`. (write_model maps width->ref_width, template_width->aux_width.)
- **`fps`** — FurthestPointSampling: greedily pick `CentroidCount` maximally-far-apart points (seeded at index 0). `points` `(N,3)` -> `(k,3)`. **L2-only on this arch** regardless of `DistanceMetric`; arch limits `k<=1024`, `N<=8192`.
- **`radius_search`** — L2 ball-query membership: `points` `(N,3)`, `centroids` `(Nc,3)` -> `(N,Nc)` uint8 membership matrix (`[i,j]==1` iff `points[i]` within L2 `radius` of `centroids[j]`). Output is `[Np rows x Nc cols]`, 2 bytes per W-cell, low byte = membership flag.
- **`dynamic_slice`** — runtime-parametric slice `x[start:start+size]` along an axis, `start` bound through a netplist constant (overwrites the index constant in weights.1). The accepted variant fixes `SliceSize=2` and `W=4` (see [op-quirks](op-quirks.md)).

Accepted unit dictionaries (examples):

```text
{"Type": "FurthestPointSampling", "Bottom": ["points"],
 "InputType": ["Float16"], "OutputChannels": 3, "OutputType": "Float16",
 "Params": {"CentroidCount": k, "DistanceMetric": "L1" | "L2"}}

{"Type": "RadiusSearch", "Bottom": ["centroids", "points"],
 "InputType": ["Float16", "Float16"], "OutputChannels": 1, "OutputType": "Float16",
 "Params": {"Radius": r}}
```

## Normalization layers

These native layers carry fp16-bit-pattern parameter quirks — a float passed where a bit pattern is expected int-truncates to ~0 and yields identity output.

### LocalResponseNormalization (`lrn`)

Classic cross-channel (AlexNet) LRN, Channel mode: `y[c] = x[c] / (K + alpha_eff * sum_{j in window} x[j]^2) ^ Beta`. Non-obvious conventions:

1. **`Alpha` is an fp16 bit pattern** (`ZinParseFP16Token`), not a float.
2. The ANE divides the parsed alpha by `KernelChannel` internally, so `alpha_eff = fp16(Alpha_bits) / KernelChannel`. For a desired alpha, pass `Alpha = fp16_bits(desired_alpha * KernelChannel)`; the bridge does this pre-multiply so callers pass the true effective alpha.
3. Only the first `KernelChannel` output channels are normalized; the rest are identity-copied. Use `KernelChannel = C` for full coverage (empirically `C==KernelChannel` works, e.g. `C=5/KC=5`).

The window is a **local** channel window of size `N = C`, asymmetric-centered on `c` and clipped at boundaries: `window(c) = [max(0, c-(N-1)//2) : min(C, c + N//2 + 1)]`. Only the center channel sees all C channels; edge channels see a partial sum — this is **not** a full-channel sum. Arch-gated to `C <= 15` (`C >= 16` fails ANECCompile).

!!! note "Two LRN paths"
    `af.local_response_norm` is the **fused MIL** op (no cut) — `x / (k + alpha/size * sum_{window} x**2) ** beta` over `size` neighbouring channels. `af.lrn` is this **netplist bridge** (graph cut) with the corrected window semantics above.

### MinMaxNormalization (`minmax_norm`)

`y = (x - min) / (max - min + eps)` over `Params.Dimension`. `x` is `[1, C, H, W]`. `Dimension="Width"` (per-row) and `"Height"` (per-column) are supported; **`"Channel"` is arch-gated** (ANECCompile fails). `Params.Epsilon` is an **fp16 bit pattern**, not a float — pass `fp16_bits(desired_eps)`.

### ScaledElementWise (`scaled_elementwise`)

`y = scale * (x OP z)` — fuses a binary elementwise op with a scalar scale. `Params.Type` selects the op (Add/Mult/Min/Max); `Params.Scale` is an **fp16 bit pattern** (use `fp16_bits`). Two arch quirks are guarded: **`Sub` is rejected** by ANECCompile, and **`Mult` ignores `scale`** on-silicon — both are rejected rather than emitting a wrong or uncompilable program.

## einsum (native single-equation layer)

The native MIL `einsum` op (distinct from the general `af.einsum` decomposer) is a single hardware layer. The only on-ANE-verified equation is `'nchw,nwhu->nchu'` — a batched matmul over the W/U dims sharing N,H: `a`=`[N,C,H,W]`, `b`=`[N,W,H,U]` (streamed weight) -> `[N,C,H,U]`. `b` is a weight array (streamed), not a graph Tensor.
