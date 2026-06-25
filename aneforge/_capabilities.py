"""The aneforge capability census - the single, closed, machine-checkable source of
truth for *what ANE capability aneforge surfaces*.

Why this module exists: the authoritative runtime op sets are `_EMIT` (the fused
e5rt-MIL emitters) and `NETPLIST_OPS` (the native Path-A bridge ops), both in
`_compile.py`. The discovery evidence behind them - which layers were proven on
silicon, which are genuine hardware walls, which are only predicted - is scattered
across the reverse-engineering corpus. Nothing reconciled the two: a new op could
appear unclassified, or a "surfaced" op could silently stop compiling unnoticed.

`build_registry()` closes the loop. It introspects the live `_EMIT` /
`NETPLIST_OPS` and merges them with curated, citation-carrying classification data
for everything that *can't* be introspected (cracked-but-not-promoted layers, the
arch-gated negatives, the predicted-but-unbuilt frontier). The result is a closed
classification of every ANE capability aneforge knows about. `tests/
test_routes.py` fails CI if the live runtime drifts from this registry either way.

Discipline (this is a publication artifact): every entry is one of
  - `fused`                - in `_EMIT`; lowers to e5rt-MIL, fuses into one program.
  - `bridge`               - in `NETPLIST_OPS`; runs as a native Path-A sub-program (a graph cut).
  - `cracked`              - reverse-engineered native layer PROVEN on silicon in
                               the reverse-engineering corpus, but NOT yet promoted to the frontend.
  - `reachable`            - the native MIL op compiles + runs correct on aneforge's own
                               e5rt path (proven by the full-vocabulary sweep,
                               `full_mil_vocabulary_sweep.json`), but is not yet
                               promoted to a frontend `_EMIT` emitter or `af.` method.
                               Reachable through the MIL door, not direct `Type=` netplist
                               authoring, so it is not a cracked native layer and does not
                               count toward the cracked-layer manifest.
  - `arch-gated-negative`  - known NOT to work, carrying the SPECIFIC wall (an authored
                               op/config that the ANE backend rejects or fails to lower).
  - `not-authorable`       - control-flow / list / state ops (cond, while_loop, make_list,
                               list_*, read_state, ...) with no single-op MIL form on the
                               single-procedure feed-forward surface; recorded, not defeated
                               (the recurrence/scan wall).
  - `predicted`            - predicted crackable by the callable-`_ANECValidate*`
                               heuristic, but NOT built and NOT verified.
`fused`/`bridge` carry `verified="silicon"`; `cracked`/`reachable` carry it too
(proven in the reverse-engineering corpus / the on-device sweep) - the distinction
from fused/bridge is *promotion*, not evidence. `predicted` is explicitly
`verified="no"`. Every `cracked` cites its bridge in the reverse-engineering corpus;
every negative cites the finding that established the wall; every
`reachable`/`not-authorable` cites the sweep matrix.

Scope: the registry is exhaustive over the full 166-op MIL vocabulary swept in
`full_mil_vocabulary_sweep.json` (116/166 reachable), not just the 50
`_ANECValidate*Layer` validators. The validator-surface accounting (the cracked-layer
manifest) is a strict subset, kept intact.
"""
from __future__ import annotations

import json
from typing import Any

from . import _compile


# --------------------------------------------------------------------------- #
# valid status vocabulary (closed set)                                        #
# --------------------------------------------------------------------------- #
STATUSES = ("fused", "bridge", "cracked", "reachable", "arch-gated-negative",
            "not-authorable", "predicted")


# --------------------------------------------------------------------------- #
# introspected: the frontend surface of each live runtime op                  #
# --------------------------------------------------------------------------- #
# runtime op name -> frontend spelling (Tensor method / af.* fn). Enriches
# introspected entries; missing op reports frontend=None (still classified).
_FRONTEND_SPELLING: dict[str, str] = {
    # fused unary activations
    "relu": "x.relu()", "silu": "x.silu()", "sigmoid": "x.sigmoid()", "tanh": "x.tanh()",
    "exp": "x.exp()", "sqrt": "x.sqrt()", "abs": "x.abs()", "sin": "x.sin()",
    "cos": "x.cos()", "erf": "x.erf()", "softplus": "x.softplus()", "relu6": "x.relu6()",
    "square": "x.square()", "gelu": "x.gelu()", "log": "x.log(eps)", "rsqrt": "x.rsqrt(eps)",
    "elu": "x.elu(alpha)", "leaky_relu": "x.leaky_relu(alpha)", "clip": "x.clip(lo,hi)",
    # fused binary / scalar
    "add": "a + b", "sub": "a - b", "mul": "a * b", "real_div": "a / b",
    "maximum": "af.maximum(a,b)", "minimum": "af.minimum(a,b)", "pow": "a ** b (pow)",
    "muls": "a * scalar",
    # fused linear algebra
    "matmul": "x @ W / x.linear(W,b)", "bmm": "a @ b (Tensor@Tensor)",
    "conv": "af.conv(...)", "conv_transpose": "af.conv_transpose(...)",
    # fused spatial / shape
    "max_pool": "x.max_pool(k)", "avg_pool": "x.avg_pool(k)", "upsample": "x.upsample(s)",
    "concat": "af.concat([...])", "reshape": "x.reshape(...)", "transpose": "x.transpose(perm)",
    "pixel_shuffle": "af.pixel_shuffle(x,r)", "pixel_unshuffle": "af.pixel_unshuffle(x,r)",
    # fused reductions / norms
    "reduce_mean": "x.mean(axes)", "reduce_sum": "x.sum(axes)", "reduce_max": "x.amax(axes)",
    "reduce_min": "x.amin(axes)", "softmax": "x.softmax(axis)", "l2_norm": "x.l2_norm(axis,eps)",
    "rms_norm": "x.rms_norm(g,eps)", "layer_norm": "x.layer_norm(g,b,eps)",
    "channel_layer_norm": "x.channel_layer_norm(g,b,eps)",
    "group_norm": "x.group_norm(g,b,groups,eps)", "batch_norm": "af.batch_norm(...)",
    # bridge ops
    "sdpa": "af.sdpa(q,k,v)", "argmax": "x.argmax(axis)", "topk": "af.topk(x,k)",
    "sort": "af.sort(x)", "cross_product": "af.cross_product(a,b)",
    "cross_correlation": "af.cross_correlation(x,t)", "cost_volume": "af.cost_volume(a,r,R)",
    "fps": "af.fps(points,k)", "radius_search": "af.radius_search(p,c,r)",
    "minmax_norm": "af.minmax_norm(x,dim,eps)", "lrn": "af.lrn(x,...)",
    "space_to_channel": "af.space_to_channel(x,r)", "channel_to_space": "af.channel_to_space(x,r)",
    "space_to_batch": "af.space_to_batch(x,bh,bw)", "batch_to_space": "af.batch_to_space(x,bh,bw)",
    "flatten": "af.flatten(x)", "input_view": "af.input_view(x,off,size)",
    "dynamic_slice": "af.dynamic_slice(x,start,size)", "scaled_elementwise": "af.scaled_elementwise(x,z,...)",
}

# The in-package bridge module each NETPLIST_OPS entry invokes (its evidence). Promoted
# out of the reverse-engineering corpus into the self-contained aneforge/_bridges/ package
# (the product no longer depends on the research corpus at runtime).
_BRIDGE_EVIDENCE: dict[str, str] = {
    "sdpa": "aneforge/_bridges/ane_sdpa_fused.py",
    "argmax": "aneforge/_bridges/ane_rank_fused.py", "topk": "aneforge/_bridges/ane_rank_fused.py",
    "sort": "aneforge/_bridges/ane_rank_fused.py",
    "cross_product": "aneforge/_bridges/ane_cross_product_fused.py",
    "cross_correlation": "aneforge/_bridges/ane_cross_correlation_fused.py",
    "cost_volume": "aneforge/_bridges/ane_cost_volume_fused.py",
    "fps": "aneforge/_bridges/ane_fps_fused.py",
    "radius_search": "aneforge/_bridges/ane_radius_search_fused.py",
    "minmax_norm": "aneforge/_bridges/minmax_norm_fused.py",
    "lrn": "aneforge/_bridges/lrn_fused.py",
    "space_to_channel": "aneforge/_bridges/ane_rearrange_fused.py",
    "channel_to_space": "aneforge/_bridges/ane_rearrange_fused.py",
    "space_to_batch": "aneforge/_bridges/ane_rearrange_fused.py",
    "batch_to_space": "aneforge/_bridges/ane_rearrange_fused.py",
    "flatten": "aneforge/_bridges/ane_structural_fused.py",
    "input_view": "aneforge/_bridges/ane_input_view_fused.py",
    "dynamic_slice": "aneforge/_bridges/ane_dynamic_slice_fused.py",
    "scaled_elementwise": "aneforge/_bridges/scaled_elementwise_fused.py",
}


def _introspect_fused() -> list[dict[str, Any]]:
  """Every op live in `_compile._EMIT` -> a `fused` registry entry."""
  return [{
    "name": name, "status": "fused",
    "route": "e5rt-MIL (fused into one program; no graph cut)",
    "frontend": _FRONTEND_SPELLING.get(name), "verified": "silicon",
    "evidence": "aneforge/_compile.py @op(...) registry; "
                "the reverse-engineering corpus (37/42 verified correct on e5rt)",
  } for name in sorted(_compile._EMIT)]


def _introspect_bridge() -> list[dict[str, Any]]:
  """Every op live in `_compile.NETPLIST_OPS` -> a `bridge` registry entry."""
  return [{
    "name": name, "status": "bridge",
    "route": "native Path-A netplist sub-program (graph cut; SegmentedModel)",
    "frontend": _FRONTEND_SPELLING.get(name), "verified": "silicon",
    "evidence": _BRIDGE_EVIDENCE.get(name, "aneforge/_compile.py NETPLIST_OPS"),
  } for name in sorted(_compile.NETPLIST_OPS)]


# --------------------------------------------------------------------------- #
# curated: classifications that can't be introspected from the runtime        #
# --------------------------------------------------------------------------- #
# (A) cracked - proven on silicon in the reverse-engineering corpus (a self-verifying
#     *_fused.py), but not promoted into the frontend _EMIT / NETPLIST_OPS. Citations are
#     the bridge files. (The promoted siblings - PixelShuffle, L2Norm, Sort/TopK/ArgMinMax,
#     the space/channel/batch rearranges, Flatten, etc. - appear above as fused/bridge.)
_CRACKED: list[dict[str, Any]] = [
    {
        "name": "Dropout",
        "route": "native netplist (Type=Dropout); identity at rate 0 (inference-only)",
        "evidence": "aneforge/_bridges/ane_structural_fused.py (bit-exact vs numpy)",
        "note": "Inference identity; not promoted (no training-time use on the inference engine).",
    },
    {
        "name": "Broadcast",
        "route": "native netplist (Type=Broadcast; BroadcastInfo replicates a length-1 axis)",
        "evidence": "aneforge/_bridges/ane_structural_fused.py (bit-exact vs numpy)",
        "note": "MIL add/sub/mul already broadcast in the fused path; standalone layer unneeded so far.",
    },
    {
        "name": "Resample",
        "route": "native netplist (Type=Resample, NearestNeighbor mode); the BILINEAR mode "
                 "is now promoted to fused (resize_bilinear/upsample_bilinear in _EMIT)",
        "evidence": "the reverse-engineering corpus; resample_dynamicgoc_numerics.py; "
                    "full_mil_vocabulary_sweep.json (resize_bilinear reachable+correct, relerr 2e-4)",
        "note": "RECONCILED 2026-06-02: the old 'Resample Bilinear is arch-gated' negative was an "
                "over-cautious wall - the direct-netplist Bilinear probe failed HWX codegen, but the "
                "MIL resize_bilinear/upsample_bilinear ops compile + run exact (relerr 2e-4) on the "
                "e5rt path and now SHIP as fused (af.resize_bilinear / af.upsample_bilinear). "
                "NearestNeighbor remains cracked-but-unpromoted (fused resize_nearest_neighbor covers it).",
    },
    {
        "name": "RandomGenerator",
        "route": "native netplist (Type=RandomGenerator; Normal/Bernoulli/Categorical/Gaussian)",
        "evidence": "capabilities.md (4 distributions compile via _ANEInMemoryModel)",
        "note": "On-device RNG compiles, but seeding/streaming semantics unsettled; host RNG used today.",
    },
    {
        "name": "Shape",
        "route": "native netplist (Type=Shape; emits shape as a 1-D output vector)",
        "evidence": "capabilities.md (3 variants compile via the netplist path)",
        "note": "Static shapes are known at compile time in aneforge, so no frontend need yet.",
    },
    {
        "name": "DynamicGOC",
        "route": "native netplist (Type=DynamicGOC; gain-offset-clamp out=idx*(data+upd))",
        "evidence": "the reverse-engineering corpus; capabilities.md",
        "note": "GOC compiles+loads; it is NOT a scatter (Mode/Axis are inert). Niche; not promoted.",
    },
    {
        "name": "Tile",
        "route": "native netplist (Type=Tile; Params is a dimension-NAME-keyed factor dict "
                 "{'Width':fx,'Height':fy,'Channel':fc} - NOT FactorX/Y/Z)",
        "evidence": "the reverse-engineering corpus (4/4 self-test PASS; bit-exact vs np.tile on M5 "
                    "across Width/Height/Channel/combined)",
        "note": "Cracked 2026-05-30 (was `predicted`): the dimension-keyed Params spelling was the "
                "missing piece. Clean NETPLIST_OPS-bridge promotion candidate (B=1 static).",
    },
    {
        "name": "slice_by_index",
        "route": "native netplist (Type=InputView; Dimension/Offset/Size) - bit-exact contiguous slice",
        "evidence": "the reverse-engineering corpus (runs bit-exact via InputView on M5)",
        "note": "Cracked 2026-05-30 (was `predicted`): SAME capability as the `input_view` bridge "
                "op - promotable as a frontend `slice`, not a new layer. Not double-counted as bridge.",
    },
]

# (B) arch-gated negatives - known not to work on this M5, each with its specific wall.
#     "Negative" means: rejected at compile, or the named configuration fails
#     ANECCompile/HWX codegen even though the validator accepts the schema. The
#     `probe` field is a tiny aneforge call the gate runs to confirm it still
#     fails (raises) - None means manual-probe-only (too slow/risky for CI).
_NEGATIVES: list[dict[str, Any]] = [
    # --- native MIL op unimplemented on the ANE backend ("Not implemented ... not
    #     supported on any backend"). Key distinction (2026-05-31 calibration): the
    #     native OP is unimplemented, but the CAPABILITY is reachable on the same e5rt
    #     path via a decomposition. These are native-op gaps, not hardware walls. ---
    {
        "name": "reduce_prod",
        "wall": "the native MIL reduce_prod op is unimplemented on the ANE backend "
                "('Not implemented ... not supported on any backend') - a NATIVE-OP gap, NOT a "
                "hardware-capability wall. The product-reduction capability is reachable on the "
                "same e5rt path via exp(reduce_sum(log(x))) (relerr ~1e-3; positive inputs, "
                "sign/zero need extra ops).",
        "evidence": "finding_op_conformance_matrix; the reverse-engineering corpus; "
                    "second-path calibration 2026-05-31 (exp-sum-log decomposition runs on ANE)",
        "probe": None,  # native op has no frontend spelling; capability via decomposition
    },
    {
        "name": "gather",
        "wall": "the native MIL gather op is unimplemented on the ANE backend - a native-op gap. "
                "The gather capability is reachable for trace-time-constant (STATIC) indices via "
                "a one-hot @ matmul (bit-exact). Only DYNAMIC large-vocab embedding lookup stays "
                "host-side (one-hot is O(vocab) and needs runtime-tensor indices); the earlier "
                "blanket 'embedding stays host-side' holds only for that dynamic case.",
        "evidence": "finding_op_conformance_matrix; reference_aneforge_frontend (host gather, "
                    "dynamic case); second-path calibration 2026-05-31 (one-hot matmul, static)",
        "probe": None,
    },
    {
        "name": "cumsum",
        "wall": "the native MIL cumsum op is unimplemented on the ANE backend - a native-op gap, "
                "NOT a 'no parallel-prefix silicon' hardware wall. The prefix-sum capability is "
                "reachable on the same e5rt path via an upper-triangular ones matmul (x @ U; "
                "relerr ~3e-4; O(N^2) in the matmul).",
        "evidence": "finding_op_conformance_matrix; the reverse-engineering corpus; "
                    "second-path calibration 2026-05-31 (triangular matmul runs on ANE)",
        "probe": None,
    },
    {
        "name": "scan",
        "wall": "no data-dependent in-graph loop: sequential recurrence / scan is "
                "inexpressible on the feed-forward engine.",
        "evidence": "finding_ane_numerical (LAPACK wall is ARCHITECTURAL); finding_ane_math (IIR)",
        "probe": None,
    },
    {
        "name": "iir_recurrence",
        "wall": "sequential IIR recurrence (each output depends on the previous) = the scan "
                "wall; only static FIR-unroll fits.",
        "evidence": "finding_ane_math (aneforge/dsp.py routes IIR-class work to FIR unroll)",
        "probe": None,
    },
    {
        "name": "scatter",
        "wall": "no scatter-write path: DynamicGOC is gain-offset-clamp only; "
                "scatter/scatter_along_axis cannot be lowered.",
        "evidence": "capabilities.md (DynamicGOC correction); finding_op_conformance_matrix",
        "probe": None,
    },
    # --- dtype walls (cleanly rejected at MIL parse / by the frontend) ---
    {
        "name": "fp32_compute",
        "wall": "fp32 compute rejected: 'not implemented on any backend'. The ANE computes "
                "in fp16 only - aneforge ACCEPTS fp32/fp64 float weights but silently casts "
                "them to fp16 (so there is no frontend raise to probe; the wall is a "
                "compute-precision fact).",
        "evidence": "reference_unentitled_menu; aneforge/graph.py _check_dtype",
        "probe": None,
    },
    {
        "name": "int32_compute",
        "wall": "int32 compute rejected at MIL parse ('not implemented').",
        "evidence": "reference_unentitled_menu; capabilities.md (Datatypes)",
        "probe": "int32_weight",
    },
    {
        "name": "bf16_compute",
        "wall": "bf16 has no path on the backend (cleanly rejected).",
        "evidence": "reference_unentitled_menu; capabilities.md (Datatypes)",
        "probe": None,  # numpy has no native bf16; no clean frontend probe
    },
    # --- quantization: no walls remain (reconciled 2026-06-02) -----------------
    # The former "blockwise_int8"/"int4_weight" negatives were a malformed probe: the
    # empty-paren spelling `constexpr_blockwise_shift_scale()[data=..]` segfaults the
    # e5rt compiler, but the named-arg form `constexpr_blockwise_shift_scale(data=..,
    # scale=..)` compiles + runs correct (relerr ~3e-4) across multi-block + int4
    # configs (same macOS 26 / M5). Ships as compress="blockwise". Caveat: the
    # blockwise output is not matmul-routable, so it dequantizes in-program to dense
    # fp16 (no weight-streaming win, unlike int8-affine / int4-LUT / sparse); exposed
    # for the smaller artifact + finer scales, not throughput. Registered via the
    # sweep-driven layer below.
    # int8-affine (constexpr_affine_dequantize), int4-LUT (constexpr_lut_to_dense) and
    # sparse (constexpr_sparse_to_dense) all run on the raw e5rt path and ship via
    # compress="int8"/"int4"/"sparse" (test_compress.py). The "sparse segfaults" claim
    # was likewise a malformed-MIL artifact. No quant wall remains.
    # --- layer/codegen walls (validator accepts schema, M5 HWX codegen fails) ---
    {
        "name": "conv_kernel_ge16",
        "wall": "conv (and the CrossCorrelation bridge that lowers to conv) is capped at "
                "kernel K<=15 on this M5: a 1xK kernel with K>=16 fails ANECCompile "
                "'Not implemented'. (aneforge/dsp.py routes wider kernels to FFT.)",
        "evidence": "finding_ane_math; the reverse-engineering corpus",
        "probe": None,  # an ANECCompile failure is not recoverable in-process; manual/subprocess only
    },
    {
        "name": "group_norm_large_featuremap",
        "wall": "RELAXED by rank-4 tiling: group_norm lowers to [1,G,C/groups,H*W] and reduces the "
                "trailing two axes, so the bound is max(C/groups, H*W) vs the ANE per-axis cap 65536, "
                "NOT the flattened (C/groups)*H*W. SD-UNet's 640ch@64 (81920) and 512ch@128 (262144) "
                "now compile + run in fp16. af.group_norm only raises when a single tiled axis > 65536.",
        "evidence": "finding_sd15 (capability boundary #2); aneforge/_compile.py _e_group_norm rank-4 tiling",
        "probe": "group_norm_large_featuremap",
    },
    {
        "name": "square_opcode_tiling",
        "wall": "the native `square` opcode fused with a following nonlinearity fails "
                "ANECCompile when (Width % 128) > 64. (aneforge emits mul(x,x) instead - "
                "the negative is the opcode, which is why aneforge never surfaces it.)",
        "evidence": "finding_ane_numerical; the reverse-engineering corpus; _compile._e_square",
        "probe": None,
    },
    {
        "name": "rank5_transpose_matmul",
        "wall": "transpose+matmul fails ANECCompile at rank>=5 (the rank-4 cap): a fully-split "
                "tensor carries at most 3 factor groups (forces matmul-FFT to 3-stage, not "
                "full log-depth radix-2).",
        "evidence": "finding_ane_math (matmul-FFT walls); aneforge/fft.py",
        "probe": None,
    },
    {
        "name": "batch_to_space_indivisible",
        "wall": "BatchToSpace with input batch N % (FactorX*FactorY) != 0 fails HWX codegen "
                "('Input batch n is not divisible by factor x * factor y'). "
                "(af.batch_to_space guards this at construction.)",
        "evidence": "finding_layer_crack_sweep; aneforge/graph.batch_to_space",
        "probe": "batch_to_space_indivisible",  # af.batch_to_space raises ValueError at construction
    },
    {
        "name": "topk_k3_k4",
        "wall": "TopK with K in {3,4} is a fixed forbidden band on this ANE (ANECCompile fails); "
                "K=1,2 and K>=5 work. (af.topk guards this at construction.)",
        "evidence": "finding_layer_crack_sweep; aneforge/graph.topk",
        "probe": "topk_k3",  # af.topk(x,3) raises ValueError at construction
    },
    {
        "name": "minmax_norm_channel_axis",
        "wall": "MinMaxNormalization over the Channel axis is arch-gated (HWX codegen fails); "
                "Width/Height work. (af.minmax_norm rejects Channel.)",
        "evidence": "finding_layer_crack_sweep; aneforge/graph.minmax_norm",
        "probe": "minmax_norm_channel",  # af.minmax_norm(dimension='Channel') raises ValueError
    },
    {
        "name": "lrn_channels_ge16",
        "wall": "LocalResponseNormalization with KernelChannel=C fails ANECCompile for C>=16 "
                "on this ANE; C<=15 works. (af.lrn guards this.)",
        "evidence": "finding_ane_numerical (bug 3); aneforge/graph.lrn",
        "probe": "lrn_c16",  # af.lrn on a 16-channel input raises ValueError at construction
    },
    {
        "name": "scaled_elementwise_sub",
        "wall": "ScaledElementWise op='Sub' is rejected by ANECCompile; op='Mult' silently "
                "ignores `scale`. (af.scaled_elementwise guards both.)",
        "evidence": "finding_ane_numerical (bug 1/2); aneforge/graph.scaled_elementwise",
        "probe": "scaled_elementwise_sub",  # raises ValueError at construction
    },
    {
        "name": "Unflatten",
        "wall": "Unflatten fails HWX codegen for all variants (composite lowering specific); "
                "bare Reshape/Transpose compile fine.",
        "evidence": "finding_layer_crack_sweep; capabilities.md",
        "probe": None,
    },
    {
        "name": "KernelRasterizer",
        "wall": "KernelRasterizer (im2col unfold) only compiles in the degenerate identity "
                "config (OutputChannels=1, output dims = input dims); real im2col "
                "(OutputChannels=K*K*C) always fails ANECCompile.",
        "evidence": "finding_layer_crack_sweep; capabilities.md",
        "probe": None,
    },
    {
        "name": "RingBufferWriter",
        "wall": "RingBufferWriter is a multi-procedure streaming-state primitive; fails from any "
                "single-procedure netplist regardless of Params.",
        "evidence": "finding_layer_crack_sweep; capabilities.md",
        "probe": None,
    },
    # note (reconciled 2026-06-02): the former "NMS" arch-gated negative was flipped to
    # reachable. The earlier wall was an over-cautious direct-NETPLIST probe (full Params
    # recovered, compile failed). The MIL `non_maximum_suppression` op compiles + runs on the
    # e5rt path (presence-only - not numerically validated against a reference, since NMS
    # output ordering is selection-dependent), so it is `reachable` (sweep-proven on silicon,
    # not promoted to a frontend primitive) via the sweep-driven layer below, not a negative.
    # See full_mil_vocabulary_sweep.json (non_maximum_suppression, reachable).
    {
        "name": "CropResize",
        "wall": "texture-family codegen gate: direct-netplist CropResize fails ANECCompile on M5 "
                "(plane-equation coefficient constraint).",
        "evidence": "finding_layer_crack_sweep; the reverse-engineering corpus",
        "probe": None,
    },
    # note (reconciled 2026-06-02): the former "AffineTransform" and "Resample_Bilinear"
    # arch-gated negatives were flipped to reachable. The earlier walls came from over-
    # cautious direct-NETPLIST (Type=AffineTransform / Type=Resample,Bilinear) probes that
    # failed HWX codegen - scoped to the netplist door. The MIL ops compile + run correct on
    # aneforge's own e5rt path (full_mil_vocabulary_sweep.json): `affine` (genuine warp,
    # relerr 2.3e-3 at align_corners=True, not a no-op) and `resize_bilinear`/
    # `upsample_bilinear` (exact bilinear, relerr 2e-4). All three now ship as fused _EMIT
    # ops (af.affine / af.resize_bilinear / af.upsample_bilinear), so they appear in the
    # census as `fused`, not negatives. Same macOS 26 / M5 (not an OS change). The remaining
    # texture-family negative is CropResize (still walled below).
    {
        "name": "MATDECOMP_MATMULT_0x19",
        "wall": "internal-optimizer-gated (NOT schema-gated): the fused matrix-decomp-matmul "
                "composite (opcode 0x19) is emitted ONLY by ZinIrOpt::BatchMatDecompMatMul on a "
                "batched topology the netplist cannot express; 7 variants all lower to 0x60.",
        "evidence": "finding_sdpa_and_frontier; the reverse-engineering corpus",
        "probe": None,
    },
    {
        "name": "PEFUSED_GOC_0x5C",
        "wall": "internal-optimizer-gated: nested-container grammar recovered (<parent>.dynamic_goc "
                "subunit) but no single-0x5C HWX emerged.",
        "evidence": "finding_sdpa_and_frontier; the reverse-engineering corpus",
        "probe": None,
    },
    {
        "name": "MatrixDecomposition_factorization",
        "wall": "the native MatrixDecomposition (NonQRGivens) factorization is "
                "not_currently_callable (carriers compile, the factorization doesn't) - "
                "the LAPACK wall is architectural, not numerical.",
        "evidence": "finding_ane_numerical; finding_sdpa_and_frontier",
        "probe": None,
    },
    {
        "name": "on_device_rng_seeded",
        "wall": "no settled on-device RNG: RandomGenerator compiles but seeding/streaming "
                "semantics are unsettled, so RNG is kept host-side.",
        "evidence": "reference_unentitled_menu (host-side RNG)",
        "probe": None,
    },
    {
        "name": "custom_hwx_load",
        "wall": "custom HWX loading hits the kernel-signature wall (unentitled).",
        "evidence": "reference_unentitled_menu",
        "probe": None,
    },
    {
        "name": "nan_to_num",
        "wall": "the ANE does not propagate NaN, so NaN-handling ops are broken (no fix).",
        "evidence": "capabilities.md (Operators with no hardware backing)",
        "probe": None,
    },
    {
        "name": "bitwise_logical",
        "wall": "bitwise/logical ops (and/or/xor/not) rejected - not in the fp16 surface; "
                "trig/hyperbolic inverses (acos/asin/sinh) are netplist dead-ends too.",
        "evidence": "capabilities.md (no hardware backing)",
        "probe": None,
    },
    # note (reconciled 2026-06-02): the former "instance_norm" arch-gated negative was
    # flipped to reachable. The earlier wall was scoped to the native InstanceNorm netplist
    # LAYER (which failed ANECCompile); the MIL `instance_norm` op compiles + runs correct
    # on the e5rt path (full_mil_vocabulary_sweep.json, reachable+correct) and now ships as
    # a fused _EMIT op (x.instance_norm(...)), so it is `fused`, not a negative. The
    # decomposition route (reduce_mean/sub/mul/rsqrt) remains available too.
    {
        "name": "Reverse",
        "wall": "the native netplist Reverse LAYER is schema-UNREACHABLE: no exported "
                "`_ANECValidateReverseLayer` and no opcode in the atlas - only an internal "
                "mlir::mps::ReverseOp (MIL-frontend, lowers away). By the callable-validator "
                "heuristic it is not netplist-crackable; the `predicted` classification was "
                "over-optimistic. NOTE (2026-06-02): this is the netplist-DOOR fact ONLY - the "
                "MIL `reverse` OP is reachable+correct on the e5rt path (registered `reachable` "
                "via the sweep), so the reversal CAPABILITY is available even though the native "
                "layer door is not.",
        "evidence": "the reverse-engineering corpus; ane_census_completeness.py (was `predicted`); "
                    "full_mil_vocabulary_sweep.json (MIL reverse reachable+correct)",
        "probe": None,
    },
    {
        "name": "RCAS",
        "wall": "internal-optimizer-gated (NOT schema-gated): opcodes 0x44 (RCAS) / 0x65 "
                "(NEFUSED_RCAS) have a ZinRCASLayer + internal ValidateSemantics_Impl but NO "
                "exported `_ANECValidateRCASLayer` and no .tbd entry (all RCAS validate symbols are "
                "local `t`, not `T`) - not netplist-reachable. Same class as MATDECOMP_MATMULT_0x19.",
        "evidence": "the reverse-engineering corpus (the one genuine opcode-surface gap; "
                    "found 2026-05-30 by the exhaustive validator/opcode census)",
        "probe": None,
    },
]

# (C) predicted - crackable by the callable-`_ANECValidate*` heuristic (a callable
#     exported validator marks a schema-gated layer reachable by direct netplist
#     authoring), but not built and not verified. Explicitly not-verified.
_PREDICTED: list[dict[str, Any]] = [
    {
        "name": "Pad",
        "route": "predicted netplist (ZinPadLayer + ZinPadValidator; netplist "
                 "kANECNetUnitPaddingInfo dict schema unrecovered)",
        "evidence": "the reverse-engineering corpus - the validator ACCEPTS the schema (status=1) "
                    "but ~13 binary-derived PaddingInfo spellings all fail ANECCompile; spatial-only "
                    "(batch/depth/channel padding documented unsupported), amount_before/after must "
                    "be constant. Next step: recover the exact dict from a CoreML-emitted Pad bundle. "
                    "NOTE (2026-06-02): this is the netplist-DOOR Pad LAYER only - the MIL `pad` OP "
                    "is reachable+correct on the e5rt path (registered `reachable` via the sweep), so "
                    "the padding CAPABILITY is available through the MIL door.",
    },
]


# --------------------------------------------------------------------------- #
# (D) full-MIL-vocabulary sweep - the data-driven exhaustiveness layer          #
# --------------------------------------------------------------------------- #
# Makes the registry exhaustive over the full 166-op MIL vocabulary swept on-device
# (full_mil_vocabulary_sweep.json). Every sweep op not already represented above
# (by exact name in _EMIT / NETPLIST_OPS / the curated lists) gets a synthesized
# entry, classified from its sweep `klass`:
#   reachable+correct / reachable -> `reachable` (runs on the e5rt MIL path; not a
#                                    promoted frontend primitive, so not a cracked
#                                    native layer and not in the cracked-manifest count).
#   not-implemented-on-ANE        -> `arch-gated-negative` (op unimplemented on backend).
#   compile-walled                -> `not-authorable` for control-flow/list/state (no
#                                    single-op MIL form), else `arch-gated-negative`.
# Curated entries win over the sweep (exact-name match skipped), preserving the
# hand-written walls/evidence/notes.
import os as _os
_SWEEP_PATH = _os.path.join(_os.path.dirname(__file__),
                            "full_mil_vocabulary_sweep.json")

# control-flow / list / state ops: walled because the single-procedure feed-forward
# netplist surface has no loop/list/state construct (the recurrence/scan wall), not a
# per-op codegen bug. Recorded as `not-authorable`, not arch-gated-negative.
_NOT_AUTHORABLE_OPS: frozenset[str] = frozenset({
    "cond", "while_loop", "classify",
    "make_list", "list_gather", "list_length", "list_read", "list_scatter", "list_write",
    "read_state", "gru", "lstm", "rnn",
})

# sweep ops that are the same capability as an op already surfaced under a different
# name (a `reachable` entry pointing at its canonical surfaced sibling -
# aliasing, no double-count of the capability as a distinct frontend primitive).
_SWEEP_CAPABILITY_ALIAS: dict[str, str] = {
    "scaled_dot_product_attention": "sdpa (bridge) - same fused-attention capability",
    "linear": "matmul / linear_activation (fused) - same affine-projection capability",
    "reduce_l2_norm": "l2_norm / reduce_l2_norm (fused) - same L2 capability",
    "resize": "resize_bilinear / resize_nearest_neighbor (fused) - generic resize entry",
    "upsample_nearest_neighbor": "upsample / resize_nearest_neighbor (fused) - same NN upsample",
    "tile": "Tile (cracked native layer) - same broadcast-replicate capability",
    "reverse": "Reverse (netplist layer is walled; the MIL reverse OP is the reachable route)",
    "pad": "Pad (netplist layer is predicted; the MIL pad OP is the reachable route)",
    "constexpr_lut_to_dense": "shipped via af.compile(compress='int4') - int4-LUT weight streaming",
    "constexpr_sparse_to_dense": "shipped via af.compile(compress='sparse') - sparse weight streaming",
}

# per-op notes for the reachable sweep entries that carry an important caveat.
_SWEEP_NOTE: dict[str, str] = {
    "constexpr_blockwise_shift_scale":
        "FLIPPED 2026-06-02 (was the 'last quant wall'): the old empty-paren spelling "
        "segfaulted (malformed probe); the named-arg form compiles + runs correct (relerr "
        "~3e-4) across multi-block + int4 configs. Shipped via af.compile(compress='blockwise'). "
        "CAVEAT: the output is NOT matmul-routable - it dequantizes IN-PROGRAM to a dense fp16 "
        "weight, so there is NO weight-streaming bandwidth win (unlike int8/int4-LUT/sparse); "
        "exposed for the smaller-on-disk artifact + finer-grained scales, not throughput.",
    "non_maximum_suppression":
        "FLIPPED 2026-06-02 (was the 'NMS' arch-gated negative, a direct-netplist probe that "
        "failed): the MIL non_maximum_suppression op compiles + runs on the e5rt path. "
        "Presence-only - NOT numerically validated against a reference (selection-ordering "
        "output); not promoted to a frontend primitive.",
}

# the three sweep klasses that mean "reachable on the e5rt MIL path".
_SWEEP_REACHABLE_KLASSES = ("reachable+correct", "reachable")


def _load_sweep() -> list[dict[str, Any]]:
  try:
    with open(_SWEEP_PATH) as f:
      return json.load(f)
  except (OSError, ValueError):
    return []  # sweep artifact absent -> registry still well-formed (just not extended)


def _introspect_sweep(already: set[str]) -> list[dict[str, Any]]:
  """Synthesize registry entries for every full-MIL-vocabulary-sweep op not already
    represented (`already` = names from _EMIT / NETPLIST_OPS / curated cracked /
    negatives / predicted). Classified from the sweep `klass`. Curated entries win - any
    op in `already` is skipped so its hand-written wall/evidence/note is preserved."""
  out: list[dict[str, Any]] = []
  for e in _load_sweep():
    op = e["op"]
    if op in already: continue
    klass = e["klass"]
    relerr = e.get("relerr")
    if klass in _SWEEP_REACHABLE_KLASSES:
      entry: dict[str, Any] = {
        "name": op, "status": "reachable",
        "route": "e5rt-MIL op (compiles + runs on aneforge's own e5rt path; "
                 "not promoted to a frontend _EMIT emitter / af. method)",
        "frontend": None, "verified": "silicon",
        "evidence": f"full_mil_vocabulary_sweep.json (klass={klass}"
                    + (f", relerr={relerr}" if klass == "reachable+correct" and relerr is not None else "")
                    + "); the reverse-engineering corpus",
      }
      if op in _SWEEP_CAPABILITY_ALIAS: entry["alias_of"] = _SWEEP_CAPABILITY_ALIAS[op]
      if op in _SWEEP_NOTE: entry["note"] = _SWEEP_NOTE[op]
      if klass == "reachable" and op not in _SWEEP_NOTE:
        entry["note"] = ((entry.get("note") or "") +
          " Sweep klass 'reachable' = compiles + runs (presence), correctness not "
          "numerically pinned against a reference.").strip()
      out.append(entry)
    elif op in _NOT_AUTHORABLE_OPS:
      out.append({
        "name": op, "status": "not-authorable",
        "route": None, "frontend": None, "verified": "negative-confirmed",
        "wall": "control-flow / list / state op with no single-op form on the "
                "single-procedure feed-forward MIL surface aneforge targets (the "
                "recurrence/scan wall); recorded, not defeated.",
        "evidence": f"full_mil_vocabulary_sweep.json (klass={klass}); "
                    "the reverse-engineering corpus; finding_ane_numerical (scan wall)",
      })
    elif klass == "not-implemented-on-ANE":
      out.append({
        "name": op, "status": "arch-gated-negative",
        "route": None, "frontend": None, "verified": "negative-confirmed",
        "wall": "the native MIL op is unimplemented on the ANE backend "
                "(ane_e5rt_program_compile fails, mask=0x4) - a native-op gap. Some "
                "of these capabilities are reachable via a decomposition on the same "
                "e5rt path (see the curated reduce_prod/gather/cumsum entries); this "
                "synthesized entry records only that the NATIVE op does not lower.",
        "evidence": f"full_mil_vocabulary_sweep.json (klass={klass}); "
                    "the reverse-engineering corpus",
      })
    else:  # compile-walled, non-control-flow -> codegen wall
      out.append({
        "name": op, "status": "arch-gated-negative",
        "route": None, "frontend": None, "verified": "negative-confirmed",
        "wall": "compiles-walled: the authored minimal MIL program fails ANECCompile "
                "(ane_e5rt_program_compile, mask=0x4) on M5 - a codegen/backend wall.",
        "evidence": f"full_mil_vocabulary_sweep.json (klass={klass}); "
                    "the reverse-engineering corpus",
      })
  return out


# --------------------------------------------------------------------------- #
# EQUIVALENCE-route registry - the closed table of alternative lowerings        #
# --------------------------------------------------------------------------- #
# Every surfaced capability is either route-selectable (the autotuner picks its
# cheapest equivalent lowering by measured cost) or single-route (one lowering, no
# alternative). This table closes that over the bridge ops: each NETPLIST_OPS bridge op
# records either an `alt` (a fused decomposition removing the graph cut, with loss-class
# + on-device evidence) or `single_route=True` with the reason it has none. Only on-
# device-validated alts appear (the builder lives in _rewrite._BRIDGE_DECOMPOSERS, keyed
# identically; reconciled by tests/test_routes.py); a candidate that failed validation is
# marked single-route with its failure evidence (e.g. rearranges hit the rank-5 wall).
#
# Loss class "lossless" = mathematically identical (bit-identical or within fp16
# op-noise), chosen purely on measured speed. Fused ops are single-route by
# construction and not listed; _routes_for() reports any op absent here as
# single-route("only lowering"). The reduce_sum->@ones rewrite is on the precision
# axis (tune_precision), documented there, not as a bridge alt.
_BRIDGE_ROUTES: dict[str, dict[str, Any]] = {
    # --- route-selectable: a validated fused decomposition removes the cut ---
    "sdpa": {
        "alt": "decomposed ((q@k^T)*scale).softmax(-1)@v - fused MIL, no cut",
        "loss": "lossless",
        "evidence": "tests/fuzz_metamorphic.py mha_vs_sdpa (bit-identical); "
                    "the reverse-engineering corpus; _rewrite._decompose_sdpa",
    },
    "minmax_norm": {
        "alt": "fused (x-amin)/(amax-amin+eps) - reduce_min/max + sub + adds + real_div, no cut",
        "loss": "lossless",
        "evidence": "tests/test_routes.py (Width/Height + degenerate row, relerr<=1.5e-3 "
                    "== fp16 op-noise); _rewrite._decompose_minmax_norm",
    },
    "flatten": {
        "alt": "reshape to the same 1-D shape - fused, bit-identical, no cut",
        "loss": "lossless",
        "evidence": "tests/test_routes.py (relerr 0.0); _rewrite._flatten_to_reshape",
    },
    "lrn": {
        "alt": "fused local_response_norm(size=C, alpha=alpha*C, beta, k) - one program, no cut",
        "loss": "lossless",
        "evidence": "tests/test_routes.py (cos 1.000000 / relerr 0.0 across C/alpha/beta/k); "
                    "_rewrite._decompose_lrn (bit-equivalent: bridge window N=C + internal "
                    "/KernelChannel == fused size=C with alpha pre-scaled by C; ~1250x cut "
                    "removal on conv->lrn->relu; fused has no C<16 cap)",
    },
    # --- single-route: candidate validated and rejected (confirmed negative) ---
    "space_to_channel": {
        "single_route": True,
        "reason": "the reshape+transpose decomposition needs a rank-6 intermediate "
                  "([N,C,H/r,r,W/r,r]); ANECCompile rejects reshape rank>5 "
                  "('Rank of the shape parameter must be between 0 and 5'). Confirmed "
                  "on-device - the rank-5 wall (see arch-gated rank5_transpose_matmul).",
    },
    "channel_to_space": {
        "single_route": True,
        "reason": "inverse of space_to_channel; the same rank-6 reshape+transpose "
                  "decomposition hits the rank-5 ANECCompile wall.",
    },
    "space_to_batch": {
        "single_route": True,
        "reason": "block-to-batch rearrange decomposes only through a rank-5+ "
                  "reshape/transpose chain (rank-5 wall); no fused equivalent compiles.",
    },
    "batch_to_space": {
        "single_route": True,
        "reason": "inverse of space_to_batch; same rank-5+ reshape/transpose wall.",
    },
    # --- single-route: structurally no fused equivalent (the op IS the capability) ---
    "argmax": {"single_route": True,
               "reason": "index-of-max has no fused reduction lowering (gather/argmax "
                         "are arch-gated); the native GlobalArgMinMax bridge is the only route."},
    "topk": {"single_route": True,
             "reason": "selection/ranking has no fused lowering (no in-graph sort); "
                       "the native sort/topk bridge is the only route."},
    "sort": {"single_route": True,
             "reason": "no fused sort primitive on the feed-forward engine; native bridge only."},
    "cross_product": {"single_route": True,
                      "reason": "3-vector cross product is a fixed native layer; the fused "
                                "permute+mul+sub form is a niche micro-op not worth a cut-free route."},
    "cross_correlation": {"single_route": True,
                          "reason": "lowers to conv (kernel<=15 arch cap); the native "
                                    "CrossCorrelation bridge is the surfaced route."},
    "cost_volume": {"single_route": True,
                    "reason": "stereo cost-volume is a fixed native layer; no fused equivalent surfaced."},
    "fps": {"single_route": True,
            "reason": "farthest-point sampling is iterative/data-dependent (greedy argmax loop) "
                      "- inexpressible as a fused feed-forward graph; native bridge only."},
    "radius_search": {"single_route": True,
                      "reason": "neighbor membership is data-dependent gather; native bridge only."},
    "input_view": {"single_route": True,
                   "reason": "contiguous slice along Width; no fused slice op surfaced "
                             "(slice_by_index is predicted-not-built). Native bridge only."},
    "dynamic_slice": {"single_route": True,
                      "reason": "data-positioned slice; no fused dynamic-slice op. Native bridge only."},
    "scaled_elementwise": {"single_route": True,
                           "reason": "fused-scale elementwise op; the equivalent muls+binary form "
                                     "is itself fused but the native layer is the surfaced single route "
                                     "(op='Sub' arch-gated - see scaled_elementwise_sub negative)."},
}


def _routes_for(name: str, status: str) -> dict[str, Any]:
  """Route classification for one op: `{"route_class": "selectable"|"single", ...}`.
    A bridge op in `_BRIDGE_ROUTES` with an `alt` is `selectable`; everything else
    (single-route bridge, or any fused op - one lowering by construction) is `single`
    with a reason."""
  info = _BRIDGE_ROUTES.get(name)
  if info and "alt" in info:
    return {"route_class": "selectable", "alt_route": info["alt"],
            "alt_loss": info["loss"], "route_evidence": info["evidence"]}
  if info and info.get("single_route"):
    return {"route_class": "single", "route_reason": info["reason"]}
  if status == "fused":
    return {"route_class": "single",
            "route_reason": "fused _EMIT op - one lowering by construction (no graph "
                            "cut to remove; no alternative route)."}
  if status == "bridge":
    # a bridge op not in the table is an oversight - surface it via the gate.
    return {"route_class": "single",
            "route_reason": "UNCLASSIFIED bridge op (add to _capabilities._BRIDGE_ROUTES)."}
  # cracked / negative / predicted: not optimizer-route-bearing.
  return {"route_class": "n/a"}


def route_registry() -> dict[str, Any]:
  """The closed equivalence-route registry over the BRIDGE ops: which are route-
    selectable (with the validated alternative + loss class + evidence) and which are
    single-route (with the reason). Driven off the live `NETPLIST_OPS` so it can never
    silently omit a bridge op. Reconciled against `_rewrite._BRIDGE_DECOMPOSERS` and the
    census by tests/test_routes.py."""
  selectable, single = [], []
  for name in sorted(_compile.NETPLIST_OPS):
    r = _routes_for(name, "bridge")
    if r["route_class"] == "selectable": selectable.append({"name": name, **r})
    else: single.append({"name": name, **r})
  return {
    "schema": "aneforge-route-registry/1",
    "selectable": selectable,
    "single_route": single,
    "summary": {"selectable": len(selectable), "single_route": len(single),
                "total_bridge": len(_compile.NETPLIST_OPS)},
  }


# --------------------------------------------------------------------------- #
# cracked-LAYER MANIFEST - the traceable artifact behind the "26 cracked"      #
# --------------------------------------------------------------------------- #
# `slice_by_index` (status=cracked) is the same hardware capability as the
# `input_view` bridge op (Type=InputView) - the registry note already records it as
# "not double-counted". The manifest folds it into `input_view` so the 26 are
# distinct native layers, not 27 with an alias.
_CRACKED_DEDUPE_ALIAS: dict[str, str] = {"slice_by_index": "input_view"}

_CRACKED_MANIFEST_DEFINITION = (
    "A cracked native hardware layer is a distinct native ANE layer reached by the "
    "netplist-authoring method (authoring `Type=<Layer>` directly in the netplist / "
    "Path-A) which Apple's user-space MIL frontend does not emit. In the registry these "
    "are exactly the entries with status in {cracked, bridge}: the `bridge` ops are "
    "Path-A netplist sub-programs authored directly, and the `cracked` ops are "
    "unpromoted direct-netplist cracks. Exactly one alias is deduped - `slice_by_index` "
    "(cracked) is the same hardware capability as `input_view` (bridge), folded in and "
    "not double-counted. Distinct cracked layers = (cracked) + (bridge) - (1 alias)."
)

_CRACKED_MANIFEST_HISTORICAL_NOTE = (
    "Historical-promotion nuance: some `fused` ops (e.g. PixelShuffle, L2Norm) were "
    "originally reached by the same netplist-authoring method before being promoted to aneforge's "
    "MIL @op emitter (the _EMIT path). They are intentionally EXCLUDED here: their "
    "current route is the MIL emitter, not direct Type= authoring, so under the "
    "membership rule above they are `fused`, not cracks. This manifest counts the "
    "current direct-netplist route only - it does not retroactively claim promoted "
    "ops as cracks."
)


def cracked_manifest() -> list[dict[str, Any]]:
  """The closed, machine-checkable manifest of the distinct cracked native hardware
    layers behind the "26 cracked native hardware layers" count.

    Membership rule (exact): a cracked native hardware layer is a registry entry whose
    `status` is in `{"cracked", "bridge"}` - `bridge` ops are Path-A netplist sub-
    programs authored directly, `cracked` ops are unpromoted direct-netplist cracks; both
    are reached by the netplist-authoring method (author `Type=<Layer>` in the netplist)
    and are not emitted by Apple's user-space MIL frontend.

    Dedupe: `slice_by_index` (status=cracked) is the SAME hardware capability as the
    `input_view` bridge op (both `Type=InputView`); it is folded into `input_view`
    and not double-counted. So distinct cracked layers = (#cracked) + (#bridge) - 1.

    `fused` ops are not cracks under this rule - they go through aneforge's MIL @op
    emitter, not direct `Type=` authoring - even ones (PixelShuffle, L2Norm) historically
    reached by the netplist-authoring method before promotion (see
    `_CRACKED_MANIFEST_HISTORICAL_NOTE`).

    Returns a list of `{name, status, route, evidence}` items, sorted by name. The
    aliased member carries its fold-in via an `aliases` key.
    """
  reg = build_registry()
  by_name = {e["name"]: e for e in reg["entries"]}

  members: list[dict[str, Any]] = []
  for e in reg["entries"]:
    if e["status"] not in ("cracked", "bridge"): continue
    if e["name"] in _CRACKED_DEDUPE_ALIAS: continue  # folded into its canonical entry below
    item = {
      "name": e["name"],
      "status": e["status"],
      "route": e["route"],
      "evidence": e["evidence"],
    }
    # attach any cracked alias that folds into this entry
    aliases = [a for a, canon in _CRACKED_DEDUPE_ALIAS.items() if canon == e["name"]]
    if aliases:
      item["aliases"] = sorted(aliases)
      item["alias_evidence"] = {
        a: by_name[a]["evidence"] for a in aliases if a in by_name
      }
    members.append(item)

  members.sort(key=lambda m: m["name"])
  return members


def cracked_count() -> int:
  """The number of distinct cracked native hardware layers (see `cracked_manifest`)."""
  return len(cracked_manifest())


def cracked_manifest_doc() -> dict[str, Any]:
  """The full committed manifest artifact: definition string, historical-promotion
    note, member list, and count. Serialised by `write_cracked_manifest` to
    `cracked_manifest.json`."""
  members = cracked_manifest()
  return {
    "schema": "aneforge-cracked-manifest/1",
    "definition": _CRACKED_MANIFEST_DEFINITION,
    "historical_promotion_note": _CRACKED_MANIFEST_HISTORICAL_NOTE,
    "dedupe": {
      "alias": _CRACKED_DEDUPE_ALIAS,
      "explanation": "slice_by_index (cracked) == input_view (bridge): same "
                     "Type=InputView hardware capability; counted once.",
    },
    "ci_assertion": "tests/test_routes.py::test_route_tables_reconcile",
    "count": len(members),
    "members": members,
  }


def write_cracked_manifest(path: str = "cracked_manifest.json") -> str:
  """Serialise the cracked-layer manifest to `path` (default `cracked_manifest.json`,
    matching where `capabilities.json` lives). Returns the path written. Reproducible -
    re-run whenever the registry changes."""
  doc = cracked_manifest_doc()
  with open(path, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=False)
    f.write("\n")
  return path


def build_registry() -> dict[str, Any]:
  """Build the closed capability registry: introspect the live `_EMIT` /
    `NETPLIST_OPS` and merge with the curated cracked/negative/predicted entries.

    Returns a dict with `entries` (list, sorted by (status-order, name)) and a
    `summary` of per-status counts. This is the source of truth `write_json`
    serialises and `tests/test_routes.py` reconciles against.
    """
  entries: list[dict[str, Any]] = []
  entries += _introspect_fused()
  entries += _introspect_bridge()

  entries += [{"name": e["name"], "status": "cracked", "route": e["route"], "frontend": None,
               "verified": "silicon", "evidence": e["evidence"], "note": e.get("note")}
              for e in _CRACKED]
  entries += [{"name": e["name"], "status": "arch-gated-negative", "route": None, "frontend": None,
               "verified": "negative-confirmed", "wall": e["wall"],
               "evidence": e["evidence"], "probe": e.get("probe")}
              for e in _NEGATIVES]
  entries += [{"name": e["name"], "status": "predicted", "route": e["route"], "frontend": None,
               "verified": "no", "evidence": e["evidence"]}
              for e in _PREDICTED]

  # (D) extend exhaustively over the full 166-op MIL vocabulary sweep: synthesize an
  # entry for every swept op not already represented (curated entries win - same name
  # is skipped so the hand-written walls/notes survive). Makes the registry exhaustive
  # over the MIL vocabulary, not just the 50 validators / surfaced ops.
  already = {e["name"] for e in entries}
  entries += _introspect_sweep(already)

  # detect any accidental duplicate names within a status (curation hygiene)
  seen: set[tuple[str, str]] = set()
  for e in entries:
    key = (e["name"], e["status"])
    if key in seen: raise ValueError(f"capability registry: duplicate entry {key}")
    seen.add(key)

  # enrich fused/bridge entries with their equivalence-route classification (the
  # source of truth for the optimizer's route choice - selectable vs single).
  for e in entries:
    if e["status"] in ("fused", "bridge"): e.update(_routes_for(e["name"], e["status"]))

  order = {s: i for i, s in enumerate(STATUSES)}
  entries.sort(key=lambda e: (order[e["status"]], e["name"]))

  summary = {s: sum(1 for e in entries if e["status"] == s) for s in STATUSES}
  summary["total"] = len(entries)
  # distinct cracked native hardware layers (cracked + bridge, minus the one alias);
  # the traceable count behind the "26 cracked" figure (see cracked_manifest()).
  summary["cracked_count"] = (
    summary["cracked"] + summary["bridge"] - len(_CRACKED_DEDUPE_ALIAS)
  )
  return {
    "schema": "aneforge-capability-census/1",
    "statuses": list(STATUSES),
    "summary": summary,
    "entries": entries,
  }


def write_json(path: str = "capabilities.json") -> str:
  """Serialise the registry to `path` (default `capabilities.json` at cwd). Returns
    the path written. Reproducible - re-run any time the runtime changes."""
  reg = build_registry()
  with open(path, "w") as f:
    json.dump(reg, f, indent=2, sort_keys=False)
    f.write("\n")
  return path


if __name__ == "__main__":
  import sys
  out = sys.argv[1] if len(sys.argv) > 1 else "capabilities.json"
  reg = build_registry()
  write_json(out)
  s = reg["summary"]
  print(f"wrote {out}: " + ", ".join(f"{k}={s[k]}" for k in (*STATUSES, "total")))
  # the cracked-layer manifest (the traceable artifact behind the "26")
  man = write_cracked_manifest()
  print(f"wrote {man}: cracked_count={cracked_count()}")
