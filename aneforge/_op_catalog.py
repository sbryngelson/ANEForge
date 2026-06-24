"""Complete ANE operation catalog - every MIL op the ANECompiler exposes, with its
category, per-device availability (M1..M5), MIL kernel class, and notes.

Auto-generated from the ANEForge reverse-engineering (the 187-op x device matrix +
the symbol-resolution op map), grounded in the HAL extraction and live M1 silicon probes.
Device columns map to ANE capability families (verified SoC->arch ladder):
    m1 = A13 (family 2)   m2 = A14 (family 3)   m3 = A15 (family 4)   m4_m5 = A16/A17 (family 5)
Each device cell is one of: 'native' (runs on-engine), 'bridge' (needs a netplist-bridge
or host/graph decomposition), 'walled' (no path - decompose on host).

This is the NATIVE MIL op vocabulary the ANE emits. aneforge's higher-level ops
(rms_norm, group_norm, channel_layer_norm, mha, sdpa, einsum, the fft/linalg/special
submodules) are COMPOSITES that lower to these - query their constituent ops here.

This dict is the single source of truth. The deeper per-op argument, shape, dtype,
and per-chip capability write-ups live in the project's companion reference guide.
"""
from __future__ import annotations

# chip alias -> catalog device key
_CHIP = {
    "m1": "m1", "a13": "m1", "h13": "m1",
    "m2": "m2", "a14": "m2", "h14": "m2",
    "m3": "m3", "a15": "m3", "h15": "m3",
    "m4": "m4_m5", "m5": "m4_m5", "a16": "m4_m5", "a17": "m4_m5", "h16": "m4_m5", "h17": "m4_m5",
}
_FAMILY = {"m1": 2, "m2": 3, "m3": 4, "m4_m5": 5}

# op -> {category, m1, m2, m3, m4_m5, mil_status, kernel, note}
OP_CATALOG: dict[str, dict] = {
    # --- Activations (incl. LUT) ---
    'ceil': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT (probed native M1)'},
    'clamped_relu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ClampedRelu', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'clip': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'user-facing `clamp`; LUT'},
    'elu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Elu', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT (effectively F2 -> A13+)'},
    'erf': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SimpleActivation', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT'},
    'exp': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'exp2': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT'},
    'floor': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT (probed native M1)'},
    'gelu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Gelu', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT (M1 probe: ~0.08 rel err vs exact - LUT approximation, still native)'},
    'leaky_relu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'LeakyRelu', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'log': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT (ln2 immediate)'},
    'prelu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'PRelu', 'mil_status': 'lut_activation_or_special_function', 'note': 'per-channel alpha (LUT); native at rank >=3 (M1-confirmed)'},
    'relu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SimpleActivation', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 SimpleActivation'},
    'relu6': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': '', 'mil_status': '', 'note': 'LUT'},
    'round': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT round-nearest (probed native M1)'},
    'scaled_tanh': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ScaledTanh', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'sigmoid': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SimpleActivation', 'mil_status': 'lut_activation_or_special_function', 'note': 'F0/LUT (incl. hard variant)'},
    'sigmoid_hard': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SigmoidHard', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'sign': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT (probed native M1)'},
    'silu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SimpleActivation', 'mil_status': 'lut_activation_or_special_function', 'note': 'a.k.a. swish; LUT'},
    'softmax': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Softmax', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT (log2e immediate)'},
    'softplus': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Softplus', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT (+ parametric)'},
    'softplus_parametric': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Softplus', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'softsign': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Softsign', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'tanh': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'threshold': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    'thresholded_relu': {'category': 'Activations (incl. LUT)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ThresholdedRelu', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT'},
    # --- Comparison / logical ---
    'equal': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare -> bool (probed native M1)'},
    'greater': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare'},
    'greater_equal': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare (probed native M1)'},
    'less': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare (probed native M1)'},
    'less_equal': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare (probed native M1)'},
    'logical_and': {'category': 'Comparison / logical', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose via `min`/`mul` on host'},
    'logical_not': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 (probed native M1)'},
    'logical_or': {'category': 'Comparison / logical', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose via `max` on host'},
    'logical_xor': {'category': 'Comparison / logical', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose via `!=` on host'},
    'not_equal': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 compare (probed native M1)'},
    'select': {'category': 'Comparison / logical', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Select', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'user-facing `where`; template_text backend'},
    # --- Control flow ---
    'call': {'category': 'Control flow', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'Call', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'function call; `mapped_no_current_hwx_case` (inlined)'},
    'cond': {'category': 'Control flow', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': '', 'mil_status': '', 'note': '`mapped_no_current_hwx_case` + `Unsupported` converter - no standalone ANE codegen; flatten on host'},
    'while_loop': {'category': 'Control flow', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': '', 'mil_status': '', 'note': '`mapped_no_current_hwx_case` + `Unsupported`/`WhileLoop` - unroll on host'},
    # --- Conv / MatMul / Pooling ---
    'avg_pool': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Pool', 'mil_status': 'pool_template_or_nepool', 'note': 'F0; window <=29 (M1) / 31 (A14+); 3D window A13+'},
    'conv': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Conv', 'mil_status': 'neconv_or_ne_lane', 'note': 'F0; M1 kernels <=29x29 (13x13 fp16), 3D depth native A13+; M5 <=32x32'},
    'conv_transpose': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Conv', 'mil_status': 'neconv_or_ne_lane', 'note': 'F0 deconv; strided axes use small-kernel caps'},
    'einsum': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Einsum', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'lowers to matmul/transpose chain'},
    'l2_pool': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Pool', 'mil_status': 'special_lut_or_pool', 'note': 'special LUT pool (1024-entry fp16)'},
    'linear': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Linear', 'mil_status': 'neconv_or_ne_lane', 'note': 'folds to conv when RHS <=2 MB SRAM working set'},
    'linear_activation': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'LinearActivation', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'fused linear+activation'},
    'matmul': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Matmul', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane / conv-fold; same tensor caps as conv'},
    'max_pool': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Pool', 'mil_status': 'pool_template_or_nepool', 'note': 'F0'},
    'ne_bypass': {'category': 'Conv / MatMul / Pooling', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'NEBypass', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'private NEBypass unit; `mapped_no_current_hwx_case`'},
    'ne_conv': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'NEConv', 'mil_status': 'neconv_or_ne_lane', 'note': 'private NEConv unit (fill=0x44/mir=0x5d)'},
    'ne_matmul': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'NEMatMul', 'mil_status': 'neconv_or_ne_lane', 'note': 'private NEMatMul unit'},
    'ne_pool': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'NEPool', 'mil_status': 'pool_template_or_nepool', 'note': 'private NEPool unit (probe-pending codegen, treated reachable)'},
    'pe_elementwise': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'PEElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'private PEElementWise unit (fill=0x49/mir=0x59)'},
    'pe_goc': {'category': 'Conv / MatMul / Pooling', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'PEGOC', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'private PEGOC unit; `mapped_no_current_hwx_case` (compiler-internal)'},
    'pe_pool': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'PEPool', 'mil_status': 'pool_template_or_nepool', 'note': 'private PEPool unit'},
    'scaled_dot_product_attention': {'category': 'Conv / MatMul / Pooling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SDPA', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2; rides matmul+softmax (NOT texture-gated) - native on M1. user-facing `sdpa`'},
    # --- Detection / sampling ---
    'argsort': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Sort', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**Sort family, A14+**; codegen-rejected on M1 (= `sort` floor)'},
    'list_gather': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': 'TensorList op - `Unsupported` everywhere'},
    'list_length': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'list_read': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'list_scatter': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'list_write': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'make_list': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'non_maximum_suppression': {'category': 'Detection / sampling', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'NonMaximumSuppression', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'template_text NMS backend'},
    'random_bernoulli': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - host RNG'},
    'random_categorical': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - host RNG'},
    'random_normal': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - host RNG'},
    'random_uniform': {'category': 'Detection / sampling', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'RandomUniform', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**RNG, A15+** (HAL 0x4a9=0 on M1/M2); aneforge uses host RNG below A15 (`dropout`/`random` decomposable)'},
    'topk': {'category': 'Detection / sampling', 'm1': 'walled', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'TopK', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**rank/sort bridge, A14+** (`_OP_FLOOR`); bridge validator callable on M1 but **codegen-rejected** (measured)'},
    # --- Elementwise arithmetic ---
    'abs': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'PEElementWise (F0)'},
    'add': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_with_text_immediates', 'note': 'const + tensor forms; text-immediate fused const'},
    'cumsum': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'CumSum', 'mil_status': 'curated', 'note': 'runs ON the ANE as a single op (verified M1 2026-06-09: cos 1.0) - NOT host-decomposed. The standard MIL `cumsum` op is unimplemented, so it is reached via the curated e5rt path (see _capabilities).'},
    'floor_div': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT-assisted (actlut:2)'},
    'inverse': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'reciprocal LUT'},
    'maximum': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'const + tensor (LUT)'},
    'minimum': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'const + tensor (LUT)'},
    'mod': {'category': 'Elementwise arithmetic', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose on host'},
    'mul': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_with_text_immediates', 'note': 'const + tensor forms'},
    'pow': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': '`pow_const`; user-facing `x ** y` (probed native M1)'},
    'real_div': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'general divide; A11/A12 = const-fp16 reciprocal only. user-facing `truediv`/`div`'},
    'rsqrt': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT'},
    'sqrt': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT activation (native A13+, decomposed on A11/A12)'},
    'square': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_template_text', 'note': 'F0 PEElementWise'},
    'sub': {'category': 'Elementwise arithmetic', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'pe_elementwise_with_text_immediates', 'note': 'lowered to add-of-negated-const'},
    # --- Image / resize / texture ---
    'affine': {'category': 'Image / resize / texture', 'm1': 'walled', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Affine', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**texture-engine only (A14+)**; "affine transform is not supported on this architecture" on M1'},
    'crop_resize': {'category': 'Image / resize / texture', 'm1': 'walled', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'CropResize', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**texture-engine only (A14+, HAL 0x81d)** - `_OP_FLOOR`; unavailable on M1, no host substitution wired'},
    'degamma': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'DeGamma', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'ISP/image op; `mapped_no_current_hwx_case`'},
    'gamma': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'Gamma', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'ISP/image op; `mapped_no_current_hwx_case`'},
    'pixel_buffer_to_tensor': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'PixelBufferToTensor', 'mil_status': 'mapped_no_current_hwx_case', 'note': '4CC image input; `mapped_no_current_hwx_case`. Does not lower on the unentitled direct path (entitlement gate, not chip gate); use `af.image_input`.'},
    'resample': {'category': 'Image / resize / texture', 'm1': 'walled', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Resample', 'mil_status': 'template_text_or_unresolved_backend', 'note': '**texture-engine only (A14+)**; warp depth=1, channelin{1,2}. Walled on M1'},
    'resize': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Resize', 'mil_status': 'neconv_or_ne_lane', 'note': 'F2 but **texture-gated**: M1 = software deconv/transpose fallback (different rounding; some modes hard-abort); native A14+'},
    'resize_bilinear': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ResizeBilinear', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane; sw-fallback on M1'},
    'resize_nearest_neighbor': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ResizeNearestNeighbor', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane; sw-fallback on M1 (1x1-source fast path exists)'},
    'tensor_to_pixel_buffer': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'TensorToPixelBuffer', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case` (compiler-internal)'},
    'upsample_bilinear': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'UpsampleBilinear', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane; sw-fallback on M1'},
    'upsample_nearest_neighbor': {'category': 'Image / resize / texture', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'UpsampleNearestNeighbor', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane; sw-fallback on M1'},
    # --- Normalization ---
    'batch_norm': {'category': 'Normalization', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'BatchNorm', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'inference fold-to-affine runs everywhere (incl. A11/A12); native stats form is A13+'},
    'instance_norm': {'category': 'Normalization', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'InstanceNorm', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F2'},
    'l2_norm': {'category': 'Normalization', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': '', 'mil_status': '', 'note': 'F2'},
    'layer_norm': {'category': 'Normalization', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'LayerNorm', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F2 (native A13+)'},
    'local_response_norm': {'category': 'Normalization', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'LRNorm', 'mil_status': 'lut_activation_or_special_function', 'note': 'LRN bridge (measured OK on M1)'},
    # --- Quantization / dtype ---
    'cast': {'category': 'Quantization / dtype', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Cast', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 format primitive. **fp16<->fp32/bool native on M1**; `cast(->int32)` is **walled on M1** (empirically confirmed) - keep dtype fp on h13'},
    'const': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case` - folded at compile, not a standalone codegen op'},
    'constexpr_affine_dequantize': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'weight-compression const; folded. **int4-LUT streams natively from M1; int8/affine fold to fp16 below A15** (HAL +0x520-0x539).'},
    'constexpr_blockwise_shift_scale': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'blockwise stream gate A15+; folds to fp16 on M1/M2'},
    'constexpr_cast': {'category': 'Quantization / dtype', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'constexpr_lut_to_dense': {'category': 'Quantization / dtype', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'palette/LUT stream gate (+0x529) is **A13-on** -> int4-LUT streams natively from M1 (*the one compressed format that wins on M1)'},
    'constexpr_lut_to_sparse': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'folded const; sparse stream A15+'},
    'constexpr_sparse_blockwise_shift_scale': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'sparse+blockwise stream A15+'},
    'constexpr_sparse_to_dense': {'category': 'Quantization / dtype', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ConstOps', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'sparse stream A15+'},
    'dequantize': {'category': 'Quantization / dtype', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Dequantize', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'quantize': {'category': 'Quantization / dtype', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Quantize', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 (not texture-gated)'},
    # --- Recurrent ---
    'gru': {'category': 'Recurrent', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - unroll to conv/matmul+activation on host'},
    'lstm': {'category': 'Recurrent', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - unroll on host'},
    'rnn': {'category': 'Recurrent', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - unroll on host'},
    # --- Reductions ---
    'reduce_argmax': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ReduceArg', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'per-axis ArgMax - F0, all chips (bridge `ArgMax` measured OK on M1)'},
    'reduce_argmin': {'category': 'Reductions', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ReduceArg', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'per-axis argmin; **M1/M2 walled on the MIL route** (HAL 0x4f2, A15+), bridge mirrors argmax. user-facing `argmin`'},
    'reduce_l1_norm': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2 Reduce'},
    'reduce_l2_norm': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2 Reduce'},
    'reduce_log_sum': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT-assisted Reduce (ln2 immediate)'},
    'reduce_log_sum_exp': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'lut_activation_or_special_function', 'note': 'LUT Reduce; aneforge wires its vjp (probed native M1)'},
    'reduce_max': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2'},
    'reduce_mean': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2'},
    'reduce_min': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2'},
    'reduce_prod': {'category': 'Reductions', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose (log-sum-exp / scan) on host'},
    'reduce_sum': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2 (native A13+; decomposed on A11/A12). reduced-axis >=192 -> transpose route (>=384 on A15+)'},
    'reduce_sum_square': {'category': 'Reductions', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reduce', 'mil_status': 'pe_elementwise_template_text', 'note': 'F2; the 0x494 `reduce->square` *fusion* is M2+ only - M1 emits an extra fp16 round (<=1-round numeric, not a wall)'},
    # --- Special / math ---
    'acos': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'acosh': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'asin': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - host decomposition'},
    'asinh': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'atan': {'category': 'Special / math', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': 'F2 LUT - **native on M1** (probe: WORKS; the one trig in vocab on h13)'},
    'atanh': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'cos': {'category': 'Special / math', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': '**F4 trig, native A15+ only** (REJECTED on M1/A14); M1/M2 Horner'},
    'cosh': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (REJECTED M1 probe)'},
    'cost_volume': {'category': 'Special / math', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'CostVolume', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'bridge `CostVolume` (measured OK on M1); `mapped_no_current_hwx_case`'},
    'cross_product': {'category': 'Special / math', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'CrossProduct', 'mil_status': 'mapped_no_current_hwx_case', 'note': 'bridge `CrossProduct` (measured OK on M1) but `mapped_no_current_hwx_case` in MIL map - reachable via bridge'},
    'matrix_decomposition': {'category': 'Special / math', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'MatrixDecomposition', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case` - no observed codegen'},
    'sin': {'category': 'Special / math', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ElementWise', 'mil_status': 'lut_activation_or_special_function', 'note': '**F4 trig, native A15+ only** (REJECTED on M1/A14 - silicon-measured); M1/M2 use `special.py` Horner'},
    'sinh': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (REJECTED M1 probe) - `(exp(x)-exp(-x))/2` on host'},
    'tan': {'category': 'Special / math', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (REJECTED M1 probe) - `sin/cos` Horner identity on host'},
    # --- Stateful (state / buffers) ---
    'circular_buffer_to_tensor': {'category': 'Stateful (state / buffers)', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'CircularBufferToTensor', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case`; ring-buffer reader'},
    'read_state': {'category': 'Stateful (state / buffers)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ReadState', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F2 stateful; reachable on M1 but needs the e5rt inout-tensor-desc plumbing for KV-cache.'},
    'tensor_buffer_to_tensor': {'category': 'Stateful (state / buffers)', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'TensorBufferToTensor', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case`; F2 ring/streaming buffer mover (A13+, reachable inside stateful graph)'},
    'tensor_to_circular_buffer': {'category': 'Stateful (state / buffers)', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'TensorToCircularBuffer', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case`; ring-buffer writer'},
    'tensor_to_tensor_buffer': {'category': 'Stateful (state / buffers)', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'TensorToTensorBuffer', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case`'},
    'write_state': {'category': 'Stateful (state / buffers)', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'WriteState', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F2 stateful'},
    # --- Structural / shape ---
    'band_part': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (mask via host)'},
    'batch_to_space': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'BatchToSpace', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'inverse of above'},
    'concat': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Concat', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 DMA'},
    'crop': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Crop', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 slice/crop (distinct from texture `crop_resize`)'},
    'depth_to_space': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'DepthToSpace', 'mil_status': 'neconv_or_ne_lane', 'note': 'F2 NE lane; user-facing `pixel_shuffle`'},
    'expand_dims': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ExpandDims', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'fill': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Fill', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'const tensor producer'},
    'fill_like': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'FillLike', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'const tensor producer'},
    'flatten2d': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': '', 'mil_status': '', 'note': 'F0'},
    'gather': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Gather', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'software gather on M1 in narrow envelope (batch=1,depth=1); **hw `gather_hw` path is A14+** (`_OP_FLOOR`)'},
    'gather_along_axis': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'GatherAlongAxis', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'template_text; same M1 envelope caveat'},
    'gather_nd': {'category': 'Structural / shape', 'm1': 'bridge', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'GatherND', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'M1 = sw-envelope only (`IsValidForH13`: batch=1,depth=1,idx-ch=3); outside it **rejected**. Native (texture) A14+'},
    'identity': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Cast', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'aliases Cast/no-op'},
    'non_zero': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (data-dependent shape)'},
    'one_hot': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose (eye-gather) on host'},
    'pad': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Pad', 'mil_status': 'neconv_or_ne_lane', 'note': 'const pad F0 (NE lane). **symmetric/reflect pad is texture-gated -> WARN/sw on M1**, native A14+'},
    'pixel_shuffle': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'PixelShuffle', 'mil_status': 'neconv_or_ne_lane', 'note': 'F2 NE lane'},
    'pixel_unshuffle': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'PixelUnshuffle', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'template_text'},
    'range_1d': {'category': 'Structural / shape', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': '', 'mil_status': '', 'note': 'template_text; **M1 raw-MIL probe: walled** (positional-encoding range rejects on h13 codegen) - host-precompute the const'},
    'reshape': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reshape', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 metadata (A11/A12 = fp16-only/Flatten-or-abort)'},
    'reshape_like': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'ReshapeLike', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'reverse': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Reverse', 'mil_status': 'neconv_or_ne_lane', 'note': 'NE lane (probed native M1; aneforge wires vjp)'},
    'reverse_sequence': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose on host'},
    'scatter': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere - decompose on host'},
    'scatter_along_axis': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'scatter_nd': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere'},
    'shape': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'Unsupported', 'mil_status': 'unsupported', 'note': '`Unsupported` everywhere (static-shape graphs only)'},
    'slice_by_index': {'category': 'Structural / shape', 'm1': 'bridge', 'm2': 'bridge', 'm3': 'bridge', 'm4_m5': 'bridge', 'kernel': 'SliceByIndex', 'mil_status': 'mapped_no_current_hwx_case', 'note': '`mapped_no_current_hwx_case`; static-offset slice folds into descriptor (reachable inside graph)'},
    'slice_by_size': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SliceBySize', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0; **pre-A16 width-offset quirk** (Q.4 x16 crop-DMA): CONCATenating multiple nonzero last-axis (width) slices returns WRONG ELEMENTS on A14 (the gather-axis-1 bug; a SINGLE width slice is exact on A14 - linalg column/element extraction is green on M2); on A13 a width slice also saturates |value|>4094 to +/-inf. A16/M5 exact. gather + conv-grad route off the last axis.'},
    'slice_update': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SliceUpdate', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'template_text backend'},
    'sliding_windows': {'category': 'Structural / shape', 'm1': 'walled', 'm2': 'walled', 'm3': 'walled', 'm4_m5': 'walled', 'kernel': 'NotImplemented', 'mil_status': 'unsupported', 'note': '`NotImplemented` on any backend - decompose on host'},
    'space_to_batch': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SpaceToBatch', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'factor in{2,3,4,8}; batch cap 4096 (older)/65536'},
    'space_to_depth': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'SpaceToDepth', 'mil_status': 'neconv_or_ne_lane', 'note': 'F2 NE lane; user-facing `pixel_unshuffle`'},
    'split': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Split', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'squeeze': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Squeeze', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'stack': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Stack', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0'},
    'tile': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Tile', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F2 (A13+); factors of {2,3,4,8}. Absent on A11/A12'},
    'transpose': {'category': 'Structural / shape', 'm1': 'native', 'm2': 'native', 'm3': 'native', 'm4_m5': 'native', 'kernel': 'Transpose', 'mil_status': 'template_text_or_unresolved_backend', 'note': 'F0 but capped by max-transpose-extent (16384 M1-A15 -> 65536 M5; 0 on A11/A12)'},
}

# headline: 187 MIL ops total

def op_info(name: str) -> dict | None:
  """Full catalog entry for a MIL op (or None if unknown)."""
  return OP_CATALOG.get(name)

def device_status(name: str, chip: str = 'm1') -> str | None:
  """'native' | 'bridge' | 'walled' for `name` on `chip` (m1/m2/m3/m4/m5 or a13..a17/h13..h17)."""
  d = OP_CATALOG.get(name)
  if d is None: return None
  return d[_CHIP.get(chip.lower(), 'm1')]

def is_native(name: str, chip: str = 'm1') -> bool:
  """True iff `name` runs natively on-engine on `chip`."""
  return device_status(name, chip) == 'native'

def ops_on(chip: str = 'm1', status: str = 'native') -> list[str]:
  """All ops with the given status (native/bridge/walled) on `chip`, sorted."""
  key = _CHIP.get(chip.lower(), 'm1')
  return sorted(n for n, d in OP_CATALOG.items() if d[key] == status)

def min_native_family(name: str) -> int | None:
  """Lowest ANE family (2=A13/M1 .. 5=A16/M5) where `name` is native; None if walled on all."""
  d = OP_CATALOG.get(name)
  if d is None: return None
  for key in ('m1', 'm2', 'm3', 'm4_m5'):
    if d[key] == 'native': return _FAMILY[key]
  return None

def walled_everywhere() -> list[str]:
  """Ops with no native/bridge path on any family (must be decomposed on host)."""
  return sorted(n for n, d in OP_CATALOG.items()
                if all(d[k] == 'walled' for k in ('m1', 'm2', 'm3', 'm4_m5')))

def categories() -> list[str]:
  return sorted({d['category'] for d in OP_CATALOG.values()})
