# MIL primer

MIL (Model Intermediate Language) is Apple's textual IR for ML programs. This
primer covers writing it by hand for ANEForge: what the parser accepts, what
each operator looks like, and what to do when something is rejected.

## Anatomy of a MIL program

The smallest legal MIL program ANEForge's compiler accepts:

```mil
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"},
                                    {"coremlc-version", "3520.5.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 4]> x) {
        tensor<fp16, [1, 4]> relu = relu(x = x)[name = string("relu")];
    } -> (relu);
}
```

Field by field:

- `program(1.3)` - MIL format version. `1.3` is what `coremltools`
  emits today and what the compiler accepts.
- `buildInfo = dict<...>(...)` - metadata header. The values must match
  the ANECompiler version on your host (run `xcrun --show-sdk-version`
  or grep system frameworks for the current `coremlc-component-MIL`
  value; `3520.4.1` is right on M5 Pro / macOS 26.5).
- `func main<ios18>(...)` - the function's opset declaration. `ios18`
  is the current target; older opsets (`ios17`, `ios16`) may work for
  legacy programs but reject newer ops.
- The body - one or more statements producing intermediate tensors.
  Each statement is `tensor<dtype, [shape]> NAME = OP(kwargs)[name = string("NAME")];`.
- `-> (output_name)` - the function's return tuple.

## Operator syntax

Every MIL operator follows the same shape:

```
tensor<RESULT_TYPE, [RESULT_SHAPE]> RESULT_NAME =
    OP_NAME(arg1 = value1, arg2 = value2, ...)[name = string("RESULT_NAME")];
```

The `[name = string("...")]` annotation is required for any op whose
output the program references later. By convention `RESULT_NAME`
matches the string inside `name`.

## Common operators

### Unary (single input, single output)

```mil
tensor<fp16, [1, 4]> y = relu(x = x)[name = string("y")];
tensor<fp16, [1, 4]> y = gelu(x = x, mode = string("EXACT"))[name = string("y")];
tensor<fp16, [1, 4]> y = sigmoid(x = x)[name = string("y")];
tensor<fp16, [1, 4]> y = sqrt(x = x)[name = string("y")];
tensor<fp16, [1, 4]> y = exp(x = x)[name = string("y")];
tensor<fp16, [1, 4]> y = log(x = x, epsilon = const(fp16(0)))[name = string("y")];
tensor<fp16, [1, 4]> y = abs(x = x)[name = string("y")];
```

`gelu` accepts `mode = "EXACT" | "TANH_APPROXIMATION" | "SIGMOID_APPROXIMATION"`.
`log` requires an `epsilon` argument (typically `fp16(0)` for ANE).

### Binary (two inputs)

```mil
tensor<fp16, [1, 4]> y = add(x = a, y = b)[name = string("y")];
tensor<fp16, [1, 4]> y = mul(x = a, y = b)[name = string("y")];
tensor<fp16, [1, 4]> y = sub(x = a, y = b)[name = string("y")];
tensor<fp16, [1, 4]> y = real_div(x = a, y = b)[name = string("y")];
```

Note: argument names are `x` and `y` (the input tensors), NOT positional.

### Comparison + select (for masks)

```mil
tensor<bool, [1, 4]> mask = less(x = a, y = b)[name = string("mask")];
tensor<fp16, [1, 4]> y = select(cond = mask, a = a, b = b)[name = string("y")];
```

Booleans are computed but cannot be input/output tensors. Use them only
as intermediates feeding into `select`.

### Matmul

```mil
tensor<fp16, [M, N]> y = matmul(
    x = a,                       // [M, K]
    y = b,                       // [K, N]
    transpose_x = const(bool(false)),
    transpose_y = const(bool(false))
)[name = string("y")];
```

### Conv

```mil
tensor<fp16, [1, Co, Ho, Wo]> y = conv(
    x = input,                   // [1, Ci, Hi, Wi]
    weight = w,                  // [Co, Ci/groups, Kh, Kw]
    strides = const(tensor<int32, [2]>([1, 1])),
    dilations = const(tensor<int32, [2]>([1, 1])),
    pad_type = string("custom"),
    pad = const(tensor<int32, [4]>([1, 1, 1, 1])),
    groups = const(int32(1))
)[name = string("y")];
```

`pad_type` is `"custom"`, `"same"`, `"valid"`. With `"custom"`, the `pad`
array is `[top, bottom, left, right]`.

### Reshape / transpose / pad

```mil
tensor<fp16, [1, 16, 4]> y = reshape(
    x = x,
    shape = const(tensor<int32, [3]>([1, 16, 4]))
)[name = string("y")];

tensor<fp16, [B, C, W, H]> y = transpose(
    x = x,                       // [B, C, H, W]
    perm = const(tensor<int32, [4]>([0, 1, 3, 2]))
)[name = string("y")];
```

### Softmax

```mil
tensor<fp16, [1, 4]> y = softmax(x = x, axis = const(int32(-1)))[name = string("y")];
```

### Reductions

```mil
tensor<fp16, [1, 1]> y = reduce_sum(
    x = x,                       // [1, 16]
    axes = const(tensor<int32, [1]>([-1])),
    keep_dims = const(bool(true))
)[name = string("y")];
```

### Scaled dot-product attention

```mil
tensor<fp16, [B, H, Sq, Dh]> y = scaled_dot_product_attention(
    query = q,                   // [B, H, Sq, Dh]
    key = k,                     // [B, H, Sk, Dh]
    value = v                    // [B, H, Sk, Dh]
)[name = string("y")];
```

No mask argument - for masked attention, decompose into matmul + softmax +
matmul with an explicit mask multiply (`af.sdpa` builds this fused route).
ANE routes the fused op when heads>=64 and seq>=~496 at d=64.

## Constants and inline weights

Tiny constants:

```mil
tensor<fp16, []> scale = const(val = fp16(0.125))[name = string("scale")];
tensor<int32, [3]> shape = const(val = tensor<int32, [3]>([1, 16, 4]))[name = string("shape")];
```

Inlined fp16 weights:

```mil
tensor<fp16, [2, 3]> w = const(val = tensor<fp16, [2, 3]>(
    [[0x3c00, 0x4000, 0x4200],
     [0x4400, 0x4500, 0x4600]]
))[name = string("w")];
```

Each `0xXXXX` is a fp16 value as a 16-bit hex literal. To convert a numpy
fp16 array:

```python
hex_lit = arr.view('uint16').tolist()
mil_text = format_as_nested_brackets(hex_lit)  # 0x%04x per element
```

## Weights via BLOBFILE (legacy)

For large weights, the legacy subprocess path emits a sibling
`weights1.bin` and references it:

```mil
tensor<fp16, [256, 256, 3, 3]> w = const(
    val = tensor<fp16, [256, 256, 3, 3]>(BLOBFILE(@path="weights1.bin", offset=0))
)[name = string("w")];
```

The compiler reads the binary file from the filesystem. This does not
work via e5rt - e5rt compiles in-process and has no sibling weight file.
For e5rt, inline weights as hex literals or use a smaller model.

## Multi-output programs

```mil
program(1.3) [buildInfo = ...] {
    func main<ios18>(tensor<fp16, [1, 4]> x) {
        tensor<fp16, [1, 4]> y_relu = relu(x = x)[name = string("y_relu")];
        tensor<fp16, [1, 4]> y_neg = sub(x = const(val = fp16(0)), y = x)[name = string("y_neg")];
    } -> (y_relu, y_neg);
}
```

The return tuple lists each output by name. e5rt's `Program` accepts
multiple outputs in its `outputs={name: shape, ...}` argument.

## Multi-input programs

Each input is listed in `func main<...>(...)`'s argument list:

```mil
func main<ios18>(
    tensor<fp16, [1, 4]> x,
    tensor<fp16, [1, 4]> y
) {
    tensor<fp16, [1, 4]> z = add(x = x, y = y)[name = string("z")];
} -> (z);
```

## State (for streaming inference)

```mil
func main<ios18>(
    tensor<fp16, [1, 1, 8]> x,
    tensor<fp16, [1, 2, 8, 4]> k_cache_in,
    tensor<fp16, [1, 2, 8, 4]> v_cache_in,
    tensor<fp16, [1, 1, 8]> pos_one_hot
) {
    // Use k_cache_in/v_cache_in to compute attention against new input x;
    // emit k_cache_out/v_cache_out updated with the new K/V slice.
    ...
} -> (y, k_cache_out, v_cache_out);
```

The runtime aliases IOSurfaces between consecutive calls so each call's
output state becomes the next call's input state.

The MIL `state<tensor<...>>` form (with `read_state`/`write_state`) is
parser-accepted, but ANEForge's streaming prototypes use the
plain paired-tensor approach: simpler and equivalent.

## Validation gotchas

These reject at MIL parse:

- fp32 / int32 / bf16 as input or output type. Only fp16 / int8 /
  uint8 / int16 / uint16 work as I/O. fp32 is acceptable as an
  intermediate via `cast(x, dtype="fp32")`.
- Any axis dim < 1 or > 65536. Hard bound on every axis (N, C, H, W, D).
- Bool as I/O. Bool tensors are compute-only.
- Missing `epsilon` on `log`. Required argument; pass `const(fp16(0))`.
- Missing `mode` on `gelu`. Required; pass `const(string("EXACT"))`.
- Empty `[name = ...]` annotation. Required for every named output.

These pass the parser but reject later, in `ANECCompile` (a codegen-optimizer
edge, not a capability or parse limit):

- `mul(reduce_output, 0.0)` - multiplying a `reduce_sum`/`reduce_mean` output
  by fp16 zero fails `ANECCompile`. It is the specific combination:
  `reduce * nonzero` compiles, and `mul`-by-zero with no preceding reduce compiles;
  only the pair trips it (a `1e-30` multiplier that underflows to fp16 zero also
  fails). Build a zero/constant from a `sub`-based zero instead - `(t - t) + c`,
  not `(t * 0.0) + c`. ANEForge's autograd does this in its `_const_like`
  helper.

When you build a graph with `import aneforge as af` you rarely write MIL by
hand, and the frontend catches many of these before the compile. Shape and
rank limits are enforced at graph-build time - constructing an out-of-bounds
Tensor raises immediately (e.g. a rank-6 tensor, or a conv with kernel width
> 15) rather than failing deep in `ANECCompile`. To check whether a built
graph compiles for a given ANE family without running it, use
`af.cross_compile_check(out, target='hXX')`, and query per-op support with
`af.op_info(name)` / `af.is_native(name)`.

See [capabilities.md](capabilities.md#dimension-bounds).

## Building MIL programmatically

Build the program text with small emitter functions, one per op family. The
pattern:

```python
def emit_relu_mil(n: int) -> str:
    return (
        f'program(1.3)\n'
        f'[buildInfo = dict<string, string>({{"coremlc-component-MIL", '
        f'"3520.4.1"}}, {{"coremlc-version", "3520.5.1"}})]\n'
        f'{{\n'
        f'    func main<ios18>(tensor<fp16, [1, {n}]> x) {{\n'
        f'        tensor<fp16, [1, {n}]> relu = relu(x = x)[name = string("relu")];\n'
        f'    }} -> (relu);\n'
        f'}}\n'
    )
```

For complex programs with many ops, build the body line by line and
concat. Each named tensor must be unique within the function.

## Ops MIL rejects (use ANECIR netplist instead)

The MIL parser rejects these hardware-native ops, but ANECIR netplists
accept them:

```
Rsqrt, Inv, Sqrt, Log2, Exp2, Sign, Erf, Swish, Sqr
```

See [capabilities.md](capabilities.md#operators-with-no-hardware-backing).
