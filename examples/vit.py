"""ViT-B/16 forward as one fused ANE program from real torchvision weights, validated vs fp32, plus an af.tune SDPA route rewrite on one attention layer. Run: python3 examples/vit.py"""
import sys, time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af

PATCH = 16
IMG = 224
DIM = 768
HEADS = 12
N_LAYERS = 12
NUM_PATCHES = (IMG // PATCH) ** 2          # 196
SEQ = NUM_PATCHES + 1                       # 197 (+ CLS)


def load_vit_weights():
    """torchvision ViT-B/16 pretrained state_dict as fp32 numpy + the torch model."""
    import torchvision
    m = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").eval()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    return m, sd


# ViT graph. Inputs (creation order): x [1,3,224,224], cls [1,768], pos [197,768].
def build_vit(sd, n_layers):
    """aneforge graph for ViT-B/16 forward -> logits [1,1000]."""
    g = lambda k: sd[k]
    L = "encoder.layers.encoder_layer_"

    x = af.input((1, 3, IMG, IMG))
    cls = af.input((1, DIM))
    pos = af.input((SEQ, DIM))

    # patch embed via space_to_depth(16) + 1x1 conv (a strided 16x16 conv is walled on the ANE)
    w_pe = np.ascontiguousarray(g("conv_proj.weight").transpose(0, 2, 3, 1)).reshape(DIM, -1, 1, 1)
    h = af.conv(af.space_to_depth(x, PATCH), w_pe, bias=g("conv_proj.bias"))
    # patchify: [1,768,14,14] -> [1,768,196] -> [196,768] (matches torch reshape+permute)
    patches = h.reshape(1, DIM, NUM_PATCHES).transpose([0, 2, 1]).reshape(NUM_PATCHES, DIM)

    seq = af.concat([cls, patches], axis=0)     # [197,768]: prepend CLS
    seq = seq + pos                             # add positional embedding

    for i in range(n_layers):
        p = f"{L}{i}."
        # pre-norm self-attention block
        x_ln = seq.layer_norm(g(p + "ln_1.weight"), g(p + "ln_1.bias"), eps=1e-6)
        Wqkv = g(p + "self_attention.in_proj_weight")    # [2304,768] stacked q,k,v
        bqkv = g(p + "self_attention.in_proj_bias")
        Wq, Wk, Wv = Wqkv[:DIM], Wqkv[DIM:2 * DIM], Wqkv[2 * DIM:]
        bq, bk, bv = bqkv[:DIM], bqkv[DIM:2 * DIM], bqkv[2 * DIM:]
        attn = af.mha(x_ln, Wq, bq, Wk, bk, Wv, bv,
                      g(p + "self_attention.out_proj.weight"),
                      g(p + "self_attention.out_proj.bias"), HEADS)
        seq = seq + attn
        # pre-norm MLP block (Linear -> GELU -> Linear)
        y_ln = seq.layer_norm(g(p + "ln_2.weight"), g(p + "ln_2.bias"), eps=1e-6)
        y = y_ln.linear(g(p + "mlp.0.weight"), g(p + "mlp.0.bias")).gelu()
        y = y.linear(g(p + "mlp.3.weight"), g(p + "mlp.3.bias"))
        seq = seq + y

    # final layer norm + classifier on the CLS token's row (row 0)
    seq = seq.layer_norm(g("encoder.ln.weight"), g("encoder.ln.bias"), eps=1e-6)
    cls_row = _row0(seq)                          # [1,768]
    return cls_row.linear(g("heads.head.weight"), g("heads.head.bias"))   # [1,1000]


def _row0(h):
    """Pick row 0 of h [M,D] -> [1,D] via a constant one-hot picker matmul."""
    M, D = h.shape
    sel = np.eye(1, M, dtype=np.float32)          # [1,M], picks row 0
    return h.transpose([1, 0]).linear(sel).transpose([1, 0])   # [D,1] -> [1,D]


# torch reference (optionally truncated to K encoder layers, same weights)
def torch_ref(m, img, n_layers):
    import torch
    with torch.no_grad():
        x = m._process_input(torch.from_numpy(img))
        cls = m.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + m.encoder.pos_embedding
        x = m.encoder.dropout(x)
        for i in range(n_layers):
            x = m.encoder.layers[i](x)
        x = m.encoder.ln(x)[:, 0]
        return m.heads(x).numpy()[0]


# optimizer tie-in: one real ViT attention layer via af.sdpa (route rewrite)
def attn_layer_graph(sd, layer=0):
    """ViT layer-`layer` self-attention as q/k/v proj -> af.sdpa -> out-proj (real weights), so af.tune can pick the SDPA route."""
    p = f"encoder.layers.encoder_layer_{layer}."
    Wqkv = sd[p + "self_attention.in_proj_weight"]; bqkv = sd[p + "self_attention.in_proj_bias"]
    Wq, Wk, Wv = Wqkv[:DIM], Wqkv[DIM:2 * DIM], Wqkv[2 * DIM:]
    bq, bk, bv = bqkv[:DIM], bqkv[DIM:2 * DIM], bqkv[2 * DIM:]
    Wo = sd[p + "self_attention.out_proj.weight"]; bo = sd[p + "self_attention.out_proj.bias"]
    dh = DIM // HEADS
    x = af.input((SEQ, DIM))
    q = x.linear(Wq, bq).reshape(1, SEQ, HEADS, dh).transpose([0, 2, 1, 3])   # [1,H,S,dh]
    k = x.linear(Wk, bk).reshape(1, SEQ, HEADS, dh).transpose([0, 2, 1, 3])
    v = x.linear(Wv, bv).reshape(1, SEQ, HEADS, dh).transpose([0, 2, 1, 3])
    o = af.sdpa(q, k, v)                                                       # native ANE SDPA
    o = o.transpose([0, 2, 1, 3]).reshape(SEQ, DIM)
    return o.linear(Wo, bo)


def bench(net, inputs, reps=30, warmup=8):
    for _ in range(warmup):
        net(*inputs)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); net(*inputs); best = min(best, (time.perf_counter() - t0) * 1e6)
    return best


def maxdiff(a, b):
    a = np.asarray(a, np.float32).ravel(); b = np.asarray(b, np.float32).ravel()
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))


# main
def main():
    _common.head("ViT-B/16 on the Apple Neural Engine (aneforge)")
    m, sd = load_vit_weights()
    print("config: torchvision vit_b_16 IMAGENET1K_V1 | "
          f"{N_LAYERS} layers x {DIM} dim x {HEADS} heads | patch {PATCH} | "
          f"{SEQ} tokens | 1000 classes (~86M params)")

    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, IMG, IMG)).astype(np.float32)
    cls_const = sd["class_token"].reshape(1, DIM).astype(np.float32)
    pos_const = sd["encoder.pos_embedding"].reshape(SEQ, DIM).astype(np.float32)

    # try the full 12-layer model as ONE program; fall back to fewer layers
    n_layers, net, why = N_LAYERS, None, ""
    for k in (N_LAYERS, 8, 6, 4, 2, 1):
        try:
            net = af.compile(build_vit(sd, k))
            n_layers = k
            why = ("full 12-layer model compiled as one program" if k == N_LAYERS else
                   f"full 12-layer model did not compile as one program; using first {k} layers "
                   f"(real pretrained weights), validated vs the SAME {k}-layer torch forward")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [compile] {k} layers failed ({type(e).__name__}: {str(e)[:90]}) -> trying fewer")
            net = None
    if net is None:
        print("FAIL: could not compile even 1 layer")
        return 1

    print(f"\nPATH: {why}")
    print(f"forward: {net.n_ops} ops fused into 1 ANE program"
          + (f" (+{net.n_sdpa} native-SDPA sub-programs)" if getattr(net, "n_sdpa", 0) else ""))

    # validate logits vs torch reference (same n_layers)
    ane = net(img, cls_const, pos_const)[0]
    ref = torch_ref(m, img, n_layers)
    cos = float(ane @ ref / (np.linalg.norm(ane) * np.linalg.norm(ref)))
    ane_top5 = ane.argsort()[-5:][::-1].tolist()
    ref_top5 = ref.argsort()[-5:][::-1].tolist()
    top1 = int(ane.argmax()) == int(ref.argmax())
    top5_overlap = len(set(ane_top5) & set(ref_top5))
    print("\nVALIDATION (ANE forward vs torchvision fp32 reference):")
    print(f"  logit cosine = {cos:.4f}")
    print(f"  ANE top-5    = {ane_top5}")
    print(f"  ref top-5    = {ref_top5}")
    print(f"  top-1 match  = {top1} | top-5 overlap = {top5_overlap}/5")
    forward_ok = cos > 0.99 and top1

    # optimizer demo: SDPA route rewrite on a real ViT attention layer
    print("\nOPTIMIZER (af.tune SDPA route rewrite on ViT layer-0 attention):")
    xin = rng.standard_normal((SEQ, DIM)).astype(np.float32)
    from aneforge._compile import compile as _compile
    base = _compile(attn_layer_graph(sd, 0), int8=False, opt=0)
    base_y = base(xin); base_us = bench(base, [xin]); base.release()

    tuned = af.tune(attn_layer_graph(sd, 0))
    tuned_y = tuned(xin); tuned_us = bench(tuned, [xin])
    diff = maxdiff(tuned_y, base_y)
    try:
        from aneforge._optimize import _config_label, _graph_key, _input_shapes, _load_cache
        gout = attn_layer_graph(sd, 0)
        cfg = _load_cache().get(_graph_key(gout, _input_shapes(gout)), {}).get("config")
        route = _config_label(cfg) if cfg else "?"
    except Exception:  # noqa: BLE001
        route = "?"
    tuned.release()

    speedup = base_us / tuned_us if tuned_us else float("nan")
    print(f"  opt=0 baseline : {base_us:8.1f} us")
    print(f"  af.tune        : {tuned_us:8.1f} us  ({speedup:.2f}x)")
    print(f"  tuner chose    : {route}")
    print(f"  maxdiff(tune, opt=0) = {diff:.6f}  (lossless route -> ~0)")
    tune_ok = diff <= 1e-2          # lossless route must be faithful to baseline

    # verdict
    print(f"forward validates (cos>0.99 & top-1): {forward_ok}")
    print(f"tune correct (maxdiff ~0, lossless):  {tune_ok}  | measured {speedup:.2f}x")
    ok = forward_ok and tune_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
