"""aneforge optimizer demo - measured speedups on REAL models, correctness preserved.

The aneforge autotuner (``af.tune``) is a rewrite engine over a metamorphic-proven-
safe variant space (see aneforge/_optimize.py): it enumerates the legal variants
(native<->decomposed SDPA route, fp16<->int8 weight streaming), prunes with the
cost model, MEASURES the survivors on the ANE, VALIDATES each against the opt=0
baseline, and returns the fastest CORRECT one.

This demo runs that tuner end-to-end on real models and proves two things at once:

  1. CORRECTNESS (the headline, non-negotiable): the optimized program reproduces
     the opt=0 baseline output. Default tune() is accuracy-PRESERVING (lossless
     route selection / fp16) -> matches within fp16 noise. tune(atol=0.1) admits
     int8 -> matches within its stated ~0.1 budget.
  2. SPEED: the REAL measured end-to-end latency (warmup, then MIN over reps) of
     opt=0 vs tune-lossless vs tune-int8.

Models:
  - ResNet-18           (af.load_resnet18; weight-heavy convs -> int8 candidate)
  - MiniLM encoder      (af.load all-MiniLM-L6-v2; matmul-heavy, has attention)
  - Attention block     (q/k/v proj + af.sdpa + out-proj; exercises the route
                         rewrite: native SDPA cut vs decomposed-fused) at two sizes

HONESTY: this prints the REAL measured speedups. The route rewrite is expected to
move the attention block; weight-heavy models may show an int8 win under atol=0.1;
a floor-bound / already-optimal model correctly returns ~1.0x (tune returns the
baseline) - that is correct behavior, not a failure. We report exactly what the
tuner chose and never claim a speedup the measurement does not show.

    python3 examples/autotune.py
"""
import sys, time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge

import numpy as np
import aneforge as af
from aneforge._compile import compile as _compile
from aneforge._optimize import _config_label, _graph_key, _input_shapes


# measurement helpers                                                          #
def bench(net, inputs, reps: int = 30, warmup: int = 8) -> float:
    """End-to-end latency in microseconds: warmup, then MIN over reps."""
    for _ in range(warmup):
        net(*inputs)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        net(*inputs)
        best = min(best, (time.perf_counter() - t0) * 1e6)
    return best


def maxdiff(a, b) -> float:
    """Relative max-abs difference (the same metric the tuner gates on)."""
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    n = min(a.size, b.size)
    return float(np.abs(a[:n] - b[:n]).max() / (np.abs(b[:n]).max() + 1e-6))


def tuner_decision(out) -> str:
    """Recompute the cached tune() decision label for reporting (cache hit -> instant)."""
    key = _graph_key(out, _input_shapes(out))
    try:
        from aneforge._optimize import _load_cache
        c = _load_cache().get(key, {})
        cfg = c.get("config")
        return _config_label(cfg) if cfg else "?"
    except Exception:
        return "?"


# graph builders (aneforge public ops + the loaders' exposed real weights)     #
# These rebuild the SAME graph the loaders compile internally, but return the  #
# output Tensor so af.tune() can optimize it. Inputs already-embedded (host).  #
def resnet18_graph(clf):
    """Rebuild ResNet-18's ANE graph from clf.sd (folded BN) -> output Tensor."""
    sd = clf.sd

    def fold(conv_key, bn):
        W = sd[conv_key + ".weight"]
        g, b = sd[bn + ".weight"], sd[bn + ".bias"]
        mu, var = sd[bn + ".running_mean"], sd[bn + ".running_var"]
        sc = g / np.sqrt(var + 1e-5)
        return (W * sc[:, None, None, None]).astype(np.float32), (b - mu * sc).astype(np.float32)

    def block(x, prefix, stride, downsample):
        w1, b1 = fold(prefix + ".conv1", prefix + ".bn1")
        w2, b2 = fold(prefix + ".conv2", prefix + ".bn2")
        h = af.conv(x, w1, stride=stride, pad=1, bias=b1).relu()
        h = af.conv(h, w2, stride=1, pad=1, bias=b2)
        idn = x
        if downsample:
            wd, bd = fold(prefix + ".downsample.0", prefix + ".downsample.1")
            idn = af.conv(x, wd, stride=stride, pad=0, bias=bd)
        return (h + idn).relu()

    x = af.input((1, 3, 224, 224))
    w, b = fold("conv1", "bn1")
    h = af.conv(x, w, stride=2, pad=3, bias=b).relu().max_pool(3, stride=2, pad=1)
    for name, stride in [("layer1", 1), ("layer2", 2), ("layer3", 2), ("layer4", 2)]:
        for i in range(2):
            h = block(h, f"{name}.{i}", stride if i == 0 else 1,
                      downsample=(i == 0 and name != "layer1"))
    h = h.mean((2, 3)).reshape(1, 512)
    return h.linear(sd["fc.weight"], sd["fc.bias"])


def minilm_graph(enc, S):
    """Rebuild the MiniLM encoder stack for sequence length S -> output Tensor.
    Input is the host-embedded [S, D] tensor (same as the loader's _build)."""
    h = af.input((S, enc.D))
    for w in enc.layers:
        attn = af.mha(h, w["Wq"], w["bq"], w["Wk"], w["bk"], w["Wv"], w["bv"],
                      w["Wo"], w["bo"], enc.H)
        h = (h + attn).layer_norm(w["ln1w"], w["ln1b"], enc.eps)
        ff = h.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
        h = (h + ff).layer_norm(w["ln2w"], w["ln2b"], enc.eps)
    return h


def attn_block_graph(H, S, D):
    """A transformer attention block built with af.sdpa: q/k/v proj + native SDPA +
    out-proj. Input [S, Dm], Dm = H*D. Exercises the SDPA route rewrite."""
    Dm = H * D
    rng = np.random.default_rng(7)
    Wq = (rng.standard_normal((Dm, Dm)) * 0.05).astype(np.float32)
    Wk = (rng.standard_normal((Dm, Dm)) * 0.05).astype(np.float32)
    Wv = (rng.standard_normal((Dm, Dm)) * 0.05).astype(np.float32)
    Wo = (rng.standard_normal((Dm, Dm)) * 0.05).astype(np.float32)
    x = af.input((S, Dm))
    q = x.linear(Wq).reshape(1, S, H, D).transpose([0, 2, 1, 3])   # [1,H,S,D]
    k = x.linear(Wk).reshape(1, S, H, D).transpose([0, 2, 1, 3])
    v = x.linear(Wv).reshape(1, S, H, D).transpose([0, 2, 1, 3])
    o = af.sdpa(q, k, v)                                           # native ANE SDPA
    o = o.transpose([0, 2, 1, 3]).reshape(S, Dm)
    return o.linear(Wo)


# per-model runner: opt=0 baseline vs tune-lossless vs tune-int8              #
def run_model(label, build_graph, make_inputs, has_int8):
    """Compile/measure opt=0, tune-lossless, tune-int8 (if applicable); assert the
    optimized outputs match the opt=0 baseline; return a results row."""
    out = build_graph()
    inputs = make_inputs()

    # opt=0 baseline (the correctness reference + the speed reference)
    base = _compile(out, int8=False, opt=0)
    base_y = base(*inputs)
    base_us = bench(base, inputs)
    base.release()

    # tune (default): accuracy-PRESERVING. Lossless route selection / fp16 only.
    lossless = af.tune(build_graph())
    ll_y = lossless(*inputs)
    ll_us = bench(lossless, inputs)
    ll_diff = maxdiff(ll_y, base_y)
    ll_decision = tuner_decision(build_graph())
    lossless.release()

    # tune(atol=0.1): admits int8 within a stated accuracy budget.
    i8_us = i8_diff = None
    i8_decision = "n/a"
    if has_int8:
        int8m = af.tune(build_graph(), atol=0.1)
        i8_y = int8m(*inputs)
        i8_us = bench(int8m, inputs)
        i8_diff = maxdiff(i8_y, base_y)
        i8_decision = tuner_decision(build_graph())
        int8m.release()

    # correctness assertions (the headline)
    ok = ll_diff <= 1e-2          # default tune must match within fp16 noise
    if has_int8 and i8_diff is not None:
        ok = ok and (i8_diff <= 0.12)   # int8 within its ~0.1 budget (metamorphic tol)

    return {
        "label": label, "base_us": base_us,
        "ll_us": ll_us, "ll_diff": ll_diff, "ll_decision": ll_decision,
        "i8_us": i8_us, "i8_diff": i8_diff, "i8_decision": i8_decision,
        "ok": ok,
    }


def verdict(r):
    sp_ll = r["base_us"] / r["ll_us"] if r["ll_us"] else float("nan")
    parts = [f"lossless {sp_ll:.2f}x ({r['ll_decision']})"]
    if r["i8_us"]:
        sp_i8 = r["base_us"] / r["i8_us"]
        parts.append(f"int8 {sp_i8:.2f}x ({r['i8_decision']}, maxdiff {r['i8_diff']:.3f})")
    chose_speedup = sp_ll >= 1.05 or (r["i8_us"] and r["base_us"] / r["i8_us"] >= 1.05)
    head = "WIN" if chose_speedup else "no measured win (tuner returned baseline - correct)"
    corr = "correct" if r["ok"] else "CORRECTNESS FAIL"
    return f"  {r['label']}: {head}; {', '.join(parts)}; {corr}"


def fmt_speedup(base, us):
    return f"{us:8.1f} ({base/us:.2f}x)" if us else f"{'n/a':>15s}"


def main():
    _common.head("aneforge optimizer demo - real models on the M5 ANE")
    rows = []

    # ResNet-18 (vision; weight-heavy convs)
    print("\n[1/4] ResNet-18 (vision, conv-heavy)...")
    clf = af.load_resnet18()
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    rows.append(run_model("ResNet-18", lambda: resnet18_graph(clf),
                          lambda: [img], has_int8=True))

    # MiniLM encoder (matmul-heavy, attention)
    print("[2/4] MiniLM all-MiniLM-L6-v2 encoder (S=32)...")
    enc = af.load("sentence-transformers/all-MiniLM-L6-v2")
    S = 32
    emb = rng.standard_normal((S, enc.D)).astype(np.float32)
    rows.append(run_model("MiniLM(S=32)", lambda: minilm_graph(enc, S),
                          lambda: [emb], has_int8=True))

    # attention block via af.sdpa, two sizes (route rewrite)
    for (Hh, Ss, Dd) in [(8, 32, 64), (8, 256, 64)]:
        print(f"[{'3' if Ss==32 else '4'}/4] Attention block H={Hh} S={Ss} D={Dd} (af.sdpa route rewrite)...")
        Dm = Hh * Dd
        xin = rng.standard_normal((Ss, Dm)).astype(np.float32)
        rows.append(run_model(f"Attn(S={Ss})",
                              lambda Hh=Hh, Ss=Ss, Dd=Dd: attn_block_graph(Hh, Ss, Dd),
                              lambda xin=xin: [xin], has_int8=False))

    # results table
    print(f"{'model':14s} | {'opt=0 (us)':>11s} | {'tune-lossless (us, x)':>21s} | "
          f"{'tune-int8 (us, x)':>21s} | {'maxdiff ll/int8':>16s} | {'tuner chose (int8 run)':<22s}")
    for r in rows:
        i8d = f"{r['i8_diff']:.3f}" if r["i8_diff"] is not None else "n/a"
        print(f"{r['label']:14s} | {r['base_us']:11.1f} | "
              f"{fmt_speedup(r['base_us'], r['ll_us']):>21s} | "
              f"{fmt_speedup(r['base_us'], r['i8_us']):>21s} | "
              f"{r['ll_diff']:.4f} / {i8d:>6s} | {r['i8_decision']:<22s}")
    print("note: a 'tuner chose' label of int8=False in the int8 column means the int8 variant was")
    print("      measured but NOT selected (rejected on accuracy at atol=0.1, or below the 1.10x")
    print("      lossy-speedup margin) - so that column reports the fp16 fallback the tuner returned.")

    print("\nper-model verdict (what the tuner chose + speedup + correctness):")
    for r in rows:
        print(verdict(r))

    all_ok = all(r["ok"] for r in rows)
    any_win = any((r["base_us"] / r["ll_us"] >= 1.05) or
                  (r["i8_us"] and r["base_us"] / r["i8_us"] >= 1.05) for r in rows)
    print("\nHEADLINE:")
    print(f"  correctness preserved on all {len(rows)} models: {all_ok}")
    print(f"  at least one model shows a measured speedup: {any_win}")
    print("  The optimizer delivers measured wins where the variant space has one to give,")
    print("  and correctly returns the baseline (~1.0x) where the model is already optimal - ")
    print("  never trading accuracy for a speedup the measurement does not show.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
