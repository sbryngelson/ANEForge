#!/usr/bin/env python3
"""Real autoregressive LLM-DECODE measurement — ANE (aneforge) vs GPU (MLX) vs CPU.

WHY THIS EXISTS. The paper reasons extensively about LLM *decode* — "GEMV AI ~ 1,
decode is bandwidth/GPU territory", the device ordering inferred from a single
synthetic GEMV roofline point (ANE 112 / GPU 75 / CPU 81 GFLOP/s; ANE winning at
B=1 up to K~4096; batched shifting toward the GPU). But the paper never MEASURES
an actual decode loop. This harness measures one and states plainly whether the
measured decode CONFIRMS or CONTRADICTS that inference.

WHAT IS MEASURED. The per-token compute of a small transformer decoder in the
canonical decode regime: seq position = 1 (the M=1 / GEMV / skinny-GEMM regime),
with a fixed-length KV cache. One "token" = one forward of the full L-layer stack
plus the vocab projection. We run:

  * B=1   single-stream decode (the canonical on-device case), AND
  * B=BATCH (default 32) batched decode (the serving case),

for each device {ANE (aneforge fp16), GPU (MLX fp16), CPU (numpy/Accelerate fp32)}.

Per device per batch we report: tokens/s, latency/token (ms), tokens/s/W and
energy/token (mJ/token) — using the idle-subtracted total-package ACTIVE power
from the device_compare_wattcomplete harness (imported, not reimplemented) — and
relerr of one token's logits vs an fp32 numpy reference (sanity).

THE PER-TOKEN LAYER. A decoder block in the decode regime:
    h  = x
    a  = attn(rmsnorm(h))          # QKV proj GEMVs + scores vs KV cache + out proj
    h  = h + a
    f  = ffn(rmsnorm(h))           # gate/up GEMVs + SiLU + down proj GEMV
    h  = h + f
then after L blocks: logits = rmsnorm(h) @ Wvocab  (the big vocab GEMV).

The attention here uses a FIXED, PRE-FILLED KV cache of length S_KV: the query is
the single new token (seq=1), scores are q @ Kcache^T  -> softmax -> @ Vcache. This
is the faithful per-token decode shape — the projections are genuine M=1 GEMVs and
dominate the FLOPs, which is exactly why decode is memory-bound. The KV cache is a
constant baked into the graph (we measure steady-state per-token compute, not the
cache-append bookkeeping, which is a memcopy on every backend and not the compute
the roofline argument is about).

ANE NOTE / PROXY LABELLING. aneforge fuses the whole block into e5rt program(s).
af.sdpa requires equal-shape q,k,v so it can't express seq=1-query-vs-S_KV-cache;
we therefore build attention from primitives (bmm scores + softmax + bmm context),
which is the decomposed-SDPA route the ANE fuses natively anyway. This is a
FULL-STACK decode forward (not a primitives-only proxy) — every per-token op of the
block + vocab head is present and measured end-to-end as one compiled program per
batch size. The only decode-loop element not measured is the KV-cache *append*
(a memcopy, identical cost on every backend).

CONFIG (default, TinyLlama/GPT-small class):
    d_model=2048, n_layers=8, n_heads=16, FFN=4x (d_ff=8192), vocab=32000,
    KV cache length S_KV=256.

Run from repo root (energy needs passwordless sudo for powermetrics):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 bench/decode_measurement.py

Writes bench/results/decode_measurement_results.json. --quick shrinks the model
(d=1024,L=4) and the power window for a smoke test. Devices run SEQUENTIALLY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

# Reuse the rigorous power harness (idle-subtracted total-package active W) — import,
# do NOT reimplement. measure_energy/sample_idle live in device_compare_wattcomplete.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_compare_wattcomplete as wc  # noqa: E402
import device_compare as dc                # noqa: E402

HAVE_ANE, HAVE_MLX, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_MLX, dc.HAVE_SUDO
relerr = dc.relerr

if HAVE_ANE:
    import aneforge as af
if HAVE_MLX:
    import mlx.core as mx


# config
class Cfg:
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab, s_kv):
        self.d = d_model
        self.L = n_layers
        self.H = n_heads
        self.dh = d_model // n_heads
        self.d_ff = d_ff
        self.vocab = vocab
        self.s_kv = s_kv

    def asdict(self):
        return {"d_model": self.d, "n_layers": self.L, "n_heads": self.H,
                "d_head": self.dh, "d_ff": self.d_ff, "vocab": self.vocab,
                "s_kv": self.s_kv}

    def flops_per_token(self, B):
        """2*MACs for the per-token forward of B streams: per layer = 4 proj GEMVs
        (QKV=3*d*d, out=d*d) + attention (2 * H * s_kv * dh for scores+context) +
        FFN (gate+up = 2*d*d_ff, down = d_ff*d); plus the vocab GEMV d*vocab."""
        d, L, H, dh, dff, V, s = self.d, self.L, self.H, self.dh, self.d_ff, self.vocab, self.s_kv
        per_layer = (4 * d * d) + (2 * H * s * dh) + (3 * d * dff)
        total_macs = L * per_layer + d * V
        return 2.0 * total_macs * B


# weights (shared across all three backends so relerr is meaningful)
def make_weights(cfg: Cfg, seed=0):
    rng = np.random.default_rng(seed)
    sc = 1.0 / np.sqrt(cfg.d)
    W = {}
    for li in range(cfg.L):
        W[li] = {
            "ln1": (rng.standard_normal(cfg.d).astype(np.float32) * 0.02 + 1.0),
            "ln2": (rng.standard_normal(cfg.d).astype(np.float32) * 0.02 + 1.0),
            "Wq": rng.standard_normal((cfg.d, cfg.d)).astype(np.float32) * sc,
            "Wk": rng.standard_normal((cfg.d, cfg.d)).astype(np.float32) * sc,
            "Wv": rng.standard_normal((cfg.d, cfg.d)).astype(np.float32) * sc,
            "Wo": rng.standard_normal((cfg.d, cfg.d)).astype(np.float32) * sc,
            "Wg": rng.standard_normal((cfg.d_ff, cfg.d)).astype(np.float32) * sc,
            "Wu": rng.standard_normal((cfg.d_ff, cfg.d)).astype(np.float32) * sc,
            "Wd": rng.standard_normal((cfg.d, cfg.d_ff)).astype(np.float32) * (1.0 / np.sqrt(cfg.d_ff)),
            # pre-filled KV cache: [H, s_kv, dh]
            "Kc": rng.standard_normal((cfg.H, cfg.s_kv, cfg.dh)).astype(np.float32) * 0.5,
            "Vc": rng.standard_normal((cfg.H, cfg.s_kv, cfg.dh)).astype(np.float32) * 0.5,
        }
    W["lnf"] = (rng.standard_normal(cfg.d).astype(np.float32) * 0.02 + 1.0)
    W["Wvocab"] = rng.standard_normal((cfg.vocab, cfg.d)).astype(np.float32) * sc
    return W


# numpy fp32/fp64 reference (one token, B streams)
def ref_decode(cfg: Cfg, W, x, dt=np.float64):
    """x: [B, d]. Returns logits [B, vocab] in dt."""
    x = x.astype(dt)
    B = x.shape[0]
    H, dh = cfg.H, cfg.dh

    def rms(h, g):
        ms = (h.astype(dt) ** 2).mean(-1, keepdims=True)
        return (h / np.sqrt(ms + 1e-5)) * g.astype(dt)

    h = x
    for li in range(cfg.L):
        w = W[li]
        a_in = rms(h, w["ln1"])                       # [B, d]
        q = a_in @ w["Wq"].astype(dt).T               # [B, d]
        k = a_in @ w["Wk"].astype(dt).T
        v = a_in @ w["Wv"].astype(dt).T
        qh = q.reshape(B, H, dh)                       # [B, H, dh]
        Kc = w["Kc"].astype(dt)                        # [H, s, dh]
        Vc = w["Vc"].astype(dt)
        # append the new token's k,v to the cache for this single decode step
        # scores for the new query against (cache + self): [B,H,1,s+1]
        kh = k.reshape(B, H, dh)
        vh = v.reshape(B, H, dh)
        out = np.empty((B, H, dh), dtype=dt)
        for b in range(B):
            Kfull = np.concatenate([Kc, kh[b][:, None, :]], axis=1)   # [H, s+1, dh]
            Vfull = np.concatenate([Vc, vh[b][:, None, :]], axis=1)
            sc = (qh[b][:, None, :] @ Kfull.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))  # [H,1,s+1]
            sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc); p = p / p.sum(-1, keepdims=True)
            out[b] = (p @ Vfull)[:, 0, :]              # [H, dh]
        ctx = out.reshape(B, H * dh)
        a = ctx @ w["Wo"].astype(dt).T
        h = h + a
        f_in = rms(h, w["ln2"])
        g = f_in @ w["Wg"].astype(dt).T
        u = f_in @ w["Wu"].astype(dt).T
        silu = g / (1.0 + np.exp(-g))
        f = (silu * u) @ w["Wd"].astype(dt).T
        h = h + f
    h = rms(h, W["lnf"])
    return h @ W["Wvocab"].astype(dt).T


# ANE graph (aneforge) — full per-token stack, batch B
def build_ane(cfg: Cfg, W, B, int8=False):
    """Build the B-stream per-token decoder forward as one aneforge graph.

    ``int8=True`` compiles the linear weights as per-output-channel symmetric int8
    streamed at half the bytes (dequantised during the tile DMA) — the verified
    int8 weight-streaming path. Decode is the AI~1 memory-bound regime, so halving
    the weight bytes is the one lever that can move decode THROUGHPUT, not just
    energy; this row tests whether it does, or whether decode stays dispatch-bound.

    aneforge has no constant-tensor leaf node, and a bmm (activation@activation) needs
    Tensor operands — so the per-layer KV cache is supplied as GRAPH INPUTS (fed each
    call). That's the only honest way to express a cache here; the cache bytes are
    constant data the caller passes in, exactly like a real decode loop reads its cache
    from memory. Returns (Model, list-of-cache-arrays) so the runner feeds x + caches.

    Attention is per (batch*head): scores = concat(q@Kc^T, q@k_self) -> softmax ->
    @ concat(Vc, v_self). Projections are the M=1 GEMVs that dominate; fp16 weights."""
    f16 = np.float16
    H, dh, s, d = cfg.H, cfg.dh, cfg.s_kv, cfg.d
    x = af.input((B, d))
    cache_inputs = []     # (Tensor, np.ndarray) feed pairs, in creation order after x
    h = x
    for li in range(cfg.L):
        w = W[li]
        a_in = h.rms_norm(w["ln1"].astype(f16))                       # [B,d]
        q = a_in.linear(w["Wq"].astype(f16))                          # [B,d]
        k = a_in.linear(w["Wk"].astype(f16))
        v = a_in.linear(w["Wv"].astype(f16))
        qh = q.reshape(B * H, 1, dh)
        kh = k.reshape(B * H, 1, dh)
        vh = v.reshape(B * H, 1, dh)
        # KV cache as graph inputs (broadcast across batch): Kc_T [B*H,dh,s], Vc [B*H,s,dh]
        Kc_T_arr = np.ascontiguousarray(
            np.tile(w["Kc"][None], (B, 1, 1, 1)).reshape(B * H, s, dh).transpose(0, 2, 1).astype(f16))
        Vc_arr = np.ascontiguousarray(
            np.tile(w["Vc"][None], (B, 1, 1, 1)).reshape(B * H, s, dh).astype(f16))
        Kc_T = af.input((B * H, dh, s)); cache_inputs.append((Kc_T, Kc_T_arr))
        Vc = af.input((B * H, s, dh)); cache_inputs.append((Vc, Vc_arr))
        sc_cache = (qh @ Kc_T) * (1.0 / dh ** 0.5)              # [B*H,1,s]
        sc_self = (qh @ kh.transpose([0, 2, 1])) * (1.0 / dh ** 0.5)  # [B*H,1,1]
        scores = af.concat([sc_cache, sc_self], axis=2)        # [B*H,1,s+1]
        p = scores.softmax(-1)
        Vfull = af.concat([Vc, vh], axis=1)                    # [B*H,s+1,dh]
        ctx = (p @ Vfull).reshape(B, H * dh)                   # [B,d]
        a = ctx.linear(w["Wo"].astype(f16))
        h = h + a
        f_in = h.rms_norm(w["ln2"].astype(f16))
        g = f_in.linear(w["Wg"].astype(f16))
        u = f_in.linear(w["Wu"].astype(f16))
        f = (g.silu() * u).linear(w["Wd"].astype(f16))
        h = h + f
    h = h.rms_norm(W["lnf"].astype(f16))
    logits = h.linear(W["Wvocab"].astype(f16))
    net = af.compile(logits, int8=int8)
    return net, [arr for (_, arr) in cache_inputs]


# MLX graph (GPU)
def build_mlx(cfg: Cfg, W, B, dt):
    H, dh, s = cfg.H, cfg.dh, cfg.s_kv
    g = {}
    for li in range(cfg.L):
        w = W[li]
        g[li] = {k: mx.array(w[k].astype(dt)) for k in
                 ("ln1", "ln2", "Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")}
        g[li]["Kc"] = mx.array(np.tile(w["Kc"][None], (B, 1, 1, 1)).reshape(B * H, s, dh).astype(dt))
        g[li]["Vc"] = mx.array(np.tile(w["Vc"][None], (B, 1, 1, 1)).reshape(B * H, s, dh).astype(dt))
    g["lnf"] = mx.array(W["lnf"].astype(dt))
    g["Wvocab"] = mx.array(W["Wvocab"].astype(dt))

    def rms(h, gg):
        ms = mx.mean(h * h, axis=-1, keepdims=True)
        return (h * mx.rsqrt(ms + 1e-5)) * gg

    def fwd(x):
        h = x
        for li in range(cfg.L):
            w = g[li]
            a_in = rms(h, w["ln1"])
            q = a_in @ w["Wq"].T
            k = a_in @ w["Wk"].T
            v = a_in @ w["Wv"].T
            qh = q.reshape(B * H, 1, dh)
            kh = k.reshape(B * H, 1, dh)
            vh = v.reshape(B * H, 1, dh)
            Kfull = mx.concatenate([w["Kc"], kh], axis=1)      # [B*H,s+1,dh]
            Vfull = mx.concatenate([w["Vc"], vh], axis=1)
            scores = (qh @ Kfull.transpose(0, 2, 1)) * (1.0 / np.sqrt(dh))
            p = mx.softmax(scores, axis=-1)
            ctx = (p @ Vfull).reshape(B, H * dh)
            a = ctx @ w["Wo"].T
            h = h + a
            f_in = rms(h, w["ln2"])
            gg = f_in @ w["Wg"].T
            u = f_in @ w["Wu"].T
            silu = gg * mx.sigmoid(gg)
            f = (silu * u) @ w["Wd"].T
            h = h + f
        h = rms(h, g["lnf"])
        return h @ g["Wvocab"].T

    return fwd


# CPU (numpy fp32 / Accelerate)
def build_cpu(cfg: Cfg, W):
    return lambda x: ref_decode(cfg, W, x, dt=np.float32)


# measurement
def min_latency(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_device(name, run_once, *, B, cfg, ref, window, reps, warmup, get_out):
    """One device/batch point: min latency/token, throughput, energy via the
    wattcomplete harness, relerr vs fp32 ref. get_out() returns the logits as np."""
    row = {"device": name, "batch": B, "status": "ok"}
    try:
        lat = min_latency(run_once, reps=reps, warmup=warmup)   # s per token-step (B streams)
    except Exception as e:
        row["status"] = f"{type(e).__name__}: {e}"
        return row
    row["latency_per_step_ms"] = lat * 1e3
    row["latency_per_token_ms"] = lat * 1e3 / B          # per individual token in the batch
    row["tokens_per_s"] = B / lat                        # aggregate decode throughput
    try:
        out = np.asarray(get_out(), dtype=np.float32)
        row["relerr"] = relerr(out, ref)
    except Exception as e:
        row["relerr"] = None
        row["relerr_err"] = f"{type(e).__name__}: {e}"

    if HAVE_SUDO:
        e = wc.measure_energy(run_once, tag=f"decode_{name}_{B}".replace(" ", ""), window=window)
        if e:
            apw = e.get("active_pkg_W", float("nan"))
            row["active_pkg_W"] = apw
            row["active_pkg_cv_pct"] = e.get("active_pkg_cv_pct")
            row["ane_active_mW"] = e.get("ane_active_mW")
            row["gpu_active_mW"] = e.get("gpu_active_mW")
            row["cpu_active_mW"] = e.get("cpu_active_mW")
            row["energy_iter_ms"] = e.get("iter_ms")
            row["flags"] = e.get("flags")
            if apw == apw and apw > 0:
                # throughput from the energy loop's own iter time (sustained), B streams
                tok_s_energy = B / (e["iter_ms"] / 1e3)
                row["tokens_per_s_sustained"] = tok_s_energy
                row["tokens_per_s_per_W"] = tok_s_energy / apw
                row["mJ_per_token"] = (apw * e["iter_ms"]) / B     # W*ms = mJ, per token
    return row


# driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small model + short window smoke test")
    ap.add_argument("--batch", type=int, default=32, help="batched-decode B (default 32)")
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32",
                    help="comma list of batch sizes to sweep (default 1,2,4,8,16,32)")
    ap.add_argument("--window", type=float, default=6.0, help="power sampling window (s)")
    args = ap.parse_args()

    if args.quick:
        cfg = Cfg(d_model=1024, n_layers=4, n_heads=16, d_ff=4096, vocab=32000, s_kv=128)
        window = 3.0
        reps, warmup = 20, 5
    else:
        cfg = Cfg(d_model=2048, n_layers=8, n_heads=16, d_ff=8192, vocab=32000, s_kv=256)
        window = args.window
        reps, warmup = 40, 8

    if args.quick:
        batches = [1, args.batch]
    else:
        batches = [int(b) for b in args.batches.split(",") if b.strip()]
    print(f"=== DECODE config: {cfg.asdict()} ===")
    print(f"batch sweep: {batches}")
    print(f"FLOPs/token-step B=1: {cfg.flops_per_token(1)/1e9:.3f} GFLOP; "
          f"B={args.batch}: {cfg.flops_per_token(args.batch)/1e9:.3f} GFLOP")
    print(f"sudo/powermetrics: {HAVE_SUDO}; ANE: {HAVE_ANE}; MLX: {HAVE_MLX}")

    W = make_weights(cfg)

    # idle baseline ONCE (the wattcomplete harness subtracts it)
    if HAVE_SUDO:
        print("sampling idle baseline...", flush=True)
        wc.sample_idle(3.0)
        print(f"  idle pkg = {wc.IDLE_PKG/1000.0:.3f} W; per-rail mW = "
              f"{ {k: round(v,1) for k,v in wc.IDLE.items()} }", flush=True)

    results = {"config": cfg.asdict(), "quick": args.quick, "batch": args.batch,
               "flops_per_token": {str(b): cfg.flops_per_token(b) for b in batches},
               "have": {"ane": HAVE_ANE, "mlx": HAVE_MLX, "sudo": HAVE_SUDO},
               "idle_pkg_W": (wc.IDLE_PKG / 1000.0) if HAVE_SUDO else None,
               "rows": []}

    rng = np.random.default_rng(123)

    for B in batches:
        x = (rng.standard_normal((B, cfg.d)).astype(np.float32) * 0.1)
        ref = np.asarray(ref_decode(cfg, W, x, dt=np.float64), dtype=np.float32)
        print(f"\n########## BATCH B={B} ##########", flush=True)

        # ---- CPU (numpy fp32) ----
        print("[CPU] numpy/Accelerate fp32 ...", flush=True)
        cpu_fwd = build_cpu(cfg, W)
        st = {"o": None}
        def cpu_run():
            st["o"] = cpu_fwd(x)
        r = measure_device("CPU", cpu_run, B=B, cfg=cfg, ref=ref, window=window,
                            reps=max(5, reps // 6), warmup=2,
                            get_out=lambda: st["o"])
        results["rows"].append(r); _print_row(r)

        # ---- GPU (MLX fp16) ----
        if HAVE_MLX:
            print("[GPU] MLX fp16 ...", flush=True)
            try:
                fwd = build_mlx(cfg, W, B, np.float16)
                xg = mx.array(x.astype(np.float16))
                stg = {"o": None}
                def gpu_run():
                    o = fwd(xg)
                    mx.eval(o)
                    stg["o"] = o
                r = measure_device("GPU", gpu_run, B=B, cfg=cfg, ref=ref, window=window,
                                   reps=reps, warmup=warmup,
                                   get_out=lambda: np.array(stg["o"], copy=False))
            except Exception as e:
                r = {"device": "GPU", "batch": B, "status": f"{type(e).__name__}: {e}"}
            results["rows"].append(r); _print_row(r)

        # ---- ANE (aneforge fp16) ----
        if HAVE_ANE:
            print("[ANE] aneforge fp16 ...", flush=True)
            try:
                net, caches = build_ane(cfg, W, B)
                xf = x.astype(np.float16)
                sta = {"o": None}
                def ane_run():
                    sta["o"] = net(xf, *caches)
                r = measure_device("ANE", ane_run, B=B, cfg=cfg, ref=ref, window=window,
                                   reps=reps, warmup=warmup,
                                   get_out=lambda: sta["o"])
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {"device": "ANE", "batch": B, "status": f"{type(e).__name__}: {e}"}
            results["rows"].append(r); _print_row(r)

        # ---- ANE-int8 (aneforge, per-channel int8 weight streaming) ----
        if HAVE_ANE:
            print("[ANE8] aneforge int8 weight-stream ...", flush=True)
            try:
                net8, caches8 = build_ane(cfg, W, B, int8=True)
                xf = x.astype(np.float16)
                sta8 = {"o": None}
                def ane8_run():
                    sta8["o"] = net8(xf, *caches8)
                r = measure_device("ANE8", ane8_run, B=B, cfg=cfg, ref=ref, window=window,
                                   reps=reps, warmup=warmup,
                                   get_out=lambda: sta8["o"])
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {"device": "ANE8", "batch": B, "status": f"{type(e).__name__}: {e}"}
            results["rows"].append(r); _print_row(r)

    out_path = Path(__file__).resolve().parent / "results" / "decode_measurement_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_path}")
    _verdict(results, cfg, args.batch)


def _print_row(r):
    if r.get("status") != "ok":
        print(f"  {r['device']} B={r['batch']}: {r['status']}")
        return
    s = (f"  {r['device']:>3} B={r['batch']:>2}: "
         f"tok/s={r.get('tokens_per_s',float('nan')):8.1f}  "
         f"lat/tok={r.get('latency_per_token_ms',float('nan')):7.3f}ms  ")
    if r.get("tokens_per_s_per_W") is not None:
        s += (f"tok/s/W={r['tokens_per_s_per_W']:7.2f}  "
              f"mJ/tok={r.get('mJ_per_token',float('nan')):8.1f}  "
              f"pkgW={r.get('active_pkg_W',float('nan')):5.2f}  ")
    s += f"relerr={r.get('relerr')}"
    if r.get("flags"):
        s += f"  flags={r['flags']}"
    print(s, flush=True)


def _verdict(results, cfg, batch):
    print("\n" + "=" * 70)
    print("VERDICT — measured decode vs the GEMV-inferred claims")
    print("=" * 70)
    def find(dev, B):
        for r in results["rows"]:
            if r.get("device") == dev and r.get("batch") == B and r.get("status") == "ok":
                return r
        return None
    for B in (1, batch):
        line = [f"B={B}:"]
        for dev in ("ANE", "GPU", "CPU"):
            r = find(dev, B)
            if r:
                line.append(f"{dev} {r.get('tokens_per_s',0):.0f} tok/s"
                            f" / {r.get('tokens_per_s_per_W') and round(r['tokens_per_s_per_W'],1)} tok/s/W")
        print("  " + " | ".join(line))
    print("\n  (Interpretation printed by the caller; see JSON for full table.)")


if __name__ == "__main__":
    main()
