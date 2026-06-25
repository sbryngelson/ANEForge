#!/usr/bin/env python3
"""Below-ridge fusion demo: fusing a memory-bound neck block slides its effective AI past the weight-path ridge. Run: PYTHONPATH=. python3 bench/below_ridge_fusion.py"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import device_compare as dc  # noqa: E402
import device_compare_wattcomplete as wc  # noqa: E402

HAVE_ANE, HAVE_SUDO = dc.HAVE_ANE, dc.HAVE_SUDO
min_latency_with_out = dc.min_latency_with_out
relerr = dc.relerr
if HAVE_ANE:
    import aneforge as af

# ANE weight-path ridge (FLOP/byte); activation-path ridge ~424.
WEIGHT_RIDGE = 71.0
ACT_RIDGE = 424.0
BYTES_FP16 = 2


def build_block_ops(C, H, W):
    """Return (op_specs, np_reference_fn) for the conv1x1->group_norm->gelu->gelu neck block."""
    rng = np.random.default_rng(17)
    Wc = (rng.standard_normal((C, C, 1, 1)).astype(np.float32)
          * np.sqrt(2.0 / C))
    gn_g = (rng.standard_normal(C).astype(np.float32) * 0.1 + 1.0)
    gn_b = (rng.standard_normal(C).astype(np.float32) * 0.1)
    GROUPS = 32 if C % 32 == 0 else 1

    def ref(u):
        u = u.astype(np.float64)
        x = np.einsum("ncHW,oc->noHW", u, Wc[..., 0, 0].astype(np.float64))
        N, Ch, Hh, Ww = x.shape
        xg = x.reshape(N, GROUPS, Ch // GROUPS, Hh, Ww)
        mu = xg.mean((2, 3, 4), keepdims=True)
        var = xg.var((2, 3, 4), keepdims=True)
        xg = (xg - mu) / np.sqrt(var + 1e-5)
        x = xg.reshape(N, Ch, Hh, Ww) * gn_g[None, :, None, None] + gn_b[None, :, None, None]
        x = gelu_np(x)
        x = gelu_np(x)
        return x

    flops = 2.0 * C * C * H * W + (8 + 10 + 10) * C * H * W
    act_bytes = C * H * W * BYTES_FP16              # one CxHxW fp16 tensor
    weight_bytes = (C * C + 2 * C) * BYTES_FP16     # conv + gn gamma/beta
    n_ops = 4   # conv, group_norm, gelu, gelu
    return {"Wc": Wc, "gn_g": gn_g, "gn_b": gn_b, "GROUPS": GROUPS, "ref": ref, "flops": flops,
                "act_bytes": act_bytes, "weight_bytes": weight_bytes, "n_ops": n_ops,
                "C": C, "H": H, "W": W}


def gelu_np(x):
    from math import sqrt, pi
    c = sqrt(2.0 / pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def fused_block(xin, spec):
    """The whole block as one aneforge graph (intermediates stay on-chip)."""
    GROUPS = spec["GROUPS"]
    h = af.conv(xin, spec["Wc"].astype(np.float16), stride=1, pad=0)
    h = h.group_norm(spec["gn_g"].astype(np.float16), spec["gn_b"].astype(np.float16), GROUPS)
    h = h.gelu()
    h = h.gelu()
    return h


def run(window=0.0):
    results = {"ridge_weight_path": WEIGHT_RIDGE, "ridge_activation_path": ACT_RIDGE,
               "blocks": []}
    if not HAVE_ANE:
        print("ANE unavailable; nothing to do.")
        return results

    for (C, H, W) in [(256, 32, 32), (512, 16, 16)]:
        spec = build_block_ops(C, H, W)
        u0 = np.random.default_rng(2).standard_normal((1, C, H, W)).astype(np.float32)
        ref = spec["ref"](u0)
        flops = spec["flops"]; act = spec["act_bytes"]; wt = spec["weight_bytes"]
        n_ops = spec["n_ops"]
        print(f"\n=== neck block C={C} {H}x{W} ===", flush=True)

        # fused: one program, one dispatch, intermediates on-chip
        try:
            xin = af.input((1, C, H, W))
            net = af.compile(fused_block(xin, spec))
            uf = u0.astype(np.float16)
            lat_f, out = min_latency_with_out(lambda: net(uf))
            err = relerr(np.asarray(out), ref)
        except Exception as e:
            print(f"  fused build FAILED: {type(e).__name__}: {e}")
            results["blocks"].append({"C": C, "HW": [H, W], "error": str(e)})
            continue
        bytes_fused = act + wt + act   # input + weights + output (intermediates on-chip)
        ai_fused = flops / bytes_fused
        gf_fused = flops / lat_f / 1e9
        print(f"  fused      {lat_f*1e3:8.3f} ms  {gf_fused:8.1f} GFLOP/s  "
              f"AI_eff={ai_fused:7.1f}  relerr {err:.2e}")

        # standalone: each op a separate compiled program (full in+out per op)
        standalone_ops = []
        lat_s_total = 0.0
        try:
            n1 = af.compile(af.conv(af.input((1, C, H, W)), spec["Wc"].astype(np.float16), stride=1, pad=0))
            l1, o1 = min_latency_with_out(lambda: n1(u0.astype(np.float16)))
            lat_s_total += l1; standalone_ops.append(("conv1x1", l1))
            mid = np.asarray(o1).astype(np.float16)
            n2 = af.compile(af.input((1, C, H, W)).group_norm(spec["gn_g"].astype(np.float16), spec["gn_b"].astype(np.float16), spec["GROUPS"]))
            l2, o2 = min_latency_with_out(lambda: n2(mid))
            lat_s_total += l2; standalone_ops.append(("group_norm", l2))
            mid2 = np.asarray(o2).astype(np.float16)
            n3 = af.compile(af.input((1, C, H, W)).gelu())
            l3, o3 = min_latency_with_out(lambda: n3(mid2))
            lat_s_total += l3; standalone_ops.append(("gelu", l3))
            l4, _ = min_latency_with_out(lambda: n3(mid2))
            lat_s_total += l4; standalone_ops.append(("gelu2", l4))
        except Exception as e:
            print(f"  standalone chain note: {type(e).__name__}: {e}")
        bytes_standalone = n_ops * (act + act) + wt   # each op moves in+out (+ weights)
        ai_standalone = flops / bytes_standalone
        gf_standalone = (flops / lat_s_total / 1e9) if lat_s_total else None
        print(f"  standalone {lat_s_total*1e3:8.3f} ms  "
              f"{(gf_standalone or 0):8.1f} GFLOP/s  AI_eff={ai_standalone:7.1f}  "
              f"(sum of {len(standalone_ops)} separate dispatches)")
        crossed = ai_standalone < WEIGHT_RIDGE <= ai_fused
        print(f"  ridge {WEIGHT_RIDGE}: standalone {'<' if ai_standalone < WEIGHT_RIDGE else '>='} ridge "
              f"(memory-bound), fused {'>=' if ai_fused >= WEIGHT_RIDGE else '<'} ridge  "
              f"==> AI lever crosses the ridge: {crossed}")

        block = {
            "C": C, "HW": [H, W], "flops": flops, "n_ops": n_ops,
            "act_bytes": act, "weight_bytes": wt,
            "fused": {"lat_ms": lat_f * 1e3, "gflops": gf_fused,
                      "bytes_moved": bytes_fused, "ai_eff": ai_fused, "relerr": err},
            "standalone": {"lat_ms": lat_s_total * 1e3, "gflops": gf_standalone,
                           "bytes_moved": bytes_standalone, "ai_eff": ai_standalone,
                           "ops": [{"op": o, "lat_ms": l * 1e3} for o, l in standalone_ops]},
            "ai_lever_crosses_weight_ridge": bool(crossed),
            "fusion_speedup": (lat_s_total / lat_f) if lat_f else None,
        }

        if HAVE_SUDO and window > 0:
            e_f = wc.measure_energy(lambda: net(uf), tag=f"neck_fused_{C}", window=window)
            if e_f and e_f.get("active_pkg_W"):
                block["fused"]["active_pkg_W"] = e_f["active_pkg_W"]
                block["fused"]["perf_per_W"] = (flops / (e_f["iter_ms"] / 1e3)) / 1e9 / e_f["active_pkg_W"]

        results["blocks"].append(block)

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=0.0,
                    help="power window seconds (0 = latency only)")
    args = ap.parse_args()
    print("=" * 80)
    print(" below-ridge fusion demo: AI lever crossing the weight-path ridge")
    print("=" * 80)
    if HAVE_SUDO and args.window > 0:
        wc.sample_idle(3.0)
    res = run(window=args.window)
    out = Path(__file__).resolve().parent / "results" / "below_ridge_fusion.json"
    out.write_text(json.dumps(res, indent=2, default=lambda o: None))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
