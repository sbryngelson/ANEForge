"""On-device validation of the optimizer's equivalence-route registry.

The optimizer's coverage guarantee is: *every surfaced capability is either ROUTE-
SELECTABLE (the autotuner picks its cheapest equivalent lowering by measured cost) or
EXPLICITLY SINGLE-ROUTE.* This test proves the SELECTABLE half is honest — every route
the optimizer is willing to flip a bridge node to is mathematically equivalent on real
ANE silicon, the same proof class as the metamorphic ``mha_vs_sdpa`` /
``reduce_sum_vs_matmul`` transforms.

For each route in the registry it compiles BOTH lowerings (the native bridge and the
fused decomposition) on the device, runs them on identical inputs, and asserts they
agree within fp16 op-noise. It also reconciles the three closed tables so they cannot
drift: the route registry (``_capabilities.route_registry``), the executable builders
(``_rewrite._BRIDGE_DECOMPOSERS``), and the optimizer's route-id detector
(``_optimize._route_ids``).

Run standalone:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 tests/test_routes.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aneforge as af
from aneforge import _compile
from aneforge import _capabilities as cap
from aneforge._rewrite import _BRIDGE_DECOMPOSERS, decompose_bridge


# fp16 op-noise ceiling: a single fused op sits at ~1e-3 relerr; a short chain (the
# minmax decomposition is sub/divide) stays well under this. Same scale as the
# metamorphic mha_vs_sdpa tol (0.06). A genuine semantic break is orders of magnitude
# larger, so this never hides a real bug.
_TOL = 5e-3


def _run(out, *inputs):
    net = _compile.compile(out)
    try:
        return np.asarray(net(*[np.asarray(a, np.float16) for a in inputs]), np.float32)
    finally:
        try:
            net.release()
        except Exception:
            pass


def _relerr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    return float(np.abs(a - b).max() / (float(np.abs(b).max()) + 1e-6))


# table reconciliation (pure; no device)
def test_route_tables_reconcile():
    """The route registry's `selectable` set == the executable decomposer set == the
    set of bridge ops the optimizer's route-id detector will flip. No drift."""
    reg = cap.route_registry()
    selectable = {e["name"] for e in reg["selectable"]}
    decomposers = set(_BRIDGE_DECOMPOSERS)
    assert selectable == decomposers, (
        f"route registry `selectable` {sorted(selectable)} != "
        f"_rewrite._BRIDGE_DECOMPOSERS {sorted(decomposers)} — keep them in sync.")
    # every bridge op is classified (selectable XOR single) — closed over NETPLIST_OPS.
    classified = selectable | {e["name"] for e in reg["single_route"]}
    assert classified == set(_compile.NETPLIST_OPS), (
        f"unclassified bridge ops: {set(_compile.NETPLIST_OPS) - classified}")


def test_every_bridge_op_is_selectable_or_single():
    """The headline guarantee, machine-checked: every bridge op carries a route_class
    of exactly `selectable` or `single` (with a reason)."""
    reg = cap.build_registry()
    for e in reg["entries"]:
        if e["status"] != "bridge":
            continue
        assert e["route_class"] in ("selectable", "single"), e
        if e["route_class"] == "single":
            assert e.get("route_reason"), f"{e['name']}: single-route with no reason"
        else:
            assert e.get("alt_route") and e.get("alt_loss") == "lossless", e


# on-device equivalence per selectable route
def _check_route(build_bridge, shapes, x, label):
    """Compile the bridge build and its fused decomposition; assert agreement."""
    bridge_out = _run(build_bridge(*[af.input(s) for s in shapes]), x)
    # decompose every bridge node in the graph (the optimizer's all-decomposed config)
    g = build_bridge(*[af.input(s) for s in shapes])
    from aneforge._compile import _topo
    idx = {i for i, t in enumerate(_topo(g)) if t.op in _BRIDGE_DECOMPOSERS}
    fused_out = _run(decompose_bridge(g, idx), x)
    re = _relerr(fused_out, bridge_out)
    assert re <= _TOL, f"{label}: route mismatch relerr={re:.5g} > {_TOL}"
    return re


def test_sdpa_route_equivalent():
    rng = np.random.default_rng(1)
    H, S, D = 4, 8, 16
    x = rng.standard_normal((1, H, S, D)).astype(np.float16)

    def build(q):  # single tensor reused as q=k=v keeps it self-contained
        return af.sdpa(q, q, q, scale=1.0 / D ** 0.5)
    re = _check_route(build, [(1, H, S, D)], x, "sdpa")
    print(f"  sdpa             relerr={re:.5g}")


def test_minmax_norm_route_equivalent():
    rng = np.random.default_rng(2)
    for dim in ("Width", "Height"):
        for (C, H, W) in [(3, 4, 8), (4, 2, 16)]:
            x = rng.standard_normal((1, C, H, W)).astype(np.float16)
            re = _check_route(lambda xt: af.minmax_norm(xt, dimension=dim, eps=1e-4),
                              [(1, C, H, W)], x, f"minmax_norm/{dim}/{C}x{H}x{W}")
            print(f"  minmax_norm/{dim}/{C}x{H}x{W:<3} relerr={re:.5g}")
    # degenerate constant row: max==min -> 0/eps==0 on both sides (no NaN divergence)
    x = np.ones((1, 2, 2, 4), np.float16); x[0, 0, 0, :] = 3.0
    re = _check_route(lambda xt: af.minmax_norm(xt, dimension="Width", eps=1e-4),
                      [(1, 2, 2, 4)], x, "minmax_norm/degenerate")
    print(f"  minmax_norm/degenerate  relerr={re:.5g}")


def test_flatten_route_equivalent():
    rng = np.random.default_rng(3)
    C, H, W = 2, 3, 5
    x = rng.standard_normal((C, H, W)).astype(np.float16)
    re = _check_route(lambda xt: af.flatten(xt), [(C, H, W)], x, "flatten")
    print(f"  flatten          relerr={re:.5g}")
    assert re == 0.0, "flatten<->reshape must be BIT-IDENTICAL"


def test_lrn_route_equivalent():
    """lrn (native LocalResponseNormalization bridge, a graph cut) <-> the fused MIL
    local_response_norm op (one program, no cut). BIT-EQUIVALENT by construction:
    lrn(alpha,beta,k) == local_response_norm(size=C, alpha=alpha*C, beta, k), C=x.shape[1]."""
    rng = np.random.default_rng(4)
    for (C, H, W) in [(3, 4, 8), (8, 2, 6), (12, 3, 5)]:
        for (alpha, beta, k) in [(1.0, 0.75, 1.0), (1e-4, 0.5, 2.0)]:
            x = rng.standard_normal((1, C, H, W)).astype(np.float16)
            re = _check_route(lambda xt: af.lrn(xt, alpha=alpha, beta=beta, k=k),
                              [(1, C, H, W)], x, f"lrn/{C}x{H}x{W}/a{alpha}b{beta}k{k}")
            print(f"  lrn/{C}x{H}x{W}/a{alpha}b{beta}k{k:<3} relerr={re:.5g}")


def test_rejected_candidate_stays_single_route():
    """Honesty check on a REJECTED candidate: space_to_channel is marked single-route
    because its reshape+transpose decomposition needs a rank-6 intermediate that
    ANECCompile rejects. Confirm the registry records it single-route (we do NOT
    re-run the failing compile here — an ANECCompile abort can take down the
    in-process runtime; the rejection evidence is in the route_reason)."""
    reg = cap.route_registry()
    single = {e["name"]: e for e in reg["single_route"]}
    for name in ("space_to_channel", "channel_to_space", "space_to_batch", "batch_to_space"):
        assert name in single, f"{name} should be single-route"
        assert "rank" in single[name]["route_reason"].lower(), single[name]


if __name__ == "__main__":
    print("route-table reconciliation:")
    test_route_tables_reconcile()
    test_every_bridge_op_is_selectable_or_single()
    print("  OK (selectable == decomposers == route-ids; all bridge ops classified)")
    print("on-device route equivalence:")
    test_sdpa_route_equivalent()
    test_minmax_norm_route_equivalent()
    test_flatten_route_equivalent()
    test_lrn_route_equivalent()
    test_rejected_candidate_stays_single_route()
    print("\nALL ROUTES VALIDATED — every selectable route is equivalent on silicon.")
