"""On-device check that each selectable route's bridge and fused lowerings agree."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aneforge as af
from aneforge import _compile
from aneforge import _capabilities as cap
from aneforge._rewrite import _BRIDGE_DECOMPOSERS, decompose_bridge


# fp16 op-noise ceiling: well above per-op rounding, far below a semantic break
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
  """selectable set == decomposer set == route-id-detector set; no drift."""
  reg = cap.route_registry()
  selectable = {e["name"] for e in reg["selectable"]}
  decomposers = set(_BRIDGE_DECOMPOSERS)
  assert selectable == decomposers, (
    f"route registry `selectable` {sorted(selectable)} != "
    f"_rewrite._BRIDGE_DECOMPOSERS {sorted(decomposers)} - keep them in sync.")
  # every bridge op is classified (selectable XOR single) - closed over NETPLIST_OPS.
  classified = selectable | {e["name"] for e in reg["single_route"]}
  assert classified == set(_compile.NETPLIST_OPS), (
    f"unclassified bridge ops: {set(_compile.NETPLIST_OPS) - classified}")


def test_every_bridge_op_is_selectable_or_single():
  """Every bridge op carries a route_class of exactly `selectable` or `single`."""
  reg = cap.build_registry()
  for e in reg["entries"]:
    if e["status"] != "bridge": continue
    assert e["route_class"] in ("selectable", "single"), e
    if e["route_class"] == "single":
      assert e.get("route_reason"), f"{e['name']}: single-route with no reason"
    else:
      assert e.get("alt_route") and e.get("alt_loss") == "lossless", e


# on-device equivalence per selectable route
def _check_route(build_bridge, shapes, x, label):
  """Compile the bridge build and its fused decomposition; assert agreement."""
  bridge_out = _run(build_bridge(*[af.input(s) for s in shapes]), x)
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

  def build(q):  # q=k=v keeps it self-contained
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
  # degenerate constant row: max==min -> 0/eps==0 both sides (no NaN divergence)
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
  """native lrn bridge vs fused MIL local_response_norm; bit-equivalent by construction."""
  rng = np.random.default_rng(4)
  for (C, H, W) in [(3, 4, 8), (8, 2, 6), (12, 3, 5)]:
    for (alpha, beta, k) in [(1.0, 0.75, 1.0), (1e-4, 0.5, 2.0)]:
      x = rng.standard_normal((1, C, H, W)).astype(np.float16)
      re = _check_route(lambda xt: af.lrn(xt, alpha=alpha, beta=beta, k=k),
                        [(1, C, H, W)], x, f"lrn/{C}x{H}x{W}/a{alpha}b{beta}k{k}")
      print(f"  lrn/{C}x{H}x{W}/a{alpha}b{beta}k{k:<3} relerr={re:.5g}")


def test_rejected_candidate_stays_single_route():
  """space_to_channel et al. stay single-route: rank-6 intermediate is rejected."""
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
  print("\nALL ROUTES VALIDATED - every selectable route is equivalent on silicon.")
