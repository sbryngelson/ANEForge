"""The fuzz harness's host-side machinery (generator, mirrors, shrinker) - CI-testable, no ANE."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from _helpers import requires_ane

_spec = importlib.util.spec_from_file_location(
  "fuzz", Path(__file__).resolve().parents[1] / "scripts" / "fuzz.py")
assert _spec is not None and _spec.loader is not None
fuzz = importlib.util.module_from_spec(_spec)
sys.modules["fuzz"] = fuzz
_spec.loader.exec_module(fuzz)


def _specs(n=40):
  out = []
  for seed in range(n):
    s = fuzz.gen_spec(seed)
    if s["nodes"]: out.append(s)
  return out


def test_gen_spec_is_deterministic():
  assert fuzz.gen_spec(1234) == fuzz.gen_spec(1234)
  assert fuzz.gen_spec(1235) != fuzz.gen_spec(1234)

def test_specs_are_json_round_trippable():
  for s in _specs(10):
    assert json.loads(json.dumps(s)) == s

def test_feeds_are_deterministic_from_spec_alone():
  for s in _specs(10):
    assert np.array_equal(fuzz._feed_for(s), fuzz._feed_for(s))

def test_int_mode_mirror_is_exactly_integral_and_bounded():
  # the exact-oracle precondition: every int-mode intermediate is an integer within fp16-exact range
  seen = 0
  for s in _specs(60):
    if s["mode"] != "int": continue
    vs, ref = fuzz._mirror(s, fuzz._feed_for(s), np.float64)
    for v in vs:
      assert np.array_equal(v, np.round(v)) and np.abs(v).max() <= fuzz.INT_MAX
    seen += 1
  assert seen >= 10

def test_float_mode_mirror_is_finite_and_bounded():
  for s in _specs(60):
    if s["mode"] != "float": continue
    vs, _ = fuzz._mirror(s, fuzz._feed_for(s), np.float64)
    assert all(np.isfinite(v).all() and np.abs(v).max() <= fuzz.FLOAT_MAX for v in vs)

def test_graph_shape_matches_mirror():
  for s in _specs(25):
    _, ref = fuzz._mirror(s, fuzz._feed_for(s), np.float64)
    assert tuple(fuzz.build_graph(s).shape) == ref.shape
    assert tuple(fuzz.build_graph(s, emi=True).shape) == ref.shape   # identity chain preserves shape

def test_boundary_dims_actually_appear():
  dims = set()
  for s in _specs(60): dims.update(s["input"])
  assert {1} & dims and ({15, 17, 31, 33} & dims)   # the off-by-one lattice is being sampled

def test_drop_node_rewires_indices():
  nodes = [{"op": "relu", "src": [0]}, {"op": "abs", "src": [1]}, {"op": "sign", "src": [2]}]
  assert fuzz._drop_node(nodes, 1) == [{"op": "relu", "src": [0]}, {"op": "sign", "src": [1]}]

def test_fingerprint_ignores_seed_but_not_mode():
  a = {"seed": 1, "mode": "int", "input": [2, 2], "nodes": [{"op": "relu", "src": [0]}]}
  b = {"seed": 9, "mode": "int", "input": [2, 2], "nodes": [{"op": "relu", "src": [0]}]}
  c = {**a, "mode": "float"}
  assert fuzz.fingerprint(a) == fuzz.fingerprint(b) != fuzz.fingerprint(c)


@requires_ane
def test_smoke_fuzz_agrees_on_device():
  # a handful of seeded graphs must dispatch and satisfy their mode's oracle (incl. the EMI probe)
  ran = ints = 0
  for s in _specs(60):
    fails = fuzz.run_case(s)
    assert not fails, f"seed {s['seed']} [{s['mode']}]: {fails}"
    ran += 1; ints += s["mode"] == "int"
    if ran >= 8 and ints >= 3: break
  assert ran >= 8 and ints >= 3
