# tests/test_rewrite_engine.py
import aneforge as af
from aneforge.graph import Tensor
from aneforge._rewrite import Rule, graph_rewrite, NUMERIC_RULES, graph_rewrite as _gr

def _noop_rules(): return []

def test_noop_returns_same_object():
  x = af.input((1, 4)); y = (x * 2.0).adds(1.0)
  assert graph_rewrite(y, _noop_rules()) is y          # opt=0 byte-identity premise

def test_diamond_stays_diamond():
  x = af.input((1, 4)); a = x * 2.0
  y = a + a                                             # a shared by both add operands
  rules = [Rule("mul3", "lossless", lambda t: t.op == "muls",
                lambda t: Tensor(t.shape, "muls", t.srcs, {"k": 3.0}))]
  out = graph_rewrite(y, rules)
  assert out.srcs[0] is out.srcs[1]                     # rewritten once, still shared

def test_select_restricts_to_original_ids():
  x = af.input((1, 4)); a = x * 2.0; b = a * 2.0       # two muls nodes
  rules = [Rule("tag", "lossless", lambda t: t.op == "muls",
                lambda t: Tensor(t.shape, "muls", t.srcs, {**t.attrs, "tagged": True}))]
  out = graph_rewrite(b, rules, select={id(a)})         # only the inner muls eligible
  assert out.attrs.get("tagged") is None                # outer untouched
  assert out.srcs[0].attrs.get("tagged") is True        # inner tagged

def test_multi_rule_one_pass_equals_sequential():
  x = af.input((1, 4)); y = (x * 2.0).adds(1.0)
  r_mul = Rule("m", "lossless", lambda t: t.op == "muls",
               lambda t: Tensor(t.shape, "muls", t.srcs, {"k": t.attrs["k"] + 1}))
  r_add = Rule("a", "lossless", lambda t: t.op == "adds",
               lambda t: Tensor(t.shape, "adds", t.srcs, {"k": t.attrs["k"] + 1}))
  one_pass = graph_rewrite(y, [r_mul, r_add])
  seq = graph_rewrite(graph_rewrite(y, [r_mul]), [r_add])
  assert one_pass.attrs["k"] == seq.attrs["k"] == 2.0
  assert one_pass.srcs[0].attrs["k"] == seq.srcs[0].attrs["k"] == 3.0


# -- back-compat wrapper tests (Task 2) -------------------------------------- #
import numpy as np
from aneforge._rewrite import (sdpa_to_decomposed, reduce_sum_to_matmul,
                               set_node_int8, decompose_bridge)
from aneforge._compile import _topo

def _ops(out): return [t.op for t in _topo(out)]

def test_reduce_sum_to_matmul_wrapper_decomposes():
  x = af.input((1, 8)); y = x.sum(-1)                  # reduce_sum, single last axis (keepdims implicit)
  out = reduce_sum_to_matmul(y)
  assert "reduce_sum" not in _ops(out) and "matmul" in _ops(out)

def test_set_node_int8_wrapper_tags_original_id():
  x = af.input((1, 8)); W = np.ones((8, 4), np.float16); y = x @ W
  order = _topo(y); mm = next(t for t in order if t.op == "matmul")
  out = set_node_int8(y, {id(mm)})
  assert any(t.op == "matmul" and t.attrs.get("int8") for t in _topo(out))

def test_noop_wrappers_keep_identity():
  x = af.input((1, 8)); y = (x * 2.0).adds(1.0)
  assert reduce_sum_to_matmul(y) is y                  # no reduce_sum -> same object
  assert set_node_int8(y, set()) is y

def test_sdpa_to_decomposed_noop_identity():
  x = af.input((1, 8)); y = (x * 2.0).adds(1.0)
  assert "sdpa" not in _ops(y)
  assert sdpa_to_decomposed(y) is y                    # no sdpa -> same object

def test_decompose_bridge_empty_set_identity():
  x = af.input((1, 8)); y = (x * 2.0).adds(1.0)
  assert decompose_bridge(y, set()) is y               # empty select -> same object


# -- Task 3: CANON_RULES + canonicalize -------------------------------------- #
from aneforge._rewrite import canonicalize

def test_canon_drops_reshape_to_same_shape():
  x = af.input((2, 3)); y = x.reshape(2, 3)            # no-op reshape
  out = canonicalize(y * 1.0)                          # muls(1.0) also a no-op
  assert "reshape" not in _ops(out) and "muls" not in _ops(out)

def test_canon_drops_mul1_and_add0():
  x = af.input((1, 4)); y = (x * 1.0).adds(0.0)
  out = canonicalize(y)
  assert out is x                                       # both no-ops removed -> x itself

def test_canon_collapses_transpose_inverse():
  x = af.input((2, 3, 4)); y = x.transpose((1, 0, 2)).transpose((1, 0, 2))  # self-inverse
  out = canonicalize(y)
  assert "transpose" not in _ops(out)

def test_canon_collapses_double_logical_not():
  import numpy as np
  x = af.input((1, 4))
  zero = Tensor((1, 4), "const_array", [], {"value": np.zeros((1, 4), dtype=np.float16)})
  y = x.greater(zero).logical_not().logical_not()
  out = canonicalize(y)
  assert _ops(out).count("logical_not") == 0

def test_canon_noop_keeps_identity():
  x = af.input((1, 4)); y = (x * 2.0).adds(1.0)        # nothing redundant
  assert canonicalize(y) is y

def test_graph_rewrite_deep_graph_no_recursion():
  # 4000-deep chain >> recursion limit: the walk must stay iterative (no RecursionError)
  x = af.input((10, 1)); y = x
  for _ in range(4000): y = y.adds(0.0)                # deep chain of no-op adds
  out = canonicalize(y)                                # must not RecursionError
  assert out is x                                      # all 4000 no-op adds dropped to x

def test_canon_drops_cast_on_compute_tensor():
  x = af.input((1, 4)); c = x * 2.0                    # a compute tensor (not an input)
  y = Tensor(c.shape, "cast", [c], {"dtype": "fp16"})  # redundant fp16->fp16 cast
  out = canonicalize(y)
  assert "cast" not in _ops(out)

def test_canon_preserves_uint8_input_cast():
  y = af.image_input((1, 3, 4, 4))                     # uint8 input -> cast(fp16) -> dequant
  assert "cast" in _ops(y) and "input" in _ops(y)     # the dequant cast is present
  out = canonicalize(y)
  assert "cast" in _ops(out)                           # NOT stripped (srcs[0].op == "input")


# -- Task 4: NUMERIC_RULES -- scalar-chain folding -------------------------- #
def _apply(out, names):
  rules = [r for r in NUMERIC_RULES if r.name in names]
  while (nxt := _gr(out, rules)) is not out: out = nxt
  return out

def test_muls_chain_folds():
  x = af.input((1, 4)); y = (x * 2.0) * 3.0
  out = _apply(y, {"muls_chain"})
  assert _ops(out).count("muls") == 1 and out.attrs["k"] == 6.0

def test_adds_chain_folds():
  x = af.input((1, 4)); y = x.adds(2.0).adds(5.0)
  out = _apply(y, {"adds_chain"})
  assert _ops(out).count("adds") == 1 and out.attrs["k"] == 7.0

def test_scalar_fold_matches_fp32_reference():
  rng = np.random.default_rng(0); xv = rng.standard_normal((1, 16)).astype(np.float16)
  x = af.input((1, 16)); y = (x * 0.5) * 4.0
  folded = _apply(y, {"muls_chain"})
  ref = (xv.astype(np.float32) * 0.5) * 4.0
  net = af.compile(folded, opt=0); got = net(xv)
  assert np.allclose(got.astype(np.float32), ref, atol=1e-2)


# -- Task 5: constant folding ------------------------------------------------ #
from aneforge._rewrite import _const_subgraph

def _const(v): return Tensor(v.shape, "const_array", [], {"value": np.asarray(v, np.float16)})

def test_const_subgraph_folds_pure_constants():
  a = _const(np.full((1, 4), 2.0)); y = (a * 3.0).adds(1.0)   # all-constant cone
  val = _const_subgraph(y)
  assert val is not None and np.allclose(val.astype(np.float32), 7.0)

def test_const_subgraph_returns_none_with_input():
  x = af.input((1, 4)); y = x * 2.0                            # depends on a runtime input
  assert _const_subgraph(y) is None

def test_const_subgraph_respects_size_bound():
  a = _const(np.ones((1, 100), np.float16))
  assert _const_subgraph(a, max_elems=10) is None             # too big to fold (leaf early-exit)
  y = a.adds(1.0)                                             # multi-op cone with a large root
  assert _const_subgraph(y, max_elems=10) is None             # bound blocks a real computation

def test_const_fold_rule_replaces_with_const_array():
  a = _const(np.full((1, 4), 2.0)); y = a.adds(5.0)
  out = _apply(y, {"const_fold"})                              # _apply from Task 4
  assert out.op == "const_array" and np.allclose(out.attrs["value"].astype(np.float32), 7.0)


# -- Task 6: tuner integration — numeric passes as gated variant axes --------- #
from aneforge import _optimize as O

def test_apply_variant_single_pass_composes_axes():
  a = Tensor((1, 4), "const_array", [], {"value": np.full((1, 4), 2.0, np.float16)})
  x = af.input((1, 4)); y = (x * 2.0) * 3.0 + a.adds(1.0)     # scalar chain + foldable const cone
  order = _topo(y)
  cfg = {"int8": False, "decomp": [], "lossy": True,
         "scalarfold": [i for i, t in enumerate(order) if t.op == "muls" and t.srcs[0].op == "muls"],
         "constfold": O._constfold_candidates(y)}
  new_out, _ = O._apply_variant(y, cfg)
  ops = _ops(new_out)
  assert ops.count("muls") == 1                                # 2*3 folded
  assert "const_array" in ops                                  # const cone folded

def test_constfold_candidates_skip_inputs():
  x = af.input((1, 4)); y = x * 2.0
  assert O._constfold_candidates(y) == []                      # nothing constant

def test_apply_variant_noop_cfg_keeps_baseline():
  x = af.input((1, 4)); y = (x * 2.0).adds(1.0)
  new_out, int8 = O._apply_variant(y, {"int8": False, "decomp": [], "lossy": False})
  assert new_out is y and int8 is False


# -- Task 7: README-example canon-equivalence (on-device) -------------------- #
def test_readme_example_canon_equivalent():
  rng = np.random.default_rng(1)
  img = rng.integers(0, 255, (1, 3, 32, 32)).astype(np.uint8).astype(np.float16)
  W = rng.standard_normal((8, 3, 3, 3)).astype(np.float16)
  x = af.input((1, 3, 32, 32)); y = af.conv(x, W, pad=1).relu().mean((2, 3))
  base = af.compile(y, opt=0)(img)                     # un-canonicalized baseline
  opt = af.compile(y, opt=1)(img)                      # canonicalized + cost-model pick
  cos = float((base.ravel() @ opt.ravel()) /
              (np.linalg.norm(base) * np.linalg.norm(opt) + 1e-12))
  assert cos > 0.9999
