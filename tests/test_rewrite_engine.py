# tests/test_rewrite_engine.py
import aneforge as af
from aneforge.graph import Tensor
from aneforge._rewrite import Rule, graph_rewrite

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
