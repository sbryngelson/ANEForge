"""Every norm op's affine must not sit between a reshape pair (#202).

#201 fixed layer_norm applying a non-uniform [1,D,1,1] const between the reshape-in
and reshape-out of its rank-4 body, which ANECCompile rejects from D=1024 up (#162).
The other norm ops are safe today only because nothing stops a refactor from
reintroducing that shape. This is the shared guardrail: for each norm op, lower the
graph and assert that no affine const mul is sandwiched between a reshape pair.
channel_layer_norm and group_norm legitimately use [1,C,1,1] consts, so the check is
dataflow, not shape-based: the affine mul's input must not descend from a reshape
whose output the mul's result (transitively) feeds.
"""
import re

import numpy as np
import pytest

import aneforge as af
from aneforge._compile import _lower_fused_to_dir

R, D = 8, 1024
C, S = 1024, 16
H, W = 4, 4

_AFFINE = re.compile(r"tensor<fp16, \[([^\]]+)\]> (\w+_[gb]) = const\(")
_OP = re.compile(r"(\w+)\(")
_ARG = re.compile(r"[a-z_]\w* ?= ?(\w+)")


def _non_uniform(dim, seed=13):
  """Non-uniform affine, the case that tripped the compile (#162)."""
  rng = np.random.default_rng(seed)
  g = (rng.standard_normal(dim).astype(np.float32) * 0.1 + 1.0).astype(np.float16)
  b = (rng.standard_normal(dim).astype(np.float32) * 0.1).astype(np.float16)
  return g, b


_NORM_CASES = [
  ("layer_norm", lambda: af.input((R, D)).layer_norm(*_non_uniform(D))),
  ("rms_norm", lambda: af.input((R, D)).rms_norm(_non_uniform(D)[0])),
  ("channel_layer_norm", lambda: af.input((2, C, 1, S)).channel_layer_norm(*_non_uniform(C))),
  ("group_norm", lambda: af.input((1, C, H, W)).group_norm(*_non_uniform(C), num_groups=8)),
]


def _mil(tmp_path, graph):
  d = _lower_fused_to_dir(graph, build_dir=str(tmp_path / "prog"))
  return (d / "model.mil").read_text()


def _graph(mil):
  """name -> (op, inputs) from emitted lines; unresolved identifiers are dropped."""
  defs, uses = {}, {}
  for line in mil.splitlines():
    line = line.split("[name", 1)[0]
    if " = " not in line:
      continue
    lhs, rhs = line.split(" = ", 1)
    m = _OP.match(rhs)
    if not m:
      continue
    name = lhs.strip().split()[-1]
    args = rhs[m.end():].rstrip(");")
    inputs = set(_ARG.findall(args)) & set(defs)
    defs[name] = (m.group(1), inputs)
    for i in inputs:
      uses.setdefault(i, set()).add(name)
  return defs, uses


def _reach(edges, start):
  seen, stack = set(), [start]
  while stack:
    cur = stack.pop()
    if cur in seen:
      continue
    seen.add(cur)
    stack.extend(edges.get(cur, ()))
  return seen


def _sandwiched_affine_muls(mil):
  """Affine muls whose input descends from a reshape and whose output feeds another reshape."""
  defs, uses = _graph(mil)
  deps = {n: ins for n, (_, ins) in defs.items()}
  reshape_names = {n for n, (op, _) in defs.items() if op == "reshape"}
  reshape_inputs = {
    i for n, (op, ins) in defs.items() if op == "reshape" for i in ins
  }
  bad = []
  for shape, c in _AFFINE.findall(mil):
    for m, (op, ins) in defs.items():
      if op != "mul" or c not in ins:
        continue
      x = next((i for i in ins if i != c), None)
      if x is None:
        continue
      if _reach(deps, x) & reshape_names and _reach(uses, m) & reshape_inputs:
        bad.append(f"affine {c} shape [{shape}] via mul {m} = mul({', '.join(sorted(ins))})")
  return bad


@pytest.mark.parametrize("op,make", _NORM_CASES, ids=[c[0] for c in _NORM_CASES])
def test_affine_not_sandwiched_between_reshapes(tmp_path, op, make):
  mil = _mil(tmp_path, make())
  bad = _sandwiched_affine_muls(mil)
  assert not bad, (
    f"{op}: {bad[0]} sits between a reshape pair; a non-uniform affine const in that "
    f"position fails ANECCompile from D=1024 up (#162, fixed in #201)"
  )
