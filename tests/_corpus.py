"""Shared harness for the aneforge correctness corpus (the green-gate infrastructure)."""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import aneforge
import aneforge as af


@dataclass
class Case:
  name: str
  category: str
  build_fn: Callable          # (*input_tensors) -> af.Tensor
  ref_fn: Callable            # (*input_arrays_fp32) -> np.ndarray
  inputs: list                # list of np.ndarray (fp16); shapes drive af.input
  tol: float = 0.02           # relerr tolerance (used unless exact/abserr set)
  exact: bool = False         # require bit-exact equality (index ops, shuffles)
  abserr: float | None = None # absolute-error tolerance (near-zero outputs)
  int8_ok: bool = False       # also run an int8-compiled variant
  int8_tol: float = 0.06      # looser tol for the int8 variant
  xfail: str = ""             # non-empty reason => expected failure (won't gate)
  input_shapes: list | None = field(default=None)  # override af.input shapes


def _build_graph(case: Case):
  """Construct the af.Tensor graph: one af.input per case input, fed to build_fn."""
  shapes = case.input_shapes or [a.shape for a in case.inputs]
  tensors = [af.input(tuple(s)) for s in shapes]
  return case.build_fn(*tensors)


def run_case(case: Case, int8: bool = False) -> np.ndarray:
  """Build -> compile -> run the case on the ANE; return the fp32 output array.

    This is the reuse hook for an optimizer diff: call with the same Case before
    and after a rewrite and compare the two returned arrays.
    """
  out_tensor = _build_graph(case)
  net = af.compile(out_tensor, int8=int8)
  ins = [np.asarray(a, np.float16) for a in case.inputs]
  return np.asarray(net(*ins), np.float32)


def compare(out: np.ndarray, ref: np.ndarray, case: Case, int8: bool = False):
  """Return (passed: bool, metric_str, tol_used, relerr) for one (out, ref) pair.

    ``relerr`` is the numeric relative error when a relerr metric applies, else None
    (carried on the record so runners need not re-parse the metric string)."""
  out = np.asarray(out, np.float32)
  ref = np.asarray(ref, np.float32)
  if out.shape != ref.shape: return False, f"shape {out.shape} != ref {ref.shape}", 0.0, None
  if case.exact:
    ok = bool(np.array_equal(out, ref))
    return ok, f"exact={ok}", 0.0, None
  if case.abserr is not None:
    err = float(np.abs(out - ref).max())
    return err < case.abserr, f"abserr {err:.5g}", case.abserr, None
  tol = case.int8_tol if int8 else case.tol
  relerr = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6))
  metric = f"relerr {relerr:.5g}"
  # Carry the *printed-precision* value so runners that aggregate it match the old
  # behaviour of re-parsing ``float(metric.split()[1])`` exactly.
  return relerr < tol, metric, tol, float(metric.split()[1])


def eval_case(case: Case):
  """Run one case (fp16, and int8 if requested). Return a list of result dicts,
    one per variant. Each dict: name, variant, status, metric, tol, err(optional).
    Status in {PASS, FAIL, XFAIL, XPASS, ERROR}."""
  results = []
  variants = [("fp16", False)] + ([("int8", True)] if case.int8_ok else [])
  ref = None
  for variant, int8 in variants:
    rec = {"name": case.name, "category": case.category, "variant": variant,
           "metric": "", "tol": 0.0, "err": "", "relerr": None}
    try:
      if ref is None:
        ref = case.ref_fn(*[np.asarray(a, np.float32) for a in case.inputs])
      out = run_case(case, int8=int8)
      passed, metric, tol, relerr = compare(out, ref, case, int8=int8)
      rec["metric"], rec["tol"], rec["relerr"] = metric, tol, relerr
      if case.xfail:
        rec["status"] = "XPASS" if passed else "XFAIL"
        rec["err"] = case.xfail
      else:
        rec["status"] = "PASS" if passed else "FAIL"
    except Exception as e:  # noqa: BLE001
      rec["err"] = f"{type(e).__name__}: {e}"
      rec["metric"] = "exception"
      rec["status"] = "XFAIL" if case.xfail else "ERROR"
      if case.xfail: rec["err"] = f"{case.xfail} | {rec['err']}"
      else: rec["traceback"] = traceback.format_exc()
    results.append(rec)
  return results


def _default_header():
  return f"{'case':36s} {'var':5s} {'status':6s}  detail"


def _default_row(rec):
  return f"{rec['name']:36s} {rec['variant']:5s} {rec['status']:6s}  {rec['metric']}"


def run_corpus(cases, verbose: bool = True, *, columns=None, verdict=None,
               sep_width: int = 78, annotate=None):
  """Run all cases, print a pass/fail table, return (results, exit_code).

    Gating rule: PASS and XFAIL are green; FAIL, ERROR, and XPASS are red.
    (XPASS = an xfail-marked case unexpectedly passed -> the marker is stale.)

    Customisation hooks (used by the per-domain runners so they need not re-implement
    the eval/count/gate skeleton):
      columns   : (header_str, row_fn(rec)->str) overriding the printed table layout.
      annotate  : annotate(case, rec) called per record before it is printed, to stamp
                  extra fields (e.g. cost/feasibility tags) onto the record.
      verdict   : verdict(all_results, relerrs) printed after the summary stats and
                  before the GATE line (a domain-specific findings block).
      sep_width : width of the '-'/'=' separator rules.
    """
  header_fn, row_fn = columns or (_default_header, _default_row)
  all_results = []
  relerrs = []
  if verbose:
    print(header_fn())
    print("-" * sep_width)
  for case in cases:
    for rec in eval_case(case):
      if annotate is not None: annotate(case, rec)
      all_results.append(rec)
      line = row_fn(rec)
      if rec["err"]: line += f"  [{rec['err']}]"
      if verbose:
        print(line)
        if rec.get("traceback"):
          print("    " + rec["traceback"].replace("\n", "\n    "))
      if rec.get("relerr") is not None:
        relerrs.append(rec["relerr"])

  n_pass = sum(r["status"] == "PASS" for r in all_results)
  n_xfail = sum(r["status"] == "XFAIL" for r in all_results)
  n_fail = sum(r["status"] == "FAIL" for r in all_results)
  n_err = sum(r["status"] == "ERROR" for r in all_results)
  n_xpass = sum(r["status"] == "XPASS" for r in all_results)
  total = len(all_results)
  red = n_fail + n_err + n_xpass

  print("\n" + "=" * sep_width)
  print(f"variants run: {total}   PASS {n_pass}   XFAIL {n_xfail}   "
        f"FAIL {n_fail}   ERROR {n_err}   XPASS {n_xpass}")
  if relerrs:
    print(f"relerr across {len(relerrs)} numeric variants: "
          f"min {min(relerrs):.2e}  median {np.median(relerrs):.2e}  max {max(relerrs):.2e}")
  if verdict is not None:
    verdict(all_results, relerrs)
  print(f"GATE: {'GREEN' if red == 0 else 'RED'}  "
        f"({n_pass + n_xfail}/{total} green, {red} red)")
  return all_results, (0 if red == 0 else 1)
