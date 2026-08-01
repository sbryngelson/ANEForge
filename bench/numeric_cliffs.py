#!/usr/bin/env python3
"""Numeric-cliff rooflines: the per-silicon CORRECTNESS ceilings ANEForge has
characterized (issue #115 and docs/cross-chip.md), measured as committed sweeps.

This complements device_saturation_sweep.py, which measures COMPUTE-throughput
falloff (how fast a big GEMM runs). This measures the opposite failure: the
magnitudes at which fp16 math on the engine silently returns the WRONG answer -
inf on a mathematically finite result, or a rounded integer sum. Each cliff is
potentially silicon-dependent, so the sweep is exactly the cross-hardware
question the roofline paper asks: does the M5 cliff sit where the M2 cliff does?

Three cliffs, all at opt=0 (the untouched-lowering contract):
  1. matmul_saturation - a matmul output flips finite->inf near fp16_max/2
     (~32752), for every contraction size K (issue #115, item 1).
  2. slice_saturation  - a width-offset slice routes through the pre-A16 Q.4 x16
     crop-DMA that clamps |value|>4094 to +/-inf (docs/cross-chip.md). Documented
     EXACT on A16/M5, so a modern part should report NO cliff - the sweep proves it.
  3. reduce_exactness  - a reshape->reduce_sum of small integers stops being
     bit-exact crossing the fp16 integer-spacing boundary at 2048 (fuzzer finding
     e5afb32f: a row summing to 2048 read 2050).

Run: PYTHONPATH=. python3 bench/numeric_cliffs.py
"""
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

import numpy as np

import aneforge as af


def _dispatch(out, feed):
    """Compile at opt=0 (untouched lowering) and dispatch a single feed to the ANE."""
    net = af.compile(out, opt=0, _check_precision=False)
    y = np.asarray(net(feed), np.float32)
    net.release()
    return y


def _bisect_cliff(finite_at, lo, hi, tol=2.0):
    """Largest magnitude in [lo, hi] that stays finite.

    Assumes finite at lo and non-finite at hi. Returns the boundary, or None with
    a note when the range does not bracket a cliff (either end violates the
    assumption) - a None here is itself a result (e.g. 'M5 has no slice cliff').
    """
    if not finite_at(lo):
        return {"cliff": None, "note": f"already non-finite at lo={lo:g}"}
    if finite_at(hi):
        return {"cliff": None, "note": f"still finite at hi={hi:g} (no cliff in range)"}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if finite_at(mid):
            lo = mid
        else:
            hi = mid
    return {"cliff": round(lo, 1)}


# --- 1. matmul output saturation (#115) --------------------------------------
def matmul_saturation(ks=(4, 8, 32, 64, 128)):
    """For each K, bisect the true output magnitude where [1,K]@[K,16] flips to inf."""
    rows = []
    for K in ks:
        W = np.ones((K, 16), np.float16)

        def finite_at(total, K=K, W=W):
            x = np.full((1, K), total / K, np.float16)  # K equal terms summing to `total`
            return np.isfinite(_dispatch(af.input(x.shape) @ W, x)).all()

        r = _bisect_cliff(finite_at, 30000.0, 40000.0)
        r.update({"K": K, "expected": 32752})
        rows.append(r)
    return {"metric": "output_magnitude_inf_cliff", "expected": 32752, "by_K": rows}


# --- 2. slice-x16 crop-DMA saturation (docs/cross-chip.md) --------------------
def slice_saturation():
    """A width-offset crop (nonzero last-axis begin) routes through the Q.4 x16
    crop-DMA. Pre-A16 clamps |value|>4094; A16/M5 is exact -> expect NO cliff."""
    def finite_at(v):
        x = np.zeros((1, 1, 1, 8), np.float16)
        x[0, 0, 0, 4] = v                       # large value inside the kept region
        out = af.crop(af.input(x.shape), 0, 0, 1, 0)  # left=1: nonzero width begin-offset
        return np.isfinite(_dispatch(out, x)).all()

    r = _bisect_cliff(finite_at, 3000.0, 60000.0)
    r.update({"expected_pre_a16": 4094,
              "interpretation": "None == exact (A16/M5); a number == the pre-A16 clamp"})
    return {"metric": "sliced_value_inf_cliff", **r}


# --- 3. integer reduce_sum bit-exactness (fuzzer finding e5afb32f) ------------
def reduce_exactness(n=8, span=(2040, 2080)):
    """reshape->reduce_sum of small integers; find where the sum stops being bit-exact.

    Above the fp16 integer-spacing boundary the representable grid coarsens (spacing
    2 past 2048), so the honest result is the largest total below which EVERY swept
    sum is exact - i.e. one less than the first failure - plus the first failing case
    for the record. (Beyond the boundary some sums land back on the grid and read
    exact again; that is the coarse grid, not restored precision, so it is not the
    boundary.)"""
    first_wrong = None
    for total in range(*span):
        base, rem = divmod(total, n)
        row = np.full(n, base, np.float16)
        row[:rem] += 1                          # `n` small integers summing to `total`
        got = float(_dispatch(af.input((n,)).reshape(1, n).sum([1]), row).reshape(-1)[0])
        if got != total and first_wrong is None:
            first_wrong = {"target": total, "got": got}
            break
    boundary = (first_wrong["target"] - 1) if first_wrong else None
    return {"metric": "last_all_exact_integer_row_sum",
            "expected_boundary": 2048, "n_terms": n,
            "last_all_exact": boundary, "first_wrong": first_wrong}


def run():
    """Return the full numeric-cliff plane as a structured dict (importable by the suite)."""
    return {
        "plane": "numeric",
        "matmul_saturation": matmul_saturation(),
        "slice_saturation": slice_saturation(),
        "reduce_exactness": reduce_exactness(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()
    result = run()
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else REPO / "bench" / "results" / "numeric_cliffs_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
