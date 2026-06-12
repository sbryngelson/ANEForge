"""BLAS test corpus for aneforge: BLAS-1/2/3 operations expressed as aneforge
graphs, each compiled + run on the Apple Neural Engine and validated against a
numpy fp32 reference at fp16-appropriate tolerance.

The point of this corpus is twofold:

  1. CORRECTNESS — confirm that the classic dense-linear-algebra kernels lower to
     the ANE and match numpy within fp16 round-off.
  2. COST-MODEL DATA — every case carries a ``cost character`` tag in its
     ``category`` field, one of:

       floor      — tiny problem, dominated by per-dispatch fixed cost
       bandwidth  — memory-bound (BLAS-1: O(n) flops over O(n) bytes, AI~1)
       compute    — arithmetic-bound (BLAS-3 GEMM: O(n^3) over O(n^2) bytes)
       reduction  — a sum/contraction collapse (dot/asum/nrm2/Gram diagonal);
                    accuracy-sensitive because the accumulation length grows

     so the corpus doubles as labelled validation data for a cost model.

Run::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 tests/test_blas.py

The shared harness (tests/_corpus.py) does build->compile->run->compare. We reuse
its ``Case`` record, ``eval_case`` and the gate accounting, and add a BLAS-specific
runner that also prints the cost-character distribution and an N/M size summary.

API notes that shape these graphs (see aneforge/graph.py):
  - scalar multiply is ``x * float`` (the ``muls`` op); binary +,-,*,/, maximum,
    minimum need TWO graph Tensors. axpy/scal therefore fold the scalar into muls.
  - ``x @ W`` with a numpy ``W`` is a streamed weight matmul (W stored as W.T);
    ``x @ y`` with two Tensors is a bmm (activation x activation).
  - reductions (sum/amax/...) keep dims (the reduced axis becomes 1). dot is
    ``(x*y).sum(axis)``; asum is ``x.abs().sum(axis)``; nrm2 is
    ``(x*x).sum(axis).sqrt()`` (axis convention: the reduced axis -> size 1).
  - ger (rank-1 outer product) is a broadcast multiply u[M,1] * v[1,N] -> [M,N].
"""
from __future__ import annotations

import numpy as np

from _corpus import Case, eval_case  # noqa: E402

rng = np.random.default_rng(1234)


def f16(*shape, scale=1.0):
    return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float16)


def fp32(a):
    return np.asarray(a, np.float32)


# BLAS-1 : vector ops (bandwidth / floor bound; dot/asum/nrm2 are reductions)
# aneforge has no rank-1 vector type; we carry vectors as [1, N] row tensors so
# they fit the 2D activation layout. The cost character is unchanged.

def _axpy(N, tag):
    # y = alpha*x + y  ->  (x * alpha) + y   (muls then add of two Tensors)
    alpha = 1.7
    x = f16(1, N); y = f16(1, N)

    def build(xt, yt):
        return (xt * alpha) + yt

    def ref(xa, ya):
        return alpha * xa + ya

    return Case(f"axpy_n{N}", tag, build, ref, [x, y], tol=0.01)


def _scal(N, tag):
    # x = alpha*x  (pure muls; the cheapest BLAS-1, read+write one vector)
    alpha = 0.5
    x = f16(1, N)

    def build(xt):
        return xt * alpha

    def ref(xa):
        return alpha * xa

    return Case(f"scal_n{N}", tag, build, ref, [x], tol=0.01)


def _dot(N, tag, tol, abserr=None):
    # dot = sum(x*y)  (a reduction; accumulation length N). The output is a single
    # scalar from a sum of signed products, so it is cancellation-prone: the true
    # value can be near zero while the absolute fp16 error stays small. For such
    # near-zero scalars relerr (|err|/|value|) is the wrong metric, so small-N dots
    # validate on an absolute tolerance instead (the accumulation is wide/fp32 on
    # the ANE, so the abs error is governed by input fp16 round-off, ~1e-3).
    x = f16(1, N, scale=0.5); y = f16(1, N, scale=0.5)

    def build(xt, yt):
        return (xt * yt).sum(1)

    def ref(xa, ya):
        return (xa * ya).sum(1, keepdims=True)

    return Case(f"dot_n{N}", tag, build, ref, [x, y], tol=tol, abserr=abserr)


def _asum(N, tag, tol):
    # asum = sum(|x|)  (reduction; all-positive accumulands, length N)
    x = f16(1, N)

    def build(xt):
        return xt.abs().sum(1)

    def ref(xa):
        return np.abs(xa).sum(1, keepdims=True)

    return Case(f"asum_n{N}", tag, build, ref, [x], tol=tol)


def _nrm2(N, tag, tol):
    # nrm2 = sqrt(sum(x^2))  (reduction then sqrt)
    x = f16(1, N, scale=0.3)

    def build(xt):
        return (xt * xt).sum(1).sqrt()

    def ref(xa):
        return np.sqrt((xa * xa).sum(1, keepdims=True))

    return Case(f"nrm2_n{N}", tag, build, ref, [x], tol=tol)


def _l2norm(N, tag, tol):
    # the fused reduce_l2_norm op: x / sqrt(sum(x^2,axis)+eps). Bandwidth-ish
    # (O(n) work, O(n) output) but contains a reduction; tagged reduction.
    x = f16(1, N, scale=0.3)

    def build(xt):
        return xt.l2_norm(1)

    def ref(xa):
        return xa / np.sqrt((xa * xa).sum(1, keepdims=True) + 1e-12)

    return Case(f"l2norm_n{N}", tag, build, ref, [x], tol=tol)


# BLAS-2 : matrix-vector (mixed; small=floor, large=bandwidth)

def _gemv(M, K, tag):
    # y = A @ x, A[M,K], x[K].  Express as x_row[1,K] @ W with W=A.T (numpy weight
    # path stores W as W.T, so x_row @ A.T == (A @ x) as a [1,M] row).
    A = f16(M, K, scale=0.3); x = f16(1, K)
    AT = A.T.astype(np.float16)

    def build(xt):
        return xt @ AT                      # [1,K] @ [K,M] -> [1,M]

    def ref(xa):
        return (fp32(A) @ fp32(xa).reshape(K)).reshape(1, M)

    return Case(f"gemv_{M}x{K}", tag, build, ref, [x], tol=0.02)


def _ger(M, N, tag):
    # rank-1 outer product: u[M,1] * v[1,N] -> [M,N] (broadcast multiply).
    u = f16(M, 1, scale=0.5); v = f16(1, N, scale=0.5)

    def build(ut, vt):
        return ut * vt

    def ref(ua, va):
        return ua * va

    return Case(f"ger_{M}x{N}", tag, build, ref, [u, v], tol=0.01)


def _symv(M, K, tag):
    # symmetric matrix-vector: y = S @ x with S = S^T. Same lowering as gemv;
    # the symmetry is in the data, not the graph. Validates A@x on a symmetric A.
    base = f16(M, K, scale=0.3)
    S = ((fp32(base) @ fp32(base).T) / K).astype(np.float16)  # [M,M] symmetric
    x = f16(1, M)
    ST = S.T.astype(np.float16)  # == S, symmetric

    def build(xt):
        return xt @ ST                      # [1,M] @ [M,M] -> [1,M]

    def ref(xa):
        return (fp32(S) @ fp32(xa).reshape(M)).reshape(1, M)

    return Case(f"symv_{M}x{M}", tag, build, ref, [x], tol=0.02)


# BLAS-3 : matrix-matrix (compute bound at size; floor when tiny)

def _gemm(M, K, N, tag, tol, int8_ok=False):
    # C = A @ B (activation A times streamed weight B). The canonical compute case.
    scale = 1.0 / max(1.0, K ** 0.5)        # keep products O(1) so fp16 is well-conditioned
    A = f16(M, K, scale=1.0); B = f16(K, N, scale=scale)

    def build(at):
        return at @ B

    def ref(aa):
        return fp32(aa) @ fp32(B)

    return Case(f"gemm_{M}x{K}x{N}", tag, build, ref, [A], tol=tol, int8_ok=int8_ok)


def _gemm_aa(M, K, N, tag, tol):
    # activation x activation GEMM (bmm path), both operands are graph inputs.
    scale = 1.0 / max(1.0, K ** 0.5)
    A = f16(M, K, scale=1.0); B = f16(K, N, scale=scale)

    def build(at, bt):
        return at @ bt

    def ref(aa, ba):
        return fp32(aa) @ fp32(ba)

    return Case(f"gemm_aa_{M}x{K}x{N}", tag, build, ref, [A, B], tol=tol)


def _syrk(M, K, tag, tol):
    # C = A @ A^T (Gram matrix), A[M,K] activation. transpose + bmm.
    scale = 1.0 / max(1.0, K ** 0.5)
    A = f16(M, K, scale=1.0) * np.float16(scale)

    def build(at):
        return at @ at.transpose([1, 0])    # [M,K] @ [K,M] -> [M,M]

    def ref(aa):
        return fp32(aa) @ fp32(aa).T

    return Case(f"syrk_{M}x{K}", tag, build, ref, [A], tol=tol)


# case list
# Sizes span regimes. Reduction tolerance loosens with accumulation length
# (fp16 round-off on long sums); the cost-model knees referenced in the project
# notes are K up to 2048 for matmul.

CASES = [
    # ---- BLAS-1 ---------------------------------------------------------- #
    # scal/axpy: read+write a vector, pure bandwidth; tiny N = floor.
    _scal(16, "floor"),
    _scal(4096, "bandwidth"),
    _axpy(16, "floor"),
    _axpy(4096, "bandwidth"),
    _axpy(65536, "bandwidth"),
    # dot/asum/nrm2: reductions, tol grows with N.
    _dot(64, "floor", tol=0.01, abserr=5e-3),
    _dot(2048, "reduction", tol=0.02),
    _dot(16384, "reduction", tol=0.03),
    _asum(2048, "reduction", tol=0.02),
    _asum(16384, "reduction", tol=0.03),
    _nrm2(2048, "reduction", tol=0.02),
    _nrm2(16384, "reduction", tol=0.03),
    _l2norm(2048, "reduction", tol=0.02),

    # ---- BLAS-2 ---------------------------------------------------------- #
    _gemv(8, 8, "floor"),
    _gemv(256, 512, "bandwidth"),
    _gemv(512, 2048, "bandwidth"),
    _ger(8, 8, "floor"),
    _ger(256, 256, "bandwidth"),
    _symv(8, 8, "floor"),
    _symv(256, 256, "bandwidth"),

    # ---- BLAS-3 ---------------------------------------------------------- #
    _gemm(4, 4, 4, "floor", tol=0.01),
    _gemm(64, 256, 256, "compute", tol=0.02, int8_ok=True),
    _gemm(64, 2048, 512, "compute", tol=0.02, int8_ok=True),
    _gemm(128, 1024, 1024, "compute", tol=0.02),
    _gemm_aa(64, 256, 64, "compute", tol=0.02),
    _gemm_aa(128, 512, 128, "compute", tol=0.02),
    _syrk(8, 8, "floor", tol=0.01),
    _syrk(128, 512, "compute", tol=0.02),
    _syrk(256, 1024, "compute", tol=0.02),
]


# runner — mirrors _corpus.run_corpus but adds cost-character + N/M summaries

def run_blas(cases, verbose: bool = True):
    all_results = []
    relerrs = []
    by_tag: dict[str, list[float]] = {}
    if verbose:
        print(f"{'case':28s} {'tag':10s} {'var':5s} {'status':6s}  detail")
        print("-" * 84)
    for case in cases:
        for rec in eval_case(case):
            all_results.append(rec)
            line = (f"{rec['name']:28s} {rec['category']:10s} {rec['variant']:5s} "
                    f"{rec['status']:6s}  {rec['metric']}")
            if rec["err"]:
                line += f"  [{rec['err']}]"
            if verbose:
                print(line)
                if rec.get("traceback"):
                    print("    " + rec["traceback"].replace("\n", "\n    "))
            m = rec["metric"]
            if m.startswith("relerr "):
                try:
                    e = float(m.split()[1])
                    relerrs.append(e)
                    by_tag.setdefault(rec["category"], []).append(e)
                except ValueError:
                    pass

    n_pass = sum(r["status"] == "PASS" for r in all_results)
    n_xfail = sum(r["status"] == "XFAIL" for r in all_results)
    n_fail = sum(r["status"] == "FAIL" for r in all_results)
    n_err = sum(r["status"] == "ERROR" for r in all_results)
    n_xpass = sum(r["status"] == "XPASS" for r in all_results)
    total = len(all_results)
    red = n_fail + n_err + n_xpass

    print("\n" + "=" * 84)
    print(f"variants run: {total}   PASS {n_pass}   XFAIL {n_xfail}   "
          f"FAIL {n_fail}   ERROR {n_err}   XPASS {n_xpass}")
    if relerrs:
        print(f"relerr across {len(relerrs)} numeric variants: "
              f"min {min(relerrs):.2e}  median {np.median(relerrs):.2e}  max {max(relerrs):.2e}")

    print("\ncost-character distribution (relerr ranges per tag):")
    print(f"  {'tag':10s} {'cases':5s}  {'min':>9s} {'median':>9s} {'max':>9s}")
    # count cases (not variants) per tag, including non-numeric (exception) ones
    tag_case_count: dict[str, int] = {}
    for c in cases:
        tag_case_count[c.category] = tag_case_count.get(c.category, 0) + 1
    for tag in sorted(tag_case_count):
        es = by_tag.get(tag, [])
        if es:
            print(f"  {tag:10s} {tag_case_count[tag]:5d}  {min(es):9.2e} "
                  f"{np.median(es):9.2e} {max(es):9.2e}")
        else:
            print(f"  {tag:10s} {tag_case_count[tag]:5d}  {'(no numeric variants)':>29s}")

    print(f"\nGATE: {'GREEN' if red == 0 else 'RED'}  "
          f"({n_pass + n_xfail}/{total} green, {red} red)")
    return all_results, (0 if red == 0 else 1)


if __name__ == "__main__":
    import sys
    _, code = run_blas(CASES)
    sys.exit(code)
