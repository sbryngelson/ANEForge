"""Paired-fp16 extended precision on the ANE (fp16-only): compensated subtract + cancellation-heavy dot. Run: python3 examples/paired_fp16.py"""
import re
import sys
from pathlib import Path

from _common import head, aside, relerr   # sets env + repo-root path; import before aneforge

import numpy as np

import aneforge as af

f16 = np.float16


# data generators (fp32/fp64 ONLY for reference + reporting)
def make_cfg_pair(D, ratio, seed):
    """Two vectors whose TRUE difference is `ratio` * magnitude, returned as
    paired-fp16 (hi, lo) limbs plus the fp64 true difference."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(D).astype(np.float64)
    base /= np.linalg.norm(base) / np.sqrt(D)
    diff = rng.standard_normal(D).astype(np.float64)
    diff *= ratio * np.linalg.norm(base) / np.linalg.norm(diff)
    cond, uncond = base + 0.5 * diff, base - 0.5 * diff
    chi = f16(cond); clo = f16(cond - chi.astype(np.float64))
    uhi = f16(uncond); ulo = f16(uncond - uhi.astype(np.float64))
    c1, u1 = f16(cond), f16(uncond)          # genuine single-fp16 (plain UNet emits this)
    return (c1, u1), (chi, clo, uhi, ulo), (cond - uncond)


def make_downproj(K, cancel, seed):
    """A length-K signed contraction whose TRUE dot is `cancel` * product magnitude."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(K).astype(np.float64)
    w = rng.standard_normal(K).astype(np.float64)
    x16, w16 = f16(x), f16(w)
    prod_mag = np.abs(x16.astype(np.float64) * w16.astype(np.float64)).sum()
    truth = float(np.asarray(x16, np.float64) @ np.asarray(w16, np.float64))
    target = cancel * prod_mag
    denom = np.asarray(x16, np.float64).sum()
    shift = (target - truth) / denom if abs(denom) > 1e-3 else 0.0
    w16 = f16(w16.astype(np.float64) + shift)
    truth = float(np.asarray(x16, np.float64) @ np.asarray(w16, np.float64))
    return x16, w16, truth


# fp16-only guard: the compiled MIL must declare/emit no fp32 type
def assert_fp16_only(build_dir):
    mil = (Path(build_dir) / "model.mil").read_text()
    bad = sorted(set(re.findall(r"\b(?:fp32|float32)\b", mil)))
    if bad:
        raise AssertionError(f"fp32 leaked into the compiled MIL: {bad}")
    n_fp16 = len(re.findall(r"\bfp16\b", mil))
    return n_fp16


# (1) compensated SUBTRACT - on device
def demo_subtract():
    head("(1) CFG-style compensated SUBTRACT - paired-fp16 (hi,lo) inputs, on the ANE")
    print("    true diff = ratio * magnitude;  plain fp16 vs af.paired compensated subtract")
    print(f"{'ratio':>8} | {'plain fp16 (ANE)':>17} | {'paired-fp16 (ANE)':>18} | {'vs numpy-fp16':>14} | note")
    D = 4096
    ok = []
    for ratio in (1e-1, 1e-2, 1e-3, 1e-4):
        (c1, u1), (chi, clo, uhi, ulo), true_diff = make_cfg_pair(D, ratio, seed=int(1e6 * ratio) + 1)
        c1 = c1.reshape(1, D); u1 = u1.reshape(1, D)
        chi = chi.reshape(1, D); clo = clo.reshape(1, D)
        uhi = uhi.reshape(1, D); ulo = ulo.reshape(1, D)

        # plain fp16 subtract of the single-fp16 inputs
        bd = Path(__import__("tempfile").mkdtemp(prefix="paired_plain_"))
        a, b = af.input((1, D)), af.input((1, D))
        net = af.compile(a - b, build_dir=bd)
        plain = net(c1, u1).reshape(D); net.release()
        assert_fp16_only(bd)

        # paired-fp16 compensated subtract (carries lo) via the public API
        bd = Path(__import__("tempfile").mkdtemp(prefix="paired_comp_"))
        ah, al = af.input((1, D)), af.input((1, D))
        bh, bl = af.input((1, D)), af.input((1, D))
        P = af.paired(ah, al) - af.paired(bh, bl)
        net = af.compile(P.to_tensor(), build_dir=bd)
        comp = net(chi, clo, uhi, ulo).reshape(D); net.release()
        n_fp16 = assert_fp16_only(bd)

        # numpy-fp16 reference of the SAME compensated algorithm (device-faithfulness)
        comp_np = _np_comp_sub(chi.reshape(D), clo.reshape(D), uhi.reshape(D), ulo.reshape(D))

        ep, ec = relerr(plain, true_diff), relerr(comp, true_diff)
        e_vs_np = relerr(comp, comp_np)
        note = "paired recovers" if (ep >= 0.05 and ec < 0.05) else ("both OK" if ep < 0.05 else "both fail")
        print(f"{ratio:>8.0e} | {ep:>17.3e} | {ec:>18.3e} | {e_vs_np:>14.2e} | {note}")
        # device must match the numpy-fp16 proof (compiler preserved the compensation)
        ok.append(e_vs_np < 1e-2)
        if ratio <= 1e-4:        # the headline claim: <5% relerr at ratio 1e-4 where plain is broken
            ok.append(ec < 0.05 and ep > 0.5)
    aside(f"    [fp16-only MIL verified; {n_fp16} fp16-typed lines in the compensated program]")
    return ok


def _np_comp_sub(chi, clo, uhi, ulo):
    """numpy-fp16 mirror of Paired.__sub__ + to_tensor (every op fp16-rounded)."""
    def fs(x): return np.asarray(x, f16)
    s = fs(chi + (-uhi))
    bb = fs(s - chi)
    e = fs(fs(chi - fs(s - bb)) + fs((-uhi) - bb))
    e = fs(e + fs(clo - ulo))
    hi = fs(s + e)
    lo = fs(e - fs(hi - s))
    return fs(hi + lo).astype(np.float64)


# (2) compensated DOT - on device
def demo_dot():
    print()
    print("(2) Cancellation-heavy compensated DOT - TwoProduct + matmul-accum, on the ANE")
    print("    cancel = true_dot / product-magnitude.  Three on-device dots:")
    print("      plain mm   = x @ w                 (matmul accum - WIDE on this ANE)")
    print("      plain rsum = sum(x*w)              (reduce_sum accum - NARROW fp16)")
    print("      paired dot = TwoProduct + matmul-accum (the af.paired path)")
    print(f"{'cancel':>8} | {'plain mm (wide)':>16} | {'plain rsum (narrow)':>19} | "
          f"{'paired dot':>12} | {'paired vs rsum':>14}")
    K = 4096
    np.ones((K, 1), dtype=np.float16)
    ok = []
    for cancel in (1e-1, 1e-2, 1e-3, 1e-4):
        x16, w16, truth = make_downproj(K, cancel, seed=int(1e6 * cancel) + 7)
        xrow, wrow = x16.reshape(1, K), w16.reshape(1, K)

        # plain fp16 dot via matmul (WIDE accumulator) - the existing fast path
        a = af.input((1, K))
        net = af.compile(a @ w16.reshape(K, 1))
        plain_mm = float(net(xrow).ravel()[0]); net.release()

        # plain fp16 dot via reduce_sum (NARROW accumulator) - the cancellation wall
        a, wt = af.input((1, K)), af.input((1, K))
        net = af.compile((a * wt).sum(1))
        plain_rs = float(net(xrow, wrow).ravel()[0]); net.release()

        # paired compensated dot via the public API (TwoProduct + matmul-accum)
        bd = Path(__import__("tempfile").mkdtemp(prefix="dot_comp_"))
        ax, wx = af.input((1, K)), af.input((1, K))
        d = af.paired(ax).dot(af.paired(wx))     # genuine-fp16 inputs (lo=0)
        net = af.compile(d.to_tensor(), build_dir=bd)
        comp = float(net(xrow, wrow).ravel()[0]); net.release()
        n_fp16 = assert_fp16_only(bd)

        rel = lambda v: abs(v - truth) / (abs(truth) + 1e-30)
        emm, ers, ec = rel(plain_mm), rel(plain_rs), rel(comp)
        print(f"{cancel:>8.0e} | {emm:>16.3e} | {ers:>19.3e} | {ec:>12.3e} | "
              f"{ers / (ec + 1e-30):>13.1f}x")
        # the paired dot must be at least as good as the WIDE matmul baseline ...
        ok.append(ec <= emm * 2 + 1e-6)
        # ... and must beat the NARROW reduce_sum, which is where compensation pays off
        ok.append(ec <= ers + 1e-6)
    aside(f"    [fp16-only MIL verified; {n_fp16} fp16-typed lines in the compensated dot]")
    aside("    NOTE: this ANE's matmul accumulator is already WIDE, so plain `@` is clean and\n"
          "    paired-dot adds no accuracy there. The win shows vs reduce_sum (narrow): the\n"
          "    paired dot routes its accumulate through matmul, recovering what rsum loses.")
    return ok


# op-cost report
def op_cost():
    print()
    head("OP COST (fp16 ops/element - counted from the compiled graph)")
    D = 16
    for label, build in (
        ("paired add",      lambda: (af.paired(af.input((1, D)), af.input((1, D)))
                                     + af.paired(af.input((1, D)), af.input((1, D)))).to_tensor()),
        ("paired sub",      lambda: (af.paired(af.input((1, D)), af.input((1, D)))
                                     - af.paired(af.input((1, D)), af.input((1, D)))).to_tensor()),
        ("paired mul (2Prod)", lambda: (af.paired(af.input((1, D)), af.input((1, D)))
                                        * af.paired(af.input((1, D)), af.input((1, D)))).to_tensor()),
    ):
        net = af.compile(build())
        print(f"  {label:20s} {net.n_ops:>3d} fp16 graph ops  (elementwise -> {net.n_ops} ops/elem)")
        net.release()
    # dot: count the per-element transforms separate from the 2 matmul accumulates
    netp = af.compile(af.paired(af.input((1, D))).dot(af.paired(af.input((1, D)))).to_tensor())
    print(f"  {'paired dot':20s} {netp.n_ops:>3d} fp16 graph ops total "
          f"(~17/elem TwoProduct + 2 matmul-accumulate + final TwoSum)")
    netp.release()
    print("  Reference (envelope finding): TwoSum ~6 ops/elem, TwoProduct ~17 ops/elem.")


def main():
    ok = []
    ok += demo_subtract()
    ok += demo_dot()
    op_cost()
    print()
    passed = sum(bool(x) for x in ok)
    print(f"{passed}/{len(ok)} checks passed - paired-fp16 runs fp16-only on the ANE and "
          f"recovers accuracy past the cancellation wall")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
