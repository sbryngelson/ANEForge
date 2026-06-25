"""Native ANE ranking ops via netplist bridges: sort / argmax / topk. Run: python3 examples/native_ranking.py"""
import sys

from _common import report   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def main():
    rng = np.random.default_rng(0)
    ok = []

    print("native ranking ops (netplist bridge - graph cut, like af.sdpa):")

    # sort: per-row ascending values over Width
    xs = rng.standard_normal((4, 8)).astype(np.float16)
    net = af.compile(af.sort(af.input((4, 8))))
    ok.append(report("sort(values)", net(xs), np.sort(xs.astype(np.float32), axis=1)))

    # sort: argsort indices (descending)
    xi = rng.standard_normal((3, 6)).astype(np.float16)
    net = af.compile(af.sort(af.input((3, 6)), descending=True, return_indices=True))
    ok.append(report("sort(indices)", net(xi), np.argsort(-xi.astype(np.float32), axis=1),
                     exact=True))

    # argmax over the last axis of a [C, W] tensor
    xa = rng.standard_normal((4, 8)).astype(np.float16)
    net = af.compile(af.input((4, 8)).argmax(axis=-1))
    ok.append(report("argmax", net(xa), xa.astype(np.float32).argmax(1, keepdims=True),
                     exact=True))

    # topk over the last axis (k=2; k in {3,4} is arch-gated)
    xt = rng.standard_normal((4, 8)).astype(np.float16)
    net = af.compile(af.topk(af.input((4, 8)), 2))
    ok.append(report("topk(k=2)", net(xt), np.sort(xt.astype(np.float32), 1)[:, ::-1][:, :2]))

    print(f"\n{sum(ok)}/{len(ok)} native ranking ops correct on the ANE")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
