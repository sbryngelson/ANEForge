"""Native ANE geometry/matching ops via netplist bridges: cross_product / cross_correlation / cost_volume / fps / radius_search. Run: python3 examples/native_geometry.py"""
import sys

from _common import report   # sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


def main():
    rng = np.random.default_rng(0)
    ok = []

    print("native geometry/matching ops (netplist bridge - graph cut, like af.sdpa):")

    # cross_product: cross(a, b) of two 3-vectors
    a = rng.standard_normal(3).astype(np.float16)
    b = rng.standard_normal(3).astype(np.float16)
    net = af.compile(af.cross_product(af.input((3,)), af.input((3,))))
    ok.append(report("cross_product", net(a, b),
                     np.cross(a.astype(np.float32), b.astype(np.float32))))

    # cross_correlation: valid (no-flip) template matching
    xc = rng.standard_normal((5, 5)).astype(np.float16)
    tmpl = rng.standard_normal((3, 3)).astype(np.float16)
    net = af.compile(af.cross_correlation(af.input((5, 5)), af.input((3, 3))))
    H, W = xc.shape; Th, Tw = tmpl.shape
    ref = np.zeros((H - Th + 1, W - Tw + 1), np.float32)
    xf, tf = xc.astype(np.float32), tmpl.astype(np.float32)
    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):
            ref[i, j] = (xf[i:i + Th, j:j + Tw] * tf).sum()
    ok.append(report("cross_correlation", net(xc, tmpl), ref))

    # cost_volume: L1 stereo matching cost, disparity range R
    Wa, R = 4, 2
    aux = rng.standard_normal(Wa).astype(np.float16)
    refrow = rng.standard_normal(Wa + R).astype(np.float16)
    net = af.compile(af.cost_volume(af.input((Wa,)), af.input((Wa + R,)), R))
    auxf, rf = aux.astype(np.float32), refrow.astype(np.float32)
    ref = np.stack([np.abs(auxf - rf[d:d + Wa]) for d in range(R + 1)], axis=0)
    ok.append(report("cost_volume", net(aux, refrow), ref))

    # fps: L2 furthest-point sampling, seeded at index 0
    N, k = 10, 4
    P = rng.integers(0, 12, size=(N, 3)).astype(np.float16); P[:, 2] = 0
    net = af.compile(af.fps(af.input((N, 3)), k))
    Pf = P.astype(np.float32); sel = [0]; d = np.full(N, np.inf)
    for _ in range(1, k):
        dist = np.sqrt(((Pf - Pf[sel[-1]]) ** 2).sum(1))
        d = np.minimum(d, dist); sel.append(int(np.argmax(d)))
    ok.append(report("fps (L2)", net(P), Pf[sel], exact=True))

    # radius_search: L2 ball-query membership matrix
    Np, Nc, radius = 6, 3, 3.0
    Pr = rng.integers(0, 6, size=(Np, 3)).astype(np.float16); Pr[:, 2] = 0
    Cr = rng.integers(0, 6, size=(Nc, 3)).astype(np.float16); Cr[:, 2] = 0
    net = af.compile(af.radius_search(af.input((Np, 3)), af.input((Nc, 3)), radius))
    d2 = np.sqrt(((Pr.astype(np.float32)[:, None, :] - Cr.astype(np.float32)[None, :, :]) ** 2).sum(-1))
    ok.append(report("radius_search", net(Pr, Cr), (d2 <= radius + 1e-3).astype(np.float32),
                     exact=True))

    print(f"\n{sum(ok)}/{len(ok)} native geometry ops correct on the ANE")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
