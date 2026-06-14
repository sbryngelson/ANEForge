"""ANE point-cloud demo - a PointNet++-style sample/group/geometry step.

Runs three private ANE point-cloud primitives that no mainstream Python
framework exposes on this hardware - all unentitled, no CoreML - wired into one
PointNet++-style set-abstraction step:

    1. FurthestPointSampling  (ane_fps_fused)            sample K centroids from N points
    2. RadiusSearch           (ane_radius_search_fused)  membership in an L2 ball per centroid
    3. CrossProduct           (ane_cross_product_fused)  surface normal from two edge vectors

These layers aren't in the aneforge frontend yet, so the demo calls the the reverse-engineering corpus
bridges directly. Each stage is validated against a numpy reference (exact for
the integer-discretised FPS/RadiusSearch, fp16-tolerance for the cross product).

Conventions worth stating honestly:
    * FPS is L2-only on this arch - the ``DistanceMetric`` netplist param is
      ignored, so the reference uses Euclidean distance regardless. Seed = point 0.
    * RadiusSearch returns a [points x centroids] uint8 membership matrix
      (1 = inside the L2 ball); we transpose to [centroids x points] for grouping.

    PYTHONPATH=. \\
        python3 examples/pointcloud.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np

from aneforge._bridges.ane_fps_fused import fps_fused, numpy_reference as fps_ref
from aneforge._bridges.ane_radius_search_fused import radius_search_fused, numpy_reference as rs_ref
from aneforge._bridges.ane_cross_product_fused import cross_product_fused, numpy_reference as xprod_ref


def main():
    rng = np.random.default_rng(0)
    N, K, radius = 64, 8, 4.0

    # Random 3D point cloud on an integer grid (keeps FPS argmax ties deterministic,
    # so the ANE's L2 greedy selection matches the numpy reference exactly).
    pts = rng.integers(0, 12, size=(N, 3)).astype(np.float16)

    # Stage 1: FurthestPointSampling on the ANE (sample K centroids)
    cent_ane = fps_fused(pts, K, metric="L2")
    cent_ref = fps_ref(pts, K, "L2")               # L2 reference == hardware (L2-only arch)
    fps_ok = np.array_equal(cent_ane, cent_ref)
    print(f"[1] FPS  (ANE)         N={N} -> K={K} centroids   exact_match={fps_ok}")

    # Stage 2: RadiusSearch on the ANE (neighborhood membership)
    memb_ane = radius_search_fused(pts, cent_ane, radius)   # [points x centroids] uint8
    memb_ref = rs_ref(pts, cent_ane, radius)
    rs_ok = np.array_equal(memb_ane, memb_ref)
    grp = memb_ane.T                                        # [centroids x points]
    nbrs = grp.sum(1)
    print(f"[2] RadiusSearch (ANE) r={radius}  membership exact_match={rs_ok}  "
          f"neighbors/centroid={nbrs.tolist()}")

    # Stage 3: CrossProduct on the ANE (surface normal per centroid)
    # For each centroid, take two neighbor edge vectors and compute their cross
    # product on the ANE -> an (unnormalised) surface normal.
    max_err, n_normals = 0.0, 0
    for j in range(K):
        members = np.flatnonzero(grp[j])
        members = members[members != _nearest_index(pts, cent_ane[j])]  # drop centroid itself
        if len(members) < 2:
            continue
        e0 = (pts[members[0]].astype(np.float32) - cent_ane[j].astype(np.float32))
        e1 = (pts[members[1]].astype(np.float32) - cent_ane[j].astype(np.float32))
        normal_ane = cross_product_fused(e0, e1).astype(np.float32)
        normal_ref = xprod_ref(e0, e1).astype(np.float32)
        max_err = max(max_err, float(np.max(np.abs(normal_ane - normal_ref))))
        n_normals += 1
    xprod_ok = n_normals > 0 and max_err < 1e-2
    print(f"[3] CrossProduct (ANE) normals computed={n_normals}  "
          f"max_abs_err={max_err:.4e}  ok={xprod_ok}")

    ok = fps_ok and rs_ok and xprod_ok
    print(f"\npipeline FPS->RadiusSearch->CrossProduct all on ANE: "
          f"{'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def _nearest_index(pts, c):
    """Index of the cloud point coincident with centroid ``c`` (FPS centroids are
    drawn from the cloud, so this is the centroid's own point)."""
    return int(np.argmin(np.abs(pts.astype(np.float32) - c.astype(np.float32)).sum(1)))


if __name__ == "__main__":
    sys.exit(main())
