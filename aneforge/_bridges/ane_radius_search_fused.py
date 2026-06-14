"""Native ANE `RadiusSearch` via a hand-authored ANECIR netplist (Path A).

RadiusSearch is point-cloud neighbor search: for each query centroid, find the
input points within a given `Radius` (an L2 ball query).  Entry point
`radius_search_fused(points, centroids, radius)` returns an
`(N_points, N_centroids)` uint8 membership matrix (entry `[i, j]` is 1 iff
`points[i]` is within L2 `radius` of `centroids[j]`).

Schema (accepted unit dictionary)::

    {"Type": "RadiusSearch",
     "Bottom": ["centroids", "points"],
     "InputType": ["Float16", "Float16"],
     "OutputChannels": 1,
     "OutputType": "Float16",
     "Params": {"Radius": r}}
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def radius_search_fused(points: np.ndarray, centroids: np.ndarray, radius: float) -> np.ndarray:
    """Return an `(N_points, N_centroids)` uint8 membership matrix from the ANE.

    Entry `[i, j]` is 1 iff `points[i]` is within L2 `radius` of
    `centroids[j]`.
    """
    from . import _netplist as g

    points = np.asarray(points, dtype=np.float16)
    centroids = np.asarray(centroids, dtype=np.float16)
    if points.ndim != 2 or points.shape[1] != 3 or centroids.shape[1] != 3:
        raise ValueError("points and centroids must be (N, 3)")
    Np, Nc = points.shape[0], centroids.shape[0]
    pts_cw = points.T.reshape(1, 3, 1, Np)
    cent_cw = centroids.T.reshape(1, 3, 1, Nc)

    d = Path(tempfile.mkdtemp(prefix="ane_rs_"))
    g.write_model(
        "radius_search", d, width=Np, template_width=Nc, radius=float(radius), output_channels=1,
    )
    (d / "points.f16").write_bytes(np.ascontiguousarray(pts_cw, dtype=np.float16).tobytes())
    (d / "centroids.f16").write_bytes(np.ascontiguousarray(cent_cw, dtype=np.float16).tobytes())
    cmd = [
        str(g.ensure_invoker("sdpa_invoker")), "--net-plist", str(d / "net.plist"), "--weights", str(d / "weights.0"),
        "--input", f"centroids={d / 'centroids.f16'}", "--input", f"points={d / 'points.f16'}",
        "--output", f"y={d / 'out.f16'}", "--repeats", "1", "--warmup", "0",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"radius_search invoker failed:\n{p.stderr}")
    raw = (d / "out.f16").read_bytes()
    # output is [Np rows x Nc cols], 2 bytes per W-cell; low byte = membership flag
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(Np, Nc * 2)
    return arr[:, :Nc].copy()


def numpy_reference(points: np.ndarray, centroids: np.ndarray, radius: float) -> np.ndarray:
    P = np.asarray(points, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)
    d2 = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    return (d2 <= radius + 1e-3).astype(np.uint8)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    all_ok = True
    for t in range(6):
        Np = int(rng.integers(3, 8)); Nc = int(rng.integers(2, 4))
        P = rng.integers(0, 6, size=(Np, 3)).astype(np.float16); P[:, 2] = 0
        C = rng.integers(0, 6, size=(Nc, 3)).astype(np.float16); C[:, 2] = 0
        r = float(rng.choice([1.0, 2.0, 3.0, 4.0]))
        ane = radius_search_fused(P, C, r)
        ref = numpy_reference(P, C, r)
        ok = np.array_equal(ane, ref)
        all_ok &= ok
        print(f"t{t} Np={Np} Nc={Nc} r={r}: match={ok}")
    assert all_ok
    print("RadiusSearch CRACKED: native L2 ball-query membership verified vs numpy.")
