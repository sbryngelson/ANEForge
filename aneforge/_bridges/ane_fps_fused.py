"""Native ANE `FurthestPointSampling` (point-cloud downsampling) on Path A. ANE uses L2 on this arch regardless of `DistanceMetric`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def fps_fused(points: np.ndarray, centroid_count: int, *, metric: str = "L1") -> np.ndarray:
  """Run native FPS on the ANE.  `points` is `(N, 3)`; returns `(k, 3)` centroids."""
  from . import _netplist as g

  points = np.asarray(points, dtype=np.float16)
  if points.ndim != 2 or points.shape[1] != 3: raise ValueError("points must be (N, 3)")
  N = points.shape[0]
  pts_cw = points.T.reshape(1, 3, 1, N)  # (B, C=3, H=1, W=N)

  d = Path(tempfile.mkdtemp(prefix="ane_fps_"))
  g.write_model(
    "furthest_point_sampling", d,
    width=N, centroid_count=int(centroid_count), distance_metric=metric, output_channels=3,
  )
  (d / "points.f16").write_bytes(np.ascontiguousarray(pts_cw, dtype=np.float16).tobytes())
  g.invoke_netplist(
    g.ensure_invoker("sdpa_invoker"), d / "net.plist",
    weights=[d / "weights.0"], inputs=[("points", d / "points.f16")],
    outputs=[("y", d / "out.f16")], warmup=0,
  )
  y = np.frombuffer((d / "out.f16").read_bytes(), dtype=np.float16)
  return y.reshape(3, centroid_count).T  # (k, 3)


def numpy_reference(points: np.ndarray, centroid_count: int, metric: str = "L2") -> np.ndarray:
  """Greedy FPS, seeded at index 0. Default L2 matches the hardware exactly."""
  P = np.asarray(points, dtype=np.float32)
  N = P.shape[0]
  sel = [0]
  d = np.full(N, np.inf)
  for _ in range(1, centroid_count):
    last = P[sel[-1]]
    dist = np.abs(P - last).sum(1) if metric == "L1" else np.sqrt(((P - last) ** 2).sum(1))
    d = np.minimum(d, dist)
    sel.append(int(np.argmax(d)))
  return P[sel].astype(np.float16)


if __name__ == "__main__":
  rng = np.random.default_rng(0)
  matches = total = 0
  for t in range(40):
    N = int(rng.integers(5, 12)); k = int(rng.integers(2, N))
    P = rng.integers(0, 12, size=(N, 3)).astype(np.float16); P[:, 2] = 0
    for metric in ("L1", "L2"):  # netplist param; ANE uses L2 either way
      ane = fps_fused(P, k, metric=metric)
      ref = numpy_reference(P, k, "L2")  # L2 reference matches the hardware
      total += 1
      matches += int(np.array_equal(ane, ref))
  print(f"FPS vs L2 numpy reference: {matches}/{total} exact matches")
  assert matches == total
  print("FPS CRACKED: native L2 point-cloud downsampling on the ANE "
        "(DistanceMetric param is L2-only on this arch).")
