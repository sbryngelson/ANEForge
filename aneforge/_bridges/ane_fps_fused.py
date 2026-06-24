"""Native ANE `FurthestPointSampling` (FPS) via a hand-authored ANECIR netplist.

FPS is point-cloud downsampling: greedily pick `CentroidCount` points that are
maximally far apart (seeded at index 0).  Entry point `fps_fused(points,
centroid_count)`; `points` is `(N, 3)`, returns `(k, 3)` centroids.  The ANE uses
L2 distance on this arch regardless of the `DistanceMetric` param.

Schema (accepted unit dictionary)::

    {"Type": "FurthestPointSampling",
     "Bottom": ["points"],
     "InputType": ["Float16"],
     "OutputChannels": 3,
     "OutputType": "Float16",
     "Params": {"CentroidCount": k, "DistanceMetric": "L1" | "L2"}}
"""

from __future__ import annotations

import subprocess
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
  cmd = [
    str(g.ensure_invoker("sdpa_invoker")), "--net-plist", str(d / "net.plist"), "--weights", str(d / "weights.0"),
    "--input", f"points={d / 'points.f16'}", "--output", f"y={d / 'out.f16'}",
    "--repeats", "1", "--warmup", "0",
  ]
  p = subprocess.run(cmd, capture_output=True, text=True)
  if p.returncode != 0: raise RuntimeError(f"fps invoker failed:\n{p.stderr}")
  y = np.frombuffer((d / "out.f16").read_bytes(), dtype=np.float16)
  return y.reshape(3, centroid_count).T  # (k, 3)


def numpy_reference(points: np.ndarray, centroid_count: int, metric: str = "L2") -> np.ndarray:
  """Greedy FPS, seeded at index 0.

    The ANE always uses L2 on this arch (see module docstring); the default
    here is therefore L2, which matches the hardware exactly.
    """
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
