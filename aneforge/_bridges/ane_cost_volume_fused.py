"""Native ANE CostVolume (stereo/optical-flow matching cost) via a netplist.

Entry point `cost_volume_fused(aux, ref, disparity_range)`.  Given single-channel
rows `aux` (width `Wa`) and `ref` (width `Wr`, with `Wr >= Wa + R`), it computes
the L1 matching cost for each disparity `d in [0, R]` and position `x in [0, Wa)`:

    `cost[d, x] = | aux[x] - ref[x + d] |`

returning `(R+1)` disparity planes each of width `Wa`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def cost_volume_fused(aux: np.ndarray, ref: np.ndarray,
                      disparity_range: int = 1) -> np.ndarray:
    """L1 cost volume of two single-channel rows.

    Args:
        aux:             `(Wa,)` fp16-castable row.
        ref:             `(Wr,)` fp16-castable row, `Wr >= Wa + R`.
        disparity_range: `R`; produces `R+1` disparity planes.
    Returns:
        `(R+1, Wa)` fp16 cost volume, `cost[d, x] = |aux[x] - ref[x+d]|`.
    """
    aux = np.asarray(aux, dtype=np.float16).reshape(-1)
    ref = np.asarray(ref, dtype=np.float16).reshape(-1)
    Wa, Wr = aux.size, ref.size
    R = int(disparity_range)
    from ._netplist import write_model, ensure_invoker  # type: ignore

    with tempfile.TemporaryDirectory(prefix="ane_costvol_") as d:
        wd = Path(d)
        # write_model maps: width->ref_width, template_width->aux_width.
        write_model(
            "cost_volume", wd,
            width=Wr, template_width=Wa, height=1, channels=1,
            disparity_direction=0, disparity_range=R,
        )
        aux.tofile(wd / "in_aux.f16")
        ref.tofile(wd / "in_ref.f16")
        cmd = [
            str(ensure_invoker("sdpa_invoker")),
            "--net-plist", str(wd / "net.plist"),
            "--weights", str(wd / "weights.0"),
            "--input", f"aux={wd / 'in_aux.f16'}",
            "--input", f"ref={wd / 'in_ref.f16'}",
            "--output", f"y={wd / 'out_y.f16'}",
            "--repeats", "1", "--warmup", "0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cost_volume invoker failed:\n{proc.stderr}\n{proc.stdout}")
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        if info.get("status") != "ok":
            raise RuntimeError(f"cost_volume non-ok: {info}")
        y = np.frombuffer((wd / "out_y.f16").read_bytes(), dtype=np.float16)
        return y.reshape(R + 1, Wa).copy()


def numpy_reference(aux: np.ndarray, ref: np.ndarray,
                    disparity_range: int = 1) -> np.ndarray:
    a = np.asarray(aux, np.float16).astype(np.float32).reshape(-1)
    r = np.asarray(ref, np.float16).astype(np.float32).reshape(-1)
    Wa, R = a.size, int(disparity_range)
    out = np.zeros((R + 1, Wa), np.float32)
    for d in range(R + 1):
        for x in range(Wa):
            out[d, x] = abs(a[x] - r[x + d])
    return out.astype(np.float16)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    max_err = 0.0
    for _ in range(6):
        Wa, R = 3, rng.integers(1, 3)
        aux = rng.standard_normal(Wa).astype(np.float16)
        ref = rng.standard_normal(Wa + R).astype(np.float16)
        got = cost_volume_fused(aux, ref, R).astype(np.float32)
        ref_out = numpy_reference(aux, ref, R).astype(np.float32)
        max_err = max(max_err, float(np.max(np.abs(got - ref_out))))
    print(json.dumps({"layer": "CostVolume", "status": "CRACKED",
                      "max_abs_err_vs_numpy": max_err}))
