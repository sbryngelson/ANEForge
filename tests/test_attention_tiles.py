"""Attention query-tile autotune: heuristic by default, exact across counts, cached.

The autotune only changes HOW the [H,S,T] score is fissioned, never the result, so the
key safety property is that the tile count does not change the output. Uses a temp cache
dir so the test sees no previously-tuned values.
"""
import os
import tempfile
os.environ["ANEFORGE_CACHE_DIR"] = tempfile.mkdtemp()      # hermetic: no prior tuned values

import numpy as np
import aneforge as af
from aneforge import _optimize as _opt
from aneforge.graph import concat


def _core(H, S, dh, nt, qn, kn, vn):
    qh, kh, vh = af.input((H, S, dh)), af.input((H, S, dh)), af.input((H, S, dh))
    kt = kh.transpose([0, 2, 1]); sc = 1.0 / dh ** 0.5
    if nt == 1:
        o = ((qh @ kt) * sc).softmax(-1) @ vh
    else:
        tile = -(-S // nt); parts = []
        for st in range(0, S, tile):
            n = min(tile, S - st)
            qt = qh.slice_by_size([0, st, 0], [H, n, dh])
            parts.append(((qt @ kt) * sc).softmax(-1) @ vh)
        o = concat(parts, axis=1)
    return np.asarray(af.compile(o)(qn, kn, vn), np.float32)


def test_heuristic_is_default():
    assert _opt._heuristic_tiles(512) == 1
    assert _opt._heuristic_tiles(768) == 2
    assert _opt._heuristic_tiles(1500) == 3
    # with an empty cache and no tuning, attention_tiles returns the heuristic
    assert af.attention_tiles(1500, 16, 64) == 3


def test_tile_count_does_not_change_output():
    H, S, dh = 8, 1500, 64
    rng = np.random.default_rng(0)
    qn, kn, vn = [rng.standard_normal((H, S, dh)).astype(np.float32) * 0.1 for _ in range(3)]
    ref = _core(H, S, dh, 1, qn, kn, vn)
    for nt in (3, 5):
        o = _core(H, S, dh, nt, qn, kn, vn)
        cos = float((ref.ravel() @ o.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(o) + 1e-30))
        assert cos > 0.999, (nt, cos)


def test_tune_picks_and_caches():
    n = af.tune_attention(512, 2, 32)                     # small shape -> quick measurement
    assert n in (1, 2, 3, 4, 5, 6, 8)
    assert af.attention_tiles(512, 2, 32) == n            # served from cache afterwards
