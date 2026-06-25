#!/usr/bin/env python3
"""Shared MIL-opcode helpers for the bench/ scripts (not run on its own)."""
from __future__ import annotations

from pathlib import Path


def mil_encoding_tally(build_dir):
    """Count weight encodings in the generated MIL, summed across model.mil files."""
    counts = {"int4_lut": 0, "sparse": 0, "int8": 0, "fp16": 0}
    for mil in Path(build_dir).rglob("model.mil"):
        txt = mil.read_text()
        counts["int4_lut"] += txt.count("constexpr_lut_to_dense")
        counts["sparse"] += txt.count("constexpr_sparse_to_dense")
        counts["int8"] += txt.count("constexpr_affine_dequantize")
        # fp16 weight constants are BLOBFILE-backed const() ops
        counts["fp16"] += sum(1 for ln in txt.splitlines()
                              if "= const()" in ln and "BLOBFILE" in ln)
    return counts
