#!/usr/bin/env python3
"""Render the ANE op catalog (op x device table) into ``op-catalog.md``.

The single source of truth for op availability is the ``OP_CATALOG`` dict in the ANEForge
package (``aneforge/_op_catalog.py``, validated by its ``tests/test_op_catalog.py``). This
script renders that data to the Markdown reference that ships here. Regenerate with:

    python docs/gen_op_catalog.py > docs/op-catalog.md

``aneforge`` must be importable: either ``pip install``ed, or with the ANEForge checkout
sitting next to this repo (a sibling ``ANEForge`` directory; auto-detected
below). Keeping the doc generated (never hand-edited) means it cannot drift from the
shipping package data.
"""
import os
import sys

try:
    from aneforge._op_catalog import OP_CATALOG
except ModuleNotFoundError:
    # aneforge isn't installed here; look for a sibling ANEForge checkout next to this repo
    _repo_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _sib in ("ANEForge", "aneforge-code"):
        _cand = os.path.join(_repo_parent, _sib)
        if os.path.isdir(os.path.join(_cand, "aneforge")):
            sys.path.insert(0, _cand)
            break
    from aneforge._op_catalog import OP_CATALOG

_CELL = {"native": "Y", "bridge": "~", "walled": "N"}


def render() -> str:
    out = []
    out.append("# ANE op catalog: every native MIL op x device (M1-M5)\n")
    out.append("> Generated from `aneforge/_op_catalog.py` in the ANEForge repository "
               "(`python docs/gen_op_catalog.py`); do not hand-edit. Query the same "
               "data at runtime via `af.op_info`, `af.is_native(op, chip)`, `af.ops_on(chip)`, "
               "`af.min_native_family(op)`, `af.walled_everywhere()`.\n")
    out.append(f"**{len(OP_CATALOG)} native MIL ops.** Device ladder: m1=A13, m2=A14, "
               "m3=A15, m4_m5=A16/A17. Cells: Y native, ~ bridge/decompose, N walled. "
               "aneforge's higher-level ops (rms_norm/group_norm/mha/sdpa/fft/linalg/...) are "
               "composites that lower to these.\n")
    # categories in first-appearance (dict) order
    cats = list(dict.fromkeys(d["category"] for d in OP_CATALOG.values()))
    for cat in cats:
        out.append(f"## {cat}")
        out.append("| op | M1 | M2 | M3 | M4/M5 | kernel | note |")
        out.append("|---|:--:|:--:|:--:|:--:|---|---|")
        for name, d in OP_CATALOG.items():
            if d["category"] != cat:
                continue
            cells = " | ".join(_CELL[d[k]] for k in ("m1", "m2", "m3", "m4_m5"))
            out.append(f"| `{name}` | {cells} | {d.get('kernel','')} | {d.get('note','')} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    sys.stdout.write(render())
