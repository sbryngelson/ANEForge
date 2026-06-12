"""DEMO: aneforge reaches the ANE from UNENTITLED user space.

Exercises:
  - that a plain Python process (no special entitlements, no CoreML) compiles + runs on the ANE
  - the split the RE found: the compile goes through the entitled ANECompilerService daemon
    (XPC), but the per-inference SUBMIT is a direct IOKit call from this unentitled process
  - a working dispatch as proof the boundary is crossed legitimately via the public daemons

Run:  python3 examples/demos/entitlement_boundary.py
"""
import sys, os, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    print(f"this process: pid={os.getpid()}, plain python, no ANE entitlements")
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
    x = af.input((1, 8, 16, 16))
    net = af.compile(af.conv(x, W, pad=1).relu().mean((2, 3)))   # compile via ANECompilerService (XPC)
    out = net(rng.standard_normal((1, 8, 16, 16)).astype(np.float32))   # SUBMIT direct from here
    print(f"compiled + dispatched on the ANE: output shape {np.asarray(out).shape}")
    print("\nThe entitlement boundary, as decoded:")
    print("  - compile: handed to the entitled ANECompilerService daemon over XPC (it owns the")
    print("    rootless model-cache + decrypt entitlements aneforge lacks).")
    print("  - submit:  a DIRECT IOKit external method (selector 2) from THIS unentitled process,")
    print("    via IOServiceOpen on 'H11ANEIn' - the kernel gates open by an Info.plist property,")
    print("    not a code entitlement, so user-space dispatch is allowed.")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
