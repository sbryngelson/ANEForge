"""The ANE capability surface: what the hardware can and can't do. Run: python3 examples/demos/capability_surface.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    print(f"OP_CATALOG: {len(af.OP_CATALOG)} ops tracked\n")

    print("per-op native support on M1 (af.is_native / af.min_native_family):")
    for op in ("conv", "matmul", "softmax", "sdpa", "topk", "sort", "layer_norm", "gelu"):
        nat = af.is_native(op, "m1")
        fam = af.min_native_family(op)
        print(f"  {op:>10}: native(m1)={str(nat):>5}  min_native_family={fam}")

    native_m1 = af.ops_on("m1", "native")
    print(f"\nnatively-runnable ops on M1: {len(native_m1)}")
    walled = af.walled_everywhere()
    print(f"ops no chip runs natively (walled everywhere): {len(walled)}")
    if walled:
        print("  e.g.", ", ".join(sorted(walled)[:8]))

    print("\nThe capability surface is queryable + machine-checked: aneforge knows which ops are")
    print("native vs decomposed vs unreachable on each ANE family - so a graph can be gated up front.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
