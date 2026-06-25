"""Hidden hardware layers: native ops the fused MIL route can't express. Run: python3 examples/demos/hidden_layers.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    H, S, D = 2, 16, 16
    rng = np.random.default_rng(0)
    Q, K, V = (rng.standard_normal((1, H, S, D)).astype(np.float16) for _ in range(3))
    q, k, v = (af.input((1, H, S, D)) for _ in range(3))
    net = af.compile(af.sdpa(q, k, v), opt=0)
    got = np.asarray(net(Q, K, V)).astype(np.float64)
    ref = np.zeros((1, H, S, D))    # numpy reference (non-causal)
    for h in range(H):
        sc = Q[0, h].astype(np.float64) @ K[0, h].astype(np.float64).T / np.sqrt(D)
        sc -= sc.max(1, keepdims=True); p = np.exp(sc); p /= p.sum(1, keepdims=True)
        ref[0, h] = p @ V[0, h].astype(np.float64)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got)*np.linalg.norm(ref)+1e-30))
    print(f"native SDPA hardware layer (ANECSDPALayerDesc) on M1: cosine vs numpy = {cos:.4f}")
    net.release()

    print("\nother hidden ops are family-gated (af.min_native_family):")
    for op in ("sdpa", "topk", "sort", "argmax"):
        fam = af.min_native_family(op)
        reach = "reachable on M1 (bridge)" if op == "sdpa" else (
            f"needs family >= {fam}" if fam else "not native on any family")
        print(f"  {op:>8}: {reach}")
    print("\nThe bridge route cuts the graph to Path-A sub-programs that target the ANE's hidden")
    print("hardware-layer pool (26 native layer types decoded) - capabilities beyond the fused MIL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
