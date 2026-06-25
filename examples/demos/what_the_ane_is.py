"""What the ANE is: a real matrix engine you can target directly. Run: python3 examples/demos/what_the_ane_is.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((1, 256)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((256, 64)) * 0.1).astype(np.float32)

    x = af.input((1, 256))
    net = af.compile(x @ B)                 # one fused ANE program
    got = np.asarray(net(A)).astype(np.float64)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    cos = float((got.ravel() @ ref.ravel()) /
                (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))

    print(f"matmul [1,256]x[256,64] on the ANE: cosine vs numpy = {cos:.4f}")
    print(f"matmul native on ANE? af.is_native('matmul') = {af.is_native('matmul')}")
    print(f"device_status('matmul') = {af.device_status('matmul')!r}")
    print("\nThe ANE is a directly-targetable matrix/conv engine: build a graph, compile to one")
    print("program, dispatch. No CoreML in the path (see 02). fp16 compute -> cosine ~1, not bit-exact.")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
