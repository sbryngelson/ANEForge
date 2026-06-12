"""DEMO: see the CoreML MIL that aneforge feeds the ANE compiler.

Reverse-engineering finding: aneforge emits a text CoreML MIL program (program(1.3),
func main<ios18>) with weights referenced by BLOBFILE offset; that MIL is what the on-device
ANECompiler lowers to a .hwx. compile(build_dir=...) writes the generated model.mil so you
can read exactly what the compiler received.

Run:  python3 examples/demos/mil_dialect.py
"""
import sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((8, 8, 3, 3)) * 0.1).astype(np.float32)
    Wf = (rng.standard_normal((8, 4)) * 0.1).astype(np.float32)
    x = af.input((1, 8, 16, 16))
    y = (af.conv(x, W, pad=1).relu().mean((2, 3)).reshape(1, 8)) @ Wf

    d = Path(tempfile.mkdtemp(prefix="aneforge_mil_"))
    af.compile(y, build_dir=str(d))
    mil = (d / "model.mil").read_text()
    print(f"generated MIL ({len(mil)} bytes) at {d/'model.mil'}:\n")
    print(mil)
    print("Note: program(1.3)/func main<ios18>; conv->relu->reduce_mean->reshape->matmul;")
    print("weights via BLOBFILE(@model_path/weights.bin, offset=...). This is the compiler input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
