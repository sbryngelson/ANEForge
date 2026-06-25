"""Resident on-device state via share_buffer: no host re-feed between steps. Run: python3 examples/demos/resident_state.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.simplefilter("ignore")
    s = af.input((1, 8))
    y = s.adds(1.0)
    net = af.compile(y)
    prog = net._prog
    in_name = net._inputs[0][0]
    out_name = net._out_name

    # alias state INPUT to the program's OWN OUTPUT buffer: step k's result feeds step k+1
    prog.share_buffer(0, out_name, 0, in_name)

    print("execute() repeatedly, feeding NOTHING - state accumulates on-device:")
    for step in range(1, 6):
        prog.execute()
        val = float(np.asarray(prog.read_output(out_name)).reshape(-1)[0])
        print(f"  step {step}: state[0] = {val:.1f}")
    print("\nThe value grows 1,2,3,... with zero host input per step - the buffer is resident.")
    print("Same primitive powers the on-device KV-cache (gpt_generate_ane.py) and resident")
    print("optimizer state in training. See aneforge Program.share_buffer.")
    net.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
