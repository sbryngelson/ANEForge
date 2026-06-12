"""DEMO: numerical / scientific compute on the ANE - a DFT as two matmuls.

Exercises:
  - expressing a scientific kernel (the discrete Fourier transform) as ANE matmuls
    (real part via a cosine basis, imaginary via a sine basis)
  - recovering the magnitude spectrum and checking it vs numpy's FFT (cosine ~1, right peaks)
  - the ANE as a general fp16 array processor, not just a neural-net accelerator

Run:  python3 examples/demos/numerical_scientific.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common  # noqa: F401
import numpy as np
import aneforge as af


def main() -> int:
    warnings.filterwarnings("ignore")
    n = 256
    t = np.arange(n)
    sig = (np.sin(2*np.pi*5*t/n) + 0.5*np.sin(2*np.pi*17*t/n)).astype(np.float32)

    # DFT as two matmuls: re = sig @ cos_basis, im = sig @ (-sin_basis)
    jk = np.outer(np.arange(n), np.arange(n))
    Wcos = np.cos(2*np.pi*jk/n).astype(np.float32)
    Wsin = (-np.sin(2*np.pi*jk/n)).astype(np.float32)

    x = af.input((1, n))
    net = af.compile((x @ Wcos).square() + (x @ Wsin).square())   # |DFT|^2, all on the ANE
    mag2 = np.asarray(net(sig.reshape(1, n))).reshape(-1).astype(np.float64)
    mag = np.sqrt(np.maximum(mag2, 0))
    ref = np.abs(np.fft.fft(sig.astype(np.float64)))

    cos = float((mag @ ref) / (np.linalg.norm(mag)*np.linalg.norm(ref)+1e-30))
    peaks = sorted(np.argsort(mag)[::-1][:4] % n)
    print(f"ANE DFT |spectrum| vs numpy FFT: cosine = {cos:.4f}")
    print(f"dominant ANE bins: {[int(p) for p in peaks]}  (expect 5 & 17 and their mirrors 239 & 251)")
    net.release()
    print("\nThe ANE runs general fp16 linear algebra - an FFT/DFT, solves, etc. are just matmuls")
    print("on the engine. fp16 is the main caveat for scientific work (af.tune_precision / paired-fp16).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
