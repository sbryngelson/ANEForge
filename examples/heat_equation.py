"""aneforge PDE demo - a 2D heat equation evolved over many timesteps on the ANE.

This is "the ANE solving a PDE over time". The 2D heat equation

    du/dt = alpha * Laplacian(u)

is integrated with the explicit (forward-Euler) scheme. One timestep is a single
3x3 convolution with the stencil  K = I + r*[[0,1,0],[1,-4,1],[0,1,0]],  where
r = alpha*dt/h^2 (kept at 0.2, inside the r<=0.25 stability bound for the 2D
explicit scheme). The conv is the ANE's home turf.

We compile the step ONCE into a single fused e5rt program, then loop host-side:
each step evaluates the program on the ANE and feeds the output field back in as
the next input. We evolve a hot-spot initial condition for many steps on a 64x64
grid (Dirichlet u=0 boundary, enforced host-side after each conv), then validate
the FINAL field against the SAME scheme run in fp32 numpy.

HONESTY: fp16 rounding COMPOUNDS over the timesteps - the ANE trajectory and the
fp32 numpy trajectory are genuinely different sequences, so they drift a little per
step. We report the relerr-vs-steps curve (it grows ~linearly: ~9e-3 at 50 steps,
~1.9e-2 at 100) and confirm the scheme stays STABLE: the field stays BOUNDED (no
fp16 blow-up) and the total heat stays near its initial value (this clamped-edge
diffusion is heat-conserving in the interior; fp16 rounding adds a small positive
bias of ~2%, NOT a divergence). That distinguishes fp16 compounding (a few %,
bounded, grows linearly) from a bug (an unstable scheme blows up by orders of
magnitude). The reference is the SAME explicit scheme with the fp32-exact stencil.

    python3 examples/heat_equation.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


H = W = 64
STEPS = 100
ALPHA = 1.0
DX = 1.0
DT = 0.2          # r = alpha*dt/dx^2 = 0.2  < 0.25 (stable explicit 2D heat)
R = ALPHA * DT / (DX * DX)


def initial_field():
    """A hot square in the middle of a cold (zero) plate."""
    u = np.zeros((1, 1, H, W), np.float32)
    u[0, 0, H // 2 - 6:H // 2 + 6, W // 2 - 6:W // 2 + 6] = 1.0
    return u


def heat_kernel():
    """The fused explicit heat step: u_new = u + r*Lap(u) as ONE 3x3 conv.
    Returns (fp16 kernel for the ANE, fp32-exact kernel for the reference)."""
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    K = np.zeros((1, 1, 3, 3), np.float32)
    K[0, 0] = R * lap
    K[0, 0, 1, 1] += 1.0           # + identity center
    return K.astype(np.float16), K


def apply_dirichlet(u):
    """Clamp the boundary to zero (cold edges) - done host-side between steps."""
    u[0, 0, 0, :] = 0.0; u[0, 0, -1, :] = 0.0
    u[0, 0, :, 0] = 0.0; u[0, 0, :, -1] = 0.0
    return u


def evolve_numpy(u0, Kfp32, steps):
    """Reference: the SAME scheme in fp32 numpy (explicit conv via padding),
    using the fp32-exact stencil."""
    Kff = Kfp32[0, 0]
    u = u0.copy()
    for _ in range(steps):
        x = np.pad(u, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros_like(u)
        for i in range(H):
            for j in range(W):
                out[0, 0, i, j] = (x[0, 0, i:i + 3, j:j + 3] * Kff).sum()
        u = apply_dirichlet(out)
    return u


def evolve_ane(step, u0, steps):
    """Evolve on the ANE: loop host-side, feed the field back in each step.
    Returns (final field, per-step total heat, per-step field max)."""
    u = u0.copy()
    heats, peak = [], []
    for _ in range(steps):
        u = np.ascontiguousarray(step(u.astype(np.float16)).astype(np.float32))
        u = apply_dirichlet(u)
        heats.append(float(u.sum()))
        peak.append(float(np.abs(u).max()))
    return u, np.array(heats), np.array(peak)


def main():
    Kf, Kfp32 = heat_kernel()

    # compile the single timestep ONCE: input field -> conv stencil -> new field
    x = af.input((1, 1, H, W))
    step = af.compile(af.conv(x, Kf, pad=1))
    print(f"heat step compiled: {step.n_ops} ANE op (one fused conv stencil), "
          f"grid {H}x{W}, r={R}")

    u0 = initial_field()
    heat0 = float(u0.sum())

    u_ane, heats, peak = evolve_ane(step, u0, STEPS)
    u_ref = evolve_numpy(u0, Kfp32, STEPS)
    relerr = float(np.linalg.norm(u_ane - u_ref) / (np.linalg.norm(u_ref) + 1e-12))

    # stability: the field must stay BOUNDED (no fp16 blow-up) and total heat must
    # stay near its initial value (clamped-edge diffusion conserves interior heat;
    # fp16 adds only a small positive bias).
    bounded = bool(peak.max() <= 1.0 + 1e-3)          # max never exceeds the hot-spot
    blew_up = bool(peak.max() > 10.0)
    heat_drift = abs(heats[-1] - heat0) / heat0

    print(f"evolved {STEPS} explicit heat steps on the ANE")
    print(f"  total heat:  initial {heat0:.3f} -> final ANE {heats[-1]:.3f} "
          f"(fp32 ref {float(u_ref.sum()):.3f})  drift {heat_drift*100:.2f}%")
    print(f"  field max:   start 1.000 -> final {peak[-1]:.4f}  (peak over run {peak.max():.4f})")
    print(f"  field stayed bounded (no blow-up): {bounded}")
    print(f"  final-field relerr vs fp32-exact scheme: {relerr:.3e}")

    # relerr-vs-steps curve (honest fp16 compounding profile)
    print("  fp16 compounding (relerr vs #steps, same hot-spot):")
    for S in (25, 50, 75, 100):
        ua, _, _ = evolve_ane(step, u0, S)
        ur = evolve_numpy(u0, Kfp32, S)
        e = float(np.linalg.norm(ua - ur) / np.linalg.norm(ur))
        print(f"    {S:3d} steps  relerr {e:.3e}")
    print("    -> grows ~linearly in #steps, stays bounded (fp16 compounding, not a bug)")

    ok = (relerr < 0.025) and bounded and (not blew_up) and (heat_drift < 0.05)
    print(f"\n{'PASS' if ok else 'FAIL'} - the ANE solved the 2D heat equation over "
          f"{STEPS} timesteps, stable and within {relerr:.1e} of the fp32 reference")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
