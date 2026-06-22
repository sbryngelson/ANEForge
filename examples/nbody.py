"""aneforge N-body demo - one gravitational force + integration step on the ANE.

This is "an N-body step on the ANE". For N point masses we compute the pairwise
inverse-square (gravity/Coulomb) acceleration via a broadcast outer-difference and a
reduction - the kernel shape behind any N-body solver:

    d_ij   = p_j - p_i                      # [N,N,3] broadcast outer difference
    a_i    = G * sum_{j != i} m_j * d_ij / (|d_ij|^2 + eps)^(3/2)

The self term j=i is removed by adding a large value on the diagonal of the squared
distance (a folded constant mask), so 1/r^3 -> ~0 there. We then take ONE explicit
leapfrog/Euler kick-drift step:  v += a*dt;  p += v*dt.

The whole force computation (broadcast diff, squared distance, rsqrt^3, masked sum)
is compiled into ONE fused e5rt program. We validate the resulting accelerations AND
the advanced positions against the SAME computation in fp32 numpy.

CAVEAT: 1/r^3 is fp16-sensitive when two bodies get close (the denominator is a
small number raised to the 3/2 power), so we seed well-separated bodies and use a
softening eps (standard in collisionless N-body) to keep r in fp16 range. With that,
the accelerations match numpy to ~1e-2. The intermediate [N,N,3] grows O(N^2), so
this is the small-N direct-summation regime (no tree/Barnes-Hut, which needs the
data-dependent control flow the feed-forward ANE lacks).

    python3 examples/nbody.py
"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af


N = 16
G = 1.0
EPS = 0.1          # gravitational softening (keeps 1/r^3 in fp16 range)
DT = 0.01


def accel_program(masses):
    """Compile the pairwise inverse-square acceleration as ONE fused ANE program.
    Inputs: positions [N,3] and the folded diagonal mask. Output: accel [N,3]."""
    p = af.input((N, 3))
    mask = af.input((N, N, 1))                       # huge value on the diagonal
    mj = af.input((N, N, 1))                         # m_j broadcast over i

    pi = p.reshape(N, 1, 3)
    pj = p.reshape(1, N, 3)
    d = pj - pi                                      # [N,N,3] outer difference
    r2 = d.square().sum(2).reshape(N, N, 1)          # [N,N,1] squared distance
    r2 = r2 + mask                                   # kill the diagonal (self term)
    # 1/(r^2+eps)^(3/2) = rsqrt(r^2+eps)^3
    inv = r2.rsqrt(eps=EPS)
    inv3 = inv * inv * inv                           # [N,N,1]
    contrib = d * (inv3 * mj) * G                    # [N,N,3] per-pair acceleration
    return af.compile(contrib.sum(1).reshape(N, 3))  # sum over j -> a_i [N,3]


def ref_accel(p, masses):
    """fp32 numpy reference for the same softened inverse-square acceleration."""
    a = p.astype(np.float32)
    d = a[None, :, :] - a[:, None, :]                # [N,N,3]
    r2 = (d ** 2).sum(2) + np.eye(N, dtype=np.float32) * 1e3
    inv3 = (1.0 / np.sqrt(r2 + EPS)) ** 3            # [N,N]
    contrib = d * (inv3 * masses[None, :])[:, :, None] * G
    return contrib.sum(1)                            # [N,3]


def main():
    rng = np.random.default_rng(3)
    p0 = (rng.standard_normal((N, 3)) * 2.0).astype(np.float16)   # well-separated
    v0 = (rng.standard_normal((N, 3)) * 0.1).astype(np.float16)
    masses = np.ones(N, np.float32)                  # equal masses

    mask = (np.eye(N, dtype=np.float32) * 1e3).astype(np.float16).reshape(N, N, 1)
    mj = np.broadcast_to(masses.reshape(1, N, 1), (N, N, 1)).astype(np.float16)
    mj = np.ascontiguousarray(mj)

    prog = accel_program(masses)
    print(f"N-body force compiled: {prog.n_ops} ANE ops "
          f"(broadcast diff + rsqrt^3 + masked reduction), N={N}")

    # acceleration on the ANE
    a_ane = np.asarray(prog(p0, mask, mj)).reshape(N, 3).astype(np.float32)
    a_ref = ref_accel(p0, masses)
    a_relerr = float(np.linalg.norm(a_ane - a_ref) / np.linalg.norm(a_ref))
    print(f"  acceleration relerr vs fp32 numpy: {a_relerr:.3e}")

    # one explicit kick-drift step (done host-side from the ANE accel)
    v1_ane = v0.astype(np.float32) + a_ane * DT
    p1_ane = p0.astype(np.float32) + v1_ane * DT
    v1_ref = v0.astype(np.float32) + a_ref * DT
    p1_ref = p0.astype(np.float32) + v1_ref * DT
    p_relerr = float(np.linalg.norm(p1_ane - p1_ref) / np.linalg.norm(p1_ref))
    # the displacement is the part that actually depends on the ANE accel (the
    # advanced position is dominated by the identical fp16 p0, so its relerr is tiny
    # by construction - report the velocity-update relerr too, which isolates accel).
    v_relerr = float(np.linalg.norm(v1_ane - v1_ref) / np.linalg.norm(v1_ref))
    print(f"  updated-velocity relerr vs fp32 numpy:  {v_relerr:.3e} (isolates the accel)")
    print(f"  advanced-position relerr vs fp32 numpy: {p_relerr:.3e} (dominated by p0)")
    print(f"  (well-separated bodies + eps={EPS} softening keep 1/r^3 in fp16 range)")

    ok = (a_relerr < 0.02) and (v_relerr < 0.02) and (p_relerr < 0.02)
    print(f"\n{'PASS' if ok else 'FAIL'} - one gravitational N-body step (N={N}) on the "
          f"ANE: accel within {a_relerr:.1e}, positions within {p_relerr:.1e} of numpy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
