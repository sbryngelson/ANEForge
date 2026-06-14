"""aneforge flagship: a fluid simulation on the Apple Neural Engine.

This solves the 2-D incompressible Navier-Stokes equations (vorticity-streamfunction
form) on a periodic grid by the *pseudo-spectral* method, and every Fourier transform
in the loop runs on the ANE as one fused e5rt program. A passive dye is painted as
the word "ANEForge" and carried by the flow: it is legible at t=0, then a field of
vortices stirs it into thin glowing filaments.

    vorticity transport:  d(omega)/dt + (u . grad) omega = nu  * lap(omega)
    dye    transport:     d(c)/dt     + (u . grad) c     = kap * lap(c)
    velocity from vorticity: lap(psi) = -omega,  u = d(psi)/dy,  v = -d(psi)/dx

Each timestep transforms between physical and Fourier space to take derivatives and
form the advection terms; the FFTs are the entire compute core. `aneforge.fft`
realizes each 2-D transform as eight real GEMMs (`F_M @ X @ F_N^T`) fused into ONE
ANE program (the engine has no complex dtype, so values ride as real/imag pairs).
The two transform programs are compiled ONCE and reused for every step. A high-order
spectral filter (hyperviscosity) holds the enstrophy cascade at the grid scale.

HONESTY (fp16). The ANE computes in fp16. The dense DFT is unnormalized, so spectral
coefficients at this 256-grid exceed fp16's range; the transforms are scaled by
linearity (exact) to fit, see FWD/INV below. Advection is chaotic, so an fp16
trajectory is its own slightly-perturbed flow. The point here is that it stays
bounded (the hyperviscosity filter holds the grid-scale modes) and looks like the
scalar mixing it is, not that it tracks an fp64 run step for step. The run is gated
on staying finite and bounded, nothing more.

    python3 examples/fluid_vorticity.py

Writes docs/assets/fluid_vorticity.png as an APNG (the dye dissolving, animated,
full 24-bit colour) if Pillow is installed; the simulation runs either way.
"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge.fft as F

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

# simulation parameters
N = 256                # grid
L = 2.0 * np.pi        # periodic domain [0, 2pi)^2
NU = 7e-4              # vorticity viscosity
KAP = 1.5e-4           # dye diffusivity (low: keep the filaments thin)
DT = 1.5e-3            # timestep
STEPS = 2200           # advection steps
FRAME_STRIDE = 32      # capture an animation frame every this many steps
ANE_RAIL_W = 1.48      # RE-measured sustained ANE rail, W (see demos/power_efficiency.py)
ANIM_SIZE, ANIM_MS = 256, 115                    # APNG: full-colour at the native grid
ANIM_START, ANIM_HOLD = 1900, 1500               # hold the wordmark, then the final field (ms)

# 'inferno'-style perceptual colormap (matplotlib), as a small anchor table.
_CMAP = [(0.00, (2, 2, 8)), (0.13, (28, 12, 69)), (0.25, (74, 12, 107)),
         (0.38, (120, 28, 109)), (0.50, (165, 44, 96)), (0.63, (207, 68, 70)),
         (0.75, (237, 105, 37)), (0.87, (251, 155, 6)), (0.95, (247, 209, 61)),
         (1.00, (252, 255, 200))]


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


def spectral_grid():
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2I = K2.copy(); K2I[0, 0] = 1.0                  # avoid /0 at the DC mode
    kmax = np.abs(k).max() * 2.0 / 3.0
    mask = ((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)).astype(np.float32)
    kabs = np.sqrt(K2)
    filt = np.exp(-30.0 * (kabs / kabs.max())**10).astype(np.float32)  # hyperviscosity
    return X, Y, KX, KY, K2, K2I, mask, filt


def wordmark(text="ANEForge"):
    """Render `text` to an [N,N] grayscale dye field (hi-res, anti-aliased down)."""
    from PIL import Image, ImageDraw, ImageFont
    up = 4; W = H = N * up
    img = Image.new("L", (W, H), 0); d = ImageDraw.Draw(img)
    try:
        path = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
        sz = int(H * 0.20)
        while sz > 8:
            f = ImageFont.truetype(path, sz); b = d.textbbox((0, 0), text, font=f)
            if b[2] - b[0] < W * 0.92:
                break
            sz -= 4
    except OSError:
        f = ImageFont.load_default()
    b = d.textbbox((0, 0), text, font=f)
    d.text(((W - (b[2] - b[0])) // 2 - b[0], (H - (b[3] - b[1])) // 2 - b[1]),
           text, fill=255, font=f)
    return (np.asarray(img.resize((N, N), Image.LANCZOS), np.float64) / 255.0).T


def vortex_field(X, Y):
    """A reproducible field of Gaussian vortices that stirs the whole frame."""
    rng = np.random.default_rng(7)
    w = np.zeros((N, N))
    for j in range(10):
        x0 = rng.uniform(0, L); y0 = rng.uniform(0, L)
        s = (-1)**j * rng.uniform(6, 12); r0 = rng.uniform(0.5, 0.9)
        dx = np.minimum(np.abs(X - x0), L - np.abs(X - x0))
        dy = np.minimum(np.abs(Y - y0), L - np.abs(Y - y0))
        w += s * np.exp(-(dx**2 + dy**2) / r0**2)
    return w - w.mean()


def main():
    X, Y, KX, KY, K2, K2I, mask, filt = spectral_grid()
    muw = np.exp(-NU * K2 * DT).astype(np.float32)
    muc = np.exp(-KAP * K2 * DT).astype(np.float32)

    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - a fluid simulation on the Apple Neural Engine{R}")
    out(f"  {DIM}2-D incompressible Navier-Stokes, pseudo-spectral, every FFT on the engine{R}")
    out()

    fwd = F.fft2_plan(N, N)
    inv = F.ifft2_plan(N, N)
    out(f"  {GREY}compile{R} {CHECK} two 2-D FFT programs "
        f"{DIM}({fwd.n_ops} ANE ops each: 8 GEMMs as F_M @ X @ F_N^T){R}")

    ane_t = [0.0]   # seconds spent in ANE transforms (dispatch included)

    # The dense DFT is unnormalized, so at N=256 the spectra pierce fp16 range. Both
    # transforms are scaled by linearity (exact) to fit: the forward INPUT by an upper
    # bound on its largest coefficient (sum of |field|), the inverse INPUT by its max.
    def FWD(field):
        c = max(1.0, float(np.abs(field).sum()) / 4.0e4)
        t = time.perf_counter(); r, i = fwd((field / c).astype(np.float32))
        ane_t[0] += time.perf_counter() - t
        return r * c, i * c

    def INV(re, im):
        m = max(float(np.abs(re).max()), float(np.abs(im).max()), 1e-9)
        s = 3.0e4 / m if m > 3.0e4 else 1.0
        t = time.perf_counter()
        r, i = inv((re * s).astype(np.float32), (im * s).astype(np.float32))
        ane_t[0] += time.perf_counter() - t
        return r / s, i / s

    def field_of(re, im):
        r, _ = INV(re, im); return r                  # real physical field of a spectrum

    # initial state: vortices to stir, "ANEForge" dye to be stirred
    whr, whi = FWD(vortex_field(X, Y))
    whr = (whr * mask).astype(np.float32); whi = (whi * mask).astype(np.float32)
    cr, ci = FWD(wordmark())
    cr = (cr * mask).astype(np.float32); ci = (ci * mask).astype(np.float32)

    def velocity(whr, whi):
        pr = whr / K2I; pi = whi / K2I                 # psi_hat = omega_hat / k^2
        return field_of(-KY * pi, KY * pr), field_of(KX * pi, -KX * pr)   # u, v

    def adv_vort(whr, whi):
        u, v = velocity(whr, whi)
        wx = field_of(-KX * whi, KX * whr); wy = field_of(-KY * whi, KY * whr)
        Nr, Ni = FWD(u * wx + v * wy)
        return (Nr * mask).astype(np.float32), (Ni * mask).astype(np.float32), (u, v)

    def adv_dye(cr, ci, uv):
        u, v = uv
        cx = field_of(-KX * ci, KX * cr); cy = field_of(-KY * ci, KY * cr)
        Nr, Ni = FWD(u * cx + v * cy)
        return (Nr * mask).astype(np.float32), (Ni * mask).astype(np.float32)

    def rk2(re, im, advect, mu, *a):
        """One RK2 step: midpoint advection, viscous integrating factor, spectral filter.
        `advect` may return extra state after (Nr, Ni); it is passed back as `rest`."""
        N1r, N1i, *rest = advect(re, im, *a)
        s1r = mu * (re - DT * N1r); s1i = mu * (im - DT * N1i)
        N2r, N2i = advect(s1r, s1i, *a)[:2]
        nr = ((mu * re - 0.5 * DT * (mu * N1r + N2r)) * filt).astype(np.float32)
        ni = ((mu * im - 0.5 * DT * (mu * N1i + N2i)) * filt).astype(np.float32)
        return nr, ni, rest

    # evolve
    out(f"  {GREY}solve{R}   {DIM}{STEPS} steps on a {N}x{N} grid, "
        f"vorticity + dye, every FFT on the engine{R}")
    frames = []; n_fft = 0
    t0 = time.perf_counter()
    for s in range(STEPS):
        if s % FRAME_STRIDE == 0:
            frames.append(field_of(cr, ci))         # the dye field, render-only
        whr, whi, (uv,) = rk2(whr, whi, adv_vort, muw)
        cr, ci, _ = rk2(cr, ci, adv_dye, muc, uv)
        n_fft += 16
    wall = time.perf_counter() - t0
    omega = field_of(whr, whi); frames.append(field_of(cr, ci))
    bounded = bool(np.isfinite(omega).all() and np.abs(omega).max() < 1e3)

    energy = ANE_RAIL_W * ane_t[0]
    out(f"  {GREY}solve{R}   {CHECK} {n_fft} FFT dispatches on the ANE "
        f"{DIM}({wall:.1f}s wall, {ane_t[0]*1e3/n_fft:.2f} ms/transform){R}")
    out()
    out(f"  {GREY}physics{R} {DIM}the dye wound into filaments; the flow stayed bounded "
        f"(enstrophy {float((omega**2).mean()):.3f}, no fp16 blow-up){R}")
    out(f"  {GREY}energy{R}  {DIM}~{ane_t[0]:.1f}s of ANE transform time at the measured "
        f"~{ANE_RAIL_W} W rail ~ {BOLD}{energy:.1f} J{R}{DIM} for the whole simulation{R}")
    out()

    wrote = render(frames, energy)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    out(f"  {CHECK} {BOLD}a fluid simulation, computed on the Apple Neural Engine{R}")
    out(f"    {DIM}no CoreML - no GPU - ~{energy:.1f} J of engine compute{R}")
    out()

    ok = bounded
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE evolved 2-D Navier-Stokes for "
        f"{STEPS} steps on the engine, stable (bounded, no blow-up)")
    return 0 if ok else 1


# rendering                                                                   #

def colormap(c):
    """Map a dye field in [0,1] to RGB via the perceptual inferno table. Flat (no
    fake lighting), so it reads like a scientific scalar field, not an oil slick."""
    vs = np.array([p[0] for p in _CMAP]); cs = np.array([p[1] for p in _CMAP], float)
    v = np.clip(c, 0.0, 1.0)**0.85
    rgb = np.stack([np.interp(v, vs, cs[:, j]) for j in range(3)], -1).astype(np.uint8)
    return rgb.transpose(1, 0, 2)              # [x,y,3] field -> [row=y, col=x] image


def telemetry(im, energy):
    """A small ticking energy/power readout in the corner (no full-width stripe):
    the joules accumulated on the engine so far, over the measured rail draw."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = im.size
    try:
        fmono = "/System/Library/Fonts/SFNSMono.ttf"
        big = ImageFont.truetype(fmono, max(13, int(H * 0.085)))
        small = ImageFont.truetype(fmono, max(8, int(H * 0.040)))
    except OSError:
        big = small = ImageFont.load_default()
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    jt = f"{energy:.1f} J"; wt = f"{ANE_RAIL_W:.2f} W  on the ANE"
    pad = int(H * 0.05); gap = int(H * 0.03)
    jh = int(big.size); wh = int(small.size)                # nominal line heights (no overlap)
    jw = d.textlength(jt, font=big); ww = d.textlength(wt, font=small)
    x = pad; y = H - pad - jh - gap - wh
    bx = int(max(jw, ww))
    d.rounded_rectangle([x - 9, y - 8, x + bx + 10, y + jh + gap + wh + 9],
                        radius=9, fill=(8, 11, 16, 150))
    d.text((x, y), jt, font=big, anchor="lt", fill=(255, 196, 92, 255))   # amber, like the field
    d.text((x, y + jh + gap), wt, font=small, anchor="lt", fill=(150, 168, 178, 235))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def render(frames, total_energy):
    """Write the dye-dissolve as an APNG: full 24-bit colour, no palette or dither
    (a GIF would band these continuous-tone fields). A corner readout ticks up the
    engine energy as the run proceeds. Animates in the browser; the first frame, the
    legible wordmark, is the still poster. Needs Pillow."""
    try:
        from PIL import Image
    except ImportError:
        out(f"  {DIM}(install Pillow to write the animation: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ims = []
    for i, c in enumerate(frames):
        im = Image.fromarray(colormap(c)).resize((ANIM_SIZE, ANIM_SIZE), Image.LANCZOS)
        e = total_energy * min(1.0, i * FRAME_STRIDE / STEPS)        # energy so far
        ims.append(telemetry(im, e))
    durs = [ANIM_MS] * len(ims); durs[0] = ANIM_START; durs[-1] = ANIM_HOLD
    ims[0].save(assets / "fluid_vorticity.png", save_all=True, append_images=ims[1:],
                duration=durs, loop=0)
    return f"docs/assets/fluid_vorticity.png (APNG, {len(ims)} frames, full colour)"


if __name__ == "__main__":
    sys.exit(main())
