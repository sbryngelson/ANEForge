"""aneforge showpiece: Gray-Scott reaction-diffusion as one fused conv-stencil program per step on the ANE, growing "ANEForge" into Turing patterns. Run: python3 examples/reaction_diffusion.py"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge as af
from aneforge import _compile as _c
from aneforge.graph import concat

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

# simulation parameters
N = 448                 # grid (periodic); supersampled, then resolved down for a crisp APNG
DU, DV = 0.16, 0.08     # diffusion of U, V
F, K = 0.042, 0.061     # feed / kill: a branching-labyrinth (circuit-trace) regime
DT = 1.0                # timestep
STEPS = 20000           # reaction steps
FRAME_STRIDE = 360      # capture an animation frame every this many steps
ANE_RAIL_W = 1.48       # measured sustained ANE rail, W (see demos/power_efficiency.py)
ANIM_SIZE, ANIM_MS = 384, 90
ANIM_START, ANIM_HOLD = 1400, 1600                   # hold the wordmark, then the final field (ms)

# 9-point Laplacian stencil (isotropic), as a 3x3 conv kernel.
LAP = np.array([[0.05, 0.20, 0.05],
                [0.20, -1.00, 0.20],
                [0.05, 0.20, 0.05]], np.float32).reshape(1, 1, 3, 3)

# 'inferno'-style perceptual colormap (matplotlib), as a small anchor table.
_CMAP = [(0.00, (2, 2, 8)), (0.13, (28, 12, 69)), (0.25, (74, 12, 107)),
         (0.38, (120, 28, 109)), (0.50, (165, 44, 96)), (0.63, (207, 68, 70)),
         (0.75, (237, 105, 37)), (0.87, (251, 155, 6)), (0.95, (247, 209, 61)),
         (1.00, (252, 255, 200))]


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


def wrap_pad(x, p=1):
    """Periodic pad H and W by p using slices of x itself (no constant leaf), so the
    Laplacian stencil wraps across the domain boundary."""
    b, c, h, w = x.shape
    x = concat([x.slice_by_size([0, 0, 0, w - p], [b, c, h, p]), x,
                x.slice_by_size([0, 0, 0, 0], [b, c, h, p])], axis=3)
    return concat([x.slice_by_size([0, 0, h - p, 0], [b, c, p, w + 2 * p]), x,
                   x.slice_by_size([0, 0, 0, 0], [b, c, p, w + 2 * p])], axis=2)


def update_program():
    """Build ONE e5rt program: (U, V) -> (U', V') for a single Gray-Scott step."""
    U = af.input((1, 1, N, N))
    V = af.input((1, 1, N, N))
    lapU = af.conv(wrap_pad(U), LAP)
    lapV = af.conv(wrap_pad(V), LAP)
    uvv = U * V * V
    # F*(1 - U) = F - F*U ; -(F+k)*V written as a scalar mul (no tensor-scalar rsub)
    dU = (lapU * DU) - uvv + (U * (-F)).adds(F)
    dV = (lapV * DV) + uvv + (V * (-(F + K)))
    Up = U + dU * DT
    Vp = V + dV * DT
    return _c.compile_multi([Up, Vp])


def wordmark(text="ANEForge"):
    """Render `text` to an [N,N] mask in [0,1] (hi-res, anti-aliased down)."""
    from PIL import Image, ImageDraw, ImageFont
    up = 4; W = Hh = N * up
    img = Image.new("L", (W, Hh), 0); d = ImageDraw.Draw(img)
    try:
        path = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
        sz = int(Hh * 0.20)
        while sz > 8:
            f = ImageFont.truetype(path, sz); b = d.textbbox((0, 0), text, font=f)
            if b[2] - b[0] < W * 0.92:
                break
            sz -= 4
    except OSError:
        f = ImageFont.load_default()
    b = d.textbbox((0, 0), text, font=f)
    d.text(((W - (b[2] - b[0])) // 2 - b[0], (Hh - (b[3] - b[1])) // 2 - b[1]),
           text, fill=255, font=f)
    return (np.asarray(img.resize((N, N), Image.LANCZOS), np.float32) / 255.0)


def initial_fields():
    """U = 1 everywhere; V seeded strongly as the wordmark (it reacts first, so the
    brand reads at t=0) over a faint global noise floor (so the labyrinth also
    nucleates across the rest of the frame and fills it, not just a central blob)."""
    try:
        mask = wordmark()
    except Exception:
        rng = np.random.default_rng(0)
        mask = np.zeros((N, N), np.float32)
        for _ in range(20):
            i, j = rng.integers(20, N - 20, 2)
            mask[i - 4:i + 4, j - 4:j + 4] = 1.0
    rng = np.random.default_rng(1)
    floor = 0.06 * rng.random((N, N)).astype(np.float32)        # global nucleation seed
    V = np.maximum(mask * (0.5 + 0.1 * rng.standard_normal((N, N))), floor)
    V = V.clip(0, 1).astype(np.float32)
    U = (1.0 - V).astype(np.float32)
    return U.reshape(1, 1, N, N), V.reshape(1, 1, N, N)


def main():
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - reaction-diffusion on the Apple Neural Engine{R}")
    out(f"  {DIM}Gray-Scott (Turing patterns), one fused conv-stencil program per step on the engine{R}")
    out()

    prog = update_program()
    onames = [n for _, n in prog.output_ports]
    out(f"  {GREY}compile{R} {CHECK} one update program "
        f"{DIM}(3x3 Laplacian conv + reaction terms, periodic in-graph, compiled once){R}")

    U, V = initial_fields()
    out(f"  {GREY}react{R}   {DIM}{STEPS} steps on a {N}x{N} grid, every step on the engine{R}")
    frames = [V[0, 0].copy()]
    ane_t = 0.0
    t0 = time.perf_counter()
    for s in range(STEPS):
        t = time.perf_counter()
        res = prog(U, V)
        ane_t += time.perf_counter() - t
        U = res[onames[0]].reshape(1, 1, N, N)
        V = res[onames[1]].reshape(1, 1, N, N)
        if (s + 1) % FRAME_STRIDE == 0:
            frames.append(V[0, 0].copy())
    wall = time.perf_counter() - t0

    v = V[0, 0]
    bounded = bool(np.isfinite(v).all() and v.max() < 2.0 and v.min() > -0.5)
    structured = bool(v.std() > 0.05)              # a flat field has no pattern
    energy = ANE_RAIL_W * ane_t

    out(f"  {GREY}react{R}   {CHECK} {STEPS} update dispatches on the ANE "
        f"{DIM}({wall:.1f}s wall, {ane_t * 1e3 / STEPS:.2f} ms/step){R}")
    out(f"  {GREY}pattern{R} {DIM}V settled into Turing filaments "
        f"(contrast {v.std():.3f}, in range [{v.min():.2f}, {v.max():.2f}]){R}")
    out(f"  {GREY}energy{R}  {DIM}~{ane_t:.1f}s of ANE step time at the measured "
        f"~{ANE_RAIL_W} W rail ~ {BOLD}{energy:.1f} J{R}{DIM} for the whole run{R}")
    out()

    vmax = max((float(f.max()) for f in frames), default=1.0) or 1.0
    wrote = render(frames, energy, vmax)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    prog.release()

    ok = bounded and structured
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE ran {STEPS} Gray-Scott steps on the "
        f"engine, stable and patterned")
    return 0 if ok else 1


# rendering
def colormap(c):
    vs = np.array([p[0] for p in _CMAP]); cs = np.array([p[1] for p in _CMAP], float)
    v = np.clip(c, 0.0, 1.0) ** 0.85
    return np.stack([np.interp(v, vs, cs[:, j]) for j in range(3)], -1).astype(np.uint8)


def telemetry(im, energy):
    from PIL import Image, ImageDraw, ImageFont
    W, Hh = im.size
    try:
        fmono = "/System/Library/Fonts/SFNSMono.ttf"
        big = ImageFont.truetype(fmono, max(13, int(Hh * 0.085)))
        small = ImageFont.truetype(fmono, max(8, int(Hh * 0.040)))
    except OSError:
        big = small = ImageFont.load_default()
    ov = Image.new("RGBA", (W, Hh), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    jt = f"{energy:.1f} J"; wt = f"{ANE_RAIL_W:.2f} W  on the ANE"
    pad = int(Hh * 0.05); gap = int(Hh * 0.03)
    jh = int(big.size); wh = int(small.size)
    jw = d.textlength(jt, font=big); ww = d.textlength(wt, font=small)
    x = pad; y = Hh - pad - jh - gap - wh
    bx = int(max(jw, ww))
    d.rounded_rectangle([x - 9, y - 8, x + bx + 10, y + jh + gap + wh + 9],
                        radius=9, fill=(8, 11, 16, 150))
    d.text((x, y), jt, font=big, anchor="lt", fill=(255, 196, 92, 255))
    d.text((x, y + jh + gap), wt, font=small, anchor="lt", fill=(150, 168, 178, 235))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def render(frames, total_energy, vmax=1.0):
    try:
        from PIL import Image
    except ImportError:
        out(f"  {DIM}(install Pillow to write the animation: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ims = []
    for i, c in enumerate(frames):
        im = Image.fromarray(colormap(c / vmax)).resize((ANIM_SIZE, ANIM_SIZE), Image.LANCZOS)
        e = total_energy * min(1.0, i * FRAME_STRIDE / STEPS)
        ims.append(telemetry(im, e))
    durs = [ANIM_MS] * len(ims); durs[0] = ANIM_START; durs[-1] = ANIM_HOLD
    ims[0].save(assets / "reaction_diffusion.png", save_all=True, append_images=ims[1:],
                duration=durs, loop=0)
    return f"docs/assets/reaction_diffusion.png (APNG, {len(ims)} frames, full colour)"


if __name__ == "__main__":
    sys.exit(main())
