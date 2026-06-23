"""aneforge flagship: a neural cellular automaton TRAINED on the Apple Neural Engine.

Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020): one small
per-cell update rule - a CNN shared across every cell and every step - is trained
so that, from a single live seed pixel, repeated application of the rule grows a
target image. It is morphogenesis as a learned local rule.

The learning runs on the engine. Each training step rolls the rule forward T times,
takes the MSE of the grown image to the target, and backpropagates through the whole
rollout - and every one of those forward and backward steps is a compiled ANE
program. The rollout is gradient-checkpointed (`aneforge.streaming.CheckpointedStack`):
the per-step forward and the per-step backward each compile ONCE and stream over the T
steps, so the rollout's depth does not bound the compile and the grid can be larger
than a single fused forward+backward+Adam program would allow. The optimizer (Adam)
runs host-side over the streamed gradients, the same path as `train_charlm_deep.py`.
The trained rule is then dispatched step by step to grow the image, again on the engine.

The rule, per cell, per step:
  1. perceive   - a fixed depthwise [identity, Sobel_x, Sobel_y] stencil, periodic
                  (wrap) padding built from slices of the state itself;
  2. a 1x1 conv (-> 128) + ReLU, then a 1x1 conv (-> C) initialised to ZERO so the
     initial update is a no-op (the dynamics start as the identity and stay stable);
  3. an alive mask: a cell updates only if it or a neighbour is alive (alpha > 0.1),
     computed as a 3x3 box conv of alpha so growth can spread into empty cells;
  4. residual: state += update * alive.

Two simplifications from the paper, both for a clean on-engine graph: the per-cell
stochastic update is dropped (deterministic updates), and the alive mask uses a
differentiable box-conv-and-sigmoid rather than a max-pool (whose overlapping backward
is not on the engine).

    python3 examples/train_neural_ca.py

Writes docs/assets/neural_ca.png as an APNG (the image growing from the seed, full
24-bit colour) if Pillow is installed; training runs either way.
"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge as af
from aneforge.graph import concat
from aneforge.streaming import CheckpointedStack

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

GRID = 64                 # cellular-automaton grid (checkpointing lets it be this large)
C = 16                    # channels: RGBA (0-3) + 12 hidden
T_TRAIN = 48              # rollout depth per training step (streamed, not fused)
T_GROW = 48               # growth-animation depth (matches the trained horizon)
STEPS = 1500              # training steps
LR = 2e-3
LR_DECAY_AT = 0.6         # drop the learning rate to 1/4 after this fraction of steps
LOSS_SCALE = 2048.0
B1, B2, EPS = 0.9, 0.999, 1e-8
TARGET = "\U0001F98E"     # the lizard, a nod to the Distill demo
ANE_RAIL_W = 1.48         # measured sustained ANE rail, W (see demos/power_efficiency.py)
ANIM_SIZE, ANIM_MS = 384, 90
ANIM_START, ANIM_HOLD = 900, 1600


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


# model                                                                       #

def perception_weight():
    """Fixed depthwise perception: each channel convolved with identity, Sobel_x,
    Sobel_y, giving 3C output channels."""
    ident = np.zeros((3, 3), np.float32); ident[1, 1] = 1.0
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
    sy = sx.T.copy()
    W = np.zeros((3 * C, C, 3, 3), np.float32)
    for c in range(C):
        for j, f in enumerate((ident, sx, sy)):
            W[3 * c + j, c] = f
    return af.conv_param(W)


def wrap_pad(x, p=1):
    """Periodic pad H and W by p using slices of x itself (no constant leaf)."""
    b, c, h, w = x.shape
    x = concat([x.slice_by_size([0, 0, 0, w - p], [b, c, h, p]), x,
                x.slice_by_size([0, 0, 0, 0], [b, c, h, p])], axis=3)
    return concat([x.slice_by_size([0, 0, h - p, 0], [b, c, p, w + 2 * p]), x,
                   x.slice_by_size([0, 0, 0, 0], [b, c, p, w + 2 * p])], axis=2)


_PERC = perception_weight()
_BOX = af.conv_param(np.ones((1, 1, 3, 3), np.float32))


def layer_fn(params, x):
    """One CA step as a graph builder (the `CheckpointedStack` layer). `params` are
    the trainable 1x1-conv weights as flat conv tensors; the fixed perception and box
    convs are module constants. Stamping `conv_shape` lets `conv2d` consume the stack's
    plain parameter leaves as 1x1 conv weights."""
    W1, b1, W2, b2 = params
    W1.attrs["conv_shape"] = (128, 3 * C, 1, 1)
    W2.attrs["conv_shape"] = (C, 128, 1, 1)
    p = af.conv2d(wrap_pad(x), _PERC)                  # [1,3C,G,G]
    h = (af.conv2d(p, W1) + b1).relu()                 # [1,128,G,G]
    ds = af.conv2d(h, W2) + b2                          # [1,C,G,G]
    alpha = x.slice_by_size([0, 3, 0, 0], [1, 1, GRID, GRID])
    life = af.conv2d(wrap_pad(alpha), _BOX)            # local alpha sum
    alive = (life.adds(-0.1) * 50.0).sigmoid()
    return x + ds * alive


def init_params():
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((3 * C, 128)) * np.sqrt(2.0 / (3 * C))).astype(np.float32)
    b1 = np.zeros((1, 128, 1, 1), np.float32)
    W2 = np.zeros((128, C), np.float32)                # zero init -> initial update is a no-op
    b2 = np.zeros((1, C, 1, 1), np.float32)
    return [W1, b1, W2, b2]


# target + seed                                                               #

def target_rgba():
    """Render TARGET to a [1,4,GRID,GRID] premultiplied-RGBA array in [0,1]."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        f = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
        img = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((90, 90), TARGET, font=f, anchor="mm", embedded_color=True)
        img = img.crop(img.getbbox()).resize((GRID, GRID), Image.LANCZOS)
        rgba = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    except Exception:
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        r = np.sqrt((xx - GRID / 2 + 0.5) ** 2 + (yy - GRID / 2 + 0.5) ** 2)
        a = (r < GRID * 0.3).astype(np.float32)
        rgba = np.stack([0.28 * a, 0.73 * a, 0.67 * a, a], 0)
    rgb, a = rgba[:3], rgba[3:4]
    return np.concatenate([rgb * a, a], 0)[None].astype(np.float32)


def seed():
    s = np.zeros((1, C, GRID, GRID), np.float32)
    s[0, 3:, GRID // 2, GRID // 2] = 1.0
    return s


def to_rgb(state):
    """Premultiplied RGBA state -> displayable RGB on black, clipped to [0,1]."""
    return np.clip(state[0, :3].transpose(1, 2, 0), 0, 1)


# training: per-step forward/backward on the engine, Adam on the host           #

def train():
    params = init_params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    out(f"  {GREY}compile{R} {DIM}per-step forward + backward (streamed, compiled once)...{R}")
    t0 = time.perf_counter()
    stack = CheckpointedStack(layer_fn, params, (1, C, GRID, GRID))
    out(f"  {GREY}compile{R} {CHECK} two reusable programs {DIM}({time.perf_counter() - t0:.0f}s; "
        f"the {T_TRAIN}-step rollout streams over them, so depth does not bound the compile){R}")

    Y = target_rgba()
    sd = seed()
    npix = 4 * GRID * GRID
    eng_t = [0.0]                                       # wall time inside ANE programs

    def loss_and_grads():
        t = time.perf_counter()
        final, ckpts = stack.forward([params] * T_TRAIN, sd)
        diff = final[:, :4] - Y
        g_out = np.zeros((1, C, GRID, GRID), np.float32)
        g_out[:, :4] = (2.0 / npix) * diff * LOSS_SCALE
        pgrads, _ = stack.backward([params] * T_TRAIN, ckpts, g_out)
        eng_t[0] += time.perf_counter() - t
        grads = [sum(pgrads[i][k] for i in range(T_TRAIN)) / LOSS_SCALE for k in range(4)]
        return float((diff ** 2).mean()), grads

    out(f"  {GREY}train{R}   {DIM}{STEPS} steps on the engine "
        f"(grid {GRID}x{GRID}, {C} channels), seed -> lizard{R}")
    out(f"\n  {'step':>6} | {'loss':>10}")
    t0 = time.perf_counter()
    loss_val = float("nan")
    for it in range(1, STEPS + 1):
        loss_val, grads = loss_and_grads()
        lr = LR * (0.25 if it > STEPS * LR_DECAY_AT else 1.0)
        for k in range(4):
            m[k] = B1 * m[k] + (1 - B1) * grads[k]
            v[k] = B2 * v[k] + (1 - B2) * grads[k] ** 2
            mh = m[k] / (1 - B1 ** it); vh = v[k] / (1 - B2 ** it)
            params[k] = params[k] - lr * mh / (np.sqrt(vh) + EPS)
        if it % 250 == 0 or it == 1:
            out(f"  {it:>6} | {loss_val:>10.5f}")
    train_wall = time.perf_counter() - t0
    return stack, params, loss_val, train_wall, eng_t[0]


def grow(stack, params, steps):
    """Dispatch the trained rule `steps` times on the engine; the stack's checkpoints
    ARE the per-step states, so they double as the growth frames."""
    t = time.perf_counter()
    final, ckpts = stack.forward([params] * steps, seed())
    eng_t = time.perf_counter() - t
    frames = [to_rgb(c) for c in ckpts] + [to_rgb(final)]
    return frames, final, eng_t


def main():
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - a neural cellular automaton trained on the Apple Neural Engine{R}")
    out(f"  {DIM}forward + backward through a {T_TRAIN}-step rollout on the engine (gradient-checkpointed){R}")
    out()

    stack, params, final_loss, train_wall, train_eng = train()
    out(f"\n  {GREY}train{R}   {CHECK} {STEPS} steps in {train_wall:.0f}s, "
        f"final loss {final_loss:.5f} {DIM}(forward+backward on the engine, Adam host-side){R}")

    out(f"  {GREY}grow{R}    {DIM}dispatching the trained rule {T_GROW} steps on the engine...{R}")
    frames, final, grow_eng = grow(stack, params, T_GROW)
    Y = target_rgba()
    grow_err = float(np.abs(np.concatenate([final[0, :3], final[0, 3:4]], 0)[None] - Y).mean())
    energy = ANE_RAIL_W * (train_eng + grow_eng)
    stack.release()

    out(f"  {GREY}grow{R}    {CHECK} grew the target from one seed cell "
        f"{DIM}(mean abs error {grow_err:.3f} vs target){R}")
    out(f"  {GREY}energy{R}  {DIM}~{BOLD}{energy:.0f} J{R}{DIM} of engine compute "
        f"(forward + backward, {STEPS} steps) at the measured ~{ANE_RAIL_W} W rail{R}")
    out()

    wrote = render(frames, energy)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    out()

    ok = bool(final_loss < 0.01 and grow_err < 0.12 and np.isfinite(final).all())
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE learned a growth rule and grew the "
        f"image from a seed")
    return 0 if ok else 1


# rendering                                                                   #

def telemetry(im, energy):
    from PIL import Image, ImageDraw, ImageFont
    W, Hh = im.size
    try:
        fmono = "/System/Library/Fonts/SFNSMono.ttf"
        big = ImageFont.truetype(fmono, max(13, int(Hh * 0.075)))
        small = ImageFont.truetype(fmono, max(8, int(Hh * 0.038)))
    except OSError:
        big = small = ImageFont.load_default()
    ov = Image.new("RGBA", (W, Hh), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    jt = f"{energy:.0f} J"; wt = "trained on the ANE"
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


def render(frames, total_energy):
    try:
        from PIL import Image
    except ImportError:
        out(f"  {DIM}(install Pillow to write the animation: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ims = []
    for i, c in enumerate(frames):
        rgb = (np.clip(c, 0, 1) * 255).astype(np.uint8)
        # crisp nearest-neighbour upscale: a cellular automaton's cells are the point,
        # so show them as clean squares rather than blurring up from the grid.
        im = Image.fromarray(rgb).resize((ANIM_SIZE, ANIM_SIZE), Image.NEAREST)
        e = total_energy * min(1.0, i / max(1, len(frames) - 1))
        ims.append(telemetry(im, e))
    durs = [ANIM_MS] * len(ims); durs[0] = ANIM_START; durs[-1] = ANIM_HOLD
    ims[0].save(assets / "neural_ca.png", save_all=True, append_images=ims[1:],
                duration=durs, loop=0)
    return f"docs/assets/neural_ca.png (APNG, {len(ims)} frames, full colour)"


if __name__ == "__main__":
    sys.exit(main())
