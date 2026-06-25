"""ANEForge hero demo: a small transformer that writes about itself, on the ANE.

A small causal char-level transformer (multi-head attention, SwiGLU, RMSNorm)
trains end to end on the Apple Neural Engine - forward, backward, and the Adam
update are all ANE graph programs - then generates text from the trained model,
one character per on-engine forward pass. No CoreML, no GPU.

    python3 examples/demo.py

This is the script recorded for the README animation (docs/assets/demo.tape). It is
a real run on real ANE silicon; the only thing staged is the typewriter pacing.
"""
import sys
import threading
import time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge import autograd as agrad

# a little colour (truecolor; brand teal + rust)
TEAL = "\033[38;2;72;187;170m"
RUST = "\033[38;2;235;130;70m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREY = "\033[38;2;150;150;150m"
R = "\033[0m"
CHECK = f"{TEAL}OK{R}"


def out(s=""):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def spinner(stop, label):
    frames = "|/-\\"
    i = 0
    while not stop.is_set():
        sys.stdout.write(f"\r  {RUST}{frames[i % len(frames)]}{R} {label}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.08)


# the model
TEXT = "the apple neural engine trains a language model on itself. " * 2
D, HEADS, dh, FF, NLAYERS, STEPS, LR = 48, 4, 12, 96, 1, 80, 0.01

chars = sorted(set(TEXT)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
ids = np.array([stoi[c] for c in TEXT], np.int64); S = len(ids) - 1


def onehot(idx):
    o = np.zeros((len(idx), V), np.float32); o[np.arange(len(idx)), idx] = 1.0
    return o


def main():
    target = "the apple neural engine trains a language model on itself."
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}")
    out(f"  {DIM}learn one line of text, then complete it from a prompt - on the Apple Neural Engine{R}")
    out()
    out(f"  {GREY}model{R}  a causal char transformer "
        f"{DIM}(multi-head attention - SwiGLU - RMSNorm){R}")
    out(f"  {GREY}task {R}  learn to write  {TEAL}\"{target}\"{R}")

    rng = np.random.default_rng(0)
    P = lambda sh, s=0.08: agrad.parameter((rng.standard_normal(sh) * s).astype(np.float32))
    W_emb, W_pos, W_out = P((V, D)), P((S, D)), P((D, V))
    fin = agrad.parameter(np.ones((1, D), np.float32))
    blocks = [{"Wq": P((D, D)), "Wk": P((D, D)), "Wv": P((D, D)), "Wo": P((D, D)),
                   "Wg": P((D, FF)), "Wu": P((D, FF)), "Wd": P((FF, D)),
                   "rn1": agrad.parameter(np.ones((1, D), np.float32)),
                   "rn2": agrad.parameter(np.ones((1, D), np.float32))} for _ in range(NLAYERS)]
    params = [W_emb, W_pos, W_out, fin] + [p for b in blocks for p in b.values()]
    heads = lambda t: t.reshape(S, HEADS, dh).transpose([1, 0, 2])
    cmask = np.triu(np.full((S, S), -1e4, np.float32), 1)

    def forward(x, mask):
        h = x @ W_emb + W_pos
        for b in blocks:
            xn = h.rms_norm(b["rn1"])
            q, k, v = heads(xn @ b["Wq"]), heads(xn @ b["Wk"]), heads(xn @ b["Wv"])
            sc = ((q @ k.transpose([0, 2, 1])) * (1.0 / dh ** 0.5) + mask).softmax(-1)
            h = h + (sc @ v).transpose([1, 0, 2]).reshape(S, D) @ b["Wo"]
            hn = h.rms_norm(b["rn2"])
            h = h + ((hn @ b["Wg"]).silu() * (hn @ b["Wu"])) @ b["Wd"]
        return h.rms_norm(fin) @ W_out

    x = af.input((S, V)); y = af.input((S, V)); mask = af.input((S, S)); mask.attrs["value"] = cmask

    # compile (forward + per-parameter backward + Adam update) into ANE programs
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(stop, "compiling into ANE programs ..."), daemon=True)
    t.start()
    t0 = time.time()
    tr = agrad.Trainer(agrad.softmax_cross_entropy(forward(x, mask), y), params, lr=LR,
                       loss_scale=1024.0, optimizer="adam",
                       data_inputs={x: onehot(ids[:S]), y: onehot(ids[1:S + 1]), mask: cmask})
    stop.set(); t.join()
    sys.stdout.write("\r" + " " * 48 + "\r")
    out(f"  {GREY}compile{R} {CHECK} forward + backward + Adam -> ANE programs "
        f"{DIM}({time.time() - t0:.1f}s, one device){R}")
    out()

    # train on the engine
    out(f"  {GREY}train{R}  {DIM}every step runs on the Apple Neural Engine{R}")
    show = {1, 10, 25, 45, STEPS}
    for i in range(1, STEPS + 1):
        tr.step()
        if i in show:
            ce = float(tr.loss())
            bar = TEAL + "#" * max(1, int(min(ce, 2.4) / 2.4 * 22)) + R
            tag = f"  {CHECK} learned" if i == STEPS else ""
            out(f"    step {i:>3}   cross-entropy {ce:6.3f}  {bar}{tag}")
            time.sleep(0.45)
    out()

    # complete the prompt on the engine, one character per on-engine forward pass.
    # the given prompt is dimmed; the model's continuation streams in bright teal.
    def complete(seed, n):
        buf = [stoi[c] for c in seed]
        out(f"  {GREY}prompt{R}      {RUST}>{R} {DIM}{seed}{R}")
        sys.stdout.write(f"  {GREY}completion{R}  {RUST}>{R} {DIM}{seed}{R}{TEAL}")
        sys.stdout.flush()
        per = []
        for _ in range(n):
            w = buf + [0] * (S - len(buf))
            tr.data[x] = onehot(np.array(w, np.int64))
            t1 = time.time()
            o = np.asarray(tr._fwd(*tr._feed(tr._fwd)))
            per.append(time.time() - t1)
            nxt = int(o[len(buf) - 1].argmax()); buf.append(nxt)
            sys.stdout.write(itos[nxt]); sys.stdout.flush()
            time.sleep(0.075)
        sys.stdout.write(f"{R}\n")
        return float(np.median(per)) * 1e3

    out(f"  {GREY}generate{R}  {DIM}give it a fragment; it writes the rest, one char per ANE forward pass{R}")
    ms = complete("the apple", 49)
    out()
    out(f"  {CHECK} {BOLD}trained and generated entirely on the Apple Neural Engine{R}")
    out(f"    {DIM}no CoreML - no GPU - {ms:.1f} ms / token on the engine{R}")
    out()
    tr.release()


if __name__ == "__main__":
    main()
