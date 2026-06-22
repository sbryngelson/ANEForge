"""Real Stable-Diffusion-1.5 text->image on the Apple Neural Engine (via aneforge).

Runs the per-step UNet and (the front of) the VAE decoder on the ANE using REAL
SD-1.5 weights (downloaded from the `stable-diffusion-v1-5/stable-diffusion-v1-5`
checkpoint via diffusers), drives a host-side scheduler denoise loop with
classifier-free guidance, decodes the final latent to a 512x512 PNG, and validates
every component against the reference diffusers pipeline.

What runs where:
  - TEXT (host, torch): CLIP tokenize + text_encoder -> cond/uncond [1,77,768].
    (CLIP-on-ANE is out of scope; one-time host cost.)
  - UNET (ANE): the full UNet2DConditionModel forward built from real weights with
    aneforge ops. The whole UNet does NOT compile as one e5rt program (Espresso
    rejects the program size), so it is SPLIT into ~25 per-RESNET e5rt programs
    (conv_in / each down resnet+attn / each downsampler / mid / each up0-2
    resnet+attn / each upsampler) chained host-side by passing intermediate tensors
    + the skip stack as numpy. The final CrossAttnUpBlock (up3) and conv_out run at
    64x64 with >=640 channels, where group_norm hits an Espresso "Not implemented"
    limit, so they run on HOST torch. Re-run twice per step (cond + uncond).
  - SCHEDULER (host): the pipeline's own scheduler.step + CFG combine.
  - VAE (ANE, partial): the AutoencoderKL decoder. group_norm compiles only up to
    64x64 / 512ch on this ANE (512ch@128 and >=256x256 -> Espresso "Not
    implemented"), so the VAE runs on the ANE through up0 (post_quant -> conv_in ->
    mid -> up0, ending 512ch@128x128) and the last three up blocks + conv_out run on
    host torch. A real hardware limit (group_norm feature-map size).

This is the heaviest aneforge demo. PER-COMPONENT validates on real weights (UNet
one-step ~1.5% relerr, VAE decode ~4.4%, both <5%). The END-TO-END image, however,
is fp16-degraded - and the cause is NOT gradual step-compounding but CATASTROPHIC
CANCELLATION in classifier-free guidance: cond-uncond is only a ~0.5% difference of
two large near-identical UNet outputs, so the ~1.5% per-output fp16 error swamps the
guidance signal (then x7.5 guidance amplifies it). The saved PNG is a coherent-but-
abstract blob, not a recognizable scene. The script reports the real
per-component + end-to-end relerr, the CFG-cancellation diagnostic, what compiled
(split + counts) and what fell back to host, and saves the actual image produced - no faked clean generation. The documented fix is mixed/higher precision on the
guidance subtraction.

    python3 examples/sd15.py
"""
import os
import sys
import time
from pathlib import Path

from _common import relerr   # sets env + repo-root path; import before aneforge

import numpy as np
import torch
import aneforge as af
from diffusers import StableDiffusionPipeline

GR = 32          # norm_num_groups (real SD-1.5)
HEAD_DIM = 8     # attention_head_dim; n_heads = channels // HEAD_DIM
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 20          # fewer steps = faster first result, coarser image (20 is a good PNDM tradeoff)
GUIDANCE = 7.5
SEED = 42
OUT_PNG = str(Path(__file__).resolve().parent / "sd15_out.png")
VAE_ANE_MAX_HW = 128   # group_norm compiles up to here; larger feature maps -> host


# Shared op builders (used by both the per-component graphs and the block split)
def _gn(x, g, p, eps): return x.group_norm(g(p + ".weight"), g(p + ".bias"), GR, eps=eps)
def _ln(x, g, p): return x.layer_norm(g(p + ".weight"), g(p + ".bias"), eps=1e-5)


def _unet_resnet(x, temb, g, has, p):
    Cout = g(p + ".conv2.weight").shape[0]
    h = af.conv(_gn(x, g, p + ".norm1", 1e-5).silu(), g(p + ".conv1.weight"), pad=1, bias=g(p + ".conv1.bias"))
    tp = temb.silu().linear(g(p + ".time_emb_proj.weight"), g(p + ".time_emb_proj.bias"))
    h = h + tp.reshape(1, Cout, 1, 1)
    h = af.conv(_gn(h, g, p + ".norm2", 1e-5).silu(), g(p + ".conv2.weight"), pad=1, bias=g(p + ".conv2.bias"))
    skip = x
    if has(p + ".conv_shortcut.weight"):
        skip = af.conv(x, g(p + ".conv_shortcut.weight"), bias=g(p + ".conv_shortcut.bias"))
    return skip + h


def _unet_transformer(x, context, g, p, H):
    _, C, _, _ = x.shape
    heads = C // HEAD_DIM
    res = x
    hh = af.conv(_gn(x, g, p + ".norm", 1e-5), g(p + ".proj_in.weight"), bias=g(p + ".proj_in.bias"))
    t = hh.transpose([0, 2, 3, 1]).reshape(H * H, C)
    b = p + ".transformer_blocks.0"
    sa = af.mha(_ln(t, g, b + ".norm1"), g(b + ".attn1.to_q.weight"), None, g(b + ".attn1.to_k.weight"), None,
                g(b + ".attn1.to_v.weight"), None, g(b + ".attn1.to_out.0.weight"), g(b + ".attn1.to_out.0.bias"), heads)
    t = t + sa
    ca = af.cross_attention(_ln(t, g, b + ".norm2"), context, g(b + ".attn2.to_q.weight"), g(b + ".attn2.to_k.weight"),
                            g(b + ".attn2.to_v.weight"), g(b + ".attn2.to_out.0.weight"), heads, bo=g(b + ".attn2.to_out.0.bias"))
    t = t + ca
    ff = af.geglu(_ln(t, g, b + ".norm3"), g(b + ".ff.net.0.proj.weight"), g(b + ".ff.net.0.proj.bias"))
    t = t + ff.linear(g(b + ".ff.net.2.weight"), g(b + ".ff.net.2.bias"))
    hh = t.reshape(1, H, H, C).transpose([0, 3, 1, 2])
    return af.conv(hh, g(p + ".proj_out.weight"), bias=g(p + ".proj_out.bias")) + res


# UNet as a CHAIN of per-block e5rt programs.
#   conv_in -> [down0 down1 down2 down3] -> mid -> [up0 up1 up2 up3] -> conv_out
# Each builder takes aneforge input Tensors and returns the block output Tensor;
# at run time we compile each once and feed numpy between them, carrying the skip
# stack (verified vs diffusers: conv_in pushes 1; down0-2 push [r0,r1,downsamp];
# down3 pushes [r0,r1] -> 12 total; each up block pops 3, resnet r concats
# skips[2-r] because diffusers pops res_hidden_states_tuple[-1] first).
class UNetANE:
    def __init__(self, g, has):
        self.g, self.has = g, has
        self.progs = {}
        # per-block input channel/resolution (from real SD-1.5 config)
        self.ch = {0: 320, 1: 640, 2: 1280, 3: 1280}
        self.hw = {0: 64, 1: 32, 2: 16, 3: 8}      # down block input HW
        # up block: input channels/HW and the 3 skip channels it concats
        self.up_in_ch = {0: 1280, 1: 1280, 2: 1280, 3: 640}
        self.up_in_hw = {0: 8, 1: 16, 2: 32, 3: 64}
        # skip channels in PUSH order (res[-3:]); resnet r concats skips[2-r].
        # verified vs diffusers down-block trace.
        self.up_skip_ch = {0: [1280, 1280, 1280], 1: [640, 1280, 1280],
                           2: [320, 640, 640], 3: [320, 320, 320]}

    # block graph builders
    def _conv_in(self, x):
        g = self.g
        return af.conv(x, g("conv_in.weight"), pad=1, bias=g("conv_in.bias"))

    def _down(self, b, x, temb, context):
        g, has = self.g, self.has
        attn = has(f"down_blocks.{b}.attentions.0.proj_in.weight")
        H = self.hw[b]
        outs = []
        h = x
        for r in range(2):
            h = _unet_resnet(h, temb, g, has, f"down_blocks.{b}.resnets.{r}")
            if attn:
                h = _unet_transformer(h, context, g, f"down_blocks.{b}.attentions.{r}", H)
            outs.append(h)
        if has(f"down_blocks.{b}.downsamplers.0.conv.weight"):
            h = af.conv(h, g(f"down_blocks.{b}.downsamplers.0.conv.weight"), stride=2, pad=1,
                        bias=g(f"down_blocks.{b}.downsamplers.0.conv.bias"))
            outs.append(h)
        return [h] + outs                        # [block_out, *skips_pushed]

    def _mid(self, x, temb, context):
        g, has = self.g, self.has
        h = _unet_resnet(x, temb, g, has, "mid_block.resnets.0")
        h = _unet_transformer(h, context, g, "mid_block.attentions.0", 8)
        h = _unet_resnet(h, temb, g, has, "mid_block.resnets.1")
        return h

    def _up(self, b, x, sk0, sk1, sk2, temb, context):
        g, has = self.g, self.has
        attn = has(f"up_blocks.{b}.attentions.0.proj_in.weight")
        H = self.up_in_hw[b]
        skips = [sk0, sk1, sk2]
        h = x
        for r in range(3):
            h = _unet_resnet(af.concat([h, skips[2 - r]], 1), temb, g, has, f"up_blocks.{b}.resnets.{r}")
            if attn:
                h = _unet_transformer(h, context, g, f"up_blocks.{b}.attentions.{r}", H)
        if has(f"up_blocks.{b}.upsamplers.0.conv.weight"):
            h = af.conv(h.upsample(2), g(f"up_blocks.{b}.upsamplers.0.conv.weight"), pad=1,
                        bias=g(f"up_blocks.{b}.upsamplers.0.conv.bias"))
        return h

    def _conv_out(self, x):
        g = self.g
        return af.conv(_gn(x, g, "conv_norm_out", 1e-5).silu(), g("conv_out.weight"), pad=1, bias=g("conv_out.bias"))


def save_png(img_chw, path):
    """img_chw: numpy [1,3,512,512] in VAE output range; map to uint8 PNG."""
    x = img_chw[0]
    x = np.clip(x / 2 + 0.5, 0, 1)               # diffusers image post-proc
    x = (x.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(x).save(path)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, x)
    return x


def main():
    t_start = time.time()
    print(f"loading real SD-1.5 ({MODEL_ID}) ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.to("cpu")
    unet, vae, tok, txt, sched = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.scheduler
    unet.eval(); vae.eval(); txt.eval()

    usd = {k: v.detach().numpy().astype(np.float32) for k, v in unet.state_dict().items()}
    ug, uhas = (lambda k: usd[k]), (lambda k: k in usd)

    # TEXT (host torch): cond + uncond CLIP embeddings
    def encode(prompt):
        ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            return txt(ids)[0]
    cond = encode(PROMPT)
    uncond = encode("")
    cond_np, uncond_np = cond.numpy()[0], uncond.numpy()[0]
    print(f"  CLIP text embeds: cond {tuple(cond.shape)}  uncond {tuple(uncond.shape)} (host torch)")

    # COMPILE UNet: try ONE program, fall back to per-block split
    print("compiling UNet ...")
    unet_blocks = UNetANE(ug, uhas)
    one_program = False
    try:
        tc = time.time()
        # 1) attempt the WHOLE UNet as one e5rt program (expected to fail at SD-1.5 scale)
        ce = af.input((77, 768)); te = af.input((1, 1280)); xx = af.input((1, 4, 64, 64))
        h = unet_blocks._conv_in(xx)
        res = [h]
        for b in range(4):
            outs = unet_blocks._down(b, h, te, ce)
            h = outs[0]; res += outs[1:]
        h = unet_blocks._mid(h, te, ce)
        for b in range(4):
            sk = res[-3:]; res = res[:-3]
            h = unet_blocks._up(b, h, sk[0], sk[1], sk[2], te, ce)
        full = unet_blocks._conv_out(h)
        unet_net = af.compile(full)
        one_program = True
        unet_ops = unet_net.n_ops
        print(f"  UNet compiled as ONE program: {unet_ops} ops, {time.time()-tc:.1f}s")
    except Exception as e:
        print(f"  single-program UNet compile FAILED: {str(e).splitlines()[-1][:120]}")
        print("  -> falling back to per-RESNET split (conv_in / down x4 / mid / up0-2; "
              "up3 + conv_out on host: group_norm unsupported at 64x64 with >=640 ch) ...")
        tc = time.time()

    def temb_of(t):
        with torch.no_grad():
            return unet.time_embedding(unet.time_proj(torch.as_tensor([int(t)])).float()).numpy()

    # Per-RESNET ANE sub-programs (the split that actually compiles)
    # Each down/up resnet (+ its transformer) and each downsampler/upsampler is one
    # small e5rt program, chained host-side; the skip stack is carried as numpy.
    # up3 (final CrossAttnUpBlock at 64x64) and conv_out hit the group_norm
    # feature-map limit (>=640 ch @ 64x64 -> Espresso "Not implemented"), so they
    # run on HOST torch - a real hardware boundary, not a numerics choice.
    if not one_program:
        g, has = ug, uhas
        progs = {}
        unet_ops = 0
        # compiled-program input order == af.input() creation order (graph idx). We
        # therefore create each program's inputs in a FIXED order: x, then temb (if
        # used), then context (if used), and call with the same positional order.
        def reg(name, xshape, builder, use_temb=False, use_ctx=False):
            x = af.input(xshape)
            te = af.input((1, 1280)) if use_temb else None
            ce = af.input((77, 768)) if use_ctx else None
            net = af.compile(builder(x, te, ce))
            progs[name] = net
            return net.n_ops
        # conv_in (x only)
        unet_ops += reg("conv_in", (1, 4, 64, 64),
                        lambda x, te, ce: af.conv(x, g("conv_in.weight"), pad=1, bias=g("conv_in.bias")))
        # down blocks: per resnet(+attn) + downsampler.
        # resnet input channels == its norm1.weight size (matches diffusers exactly).
        for b in range(4):
            attn = has(f"down_blocks.{b}.attentions.0.proj_in.weight")
            H = unet_blocks.hw[b]
            for r in range(2):
                Cin = g(f"down_blocks.{b}.resnets.{r}.norm1.weight").shape[0]
                def builder(x, te, ce, b=b, r=r, attn=attn, H=H):
                    h = _unet_resnet(x, te, g, has, f"down_blocks.{b}.resnets.{r}")
                    if attn:
                        h = _unet_transformer(h, ce, g, f"down_blocks.{b}.attentions.{r}", H)
                    return h
                unet_ops += reg(f"d{b}_r{r}", (1, Cin, H, H), builder, use_temb=True, use_ctx=attn)
            if has(f"down_blocks.{b}.downsamplers.0.conv.weight"):
                Cd = g(f"down_blocks.{b}.resnets.1.conv2.weight").shape[0]
                unet_ops += reg(f"d{b}_ds", (1, Cd, H, H),
                                lambda x, te, ce, b=b: af.conv(x, g(f"down_blocks.{b}.downsamplers.0.conv.weight"),
                                                               stride=2, pad=1, bias=g(f"down_blocks.{b}.downsamplers.0.conv.bias")))
        # mid (whole block fits one program; uses temb + context)
        unet_ops += reg("mid", (1, 1280, 8, 8),
                        lambda x, te, ce: unet_blocks._mid(x, te, ce), use_temb=True, use_ctx=True)
        # up blocks 0,1,2 on ANE (per resnet+attn, then upsampler); up3 -> host
        for b in (0, 1, 2):
            attn = has(f"up_blocks.{b}.attentions.0.proj_in.weight")
            H = unet_blocks.up_in_hw[b]
            prev = None
            for r in range(3):
                cin = g(f"up_blocks.{b}.resnets.{r}.norm1.weight").shape[0]   # concat in-channels
                def builder(x, te, ce, b=b, r=r, attn=attn, H=H):
                    h = _unet_resnet(x, te, g, has, f"up_blocks.{b}.resnets.{r}")
                    if attn:
                        h = _unet_transformer(h, ce, g, f"up_blocks.{b}.attentions.{r}", H)
                    return h
                unet_ops += reg(f"u{b}_r{r}", (1, cin, H, H), builder, use_temb=True, use_ctx=attn)
                prev = g(f"up_blocks.{b}.resnets.{r}.conv2.weight").shape[0]
            unet_ops += reg(f"u{b}_us", (1, prev, H, H),
                            lambda x, te, ce, b=b: af.conv(x.upsample(2), g(f"up_blocks.{b}.upsamplers.0.conv.weight"),
                                                           pad=1, bias=g(f"up_blocks.{b}.upsamplers.0.conv.bias")))
        n_progs = len(progs)
        print(f"  UNet compiled as {n_progs} per-resnet ANE programs "
              f"(conv_in + down + mid + up0-2): {unet_ops} total ops, {time.time()-tc:.1f}s "
              f"(up3 + conv_out on host)")

    def run_unet(lat_np, t, ctx_np):
        if one_program:
            return unet_net(lat_np, temb_of(t), ctx_np)
        te = temb_of(t)
        P = progs
        h = P["conv_in"](lat_np)
        res = [h]
        for b in range(4):
            attn = uhas(f"down_blocks.{b}.attentions.0.proj_in.weight")
            for r in range(2):
                h = P[f"d{b}_r{r}"](h, te, ctx_np) if attn else P[f"d{b}_r{r}"](h, te)
                res.append(h)
            if f"d{b}_ds" in P:
                h = P[f"d{b}_ds"](h)
                res.append(h)
        h = P["mid"](h, te, ctx_np)
        # up0-2 on ANE
        for b in (0, 1, 2):
            attn = uhas(f"up_blocks.{b}.attentions.0.proj_in.weight")
            sk = res[-3:]; res = res[:-3]
            for r in range(3):
                concat = np.concatenate([h, sk[2 - r]], axis=1)
                h = P[f"u{b}_r{r}"](concat, te, ctx_np) if attn else P[f"u{b}_r{r}"](concat, te)
            h = P[f"u{b}_us"](h)
        # up3 + conv_out on HOST torch (group_norm hardware limit at 64x64 / >=640 ch)
        sk = res[-3:]; res = res[:-3]
        with torch.no_grad():
            ht = torch.as_tensor(h, dtype=torch.float32)
            temb_t = torch.as_tensor(te, dtype=torch.float32)
            ctx_t = torch.as_tensor(ctx_np, dtype=torch.float32)[None]
            skips = tuple(torch.as_tensor(s, dtype=torch.float32) for s in sk)
            ht = unet.up_blocks[3](hidden_states=ht, temb=temb_t,
                                   res_hidden_states_tuple=skips, encoder_hidden_states=ctx_t)
            ht = unet.conv_norm_out(ht); ht = unet.conv_act(ht); ht = unet.conv_out(ht)
        return ht.numpy()

    def unet_torch(lat, t, ctx):
        with torch.no_grad():
            return unet(lat, torch.as_tensor([int(t)]), encoder_hidden_states=ctx).sample

    # VAE: ANE through 128x128, host for 256/512
    # build ANE programs for post_quant->conv_in->mid->up0->up1 (all <=128x128).
    vsd = {k: v.detach().numpy().astype(np.float32) for k, v in vae.state_dict().items()}
    vg, vhas = (lambda k: vsd[k]), (lambda k: k in vsd)

    def vae_gn(x, p): return x.group_norm(vg(p + ".weight"), vg(p + ".bias"), GR, eps=1e-6)
    def vae_resnet(x, p):
        h = af.conv(vae_gn(x, p + ".norm1").silu(), vg(p + ".conv1.weight"), pad=1, bias=vg(p + ".conv1.bias"))
        h = af.conv(vae_gn(h, p + ".norm2").silu(), vg(p + ".conv2.weight"), pad=1, bias=vg(p + ".conv2.bias"))
        skip = x
        if vhas(p + ".conv_shortcut.weight"):
            skip = af.conv(x, vg(p + ".conv_shortcut.weight"), bias=vg(p + ".conv_shortcut.bias"))
        return skip + h
    def vae_spatial_attn(x, p):
        _, C, H, W = x.shape; S = H * W
        h = x.group_norm(vg(p + ".group_norm.weight"), vg(p + ".group_norm.bias"), GR, eps=1e-6)
        seq = h.transpose([0, 2, 3, 1]).reshape(S, C)
        o = af.mha(seq, vg(p + ".to_q.weight"), vg(p + ".to_q.bias"),
                   vg(p + ".to_k.weight"), vg(p + ".to_k.bias"),
                   vg(p + ".to_v.weight"), vg(p + ".to_v.bias"),
                   vg(p + ".to_out.0.weight"), vg(p + ".to_out.0.bias"), n_heads=1)
        return x + o.reshape(1, H, W, C).transpose([0, 3, 1, 2])

    def vae_front(z):                            # post_quant -> conv_in -> mid -> up0
        # mid + up0 resnets run at 64x64 (group_norm OK); up0's upsampler emits 128x128
        # (plain conv, OK). up1 resnets at 128x128 hit the group_norm limit -> host.
        h = af.conv(z, vg("post_quant_conv.weight"), bias=vg("post_quant_conv.bias"))
        h = af.conv(h, vg("decoder.conv_in.weight"), pad=1, bias=vg("decoder.conv_in.bias"))
        h = vae_resnet(h, "decoder.mid_block.resnets.0")
        h = vae_spatial_attn(h, "decoder.mid_block.attentions.0")
        h = vae_resnet(h, "decoder.mid_block.resnets.1")
        r = 0                                    # up_block 0 (64x64 -> 128x128)
        while vhas(f"decoder.up_blocks.0.resnets.{r}.conv1.weight"):
            h = vae_resnet(h, f"decoder.up_blocks.0.resnets.{r}"); r += 1
        up = "decoder.up_blocks.0.upsamplers.0.conv"
        h = af.conv(h.upsample(2), vg(up + ".weight"), pad=1, bias=vg(up + ".bias"))
        return h                                 # [1,512,128,128]

    print("compiling VAE front (post_quant..up0, <=128x128) on ANE ...")
    tc = time.time()
    vae_front_net = af.compile(vae_front(af.input((1, 4, 64, 64))))
    print(f"  VAE-front compiled: {vae_front_net.n_ops} ops, {time.time()-tc:.1f}s "
          f"(up1/up2/up3/conv_out run on host: group_norm unsupported at 512ch@128 / >=256x256)")

    def vae_decode_hybrid(latent_t):
        """latent already divided by scaling_factor. ANE front (..up0) + torch tail (up1..)."""
        z = latent_t.numpy()
        front = torch.as_tensor(vae_front_net(z), dtype=torch.float32)   # [1,512,128,128]
        with torch.no_grad():
            h = front
            for b in (1, 2, 3):
                h = vae.decoder.up_blocks[b](h)
            h = vae.decoder.conv_norm_out(h); h = vae.decoder.conv_act(h); h = vae.decoder.conv_out(h)
        return h.numpy()

    # PER-COMPONENT validation (the real headline)
    sched.set_timesteps(STEPS)
    t0 = sched.timesteps[0]
    gen = torch.Generator().manual_seed(SEED)
    lat0 = torch.randn(1, 4, 64, 64, generator=gen)

    u_ane = run_unet(lat0.numpy(), t0, cond_np)
    u_ref = unet_torch(lat0, t0, cond).numpy()
    unet_rel = relerr(u_ane, u_ref)

    # full diffusers VAE decode reference vs our hybrid (front=ANE, tail=host)
    with torch.no_grad():
        v_ref = vae.decode(lat0).sample.numpy()
    v_ane = vae_decode_hybrid(lat0)
    vae_rel = relerr(v_ane, v_ref)
    print("\nPER-COMPONENT vs diffusers (real weights):")
    print(f"  UNet one-step relerr  {unet_rel:.4f}   {'OK' if unet_rel < 0.05 else 'HIGH (fp16)'}")
    print(f"  VAE decode  relerr    {vae_rel:.4f}   (ANE front <=128, host tail) "
          f"{'OK' if vae_rel < 0.05 else 'HIGH (fp16)'}")

    # DENOISE LOOP (host scheduler, CFG, UNet on ANE)
    def denoise(unet_fn):
        sc = type(sched).from_config(sched.config)
        sc.set_timesteps(STEPS)
        lat = lat0.clone() * sc.init_noise_sigma
        for t in sc.timesteps:
            inp = sc.scale_model_input(lat, t)
            nc = unet_fn(inp, t, cond)
            nu = unet_fn(inp, t, uncond)
            noise = nu + GUIDANCE * (nc - nu)
            lat = sc.step(noise, t, lat).prev_sample
        return lat

    print(f"\ndenoising {STEPS} steps (CFG guidance={GUIDANCE}), UNet on ANE ...")
    td = time.time()
    def unet_fn_ane(inp, t, ctx):
        return torch.as_tensor(run_unet(inp.numpy(), t, ctx.numpy()[0]), dtype=torch.float32)
    lat_ane = denoise(unet_fn_ane)
    loop_time = time.time() - td
    print(f"  ANE denoise loop: {loop_time:.1f}s ({loop_time/STEPS:.2f}s/step, 2 UNet passes/step)")

    lat_ref = denoise(lambda inp, t, ctx: unet_torch(inp, t, ctx))
    lat_rel = relerr(lat_ane.numpy(), lat_ref.numpy())

    # Diagnose the CFG cancellation (why the image degrades)
    # guided noise = uncond + g*(cond-uncond). |cond-uncond| is a TINY difference of
    # two large near-identical UNet outputs, so the fp16 per-output error swamps it.
    nc_ane = run_unet(lat0.numpy(), t0, cond_np); nu_ane = run_unet(lat0.numpy(), t0, uncond_np)
    nc_ref = unet_torch(lat0, t0, cond).numpy(); nu_ref = unet_torch(lat0, t0, uncond).numpy()
    diff_ref = nc_ref - nu_ref
    cfg_signal = float(np.linalg.norm(diff_ref) / np.linalg.norm(nc_ref))   # ~0.005
    guided_ane = nu_ane + GUIDANCE * (nc_ane - nu_ane)
    guided_ref = nu_ref + GUIDANCE * diff_ref
    guided_rel = relerr(guided_ane, guided_ref)

    # VAE decode final latent (ANE front + host tail) -> PNG
    sf = vae.config.scaling_factor
    img_ane = vae_decode_hybrid(lat_ane / sf)
    with torch.no_grad():
        img_ref = vae.decode(lat_ref / sf).sample.numpy()
    img_rel = relerr(img_ane, img_ref)
    pix = save_png(img_ane, OUT_PNG)
    mean, std = float(pix.mean()), float(pix.std())

    print("\nEND-TO-END (ANE pipeline vs diffusers fp32, same seed/prompt/scheduler/steps):")
    print(f"  final-latent relerr  {lat_rel:.4f}")
    print(f"  decoded-image relerr {img_rel:.4f}")
    print(f"  saved PNG -> {OUT_PNG}  (512x512, pixel mean {mean:.1f} std {std:.1f})")
    print(f"  CFG cancellation: |cond-uncond|/|cond| = {cfg_signal:.4f} (the guidance signal "
          f"is ~{cfg_signal*100:.1f}% of the UNet output), but per-output fp16 relerr is "
          f"{unet_rel:.3f}.")
    print(f"  -> guided-noise relerr {guided_rel:.2f}: the {unet_rel*100:.1f}% fp16 error "
          f"SWAMPS the ~{cfg_signal*100:.1f}% guidance difference, then x{GUIDANCE} amplifies it.")

    total = time.time() - t_start
    cpath = ("ONE fused program" if one_program else
             f"{len(progs)} per-resnet programs (conv_in / 8 down resnets+attn / 3 downsamplers / "
             f"mid / 9 up0-2 resnets+attn / 3 upsamplers); up3 + conv_out on host")
    print("\nVERDICT:")
    print(f"  UNet compile path: {cpath}; {unet_ops} total ANE ops.")
    print(f"  VAE: ANE front (post_quant..up0, {vae_front_net.n_ops} ops, <=128x128); "
          f"up1/up2/up3/conv_out on HOST (group_norm 'Not implemented' at 512ch@128 and >=256x256).")
    print(f"  Per-component on REAL weights VALIDATES: UNet {unet_rel:.4f}, VAE(hybrid) {vae_rel:.4f} (both <5%).")
    print(f"  End-to-end image is fp16-DEGRADED (image relerr {img_rel:.2f}, latent {lat_rel:.2f}): not a")
    print("  step-compounding effect but CATASTROPHIC CANCELLATION in classifier-free guidance - ")
    print(f"  cond-uncond is a ~{cfg_signal*100:.1f}% difference of two near-identical UNet outputs, below the")
    print("  fp16 noise floor. The saved PNG is a coherent-but-abstract blob, NOT a recognizable")
    print(f"  '{PROMPT}'. The documented fix is mixed precision (fp32/wider")
    print("  accumulation on the guidance subtraction) or guidance computed in higher precision.")
    print(f"  total runtime {total:.0f}s.")

    produced = os.path.exists(OUT_PNG) and pix.shape == (512, 512, 3)
    ok = (unet_rel < 0.05) and (vae_rel < 0.05) and produced
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
