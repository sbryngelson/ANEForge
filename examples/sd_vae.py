"""Stable-Diffusion VAE decoder on the ANE - real diffusers weights.

Builds the AutoencoderKL decode path (post_quant_conv -> conv_in -> mid block
[resnet, spatial self-attention, resnet] -> up blocks [resnets + nearest-upsample
+ conv] -> group_norm + silu + conv_out) in aneforge, fuses it into one ANE
program, and validates the decoded image against diffusers.

The VAE is the missing half of a text->image pipeline (the UNet is in
sd_unet.py). Tiny config, same architecture as SD-1.5.

    python3 examples/sd_vae.py
"""
import sys

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge

import numpy as np
import torch
import aneforge as af
from diffusers import AutoencoderKL

GR = 8  # norm_num_groups


def build_decoder(z, g, has):
    """aneforge graph for AutoencoderKL.decode given latent input Tensor ``z``."""
    def gn(x, p):
        return x.group_norm(g(p + ".weight"), g(p + ".bias"), GR, eps=1e-6)

    def resnet(x, p):
        g(p + ".conv2.weight").shape[0]
        h = af.conv(gn(x, p + ".norm1").silu(), g(p + ".conv1.weight"), pad=1, bias=g(p + ".conv1.bias"))
        h = af.conv(gn(h, p + ".norm2").silu(), g(p + ".conv2.weight"), pad=1, bias=g(p + ".conv2.bias"))
        skip = x
        if has(p + ".conv_shortcut.weight"):
            skip = af.conv(x, g(p + ".conv_shortcut.weight"), bias=g(p + ".conv_shortcut.bias"))
        return skip + h

    def spatial_attn(x, p):
        _, C, H, W = x.shape
        S = H * W
        h = x.group_norm(g(p + ".group_norm.weight"), g(p + ".group_norm.bias"), GR, eps=1e-6)
        seq = h.transpose([0, 2, 3, 1]).reshape(S, C)            # [H*W, C]
        o = af.mha(seq, g(p + ".to_q.weight"), g(p + ".to_q.bias"),
                   g(p + ".to_k.weight"), g(p + ".to_k.bias"),
                   g(p + ".to_v.weight"), g(p + ".to_v.bias"),
                   g(p + ".to_out.0.weight"), g(p + ".to_out.0.bias"), n_heads=1)
        return x + o.reshape(1, H, W, C).transpose([0, 3, 1, 2])

    h = af.conv(z, g("post_quant_conv.weight"), bias=g("post_quant_conv.bias"))   # 1x1
    h = af.conv(h, g("decoder.conv_in.weight"), pad=1, bias=g("decoder.conv_in.bias"))
    # mid: resnet -> spatial self-attention -> resnet
    h = resnet(h, "decoder.mid_block.resnets.0")
    h = spatial_attn(h, "decoder.mid_block.attentions.0")
    h = resnet(h, "decoder.mid_block.resnets.1")
    # up blocks: resnets, then nearest-upsample + conv (last block has no upsampler)
    b = 0
    while has(f"decoder.up_blocks.{b}.resnets.0.conv1.weight"):
        r = 0
        while has(f"decoder.up_blocks.{b}.resnets.{r}.conv1.weight"):
            h = resnet(h, f"decoder.up_blocks.{b}.resnets.{r}")
            r += 1
        up = f"decoder.up_blocks.{b}.upsamplers.0.conv"
        if has(up + ".weight"):
            h = af.conv(h.upsample(2), g(up + ".weight"), pad=1, bias=g(up + ".bias"))
        b += 1
    # out
    h = gn(h, "decoder.conv_norm_out").silu()
    return af.conv(h, g("decoder.conv_out.weight"), pad=1, bias=g("decoder.conv_out.bias"))


def main():
    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4, layers_per_block=1, norm_num_groups=GR).eval()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in vae.state_dict().items()}
    g = lambda k: sd[k]
    has = lambda k: k in sd

    z = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        ref = vae.decode(z).sample.numpy()

    out_t = build_decoder(af.input((1, 4, 8, 8)), g, has)
    net = af.compile(out_t)
    out = net(z.numpy())
    rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    print(f"SD VAE decoder (real diffusers weights): {net.n_ops} ops -> 1 ANE program")
    print(f"  decoded image {out.shape} vs diffusers {ref.shape} | relerr {rel:.4f}  "
          f"{'OK' if rel < 0.05 else 'MISMATCH'}")
    return 0 if rel < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
