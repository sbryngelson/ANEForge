"""Full SD UNet2DConditionModel forward as one ANE program from real diffusers weights (tiny config), validated vs diffusers. Run: python3 examples/sd_unet.py"""
import sys
import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np, torch
import aneforge as af
from diffusers import UNet2DConditionModel

GR = 8         # norm_num_groups
HEAD_DIM = 8


def build_unet(x, temb, context, g, has):
    """aneforge graph for the tiny SD UNet2DConditionModel forward.
    ``x`` [1,4,8,8] latent, ``temb`` [1,128] time embedding, ``context`` [77,16] text.
    ``g(key)`` returns a weight array; ``has(key)`` tests presence."""
    def gn(x, p, C): return x.group_norm(g(p + ".weight"), g(p + ".bias"), GR, eps=1e-5)
    def ln(x, p):    return x.layer_norm(g(p + ".weight"), g(p + ".bias"), eps=1e-5)

    def resnet(x, temb, p, Cout):
        h = af.conv(gn(x, p + ".norm1", None).silu(), g(p + ".conv1.weight"), pad=1, bias=g(p + ".conv1.bias"))
        tp = temb.silu().linear(g(p + ".time_emb_proj.weight"), g(p + ".time_emb_proj.bias"))
        h = h + tp.reshape(1, Cout, 1, 1)
        h = af.conv(gn(h, p + ".norm2", None).silu(), g(p + ".conv2.weight"), pad=1, bias=g(p + ".conv2.bias"))
        skip = x
        if has(p + ".conv_shortcut.weight"):
            skip = af.conv(x, g(p + ".conv_shortcut.weight"), bias=g(p + ".conv_shortcut.bias"))
        return skip + h

    def transformer(x, p, C, H):
        heads = C // HEAD_DIM
        res = x
        hh = af.conv(gn(x, p + ".norm", None), g(p + ".proj_in.weight"), bias=g(p + ".proj_in.bias"))
        t = hh.transpose([0, 2, 3, 1]).reshape(H * H, C)
        b = p + ".transformer_blocks.0"
        sa = af.mha(ln(t, b + ".norm1"), g(b + ".attn1.to_q.weight"), None, g(b + ".attn1.to_k.weight"), None,
                    g(b + ".attn1.to_v.weight"), None, g(b + ".attn1.to_out.0.weight"), g(b + ".attn1.to_out.0.bias"), heads)
        t = t + sa
        ca = af.cross_attention(ln(t, b + ".norm2"), context, g(b + ".attn2.to_q.weight"), g(b + ".attn2.to_k.weight"),
                                g(b + ".attn2.to_v.weight"), g(b + ".attn2.to_out.0.weight"), heads, bo=g(b + ".attn2.to_out.0.bias"))
        t = t + ca
        ff = af.geglu(ln(t, b + ".norm3"), g(b + ".ff.net.0.proj.weight"), g(b + ".ff.net.0.proj.bias"))
        t = t + ff.linear(g(b + ".ff.net.2.weight"), g(b + ".ff.net.2.bias"))
        hh = t.reshape(1, H, H, C).transpose([0, 3, 1, 2])
        return af.conv(hh, g(p + ".proj_out.weight"), bias=g(p + ".proj_out.bias")) + res

    h = af.conv(x, g("conv_in.weight"), pad=1, bias=g("conv_in.bias"))      # 32@8
    res = [h]
    # down0: CrossAttnDownBlock2D (resnet + transformer + downsampler)
    h = transformer(resnet(h, temb, "down_blocks.0.resnets.0", 32), "down_blocks.0.attentions.0", 32, 8); res.append(h)
    h = af.conv(h, g("down_blocks.0.downsamplers.0.conv.weight"), stride=2, pad=1, bias=g("down_blocks.0.downsamplers.0.conv.bias")); res.append(h)  # 32@4
    # down1: DownBlock2D (resnet only, final block has no downsampler)
    h = resnet(h, temb, "down_blocks.1.resnets.0", 64); res.append(h)        # 64@4
    # mid: resnet + transformer + resnet
    h = resnet(h, temb, "mid_block.resnets.0", 64)
    h = transformer(h, "mid_block.attentions.0", 64, 4)
    h = resnet(h, temb, "mid_block.resnets.1", 64)
    # up0: UpBlock2D (2 resnets w/ skip-concat + upsampler)
    r = res[-2:]; res = res[:-2]
    h = resnet(af.concat([h, r[1]], 1), temb, "up_blocks.0.resnets.0", 64)
    h = resnet(af.concat([h, r[0]], 1), temb, "up_blocks.0.resnets.1", 64)
    h = af.conv(h.upsample(2), g("up_blocks.0.upsamplers.0.conv.weight"), pad=1, bias=g("up_blocks.0.upsamplers.0.conv.bias"))  # 64@8
    # up1: CrossAttnUpBlock2D (2 (resnet+transformer) w/ skip-concat, no upsampler)
    r = res[-2:]; res = res[:-2]
    h = transformer(resnet(af.concat([h, r[1]], 1), temb, "up_blocks.1.resnets.0", 32), "up_blocks.1.attentions.0", 32, 8)
    h = transformer(resnet(af.concat([h, r[0]], 1), temb, "up_blocks.1.resnets.1", 32), "up_blocks.1.attentions.1", 32, 8)
    # out
    return af.conv(gn(h, "conv_norm_out", None).silu(), g("conv_out.weight"), pad=1, bias=g("conv_out.bias"))


def main():
    torch.manual_seed(0)
    u = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
        norm_num_groups=GR, attention_head_dim=HEAD_DIM).eval()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in u.state_dict().items()}
    has = lambda k: k in sd
    g = lambda k: sd[k]

    sample = torch.randn(1, 4, 8, 8); t = torch.tensor([10]); ctx = torch.randn(1, 77, 16)
    with torch.no_grad():
        emb = u.time_embedding(u.time_proj(t).to(sample.dtype))
        ref = u(sample, t, encoder_hidden_states=ctx).sample.numpy()
    sample_np, emb_np, ctx_np = sample.numpy(), emb.numpy(), ctx.numpy()[0]

    out = build_unet(af.input((1, 4, 8, 8)), af.input((1, 128)), af.input((77, 16)), g, has)
    net = af.compile(out)
    o = net(sample_np, emb_np, ctx_np)
    rel = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
    print(f"FULL SD UNet forward (real diffusers weights): {net.n_ops} ops -> 1 ANE program")
    print(f"  out {o.shape} vs diffusers {ref.shape} | relerr {rel:.4f}  {'OK' if rel < 0.05 else 'MISMATCH'}")


if __name__ == "__main__":
    sys.exit(main())
