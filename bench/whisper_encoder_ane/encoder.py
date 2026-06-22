"""A whisper-tiny encoder built two ways from one state dict: the HuggingFace
reference, and the equivalent ANEForge graph that runs on the Apple Neural Engine.

Weights are randomly initialised (fixed seed), so nothing is downloaded: latency and
energy are weight-independent, and fidelity only needs the two graphs to share
weights. Pass real whisper-tiny weights (same key names) to run the published
checkpoint.

Mapping notes:
  - The whole transformer stack runs as 2-D [seq, d_model] (batch 1), which is what
    af.mha / af.layer_norm expect.
  - Whisper's conv1d stem maps to af.conv (2-D) with a singleton height axis. af.conv
    pads symmetrically, so the time axis is zero-padded explicitly (_pad_time) and the
    conv runs with pad=0, leaving the singleton axis alone.
  - Whisper's k_proj has no bias (passed as None); attention scale is 1/sqrt(d_head),
    which is af.mha's default.
"""
from __future__ import annotations

import numpy as np
import aneforge as af

# whisper-tiny encoder dimensions.
D, LAYERS, HEADS, FFN, MELS, CTX, FRAMES = 384, 4, 6, 1536, 80, 1500, 3000


def make_encoder(seed: int = 0):
    """A randomly-initialised HF WhisperEncoder (tiny) and its numpy state dict."""
    import torch
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    torch.manual_seed(seed)
    cfg = WhisperConfig(num_mel_bins=MELS, d_model=D, encoder_layers=LAYERS,
                        encoder_attention_heads=HEADS, encoder_ffn_dim=FFN,
                        decoder_layers=LAYERS, decoder_attention_heads=HEADS, decoder_ffn_dim=FFN,
                        max_source_positions=CTX, max_target_positions=448,
                        activation_function="gelu", dropout=0.0, attention_dropout=0.0,
                        scale_embedding=False)
    enc = WhisperEncoder(cfg).eval()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in enc.state_dict().items()}
    return enc, sd


def real_encoder():
    """The trained whisper-tiny encoder and its numpy state dict (downloads the
    checkpoint). Use this for performance numbers: ANE latency is weight-dependent,
    and the trained weights run materially slower than random init."""
    from transformers import WhisperForConditionalGeneration
    enc = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval().model.encoder
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in enc.state_dict().items()}
    return enc, sd


def torch_reference(enc, mel: np.ndarray) -> np.ndarray:
    """The HF encoder output for `mel` ([1, 80, 3000] fp32); returns [1500, 384]."""
    import torch
    with torch.no_grad():
        return enc(torch.from_numpy(mel)).last_hidden_state.numpy()[0]


def mel_input(seed: int = 0) -> np.ndarray:
    """A deterministic random log-mel batch, [1, 80, 3000] fp32."""
    return np.random.default_rng(seed).standard_normal((1, MELS, FRAMES)).astype(np.float32) * 0.5


def _pad_time(x, p: int):
    """Zero-pad the time (width) axis by `p` each side; the singleton height axis is
    left untouched. (af.conv pads symmetrically, which a 1-D conv must avoid; the zeros
    are a width-`p` slice of `x` multiplied by zero, concatenated on either side.)"""
    w = x.shape[3]
    z = af.crop(x, 0, 0, 0, w - p) * 0.0
    return af.concat([z, x, z], axis=3)


def _attention(x, sd, p: str, kind: str):
    wq, bq = sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.q_proj.bias"]
    wk = sd[p + "self_attn.k_proj.weight"]                       # whisper k_proj: no bias
    wv, bv = sd[p + "self_attn.v_proj.weight"], sd[p + "self_attn.v_proj.bias"]
    wo, bo = sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"]
    if kind == "mha":
        return af.mha(x, wq, bq, wk, None, wv, bv, wo, bo, HEADS)
    s, dh = x.shape[0], D // HEADS
    def bhsd(t):
        return t.reshape(s, HEADS, dh).transpose([1, 0, 2]).reshape(1, HEADS, s, dh)
    o = af.sdpa(bhsd(x.linear(wq, bq)), bhsd(x.linear(wk, None)), bhsd(x.linear(wv, bv)))
    return o.reshape(HEADS, s, dh).transpose([1, 0, 2]).reshape(s, D).linear(wo, bo)


def build(sd, attn: str = "mha", build_dir: str | None = None):
    """The whisper-tiny encoder as one ANEForge graph; returns the compiled model.

    attn="mha" uses the decomposed multi-head attention. attn="sdpa" routes through
    af.sdpa, which at seq=1500 also decomposes: the native fused-attention layer is
    reliable only when the smaller attention axis is < 512, so it does not apply here.
    Pass `build_dir` to persist model.mil + weights.bin + the compiled bundle there.
    """
    mel = af.input((1, MELS, 1, FRAMES))              # log-mel, created first -> fed first
    pos = af.input((CTX, D))                          # positional embedding, fed as a constant
    h = _pad_time(mel, 1)
    h = af.conv(h, sd["conv1.weight"].reshape(D, MELS, 1, 3), stride=1, pad=0, bias=sd["conv1.bias"]).gelu()
    h = _pad_time(h, 1)
    h = af.conv(h, sd["conv2.weight"].reshape(D, D, 1, 3), stride=2, pad=0, bias=sd["conv2.bias"]).gelu()
    h = h.reshape(D, CTX).transpose([1, 0]) + pos     # [1, 384, 1, 1500] -> [1500, 384] + pos
    for i in range(LAYERS):
        p = f"layers.{i}."
        xn = h.layer_norm(sd[p + "self_attn_layer_norm.weight"], sd[p + "self_attn_layer_norm.bias"])
        h = h + _attention(xn, sd, p, attn)
        fn = h.layer_norm(sd[p + "final_layer_norm.weight"], sd[p + "final_layer_norm.bias"])
        h = h + fn.linear(sd[p + "fc1.weight"], sd[p + "fc1.bias"]).gelu().linear(sd[p + "fc2.weight"], sd[p + "fc2.bias"])
    h = h.layer_norm(sd["layer_norm.weight"], sd["layer_norm.bias"])
    return af.compile(h, build_dir=build_dir)


def _conv1x1(x, W, b):
    """A Linear [out, in] as a 1x1 conv over channels-first [N, in, 1, S]."""
    return af.conv(x, W.reshape(W.shape[0], W.shape[1], 1, 1), bias=b)


def build_cf(sd, build_dir: str | None = None):
    """The whisper-tiny encoder in the ANE-native channels-first layout: the whole
    transformer stack stays [1, d_model, 1, S], projections are 1x1 convolutions, the
    norm is `channel_layer_norm`, and attention is einsum, so there are no [seq, d]
    transposes. About 3x faster than `build` on the trained checkpoint (it avoids the
    layout the ANE handles poorly), at the same fidelity.

    Inputs are (mel [1, 80, 1, 3000], positional-embedding [1, 384, 1, 1500]); feed with
    `run_cf`. Output is [1500, 384] to match the reference and whisper.cpp's decoder.
    """
    mel = af.input((1, MELS, 1, FRAMES))
    pos = af.input((1, D, 1, CTX))                    # positional embedding, channels-first
    h = _pad_time(mel, 1)
    h = af.conv(h, sd["conv1.weight"].reshape(D, MELS, 1, 3), stride=1, pad=0, bias=sd["conv1.bias"]).gelu()
    h = _pad_time(h, 1)
    h = af.conv(h, sd["conv2.weight"].reshape(D, D, 1, 3), stride=2, pad=0, bias=sd["conv2.bias"]).gelu()
    h = h + pos                                       # already [1, 384, 1, 1500], no transpose
    scale = (D // HEADS) ** -0.5
    for i in range(LAYERS):
        p = f"layers.{i}."
        xn = h.channel_layer_norm(sd[p + "self_attn_layer_norm.weight"], sd[p + "self_attn_layer_norm.bias"])
        q = _conv1x1(xn, sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.q_proj.bias"])
        k = _conv1x1(xn, sd[p + "self_attn.k_proj.weight"], None)
        v = _conv1x1(xn, sd[p + "self_attn.v_proj.weight"], sd[p + "self_attn.v_proj.bias"])
        qh, kh, vh = af.split(q, HEADS, 1), af.split(k, HEADS, 1), af.split(v, HEADS, 1)
        heads = []
        for j in range(HEADS):
            w = (af.einsum("bchq,bchk->bkhq", qh[j], kh[j]) * scale).softmax(1)
            heads.append(af.einsum("bkhq,bchk->bchq", w, vh[j]))
        a = _conv1x1(af.concat(heads, axis=1), sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"])
        h = h + a
        fn = h.channel_layer_norm(sd[p + "final_layer_norm.weight"], sd[p + "final_layer_norm.bias"])
        f = _conv1x1(_conv1x1(fn, sd[p + "fc1.weight"], sd[p + "fc1.bias"]).gelu(),
                     sd[p + "fc2.weight"], sd[p + "fc2.bias"])
        h = h + f
    h = h.channel_layer_norm(sd["layer_norm.weight"], sd["layer_norm.bias"])
    return af.compile(h.reshape(D, CTX).transpose([1, 0]), build_dir=build_dir)


def run(net, sd, mel: np.ndarray) -> np.ndarray:
    """Feed (mel, positional-embedding) in creation order; returns fp32 [1500, 384]."""
    mel4 = mel[:, :, None, :].astype(np.float16)
    pos = sd["embed_positions.weight"].astype(np.float16)
    return net(mel4, pos)


def run_cf(net, sd, mel: np.ndarray) -> np.ndarray:
    """Feed the channels-first encoder (mel, positional-embedding transposed)."""
    mel4 = mel[:, :, None, :].astype(np.float16)
    pos = sd["embed_positions.weight"].T.reshape(1, D, 1, CTX).astype(np.float16)
    return net(mel4, pos)


def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
