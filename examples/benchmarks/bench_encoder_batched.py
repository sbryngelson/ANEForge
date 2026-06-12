"""Batched encoder throughput on the ANE - the serving number for embeddings.

Encoders are the ANE's niche (compute-bound, fp16-tolerant, one forward). This
processes B sequences of the SAME length as a width-B batch (length-bucketing - a
real serving pattern that needs no padding/mask), measuring embeddings/sec vs B.
Batched ANE matmuls are flat in B, so per-stream cost should fall sharply.

Reuses the weights loaded by af.Encoder; builds a batched [B,S,D] encoder graph and
fuses it into one program per (B,S). Correctness is checked against the
transformers fp32 reference; throughput is reported vs B.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \\
        python3 examples/benchmarks/bench_encoder_batched.py
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import aneforge as af

NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_batched(enc: af.Encoder, B: int, S: int) -> af.Model:
    """Batched [B,S,D] encoder: attention in [B,H,S,dh]; linears/LayerNorm in [B*S,D]."""
    D, H, eps = enc.D, enc.H, enc.eps
    dh = D // H; scale = 1.0 / (dh ** 0.5)
    x = af.input((B, S, D))
    h = x
    for w in enc.layers:
        hf = h.reshape(B * S, D)
        q = hf.linear(w["Wq"], w["bq"]).reshape(B, S, H, dh).transpose([0, 2, 1, 3])  # [B,H,S,dh]
        k = hf.linear(w["Wk"], w["bk"]).reshape(B, S, H, dh).transpose([0, 2, 1, 3])
        v = hf.linear(w["Wv"], w["bv"]).reshape(B, S, H, dh).transpose([0, 2, 1, 3])
        a = ((q @ k.transpose([0, 1, 3, 2])) * scale).softmax(-1)                      # [B,H,S,S]
        o = (a @ v).transpose([0, 2, 1, 3]).reshape(B * S, D)                          # [B*S,D]
        attn = o.linear(w["Wo"], w["bo"])
        h1 = (hf + attn).layer_norm(w["ln1w"], w["ln1b"], eps)
        ff = h1.linear(w["Wi"], w["bi"]).gelu().linear(w["Wd"], w["bd"])
        h = (h1 + ff).layer_norm(w["ln2w"], w["ln2b"], eps).reshape(B, S, D)
    return af.compile(h, int8=enc.int8)


def host_embed(enc, ids):  # ids [S] -> [S,D]
    S = len(ids)
    e = enc.word[ids] + enc.pos[np.arange(S)] + enc.typ[0]
    m = e.mean(-1, keepdims=True); v = ((e - m) ** 2).mean(-1, keepdims=True)
    return ((e - m) / np.sqrt(v + enc.eps) * enc.eln_w + enc.eln_b).astype(np.float32)


def main():
    enc = af.Encoder(NAME)
    texts = ["The Apple Neural Engine accelerates neural networks at very low power consumption indeed today",
             "A small striped cat slept peacefully on the warm windowsill all afternoon in the bright sun",
             "Transformer encoders process an entire sequence in a single forward pass without any recurrence",
             "Convolutional neural networks remain the strongest fit for the Apple Neural Engine hardware overall"]
    toks = [np.asarray(enc.tok(t)["input_ids"], dtype=np.int64) for t in texts]
    S = min(len(t) for t in toks)                       # length-bucket: truncate all to common S
    ids = [t[:S] for t in toks]
    X = np.stack([host_embed(enc, i) for i in ids])     # [n, S, D]
    n = len(ids)

    # correctness vs transformers reference (same truncated ids)
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(NAME).eval()
    refs = []
    with torch.no_grad():
        for i in ids:
            o = m(input_ids=torch.tensor(i)[None]).last_hidden_state[0].numpy()
            refs.append((lambda v: v / np.linalg.norm(v))(o.mean(0)))
    refs = np.array(refs)

    net = build_batched(enc, n, S)
    out = net(X)                                         # [n, S, D]
    pooled = out.mean(1); pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    cos = [float(pooled[i] @ refs[i]) for i in range(n)]
    print(f"batched encode (B={n}, S={S}): {net.n_ops} ops -> 1 program")
    print(f"  correctness cosine vs reference: {[round(c,4) for c in cos]}")
    net.release()

    # throughput vs B (B identical sequences; speed is what we measure)
    print(f"\n{'B':>3} | {'ms/batch':>9} | {'embeds/sec':>11} | {'per-stream ms':>13} | {'vs B=1':>7}")
    base = None
    for B in [1, 2, 4, 8, 16]:
        net = build_batched(enc, B, S)
        xb = np.repeat(X[:1], B, axis=0)
        for _ in range(5): net(xb)
        t0 = time.perf_counter()
        for _ in range(50): net(xb)
        dt = (time.perf_counter() - t0) / 50
        net.release()
        eps_ = B / dt
        if base is None: base = eps_
        print(f"{B:>3} | {dt*1e3:>9.2f} | {eps_:>11.0f} | {dt*1e3/B:>13.2f} | {eps_/base:>6.2f}x")
    print("\nBatching amortizes the fixed per-program dispatch across B sequences:")
    print("per-stream cost falls and embeddings/sec rises - encoder serving on the ANE.")


if __name__ == "__main__":
    sys.exit(main())
