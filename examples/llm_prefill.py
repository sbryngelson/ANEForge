"""LLM prefill on the Apple Neural Engine -- the compute-bound phase where the ANE is most energy-efficient.
Loads a Llama/Qwen-class model (a Hugging Face name, or a small random demo by default), prefills a prompt
as ONE fused ANE program, and reports prompt-tokens/sec. With `--energy` (needs sudo) it also samples ANE
power via `powermetrics` and reports joules/token. Run: python3 examples/llm_prefill.py [hf-model-name] [--energy]"""
import subprocess
import sys
import time

import _common   # noqa: F401 - sets env + repo-root path; import before aneforge
import numpy as np
import aneforge as af
from aneforge.llm import LlamaConfig, LlamaPrefill, _weights_from_state_dict


def _demo_model():
    """A small random Llama-style model (no download) so the example runs anywhere."""
    cfg = LlamaConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=8, ffn_dim=2048, vocab=4096)
    rng = np.random.default_rng(0); dh = cfg.dim // cfg.n_heads
    R = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32)
    sd = {"model.embed_tokens.weight": R(cfg.vocab, cfg.dim), "model.norm.weight": np.ones(cfg.dim, np.float32),
          "lm_head.weight": R(cfg.vocab, cfg.dim)}
    for L in range(cfg.n_layers):
        p = f"model.layers.{L}."
        for nm, sh in [("self_attn.q_proj", (cfg.n_heads * dh, cfg.dim)), ("self_attn.k_proj", (cfg.n_kv_heads * dh, cfg.dim)),
                       ("self_attn.v_proj", (cfg.n_kv_heads * dh, cfg.dim)), ("self_attn.o_proj", (cfg.dim, cfg.n_heads * dh)),
                       ("mlp.gate_proj", (cfg.ffn_dim, cfg.dim)), ("mlp.up_proj", (cfg.ffn_dim, cfg.dim)), ("mlp.down_proj", (cfg.dim, cfg.ffn_dim))]:
            sd[p + nm + ".weight"] = R(*sh)
        sd[p + "input_layernorm.weight"] = np.ones(cfg.dim, np.float32); sd[p + "post_attention_layernorm.weight"] = np.ones(cfg.dim, np.float32)
    return LlamaPrefill(cfg, _weights_from_state_dict(sd, cfg)), None


def _ane_energy_mj(fn, repeats):
    """Run `fn` `repeats` times under `powermetrics` and return (millijoules of ANE energy, seconds)."""
    proc = subprocess.Popen(["sudo", "-n", "powermetrics", "--samplers", "ane_power", "-i", "50", "--format", "text"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.perf_counter()
    for _ in range(repeats): fn()
    dt = time.perf_counter() - t0
    proc.terminate(); out = proc.stdout.read() if proc.stdout else ""
    mw = [float(l.split(":")[1].strip().split()[0]) for l in out.splitlines() if "ANE Power" in l]
    return (np.mean(mw) * dt if mw else float("nan")), dt          # avg power (mW) * time (s) = mJ


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_energy = "--energy" in sys.argv
    if args:
        print(f"loading {args[0]} from Hugging Face ...")
        model = af.load_llm(args[0]); from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args[0]); ids = tok("The Apple Neural Engine is", return_tensors="np")["input_ids"][0]
    else:
        print("no model name given -> small random demo model (pass a HF Llama/Qwen name for a real one)")
        model, tok = _demo_model(); ids = np.random.default_rng(0).integers(0, model.cfg.vocab, 64)

    S = len(ids); model.compile(S)
    nxt = model.prefill(ids)[0]
    print(f"\nmodel: {model.cfg.n_layers} layers, dim {model.cfg.dim}, {model.cfg.n_heads}h/{model.cfg.n_kv_heads}kv, vocab {model.cfg.vocab}")
    print(f"prefilled {S} tokens -> next-token logits {nxt.shape}, argmax id {int(nxt.argmax())}")
    if tok is not None:
        print(f"  predicted next token: {tok.decode([int(nxt.argmax())])!r}")

    model.prefill(ids)                                             # warm the compile cache
    N = 10; t = time.perf_counter()
    for _ in range(N): model.prefill(ids)
    dt = (time.perf_counter() - t) / N
    print(f"\nprefill: {dt*1000:.1f} ms  ->  {S/dt:,.0f} prompt-tokens/sec  (one fused ANE program)")

    if want_energy:
        mj, _ = _ane_energy_mj(lambda: model.prefill(ids), 50)
        if np.isnan(mj): print("energy: powermetrics needs sudo (run with: sudo python3 examples/llm_prefill.py --energy)")
        else: print(f"energy: {mj/(50*S):.3f} mJ/token of ANE energy ({mj/1000:.2f} J over {50*S} tokens)")


if __name__ == "__main__":
    sys.exit(main())
