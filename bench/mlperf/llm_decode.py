"""LLM decode workload for the MLPerf-lite harness: an aneforge LLM (KV-cache decode on the ANE) wrapped as a
SUT that greedily generates tokens and reports the MLPerf-LLM shape -- TTFT (time to first token), TPOT
(per-output-token latency), and output tokens/s.

Greedy decode is token-content-independent per step, so a synthetic prompt of arbitrary token ids gives a valid
throughput/latency measurement without a tokenizer; `eos_id=None` generates the full requested length.
"""
from __future__ import annotations
import glob
import os

import loadgen_lite as lg


class LLMDecodeSUT(lg.SUT):
    """Wrap an aneforge LLM (`LlamaPrefill`-style with `.generate(on_token=...)`) as a decode SUT. `decode`
    timestamps each emitted token via the generate callback and returns (ttft_s, per_token_s, n_tokens)."""
    def __init__(self, model, name, max_len=512):
        self.model = model
        self.name = name
        self.max_len = max_len

    def decode(self, prompt_ids, gen_len, clock):
        times = []
        t0 = clock()
        self.model.generate(list(prompt_ids), max_new_tokens=gen_len, max_len=self.max_len,
                            eos_id=None, on_token=lambda t: times.append(clock()), temperature=0.0)
        ttft = (times[0] - t0) if times else float("nan")          # prefill + first decoded token
        per_token = [times[i] - times[i - 1] for i in range(1, len(times))]
        return ttft, per_token, len(times)


def default_model():
    """A cached small LLM to default to: Qwen3-0.6B under the HF cache or ~/Models."""
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3-0.6B*/snapshots/*"))
    if hits:
        return hits[0]
    p = os.path.expanduser("~/Models/Qwen3-0.6B")
    return p if os.path.isdir(p) else None


def build_sut(model=None, compress=None, max_len=512):
    """Load an LLM for ANE decode and return the SUT (decode program compiled + cached via warmup). `model` is
    an HF name/path; default is a cached Qwen3-0.6B. `compress` ("int8"/...) quantizes the ANE weights."""
    import aneforge as af
    name = model or default_model()
    if not name:
        raise SystemExit("no LLM found; pass --llm <hf-model-or-path> (e.g. a cached Qwen3-0.6B)")
    m = af.load_llm(name, compress=compress)
    m.warmup(max_len)
    label = "qwen3-0.6b" if "Qwen3-0.6B" in name else os.path.basename(name.rstrip("/")) or "llm"
    return LLMDecodeSUT(m, name=f"{label}-ane-" + (compress or "fp16"), max_len=max_len)
