#!/usr/bin/env python3
"""MLPerf-style LLM summarization on the ANE (quick look).

Mirrors the MLPerf Inference small-LLM benchmark (v5.1: Llama3.1-8B, formerly GPT-J-6B) -- CNN/DailyMail
articles summarized greedily, scored by ROUGE -- run here on the ANE via aneforge's chunked decode. This is a
subset preview to see whether the summaries are coherent and ROUGE is in range; it is NOT the full 5,000-article
edge set, NOT the canonical rouge_score/stemmer, and NOT an official MLPerf submission (see README.md).

  PYTHONPATH=. python3 bench/mlperf/llm_summarize.py                        # cached Qwen3-8B, int8, 4 articles
  PYTHONPATH=. python3 bench/mlperf/llm_summarize.py --n 20 --fp16
  PYTHONPATH=. python3 bench/mlperf/llm_summarize.py --llm ~/Models/Qwen3-8B --n 8 --max-new 128
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_PROMPT = ("Summarize the news article below in three short sentences. State only the key facts from the "
           "article; add no commentary, and do not begin with phrases like \"The article\" or \"Summary\".\n\n"
           "{article}")


def _clean(text):
    """Strip a leading 'Summary:' / '**Summary:**' preamble and markdown bold the chat model tends to add."""
    t = re.sub(r"^\**\s*summary\s*:?\**\s*", "", text.strip(), flags=re.I)
    return t.replace("**", "").strip()


# --- ROUGE: canonical rouge_score (Porter stemmer, as MLPerf uses) if present, else a self-contained F1 ---
def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _rouge_inline(pred, ref):
    def grams(t, n):
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

    def f1n(n):
        p, r = grams(_toks(pred), n), grams(_toks(ref), n)
        overlap, pt, rt = sum((p & r).values()), sum(grams(_toks(pred), n).values()), sum(grams(_toks(ref), n).values())
        prec, rec = (overlap / pt if pt else 0.0), (overlap / rt if rt else 0.0)
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    a, b = _toks(pred), _toks(ref)
    dp = [0] * (len(b) + 1)                                   # ROUGE-L: LCS via rolling DP
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    lcs = dp[-1]
    pl, rl = (lcs / len(a) if a else 0.0), (lcs / len(b) if b else 0.0)
    rougeL = 2 * pl * rl / (pl + rl) if pl + rl else 0.0
    return {"rouge1": f1n(1), "rouge2": f1n(2), "rougeL": rougeL}


try:
    from rouge_score import rouge_scorer as _rs
    _SCORER = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def rouge(pred, ref):
        s = _SCORER.score(ref, pred)
        return {k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}
    _ROUGE = "rouge_score (stemmer)"
except Exception:
    rouge = _rouge_inline
    _ROUGE = "inline (no stemmer)"


def load_cnndm(n):
    """First `n` CNN/DailyMail validation articles (streamed, no full download). Returns [(article, reference)]."""
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation", streaming=True)
    out = []
    for ex in ds:
        out.append((ex["article"], ex["highlights"]))
        if len(out) >= n:
            break
    return out


class SummarizeSUT:
    """An aneforge LLM (chunked ANE decode) + its HF tokenizer, wrapped to summarize one article and time it."""
    def __init__(self, model_path, compress=None, max_input=512, max_new=96):
        import aneforge as af
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = af.load_llm(model_path, compress=compress)
        self.max_input, self.max_new = max_input, max_new
        self.prefill_pad = max_input + 96          # fixed bucket so batched prefill compiles once, reused per article
        self.name = os.path.basename(str(model_path).rstrip("/")) + "-ane-" + (compress or "fp16")

    def _prompt_ids(self, article):
        art = self.tok(article, truncation=True, max_length=self.max_input)["input_ids"]
        text = _PROMPT.format(article=self.tok.decode(art, skip_special_tokens=True))
        msgs = [{"role": "user", "content": text}]
        try:                                                 # Qwen3: no chain-of-thought, just the summary
            s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        except TypeError:
            s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return [int(t) for t in self.tok(s, add_special_tokens=False)["input_ids"]]   # template already has specials

    def summarize(self, article):
        ids = self._prompt_ids(article)
        times = []
        t0 = time.perf_counter()
        out = self.model.generate(list(ids), max_new_tokens=self.max_new, max_len=len(ids) + self.max_new,
                                  eos_id=self.tok.eos_token_id, temperature=0.0, prefill_pad=self.prefill_pad,
                                  on_token=lambda _t: times.append(time.perf_counter()))
        summary = _clean(self.tok.decode(out, skip_special_tokens=True))
        ttft = (times[0] - t0) if times else float("nan")
        dtoks = max(0, len(times) - 1)
        tok_s = dtoks / (times[-1] - times[0]) if dtoks > 0 else float("nan")
        return {"summary": summary, "prompt_tokens": len(ids), "gen_tokens": len(out),
                "ttft_s": ttft, "decode_tok_s": tok_s}


def main():
    ap = argparse.ArgumentParser(description="MLPerf-style LLM summarization on the ANE (quick look)")
    ap.add_argument("--llm", default=os.path.expanduser("~/Models/Qwen3-8B"), help="HF model dir/name")
    ap.add_argument("--n", type=int, default=4, help="articles (subset preview; the edge set is 5000)")
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--fp16", action="store_true", help="fp16 weights (default int8, for memory headroom)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    compress = None if args.fp16 else "int8"
    print(f"loading {os.path.basename(args.llm)} ({compress or 'fp16'}, chunked ANE decode) ...", flush=True)
    sut = SummarizeSUT(args.llm, compress=compress, max_input=args.max_input_tokens, max_new=args.max_new)
    data = load_cnndm(args.n)
    print(f"summarizing {len(data)} CNN/DailyMail articles ...\n", flush=True)

    rows = []
    for i, (article, ref) in enumerate(data):
        r = sut.summarize(article)
        sc = rouge(r["summary"], ref)
        rows.append({**r, **sc})
        print(f"[{i + 1}/{len(data)}] prompt {r['prompt_tokens']} tok, gen {r['gen_tokens']} tok, "
              f"TTFT {r['ttft_s']:.2f}s, {r['decode_tok_s']:.1f} tok/s, "
              f"R1 {sc['rouge1']:.3f} R2 {sc['rouge2']:.3f} RL {sc['rougeL']:.3f}", flush=True)
        print(f"      summary: {r['summary'][:220]}", flush=True)

    n = len(rows)
    mean = {k: sum(x[k] for x in rows) / n for k in ("rouge1", "rouge2", "rougeL")}
    mtok = sum(x["decode_tok_s"] for x in rows) / n
    print(f"\nmean ROUGE-1 {mean['rouge1']:.3f}  ROUGE-2 {mean['rouge2']:.3f}  ROUGE-L {mean['rougeL']:.3f}"
          f"   |  mean decode {mtok:.1f} tok/s   ({sut.name}, {n} articles, ROUGE: {_ROUGE})")
    print("(subset preview -- not the official 5000-article MLPerf gate)")

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", "llm_summarize_preview.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": sut.name, "n": n, "mean_rouge": mean, "mean_decode_tok_s": mtok, "rows": rows},
                  f, indent=2, allow_nan=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
