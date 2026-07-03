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

# MLPerf's exact reference prompt for the Llama3.1-8B CNN/DailyMail task (mlcommons/inference
# language/llama3.1-8b/download_cnndm.py), fed as a plain COMPLETION string (no chat template) -- the model just
# continues after "Summary:", which is why it neither refuses nor adds a "Here is a summary" preamble.
_PROMPT = ("Summarize the following news article in 128 tokens. Please output the summary only, without any "
           "other text.\n\nArticle:\n{article}\n\nSummary:")


def _clean(text):
    """Strip a leading chat preamble ('Here is a 3-sentence summary:', 'Sure,', '**Summary:**') + markdown bold."""
    t = re.sub(r"^\**\s*(here (is|are)[^\n:]*|sure[^\n:]*|below is[^\n:]*|summary)\s*:?\**\s*\n*", "",
               text.strip(), flags=re.I)
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


# In-harness scoring is the self-contained inline ROUGE: the canonical `rouge_score` (via nltk) cannot share a
# process with the ANE dispatch -- it loads fine but SEGFAULTS at dispatch time. Score canonically out of process
# with `score_rouge.py` (reads the summary/reference pairs this harness writes to the result JSON).
rouge = _rouge_inline
_ROUGE = "inline (no stemmer); canonical via score_rouge.py"


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
        from transformers import AutoTokenizer, GenerationConfig
        self.tok = AutoTokenizer.from_pretrained(model_path)
        try:                                                   # stop on the model's FULL eos set, like vLLM/MLPerf --
            e = GenerationConfig.from_pretrained(model_path).eos_token_id   # Llama-3.1: [128001, 128008, 128009], not
            self.eos = list(e) if isinstance(e, (list, tuple)) else [e]     # just the tokenizer's <|eot_id|> (128009),
        except Exception:                                                   # which never fires in completion mode
            self.eos = [self.tok.eos_token_id]
        self.model = af.load_llm(model_path, compress=compress)
        self.max_input, self.max_new = max_input, max_new
        self.prefill_pad = max_input + 96          # fixed bucket so batched prefill compiles once, reused per article
        self.name = os.path.basename(str(model_path).rstrip("/")) + "-ane-" + (compress or "fp16")

    def _prompt_ids(self, article):
        art = self.tok(article, truncation=True, max_length=self.max_input)["input_ids"]
        text = _PROMPT.format(article=self.tok.decode(art, skip_special_tokens=True))
        return [int(t) for t in self.tok(text)["input_ids"]]   # MLPerf: plain encode (BOS added), NO chat template

    def summarize(self, article):
        ids = self._prompt_ids(article)
        times = []
        t0 = time.perf_counter()
        out = self.model.generate(list(ids), max_new_tokens=self.max_new,
                                  max_len=self.prefill_pad + self.max_new,   # fixed bucket -> decode compiles ONCE
                                  eos_id=self.eos, temperature=0.0, prefill_pad=self.prefill_pad,
                                  on_token=lambda _t: times.append(time.perf_counter()))
        summary = _clean(self.tok.decode(out, skip_special_tokens=True))
        ttft = (times[0] - t0) if times else float("nan")
        dtoks = max(0, len(times) - 1)
        tok_s = dtoks / (times[-1] - times[0]) if dtoks > 0 else float("nan")
        return {"summary": summary, "prompt_tokens": len(ids), "gen_tokens": len(out),
                "ttft_s": ttft, "decode_tok_s": tok_s}


def _save(out, name, rows):
    """Write the result JSON (mean ROUGE + per-row summary/reference). Called at checkpoints and at the end."""
    n = len(rows)
    mean = {k: sum(x[k] for x in rows) / n for k in ("rouge1", "rouge2", "rougeL")} if n else {}
    mtok = (sum(x["decode_tok_s"] for x in rows) / n) if n else float("nan")
    with open(out, "w") as f:
        json.dump({"model": name, "n": n, "mean_rouge": mean, "mean_decode_tok_s": mtok, "rows": rows},
                  f, indent=2, allow_nan=True)


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

    out = args.out or os.path.join(Path(__file__).resolve().parent, "results", "llm_summarize_preview.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    for i, (article, ref) in enumerate(data):
        try:
            r = sut.summarize(article)
        except RuntimeError as e:                    # occasional transient ANE "Program Inference error"; retry once
            print(f"[{i + 1}/{len(data)}] ANE error, retry: {str(e).splitlines()[0][:60]}", flush=True)
            try:
                r = sut.summarize(article)
            except RuntimeError:
                print(f"[{i + 1}/{len(data)}] skipped (ANE error twice)", flush=True); continue
        sc = rouge(r["summary"], ref)
        rows.append({**r, **sc, "reference": ref})   # keep the reference so score_rouge.py can re-score canonically
        print(f"[{i + 1}/{len(data)}] prompt {r['prompt_tokens']} tok, gen {r['gen_tokens']} tok, "
              f"TTFT {r['ttft_s']:.2f}s, {r['decode_tok_s']:.1f} tok/s, "
              f"R1 {sc['rouge1']:.3f} R2 {sc['rouge2']:.3f} RL {sc['rougeL']:.3f}", flush=True)
        print(f"      summary: {r['summary'][:220]}", flush=True)
        if (i + 1) % 20 == 0:
            _save(out, sut.name, rows)               # checkpoint so a long run's partial results survive teardown

    n = len(rows)
    mean = {k: sum(x[k] for x in rows) / n for k in ("rouge1", "rouge2", "rougeL")}
    mtok = sum(x["decode_tok_s"] for x in rows) / n
    print(f"\nmean ROUGE-1 {mean['rouge1']:.3f}  ROUGE-2 {mean['rouge2']:.3f}  ROUGE-L {mean['rougeL']:.3f}"
          f"   |  mean decode {mtok:.1f} tok/s   ({sut.name}, {n} articles, ROUGE: {_ROUGE})")
    print("(subset preview -- not the official 5000-article MLPerf gate)")
    _save(out, sut.name, rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
